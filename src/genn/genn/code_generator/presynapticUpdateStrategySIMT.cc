#include "code_generator/presynapticUpdateStrategySIMT.h"

// Standard C++ includes
#include <numeric>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/synapseUpdateGroupMerged.h"

using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
bool isSmallSharedMemoryPop(const PresynapticUpdateGroupMerged &sg,
                            const BackendSIMT &backend)
{
    // If shared memory atomics are slow
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);
    if(backend.areSharedMemAtomicsSlow()) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.getArchetype().isDendriticOutputDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if all neuron groups targetted
    // by synapse groups within merged group are small enough that input to then can be stored in a shared memory array
    else {
        return std::all_of(sg.getGroups().cbegin(), sg.getGroups().cend(),
                           [blockSize](const auto &sg)
                           {
                               return (sg.get().getTrgNeuronGroup()->getNumNeurons() <= blockSize);
                           });
    }
}
//----------------------------------------------------------------------------
bool isPresynapticOutputRequired(const PresynapticUpdateGroupMerged &sg, bool trueSpike)
{
    const auto& tokens = trueSpike ? sg.getArchetype().getWUInitialiser().getPreSpikeSynCodeTokens() : sg.getArchetype().getWUInitialiser().getPreEventSynCodeTokens();
    return GeNN::Utils::isIdentifierReferenced("addToPre", tokens);
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpan
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
{
size_t PreSpan::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpan::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return ((sg.getParallelismHint() == SynapseGroup::ParallelismHint::PRESYNAPTIC)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE));
}
//----------------------------------------------------------------------------
size_t PreSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                        unsigned int batchSize, double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_eevnt";
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();

    if(numThreadsPerSpike > 1) {
        env.getStream() << "const unsigned int spike = " << env["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        env.getStream() << "const unsigned int thread = " << env["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
    }
    else {
        env.getStream() << "const unsigned int spike = " << env["id"] << ";" << std::endl;
    }

    if(isPresynapticOutputRequired(sg, trueSpike)) {
        env.getStream() << sg.getScalarType().getName() << " lOutPre = " << Type::writeNumeric(0.0, sg.getScalarType()) << ";" << std::endl;
    }
    
    env.print("if (spike < $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "])");
    {
        CodeStream::Scope b(env.getStream());

        env.printLine("const unsigned int preInd = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "spike") + "];");

        const auto indexType = backend.getSynapseIndexType(sg);
        const auto indexTypeName = indexType.getName();
        if(numThreadsPerSpike > 1) {
            env.printLine(indexTypeName + " synAddress = ((" + indexTypeName + ")preInd * $(_row_stride)) + thread;");
        }
        else {
            env.printLine(indexTypeName + " synAddress = (" + indexTypeName + ")preInd * $(_row_stride);");
        }
        env.printLine("const unsigned int npost = $(_row_length)[preInd];");

        if(numThreadsPerSpike > 1) {
            env.getStream() << "for(unsigned int i = thread; i < npost; i += " << numThreadsPerSpike << ", synAddress += " << numThreadsPerSpike << ")";
        }
        else {
            env.getStream() << "for(unsigned int i = 0; i < npost; i++, synAddress++)";
        }
        {
            CodeStream::Scope b(env.getStream());

            // Create substitution stack for presynaptic simulation code
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, sg);
            synEnv.add(Type::Uint32.addConst(), "id_pre", "preInd");
            synEnv.add(Type::Uint32.addConst(), "id_post", "ipost",
                       {synEnv.addInitialiser("const unsigned int ipost = $(_ind)[synAddress];")});
            synEnv.add(indexType.addConst(), "id_syn", "synAddress");

            synEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay",
                       backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
            synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                       backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
            synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");
            
            if(trueSpike) {
                sg.generateSpikeUpdate(backend, synEnv, batchSize, dt);
            }
            else {
                sg.generateSpikeEventUpdate(backend, synEnv, batchSize, dt);
            }
            
        }

        // Add lOutPre to global memory
        if(isPresynapticOutputRequired(sg, trueSpike)) {
            env.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "preInd") + "], lOutPre);");
        }
        
    }
}
//----------------------------------------------------------------------------
void PreSpan::genPostamble(EnvironmentExternalBase &, PresynapticUpdateGroupMerged&, const BackendSIMT&, unsigned int) const
{
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpan
//----------------------------------------------------------------------------
size_t PostSpan::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // **NOTE** we don't really care about extra padding i.e. stride here
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else if(sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
bool PostSpan::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getParallelismHint() == SynapseGroup::ParallelismHint::POSTSYNAPTIC)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ));
}
//----------------------------------------------------------------------------
void PostSpan::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // If synapse group provides any postsynaptic output
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // If data structure is dense, we can accumulate output directly into register
        if(shouldAccumulateInRegister(sg)) {
            env.getStream() << sg.getScalarType().getName() << " linSyn = 0;" << std::endl;
        }
        else if(isSmallSharedMemoryPop(sg, backend)) {
            env.print("if(" + backend.getThreadID() + " < $(num_post))");
            {
                CodeGenerator::CodeStream::Scope b(env.getStream());
                env.printLine("$(_sh_out_post)[" + backend.getThreadID() + "] = 0;");
            }
            backend.genSharedMemBarrier(env.getStream());
        }
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                         unsigned int batchSize, double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();
    env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "];");
    env.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

    env.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        backend.genSharedMemBarrier(env.getStream());
        env.getStream() << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(env.getStream());
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, index) + "];");
            env.printLine("$(_sh_spk" +  eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                env.printLine("$(_sh_row_length)[" + backend.getThreadID() + "] = $(_row_length)[spk];");
            }
        }
        backend.genSharedMemBarrier(env.getStream());

        env.getStream() << "// loop through all incoming spikes" << std::endl;
        env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(env.getStream());
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> spikeEnv(env, sg);

            spikeEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

            // Create local variable to hold presynaptic output from all threads in warp
            if (isPresynapticOutputRequired(sg, trueSpike)) {
                spikeEnv.printLine(sg.getScalarType().getName() + " lOutPre = " + Type::writeNumeric(0.0, sg.getScalarType()) + ";");
            }

            spikeEnv.getStream() << "// only work on existing neurons" << std::endl;
            spikeEnv.print("if ($(id) < $(_row_stride))");
            {
                CodeStream::Scope b(spikeEnv.getStream());
                EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(spikeEnv, sg);

                const auto indexType = backend.getSynapseIndexType(sg);
                const auto indexTypeName = indexType.getName();

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    synEnv.printLine("const " + indexTypeName + " gid = ((" + indexTypeName + ")$(_sh_spk" +  eventSuffix + ")[j] * $(_row_stride)) + $(id);");
                    
                }

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    synEnv.print("if($(_gp)[gid / 32] & (0x80000000 >> (gid & 31)))");
                    synEnv.getStream() << CodeStream::OB(135);
                }

                synEnv.add(indexType.addConst(), "id_syn", "synAddress",
                           {synEnv.addInitialiser( "const " + indexTypeName + " synAddress = ((" + indexTypeName + ")$(_sh_spk" + eventSuffix + ")[j] * $(_row_stride)) + $(id);")});

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.printLine("const unsigned int npost = $(_sh_row_length)[j];");

                    synEnv.print("if ($(id) < npost)");
                    synEnv.getStream() << CodeStream::OB(140);
                    synEnv.add(Type::Uint32.addConst(), "id_post", "ipost",
                               {synEnv.addInitialiser("const unsigned int ipost = $(_ind)[$(id_syn)];")});
                }
                else { // DENSE
                    synEnv.add(Type::Uint32.addConst(), "id_post", "$(id)");
                }
       
                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                synEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay",
                           backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
                // If we should accumulate in register, add parameter to register
                if(shouldAccumulateInRegister(sg)) {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "linSyn += ($(0))");
                }
                // Otherwise, if we should use shared memory, add to shared memory
                // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                else if(isSmallSharedMemoryPop(sg, backend)) {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "$(_sh_out_post)[$(id_post)] += ($(0))");
                }
                // Otherwise, use global memory atomic
                else {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                               backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                }

                // Add presynaptic output to local variable
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");

                if(trueSpike) {
                    sg.generateSpikeUpdate(backend, synEnv, batchSize, dt);
                }
                else {
                    sg.generateSpikeEventUpdate(backend, synEnv, batchSize, dt);
                }

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.getStream() << CodeStream::CB(140); // end if (id < npost)
                }

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    synEnv.getStream() << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }

            if (isPresynapticOutputRequired(sg, trueSpike)) {
                // Perform warp reduction into first lane
                backend.genWarpReduction(spikeEnv.getStream(), "lOutPre", VarAccessMode::REDUCE_SUM, sg.getScalarType());

                // Issue atomic add on first lane of warp
                spikeEnv.print("if((" + backend.getThreadID() + " % " + std::to_string(backend.getNumLanes()) + ") == 0)");
                {
                    CodeStream::Scope b(spikeEnv.getStream());
                    spikeEnv.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], lOutPre);");
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpan::genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                            const BackendSIMT &backend, unsigned int batchSize) const
{
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // If we should accumulate output directly into register
        if(shouldAccumulateInRegister(sg)) {
            env.getStream() << "// only do this for existing neurons" << std::endl;
            env.print("if ($(id) < $(num_post))");
            {
                CodeStream::Scope b(env.getStream());
                const std::string inSyn = printSubs("$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id)") + "]", env);
                if(sg.getArchetype().isPSModelFused()) {
                    env.getStream() << backend.getAtomic(sg.getScalarType()) << "(&" << inSyn << ", linSyn);" << std::endl;
                }
                else {
                    env.getStream() << inSyn << " += linSyn;" << std::endl;
                }
            }
        }
        // Otherwise, if we should accumulate into shared memory and synapse group provides any postsynaptic output
        else if(isSmallSharedMemoryPop(sg, backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
            backend.genSharedMemBarrier(env.getStream());
            env.print("if(" + backend.getThreadID() + " < $(num_post))");
            {
                CodeGenerator::CodeStream::Scope b(env.getStream());
                env.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, backend.getThreadID()) + "], $(_sh_out_post)[" + backend.getThreadID() + "]);");
            }
        }
    }
}
// ----------------------------------------------------------------------------
bool PostSpan::shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const
{
    // If no dendritic delays are required and data structure is dense, we can accumulate output directly into register
    const auto matrixType = sg.getArchetype().getMatrixType();
    return (!sg.getArchetype().isDendriticOutputDelayRequired()
            && ((matrixType & SynapseMatrixConnectivity::DENSE) || (matrixType & SynapseMatrixConnectivity::BITMASK)));
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpan
//----------------------------------------------------------------------------
size_t PostSpanVectorised::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                         const Type::TypeContext &typeContext) const
{
    // Get vector width
    const size_t vectorWidth = backend.getPresynapticUpdateVectorWidth(sg, typeContext);
    assert(vectorWidth > 1);

    // Pad max connections / num neurons to vector width
    return padSize((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) 
                   ? sg.getMaxConnections() : sg.getTrgNeuronGroup()->getNumNeurons(), 
                   vectorWidth);
}
//----------------------------------------------------------------------------
size_t PostSpanVectorised::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                                      const Type::TypeContext &typeContext) const
{
    // Get vector width
    const size_t vectorWidth = backend.getPresynapticUpdateVectorWidth(sg, typeContext);
    assert(vectorWidth > 1);

    // Pad max connections / num neurons to vector width
    return padSize((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) 
                   ? sg.getMaxConnections() : sg.getTrgNeuronGroup()->getNumNeurons(), 
                   vectorWidth);
}
//----------------------------------------------------------------------------
bool PostSpanVectorised::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                      const Type::TypeContext &typeContext) const
{
    // Postsynatic vectorised parallelism is used on SPARSE and DENSE 
    // connectivity if synapse group is vectorisable and hint is towards postsynaptic parallelism
    return ((backend.getPresynapticUpdateVectorWidth(sg, typeContext) > 1)
            && (sg.getParallelismHint() == SynapseGroup::ParallelismHint::POSTSYNAPTIC)
            && ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE)
                || (sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)));
}
//----------------------------------------------------------------------------
void PostSpanVectorised::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                     const BackendSIMT &backend) const
{
    // If synapse group provides any postsynaptic output
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // If data structure is dense, we can accumulate output directly into register
        if(shouldAccumulateInRegister(sg)) {
            env.getStream() << sg.getScalarType().getName() << " linSyn = 0;" << std::endl;
        }
        else if(isSmallSharedMemoryPop(sg, backend)) {
            env.print("if(" + backend.getThreadID() + " < $(num_post))");
            {
                CodeGenerator::CodeStream::Scope b(env.getStream());
                env.printLine("$(_sh_out_post)[" + backend.getThreadID() + "] = 0;");
            }
            backend.genSharedMemBarrier(env.getStream());
        }
    }
}
//----------------------------------------------------------------------------
size_t PostSpanVectorised::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpanVectorised::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                                   unsigned int batchSize, double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();
    env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "];");
    env.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

    env.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        backend.genSharedMemBarrier(env.getStream());
        env.getStream() << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(env.getStream());
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, index) + "];");
            env.printLine("$(_sh_spk" +  eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
            if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                env.printLine("$(_sh_row_length)[" + backend.getThreadID() + "] = $(_row_length)[spk];");
            }
        }
        backend.genSharedMemBarrier(env.getStream());

        env.getStream() << "// loop through all incoming spikes" << std::endl;
        env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(env.getStream());
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> spikeEnv(env, sg);

            spikeEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

            // Create local variable to hold presynaptic output from all threads in warp
            if (isPresynapticOutputRequired(sg, trueSpike)) {
                spikeEnv.printLine(sg.getScalarType().getName() + " lOutPre = " + Type::writeNumeric(0.0, sg.getScalarType()) + ";");
            }

            spikeEnv.getStream() << "// only work on existing neurons" << std::endl;
            spikeEnv.print("if ($(id) < $(_row_stride))");
            {
                CodeStream::Scope b(spikeEnv.getStream());
                EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(spikeEnv, sg);

                const auto indexType = backend.getSynapseIndexType(sg);
                const auto indexTypeName = indexType.getName();

                synEnv.add(indexType.addConst(), "id_syn", "synAddress",
                           {synEnv.addInitialiser( "const " + indexTypeName + " synAddress = ((" + indexTypeName + ")$(_sh_spk" + eventSuffix + ")[j] * $(_row_stride)) + $(id);")});

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.printLine("const unsigned int npost = $(_sh_row_length)[j];");

                    synEnv.print("if ($(id) < npost)");
                    synEnv.getStream() << CodeStream::OB(140);
                    synEnv.add(Type::Uint32.addConst(), "id_post", "ipost",
                               {synEnv.addInitialiser("const unsigned int ipost = $(_ind)[$(id_syn)];")});
                }
                else { // DENSE
                    synEnv.add(Type::Uint32.addConst(), "id_post", "$(id)");
                }
       
                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                synEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay",
                           backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
                // If we should accumulate in register, add parameter to register
                if(shouldAccumulateInRegister(sg)) {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "linSyn += ($(0))");
                }
                // Otherwise, if we should use shared memory, add to shared memory
                // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                else if(isSmallSharedMemoryPop(sg, backend)) {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "$(_sh_out_post)[$(id_post)] += ($(0))");
                }
                // Otherwise, use global memory atomic
                else {
                    synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                               backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                }

                // Add presynaptic output to local variable
                synEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");

                if(trueSpike) {
                    sg.generateSpikeUpdate(backend, synEnv, batchSize, dt);
                }
                else {
                    sg.generateSpikeEventUpdate(backend, synEnv, batchSize, dt);
                }

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.getStream() << CodeStream::CB(140); // end if (id < npost)
                }
            }

            if (isPresynapticOutputRequired(sg, trueSpike)) {
                // Perform warp reduction into first lane
                backend.genWarpReduction(spikeEnv.getStream(), "lOutPre", VarAccessMode::REDUCE_SUM, sg.getScalarType());

                // Issue atomic add on first lane of warp
                spikeEnv.print("if((" + backend.getThreadID() + " % " + std::to_string(backend.getNumLanes()) + ") == 0)");
                {
                    CodeStream::Scope b(spikeEnv.getStream());
                    spikeEnv.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], lOutPre);");
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpanVectorised::genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                      const BackendSIMT &backend, unsigned int batchSize) const
{
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // If we should accumulate output directly into register
        if(shouldAccumulateInRegister(sg)) {
            env.getStream() << "// only do this for existing neurons" << std::endl;
            env.print("if ($(id) < $(num_post))");
            {
                CodeStream::Scope b(env.getStream());
                const std::string inSyn = printSubs("$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id)") + "]", env);
                if(sg.getArchetype().isPSModelFused()) {
                    env.getStream() << backend.getAtomic(sg.getScalarType()) << "(&" << inSyn << ", linSyn);" << std::endl;
                }
                else {
                    env.getStream() << inSyn << " += linSyn;" << std::endl;
                }
            }
        }
        // Otherwise, if we should accumulate into shared memory and synapse group provides any postsynaptic output
        else if(isSmallSharedMemoryPop(sg, backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
            backend.genSharedMemBarrier(env.getStream());
            env.print("if(" + backend.getThreadID() + " < $(num_post))");
            {
                CodeGenerator::CodeStream::Scope b(env.getStream());
                env.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, backend.getThreadID()) + "], $(_sh_out_post)[" + backend.getThreadID() + "]);");
            }
        }
    }
}
// ----------------------------------------------------------------------------
bool PostSpanVectorised::shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const
{
    // If no dendritic delays are required and data structure is dense, we can accumulate output directly into register
    const auto matrixType = sg.getArchetype().getMatrixType();
    return (!sg.getArchetype().isDendriticOutputDelayRequired()
            && (matrixType & SynapseMatrixConnectivity::DENSE));
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpanProcedural
//--------------------------------------------------------------------------
size_t PreSpanProcedural::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpanProcedural::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Presynaptic procedural parallelism can be used when synapse groups have 
    // procedural connectivity and there are either no variables or variables are PROCEDURAL or KERNEL
    const auto matrixType = sg.getMatrixType();
    return ((matrixType & SynapseMatrixConnectivity::PROCEDURAL)
            && (sg.getWUInitialiser().getSnippet()->getVars().empty() || (matrixType & SynapseMatrixWeight::PROCEDURAL)
                || (matrixType & SynapseMatrixWeight::KERNEL)));
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPreamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, 
                                    const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                                  unsigned int batchSize, double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();
    const std::string numThreadsPerSpikeStr = std::to_string(numThreadsPerSpike);
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();

    EnvironmentExternal groupEnv(env);
    if(numThreadsPerSpike > 1) {
        groupEnv.add(Type::Uint32.addConst(), "_spike", "spike",
                     {groupEnv.addInitialiser("const unsigned int spike = $(id) / " + numThreadsPerSpikeStr + ";")});
        groupEnv.add(Type::Uint32.addConst(), "_thread", "thread",
                     {groupEnv.addInitialiser("const unsigned int thread = $(id) % " + numThreadsPerSpikeStr + ";")});
    }
    else {
        groupEnv.add(Type::Uint32.addConst(), "_spike", "$(id)");
    }

    // If there is a spike for this thread to process
    groupEnv.print("if ($(_spike) < $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "])");
    {
        CodeStream::Scope b(groupEnv.getStream());
        
        if(isPresynapticOutputRequired(sg, trueSpike)) {
            groupEnv.getStream() << sg.getScalarType().getName() << " lOutPre = " << Type::writeNumeric(0.0, sg.getScalarType()) << ";" << std::endl;
        }

        // Create environment and add presynaptic index
        EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(groupEnv, sg);
        synEnv.add(Type::Uint32.addConst(), "id_pre", "preInd",
                   {synEnv.addInitialiser("const unsigned int preInd = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(_spike)") + "];")});

        // **YUCK** add a hidden copy of num_post so we can overwrite deeper in here without losing access to original
        synEnv.add(Type::Uint32.addConst(), "_num_post", "$(num_post)");

        // If this connectivity requires an RNG for initialisation, make copy of connect Phillox RNG
        // and skip ahead to id that would have been used to initialize any variables associated with it
        if(Utils::isRNGRequired(sg.getArchetype().getSparseConnectivityInitialiser().getRowBuildCodeTokens())
           || ((sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) && Utils::isRNGRequired(sg.getArchetype().getWUInitialiser().getVarInitialisers())))
        {
            std::ostringstream skipAhead;
            if(numThreadsPerSpike > 1) {
                skipAhead << "($(id_pre) * " << numThreadsPerSpike << ") + $(_thread)";
            }
            else {
                skipAhead << "$(id_pre)";
            }

            // **FIXME**
            skipAhead << " + " << "$(_group_start_id) + " << (0/*backend.getNumInitialisationRNGStreams(modelMerged)*/ * batchSize);

            synEnv.add(Type::Void, "_rng", backend.genGlobalRNGSkipAhead(synEnv.getStream(), 
                       printSubs(skipAhead.str(), synEnv)));
        }

        // Create environment for generating presynaptic update code into seperate CodeStream
        std::ostringstream preUpdateStream;
        CodeStream preUpdate(preUpdateStream);
        {
            CodeStream::Scope b(preUpdate);
            EnvironmentExternal preUpdateEnv(synEnv, preUpdate);

            // Replace $(id_post) with first 'function' parameter as simulation code is
            // going to be, in turn, substituted into Toeplitz connectivity generation code
            // **YUCK** we need to do this in an initialiser so the $(0) doesn't get confused with those used in AddToXXXX
            preUpdateEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                             {preUpdateEnv.addInitialiser("const unsigned int idPost = $(0);")});

            // Replace kernel indices with the subsequent 'function' parameters
            // **YUCK** these also need doing in initialisers so the $(1) doesn't get confused with those used in addToPostDelay
            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                const std::string iStr = std::to_string(i);
                preUpdateEnv.add(Type::Uint32.addConst(), "id_kernel_" + iStr, "idKernel" + iStr,
                                 {preUpdateEnv.addInitialiser("const unsigned int idKernel" + iStr + " = $(" + std::to_string(i + 1) + ");")});
            }
                    
            // Add correct functions for applying synaptic input
            preUpdateEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay",
                             backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
            preUpdateEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                             backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
            preUpdateEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");

            // Generate spike update
            if(trueSpike) {
                sg.generateSpikeUpdate(backend, preUpdateEnv, batchSize, dt);
            }
            else {
                sg.generateSpikeEventUpdate(backend, preUpdateEnv, batchSize, dt);
            }
        }

        {
            // Create second environment for initialising procedural connectivity
            EnvironmentExternal connEnv(synEnv);
            connEnv.add(Type::Uint32.addConst(), "num_threads", numThreadsPerSpikeStr);

            // If we are using more than one thread to process each row
            if(numThreadsPerSpike > 1) {
                // Calculate the starting position and length of the sub-row to process on this thread
                // **TODO** fast-divide style optimisations here
                const size_t numPostPerThreadInit = connEnv.addInitialiser(
                    "const unsigned int numPostPerThread =  ($(_num_post) + " + numThreadsPerSpikeStr + " - 1) / " + numThreadsPerSpikeStr + ";");

                connEnv.add(Type::Uint32.addConst(), "id_post_begin", "idPostBegin",
                            {numPostPerThreadInit,
                             connEnv.addInitialiser("const unsigned int idPostBegin = $(_thread) * numPostPerThread;")});
                connEnv.add(Type::Uint32.addConst(), "id_thread", "$(_thread)");
                connEnv.add(Type::Uint32.addConst(), "num_post", "numPost",
                            {numPostPerThreadInit,
                             connEnv.addInitialiser("const unsigned int postRemainder = $(_num_post) % numPostPerThread;"),
                             connEnv.addInitialiser("const unsigned int numPost = (postRemainder == 0 || thread < " + std::to_string(numThreadsPerSpike - 1) + ") ? numPostPerThread : postRemainder;")});
            }
            else {
                connEnv.add(Type::Uint32.addConst(), "id_post_begin", "0");
                connEnv.add(Type::Uint32.addConst(), "id_thread", "0");
            }

            // When a synapse should be 'added', substitute in presynaptic update code
            const auto addSynapseType = Type::ResolvedType::createFunction(
                Type::Void, std::vector<Type::ResolvedType>{1ull + sg.getArchetype().getKernelSize().size(), Type::Uint32});
            connEnv.add(addSynapseType, "addSynapse", preUpdateStream.str());

            // Generate procedural connectivity code
            sg.generateProceduralConnectivity(connEnv);

        }

        // Write sum of presynaptic output to global memory
        if(isPresynapticOutputRequired(sg, trueSpike)) {
            groupEnv.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], lOutPre);");
        }

    }
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPostamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, 
                                     const BackendSIMT&, unsigned int) const
{
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanBitmask
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Pad each row to a word boundary
    return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
bool PostSpanBitmask::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    // Postsynaptic bitmask parallelism can be used if bitmask optimisations are enabled and
    // if synapse groups with bitmask connectivity and no dendritic delays request postsynaptic parallelism
    return ((sg.getParallelismHint() == SynapseGroup::ParallelismHint::WORD_PACKED_BITMASK)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)
            && !sg.isDendriticOutputDelayRequired());
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                  const BackendSIMT &backend) const
{
    // If synapse group provides any postsynaptic output
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // Loop through bits written by this thread
        for(size_t i = 0; i < 32; i++) {
            // Zero entries in this thread's shared memory array
            // **NOTE** this is ordered to prevent bank conflicts
            env.printLine("$(_sh_out_post)[" + std::to_string(i * backend.getKernelBlockSize(KernelPresynapticUpdate)) + " + " + backend.getThreadID() + "] = 0;");
        }
        backend.genSharedMemBarrier(env.getStream());
    }
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                                unsigned int batchSize, double dt, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_event";

    // Get blocksize
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();

    env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "];");
    env.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;


    env.printLine("const unsigned int rowWords =  $(_row_stride) / 32;");
    env.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << blockSize << ") + 1 : " << blockSize << ";" << std::endl;

        backend.genSharedMemBarrier(env.getStream());
        env.getStream() << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(env.getStream());
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, index) + "];");
            env.printLine("$(_sh_spk" + eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
        }
        backend.genSharedMemBarrier(env.getStream());

        env.getStream() << "// loop through all incoming spikes" << std::endl;
        env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(env.getStream());
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> spikeEnv(env, sg);
            spikeEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

            // Create local variable to hold presynaptic output from all threads in warp
            if (isPresynapticOutputRequired(sg, trueSpike)) {
                spikeEnv.printLine(sg.getScalarType().getName() + " lOutPre = " + Type::writeNumeric(0.0, sg.getScalarType()) + ";");
            }
            
            spikeEnv.getStream() << "// only work on existing neurons" << std::endl;
            spikeEnv.print("if ($(id) < rowWords)");
            {
                CodeStream::Scope b(spikeEnv.getStream());

                // Read row word
                spikeEnv.printLine("uint32_t connectivityWord = $(_gp)[($(_sh_spk" + eventSuffix + ")[j] * rowWords) + $(id)];");

                // While there any bits left
                env.getStream() << "unsigned int ibit = 0;" << std::endl;
                spikeEnv.getStream() << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(spikeEnv.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(spikeEnv, sg);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    synEnv.getStream() << "const int numLZ = " << backend.getCLZ() << "(connectivityWord);" << std::endl;

                    // Shift off zeros and the one just discovered
                    // **NOTE** if numLZ == 31, undefined behaviour results in C++, BUT in CUDA this PRESUMABLY emits
                    // a 'shl' PTX instruction where "Shift amounts greater than the register width N are clamped to N."
                    synEnv.getStream() << "connectivityWord <<= (numLZ + 1);" << std::endl;

                    // Add to bit index
                    synEnv.getStream() << "ibit += numLZ;" << std::endl;

                    synEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

                    // **YUCK** add inner environment so id_post initialisation is inserted at correct place
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> postEnv(synEnv, sg);
                    postEnv.add(Type::Uint32.addConst(), "id_post", "ipost",
                                {postEnv.addInitialiser("const unsigned int ipost = ibit + ($(id) * 32);")});

                    postEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                                "$(_sh_out_post)[(ibit * " + std::to_string(blockSize) + ") + " + backend.getThreadID() + "] += ($(0))");
                    postEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");

                    if(trueSpike) {
                        sg.generateSpikeUpdate(backend, postEnv, batchSize, dt);
                    }
                    else {
                        sg.generateSpikeEventUpdate(backend, postEnv, batchSize, dt);
                    }

                    postEnv.getStream() << "ibit++;" << std::endl;
                }
            }

            if (isPresynapticOutputRequired(sg, trueSpike)) {
                // Perform warp reduction into first lane
                backend.genWarpReduction(spikeEnv.getStream(), "lOutPre", VarAccessMode::REDUCE_SUM, sg.getScalarType());

                // Issue atomic add on first lane of warp
                spikeEnv.print("if((" + backend.getThreadID() + " % " + std::to_string(backend.getNumLanes()) + ") == 0)");
                {
                    CodeStream::Scope b(spikeEnv.getStream());
                    spikeEnv.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], lOutPre);");
                }
            }
        }
    }
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   const BackendSIMT &backend, unsigned int batchSize) const
{
    // If synapse group provides any postsynaptic output
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        backend.genSharedMemBarrier(env.getStream());
        const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

        // Use first 32 threads in each block to write shared memory back to global memory
        env.getStream() << "if (" << backend.getThreadID() << " < 32)";
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("unsigned int glbIdx = ((" + backend.getBlockID() + " - ($(_group_start_id) / " + std::to_string(blockSize) + ")) * " + std::to_string(32 * blockSize) + ") + " + backend.getThreadID() + ";");
            env.getStream() << "unsigned int shIdx = " << backend.getThreadID() << " * " << blockSize << ";" << std::endl;
            env.getStream() << "const unsigned int endShIdx = shIdx + 32;" << std::endl;
            env.print("for(;shIdx < endShIdx && glbIdx < $(num_post); shIdx++, glbIdx += 32)");
            {
                CodeStream::Scope b(env.getStream());
                const std::string inSyn = "$(_out_post)[" + sg.getPostISynIndex(batchSize, "glbIdx") +"]";
                if(sg.getArchetype().isPSModelFused()) {
                    env.printLine(backend.getAtomic(sg.getScalarType()) + "(&" + inSyn + ", $(_sh_out_post)[shIdx]);");
                }
                else {
                    env.printLine(inSyn + " += $(_sh_out_post)[shIdx];");
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanToeplitz
//--------------------------------------------------------------------------
size_t PostSpanToeplitz::getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
size_t PostSpanToeplitz::getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PostSpanToeplitz::isCompatible(const SynapseGroupInternal &sg, const BackendSIMT&, const Type::TypeContext&) const
{
    return (sg.getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ);
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   const BackendSIMT &backend) const
{
    if(isSmallSharedMemoryPop(sg, backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
        env.print("if(" + backend.getThreadID() + " < $(num_post))");
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            env.printLine("$(_sh_out_post)[" + backend.getThreadID() + "] = 0;");
        }
        backend.genSharedMemBarrier(env.getStream());
    }
}
//----------------------------------------------------------------------------
size_t PostSpanToeplitz::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                                 unsigned int batchSize, double dt, bool trueSpike) const
{
    const bool delay = trueSpike ? sg.getArchetype().getSrcNeuronGroup()->isSpikeQueueRequired() : sg.getArchetype().getSrcNeuronGroup()->isSpikeEventQueueRequired();

    // Create environment for generating presynaptic update code into seperate CodeStream
    std::ostringstream preUpdateStream;
    CodeStream preUpdate(preUpdateStream);
    {
        CodeStream::Scope b(preUpdate);
        EnvironmentExternal preUpdateEnv(env, preUpdate);
        preUpdateEnv.add(Type::Uint32.addConst(), "id_pre", "ipre");

        // Replace $(id_post) with first 'function' parameter as simulation code is
        // going to be, in turn, substituted into Toeplitz connectivity generation code
        // **YUCK** we need to do this in an initialiser so the $(0) doesn't get confused with those used in AddToXXXX
        preUpdateEnv.add(Type::Uint32.addConst(), "id_post", "idPost",
                         {preUpdateEnv.addInitialiser("const unsigned int idPost = $(0);")});

        // Replace kernel indices with the subsequent 'function' parameters
        // **YUCK** these also need doing in initialisers so the $(1) doesn't get confused with those used in addToPostDelay
        for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
            const std::string iStr = std::to_string(i);
            preUpdateEnv.add(Type::Uint32.addConst(), "id_kernel_" + iStr, "idKernel" + iStr,
                             {preUpdateEnv.addInitialiser("const unsigned int idKernel" + iStr + " = $(" + std::to_string(i + 1) + ");")});
        }
                    
        // Add correct functions for apply synaptic input
        preUpdateEnv.add(Type::getAddToPrePostDelay(sg.getScalarType()), "addToPostDelay",
                         backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
        // If we should use shared memory, add to shared memory
        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
        if(isSmallSharedMemoryPop(sg, backend)) {
            preUpdateEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost", "$(_sh_out_post)[$(id_post)] += ($(0))");
        }
        // Otherwise, use global memory atomic
        else {
            preUpdateEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPost",
                             backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
        }

        preUpdateEnv.add(Type::getAddToPrePost(sg.getScalarType()), "addToPre", "lOutPre += ($(0))");

        // Generate spike update
        if(trueSpike) {
            sg.generateSpikeUpdate(backend, preUpdateEnv, 1, dt);
        }
        else {
            sg.generateSpikeEventUpdate(backend, preUpdateEnv, 1, dt);
        }
    }

    // Create second environment for initialising Toeplitz connectivity
    EnvironmentExternal toeplitzEnv(env);
    toeplitzEnv.add(Type::Uint32.addConst(), "id_diag", "$(id)");
            
    // Define type
    const auto addSynapseType = Type::ResolvedType::createFunction(
        Type::Void, std::vector<Type::ResolvedType>{1ull + sg.getArchetype().getKernelSize().size(), Type::Uint32});

    // Generate toeplitz connectivity generation code using custom for_each_synapse loop
    sg.generateToeplitzConnectivity(
        toeplitzEnv,
        // Within for_each_synapse loops, define addSynapse function and id_pre
        [addSynapseType](auto &env, auto &errorHandler)
        {
            env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "addSynapse", 0}, addSynapseType, errorHandler);
            env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_pre", 0}, Type::Uint32.addConst(), errorHandler);
        },
        [addSynapseType, batchSize, delay, trueSpike, &preUpdateStream, &backend, &sg]
        (auto &env, auto generateBody)
        {
            // Get suffix based on type of events
            const std::string eventSuffix = trueSpike ? "" : "_event";

            env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(delay, batchSize) + "];");
            env.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

            env.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
            {
                CodeStream::Scope b(env.getStream());
                env.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

                backend.genSharedMemBarrier(env.getStream());
                env.getStream() << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
                {
                    CodeStream::Scope b(env.getStream());
                    const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
                    env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(delay, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, index) + "];");
                    env.printLine("$(_sh_spk" +  eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
                }
                backend.genSharedMemBarrier(env.getStream());

                env.getStream() << "// loop through all incoming spikes" << std::endl;
                env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                {
                    CodeStream::Scope b(env.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> spikeEnv(env, sg);
                    spikeEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

                    // Create local variable to hold presynaptic output from all threads in warp
                    if (isPresynapticOutputRequired(sg, trueSpike)) {
                        spikeEnv.printLine(sg.getScalarType().getName() + " lOutPre = " + Type::writeNumeric(0.0, sg.getScalarType()) + ";");
                    }

                    spikeEnv.getStream() << "// only work on existing neurons" << std::endl;
                    spikeEnv.print("if ($(id) < $(_row_stride))");
                    {
                        CodeStream::Scope b(spikeEnv.getStream());
                        EnvironmentExternal bodyEnv(spikeEnv);

                        // Add function substitution with parameters to add 
                        bodyEnv.add(addSynapseType, "addSynapse", preUpdateStream.str());

                        // Generate body of for_each_synapse loop within this new environment
                        generateBody(bodyEnv);
                    }

                    if (isPresynapticOutputRequired(sg, trueSpike)) {
                        // Perform warp reduction into first lane
                        backend.genWarpReduction(spikeEnv.getStream(), "lOutPre", VarAccessMode::REDUCE_SUM, sg.getScalarType());

                        // Issue atomic add on first lane of warp
                        spikeEnv.print("if((" + backend.getThreadID() + " % " + std::to_string(backend.getNumLanes()) + ") == 0)");
                        {
                            CodeStream::Scope b(spikeEnv.getStream());
                            spikeEnv.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], lOutPre);");
                        }
                    }
                }
            }
        });
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                    const BackendSIMT &backend, unsigned int batchSize) const
{
    // If we should accumulate into shared memory and synapse group provides postsynaptic output
    if(isSmallSharedMemoryPop(sg, backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
        backend.genSharedMemBarrier(env.getStream());
        env.print("if(" + backend.getThreadID() + " < $(num_post))");
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            const std::string idx = sg.getPostISynIndex(batchSize, backend.getThreadID());
            env.printLine(backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + idx + "], $(_sh_out_post)[" + backend.getThreadID() + "]);");
        }
    }
}
}   // namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
