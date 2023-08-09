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

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
bool isSmallSharedMemoryPop(const SynapseGroupInternal &sg,
                            const BackendSIMT &backend)
{
    // If shared memory atomics are slow
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);
    if(backend.areSharedMemAtomicsSlow()) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if all neuron groups targetted
    // by synapse groups within merged group are small enough that input to then can be stored in a shared memory array
    else {
        return (sg.getTrgNeuronGroup()->getNumNeurons() <= blockSize);
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpan
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
{
size_t PreSpan::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpan::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase&) const
{
    // Presynaptic parallelism can be used when synapse groups request it and they have sparse connectivity
    return ((sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE));
}
//----------------------------------------------------------------------------
size_t PreSpan::getSharedMemoryPerThread(const SynapseGroupInternal&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, 
                          const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                        const BackendSIMT &backend, unsigned int batchSize, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_evnt";
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(numThreadsPerSpike > 1) {
        env.getStream() << "const unsigned int spike = " << env["id"] << " / " << numThreadsPerSpike << ";" << std::endl;
        env.getStream() << "const unsigned int thread = " << env["id"] << " % " << numThreadsPerSpike << ";" << std::endl;
    }
    else {
        env.getStream() << "const unsigned int spike = " << env["id"] << ";" << std::endl;
    }

    if(sg.getArchetype().isPresynapticOutputRequired()) {
        env.getStream() << sg.getScalarType().getName() << " lrevInSyn = 0.0;" << std::endl;
    }
    
    env.print("if (spike < $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "])");
    {
        CodeStream::Scope b(env.getStream());

        env.printLine("const unsigned int preInd = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, "spike") + "];");

        if(numThreadsPerSpike > 1) {
            env.printLine("unsigned int synAddress = (preInd * $(_row_stride)) + thread;");
        }
        else {
            env.printLine("unsigned int synAddress = preInd * $(_row_stride);");
        }
        env.printLine("const unsigned int npost = $(_row_length)[preInd];");

        /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << "if(";

            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");

            // Generate weight update threshold condition
            sg.generateSpikeEventThreshold(backend, os, modelMerged, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }*/

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
            synEnv.add(Type::Uint32.addConst(), "id_syn", "synAddress");

            synEnv.add(Type::AddToPostDenDelay, "addToPostDelay",
                       backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
            synEnv.add(Type::AddToPost, "addToPost",
                       backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
            synEnv.add(Type::AddToPre, "addToPre", "lrevInSyn += $(0)");
            
            if(trueSpike) {
                sg.generateSpikeUpdate(backend, synEnv, batchSize);
            }
            else {
                sg.generateSpikeEventUpdate(backend, synEnv, batchSize);
            }
            
        }

        /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }*/
        
        // Should this be in the Postamble?
        if(sg.getArchetype().isPresynapticOutputRequired()) {
            // write lrevInSyn to global memory if not 0
            env.printLine("if(lrevInSyn != 0.0) " + backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "preInd") + "], lrevInSyn);");
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
size_t PostSpan::getNumThreads(const SynapseGroupInternal &sg) const
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
size_t PostSpan::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        return sg.getMaxConnections();
    }
    else {
        return sg.getTrgNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
bool PostSpan::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase&) const
{
    // Postsynatic parallelism can be used when synapse groups request it
    return ((sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL)
            && !(sg.getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ));
}
//----------------------------------------------------------------------------
void PostSpan::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                           const BackendSIMT &backend) const
{
    // If synapse group provides any postsynaptic output
    if(sg.getArchetype().isPostsynapticOutputRequired()) {
        // If data structure is dense, we can accumulate output directly into register
        if(shouldAccumulateInRegister(sg)) {
            env.getStream() << sg.getScalarType().getName() << " linSyn = 0;" << std::endl;
        }
        else if(isSmallSharedMemoryPop(sg.getArchetype(), backend)) {
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
size_t PostSpan::getSharedMemoryPerThread(const SynapseGroupInternal &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                         const BackendSIMT &backend, unsigned int batchSize, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_evnt";

    env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "];");
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
            env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) + "];");
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
            env.getStream() << "// only work on existing neurons" << std::endl;
            env.print("if ($(id) < $(_row_stride))");
            {
                CodeStream::Scope b(env.getStream());
                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // If this can only be represented using a 64-bit number
                    if(backend.areSixtyFourBitSynapseIndicesRequired(sg)) {
                        env.printLine("const uint64_t gid = ($(_sh_spk" +  eventSuffix + ")[j] * (uint64_t)$(_row_stride)) + $(id);");
                    }
                    else {
                        env.printLine("const unsigned int gid = ($(_sh_spk" +  eventSuffix + ")[j] * $(_row_stride)) + $(id);");
                    }
                }

                /*if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }*/
                /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    env.getStream() << "if(";
                    if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        env.getStream() << "(B(group->gp[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    sg.generateSpikeEventThreshold(backend, os, modelMerged, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else */if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    env.getStream() << "if (B(" << env["_gp"] << "[gid / 32], gid & 31))" << CodeStream::OB(135);
                }

                EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, sg);

                synEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");
                synEnv.add(Type::Uint32.addConst(), "id_syn", "synAddress",
                           {synEnv.addInitialiser( "const unsigned int synAddress = ($(_sh_spk" + eventSuffix + ")[j] * $(_row_stride)) + $(id);")});

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
                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay",
                           backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
                // If we should accumulate in register, add parameter to register
                if(shouldAccumulateInRegister(sg)) {
                    synEnv.add(Type::AddToPost, "addToPost", "linSyn += $(0)");
                }
                // Otherwise, if we should use shared memory, add to shared memory
                // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                else if(isSmallSharedMemoryPop(sg.getArchetype(), backend)) {
                    synEnv.add(Type::AddToPost, "addToPost", "$(_sh_out_post)[$(id_post)] += $(0)");
                }
                // Otherwise, use global memory atomic
                else {
                    synEnv.add(Type::AddToPost, "addToPost",
                               backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                }

                if(sg.getArchetype().isPresynapticOutputRequired()) {
                    synEnv.add(Type::AddToPre, "addToPre",
                               backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");
                }
                
                if(trueSpike) {
                    sg.generateSpikeUpdate(backend, synEnv, batchSize);
                }
                else {
                    sg.generateSpikeEventUpdate(backend, synEnv, batchSize);
                }

                if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    synEnv.getStream() << CodeStream::CB(140); // end if (id < npost)
                }

                /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else */if(sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    env.getStream() << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
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
        else if(isSmallSharedMemoryPop(sg.getArchetype(), backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
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
    return (!sg.getArchetype().isDendriticDelayRequired()
            && ((matrixType & SynapseMatrixConnectivity::DENSE) || (matrixType & SynapseMatrixConnectivity::BITMASK)));
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpanProcedural
//--------------------------------------------------------------------------
size_t PreSpanProcedural::getNumThreads(const SynapseGroupInternal &sg) const
{
    // Use specified number of threads for each presynaptic neuron
    return (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * (size_t)sg.getNumThreadsPerSpike();
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PreSpanProcedural::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &) const
{
    // Presynaptic procedural parallelism can be used when synapse groups have 
    // procedural connectivity and there are either no variables or variables are PROCEDURAL or KERNEL
    const auto matrixType = sg.getMatrixType();
    return ((matrixType & SynapseMatrixConnectivity::PROCEDURAL)
            && (sg.getWUModel()->getVars().empty() || (matrixType & SynapseMatrixWeight::PROCEDURAL)
                || (matrixType & SynapseMatrixWeight::KERNEL)));
}
//----------------------------------------------------------------------------
size_t PreSpanProcedural::getSharedMemoryPerThread(const SynapseGroupInternal&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genPreamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, 
                                    const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpanProcedural::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                  const BackendSIMT&, unsigned int batchSize, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_evnt";
    const size_t numThreadsPerSpike = sg.getArchetype().getNumThreadsPerSpike();

    if(numThreadsPerSpike > 1) {
        const std::string numThreadsPerSpikeStr = std::to_string(numThreadsPerSpike);
        env.printLine("const unsigned int spike = $(id) / " + numThreadsPerSpikeStr + ";");
        env.printLine("const unsigned int thread = $(id) % " + numThreadsPerSpikeStr + ";");
        env.printLine("const unsigned int numPostPerThread =  ($(num_post) + " + numThreadsPerSpikeStr + " - 1) / " + numThreadsPerSpikeStr + ";");

        // Calculate the starting position and length of the sub-row to process on this thread
        // **TODO** fast-divide style optimisations here
        env.getStream() << "const unsigned int idPostStart = thread * numPostPerThread;" << std::endl;
        env.getStream() << "const unsigned int postRemainder = " << env["num_post"] << " % numPostPerThread;" << std::endl;
        env.getStream() << "const unsigned int numPost = (postRemainder == 0 || thread < " << (numThreadsPerSpike - 1) << ") ? numPostPerThread : postRemainder;" << std::endl;
    }
    else {
        env.printLine("const unsigned int spike = $(id);");
    }

    if(sg.getArchetype().isPresynapticOutputRequired()) {
        env.getStream() << "scalar lrevInSyn = 0.0;" << std::endl;
    }

    // If there is a spike for this thread to process
    env.print("if (spike < $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "])");
    {
        CodeStream::Scope b(env.getStream());

        // Create environment and add presynaptic index
        EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, sg);
        synEnv.add(Type::Uint32.addConst(), "id_pre", "preInd",
                   {synEnv.addInitialiser("const unsigned int preInd = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, "spike") + "];")});

        /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << "if(";

            // Generate weight update threshold condition
            Substitutions threshSubs(&synSubs);
            sg.generateSpikeEventThreshold(backend, os, modelMerged, threshSubs);

            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }*/

        // Create substitution stack for generating procedural connectivity code
        assert(false);
        /*Substitutions connSubs(&synSubs);
        synEnv.add("num_threads", std::to_string(numThreadsPerSpike));

        // If this connectivity requires an RNG for initialisation,
        // make copy of connect Phillox RNG and skip ahead to id that would have been used to initialize any variables associated with it
        if(Utils::isRNGRequired(sg.getArchetype().getConnectivityInitialiser().getSnippet()->getRowBuildCode())
           || ((sg.getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL) && Utils::isRNGRequired(sg.getArchetype().getWUVarInitialisers())))
        {
            std::stringstream skipAhead;
            if(numThreadsPerSpike > 1) {
                skipAhead << "(preInd * " << numThreadsPerSpike << ") + thread";
            }
            else {
                skipAhead << "preInd";
            }
            skipAhead << " + " << connSubs["group_start_id"] << " + " << (backend.getNumInitialisationRNGStreams(modelMerged) * model.getBatchSize());

            // **NOTE** add RNG to synSubs so it can be correctly referenced in presynapticUpdateSubs below
            backend.genGlobalRNGSkipAhead(os, synSubs, skipAhead.str());
        }

        // If we are using more than one thread to process each row
        if(numThreadsPerSpike > 1) {
            connSubs.addVarSubstitution("id_post_begin", "idPostStart");
            connSubs.addVarSubstitution("id_thread", "thread");
            connSubs.addVarSubstitution("num_post", "numPost");
            connSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        }
        else {
            connSubs.addVarSubstitution("id_post_begin", "0");
            connSubs.addVarSubstitution("id_thread", "0");
            connSubs.addVarSubstitution("num_post", "group->numTrgNeurons");
            connSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        }

        // Create another substitution stack for generating presynaptic simulation code
        Substitutions presynapticUpdateSubs(&synSubs);

        // Replace $(id_post) with first 'function' parameter as simulation code is
        // going to be, in turn, substituted into procedural connectivity generation code
        presynapticUpdateSubs.addVarSubstitution("id_post", "$(0)");

        // If weights are provided by a kernel
        if(!sg.getArchetype().getKernelSize().empty()) {
            // Replace kernel indices with the subsequent 'function' parameters
            for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                presynapticUpdateSubs.addVarSubstitution("id_kernel_" + std::to_string(i),
                                                         "$(" + std::to_string(i + 1) + ")");
            }
        }

        // If dendritic delay is required, use atomic operation to update dendritic delay buffer
        if(sg.getArchetype().isDendriticDelayRequired()) {
            presynapticUpdateSubs.addFuncSubstitution("addToInSynDelay", 2, 
                                                      backend.getAtomic(model.getPrecision()) + "(&group->denDelay[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
        }
        // Otherwise, substitute global memory array for $(inSyn)
        else {
            presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, 
                                                      backend.getAtomic(model.getPrecision()) + "(&group->inSyn[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
        }
        
        if(sg.getArchetype().isPresynapticOutputRequired()) {
            synSubs.addFuncSubstitution("addToPre", 1, "lrevInSyn += $(0)");
        }
        
        // Generate presynaptic simulation code into new stringstream-backed code stream
        std::ostringstream presynapticUpdateStream;
        CodeStream presynapticUpdate(presynapticUpdateStream);
        if(trueSpike) {
            sg.generateSpikeUpdate(backend, presynapticUpdate, modelMerged, presynapticUpdateSubs);
        }
        else {
            sg.generateSpikeEventUpdate(backend, presynapticUpdate, modelMerged, presynapticUpdateSubs);
        }

        // When a synapse should be 'added', substitute in presynaptic update code
        connSubs.addFuncSubstitution("addSynapse", 1 + (unsigned int)sg.getArchetype().getKernelSize().size(), presynapticUpdateStream.str());

        // Generate procedural connectivity code
        sg.generateProceduralConnectivity(backend, os, connSubs);

        //if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
        //    os << CodeStream::CB(130);
        //}

        // Should this be in the Postamble?
        if(sg.getArchetype().isPresynapticOutputRequired()) {
            // write lrevInSyn to global memory if not 0
            os << "if(lrevInSyn != 0.0) " << backend.getAtomic(model.getPrecision()) + "(&group->revInSyn[" + sg.getPreISynIndex(batchSize, "preInd") + "], lrevInSyn);" << std::endl;
        }*/

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
size_t PostSpanBitmask::getNumThreads(const SynapseGroupInternal &sg) const
{
    return ceilDivide(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    // Pad each row to a word boundary
    return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), 32);
}
//----------------------------------------------------------------------------
bool PostSpanBitmask::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const
{
    // Postsynaptic bitmask parallelism can be used if bitmask optimisations are enabled and
    // if synapse groups with bitmask connectivity and no dendritic delays request postsynaptic parallelism
    return (preferences.enableBitmaskOptimisations
            && (sg.getSpanType() == SynapseGroup::SpanType::POSTSYNAPTIC)
            && (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK)
            && !sg.isDendriticDelayRequired());
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
size_t PostSpanBitmask::getSharedMemoryPerThread(const SynapseGroupInternal&, const BackendSIMT&) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                const BackendSIMT &backend, unsigned int batchSize, bool trueSpike) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "_evnt";

    // Get blocksize
    const size_t blockSize = backend.getKernelBlockSize(KernelPresynapticUpdate);

    env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "];");
    env.getStream() << "const unsigned int numSpikeBlocks = (numSpikes + " << blockSize << " - 1) / " << blockSize << ";" << std::endl;


    env.printLine("const unsigned int rowWords =  ($(num_post) + 32 - 1) / 32;");
    env.getStream() << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << blockSize << ") + 1 : " << blockSize << ";" << std::endl;

        backend.genSharedMemBarrier(env.getStream());
        env.getStream() << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(env.getStream());
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) + "];");
            env.printLine("$(_sh_spk" + eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
        }
        backend.genSharedMemBarrier(env.getStream());

        env.getStream() << "// loop through all incoming spikes" << std::endl;
        env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(env.getStream());
            env.getStream() << "// only work on existing neurons" << std::endl;
            env.print("if ($(id) < rowWords)");
            {
                CodeStream::Scope b(env.getStream());

                /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if(";

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                    // Generate weight update threshold condition
                    sg.generateSpikeEventThreshold(backend, os, modelMerged, threshSubs);

                    os << ")";
                    os << CodeStream::OB(130);
                }*/

                // Read row word
                env.printLine("uint32_t connectivityWord = $(_gp)[($(_sh_spk" + eventSuffix + ")[j] * rowWords) + $(id)];");

                // While there any bits left
                env.getStream() << "unsigned int ibit = 0;" << std::endl;
                env.getStream() << "while(connectivityWord != 0)";
                {
                    CodeStream::Scope b(env.getStream());
                    EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, sg);

                    // Cound leading zeros (as bits are indexed backwards this is index of next synapse)
                    synEnv.getStream() << "const int numLZ = " << backend.getCLZ() << "(connectivityWord);" << std::endl;

                    // Shift off zeros and the one just discovered
                    // **NOTE** if numLZ == 31, undefined behaviour results in C++, BUT in CUDA this PRESUMABLY emits
                    // In a 'shl' PTX instruction where "Shift amounts greater than the register width N are clamped to N."
                    synEnv.getStream() << "connectivityWord <<= (numLZ + 1);" << std::endl;

                    // Add to bit index
                    synEnv.getStream() << "ibit += numLZ;" << std::endl;

                    synEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");
                    synEnv.add(Type::Uint32.addConst(), "id_post", "ipost",
                               {synEnv.addInitialiser("const unsigned int ipost = ibit + ($(id) * 32);")});


                    synEnv.add(Type::AddToPost, "addToPost",
                        "$(_sh_out_post)[(ibit * " + std::to_string(blockSize) + ") + " + backend.getThreadID() + "] += $(0)");
                    synEnv.add(Type::AddToPre, "addToPre",
                               backend.getAtomic(sg.getScalarType()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");

                    if(trueSpike) {
                        sg.generateSpikeUpdate(backend, synEnv, batchSize);
                    }
                    else {
                        sg.generateSpikeEventUpdate(backend, synEnv, batchSize);
                    }

                    synEnv.getStream() << "ibit++;" << std::endl;
                }


                /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }*/
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
size_t PostSpanToeplitz::getNumThreads(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
size_t PostSpanToeplitz::getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const
{
    return sg.getMaxConnections();
}
//----------------------------------------------------------------------------
bool PostSpanToeplitz::isCompatible(const SynapseGroupInternal &sg, const PreferencesBase&) const
{
    return (sg.getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ);
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                   const BackendSIMT &backend) const
{
    if(isSmallSharedMemoryPop(sg.getArchetype(), backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
        env.print("if(" + backend.getThreadID() + " < $(num_post))");
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            env.printLine("$(_sh_out_post)[" + backend.getThreadID() + "] = 0;");
        }
        backend.genSharedMemBarrier(env.getStream());
    }
}
//----------------------------------------------------------------------------
size_t PostSpanToeplitz::getSharedMemoryPerThread(const SynapseGroupInternal &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                                 const BackendSIMT &backend, unsigned int batchSize, bool trueSpike) const
{
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
        preUpdateEnv.add(Type::AddToPostDenDelay, "addToPostDelay",
                         backend.getAtomic(sg.getScalarType()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
        // If we should use shared memory, add to shared memory
        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
        if(isSmallSharedMemoryPop(sg.getArchetype(), backend)) {
            preUpdateEnv.add(Type::AddToPost, "addToPost", "$(_sh_out_post)[$(id_post)] += $(0)");
        }
        // Otherwise, use global memory atomic
        else {
            preUpdateEnv.add(Type::AddToPost, "addToPost",
                             backend.getAtomic(sg.getScalarType()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
        }

        // Generate spike update
        if(trueSpike) {
            sg.generateSpikeUpdate(backend, preUpdateEnv, 1);
        }
        else {
            sg.generateSpikeEventUpdate(backend, preUpdateEnv, 1);
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
        backend, toeplitzEnv,
        // Within for_each_synapse loops, define add_synapse function and id_pre
        [addSynapseType](auto &env, auto &errorHandler)
        {
            env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "add_synapse", 0}, addSynapseType, errorHandler);
            env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, "id_pre", 0}, Type::Uint32.addConst(), errorHandler);
        },
        [addSynapseType, batchSize, trueSpike, &preUpdateStream, &backend, &sg]
        (auto &env, auto generateBody)
        {
            // Get suffix based on type of events
            const std::string eventSuffix = trueSpike ? "" : "_evnt";

            env.printLine("const unsigned int numSpikes = $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "];");
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
                    env.printLine("const unsigned int spk = $(_src_spk" + eventSuffix + ")[" + sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) + "];");
                    env.printLine("$(_sh_spk" +  eventSuffix + ")[" + backend.getThreadID() + "] = spk;");
                }
                backend.genSharedMemBarrier(env.getStream());

                env.getStream() << "// loop through all incoming spikes" << std::endl;
                env.getStream() << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
                {
                    CodeStream::Scope b(env.getStream());
                    env.getStream() << "// only work on existing neurons" << std::endl;
                    env.print("if ($(id) < $(_row_stride))");
                    {
                        CodeStream::Scope b(env.getStream());
                        EnvironmentExternal bodyEnv(env);

                        // Add presynaptic index
                        bodyEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");

                        // Add function substitution with parameters to add 
                        bodyEnv.add(addSynapseType, "add_synapse", preUpdateStream.str());

                        // Generate body of for_each_synapse loop within this new environment
                        generateBody(bodyEnv);
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
    if(isSmallSharedMemoryPop(sg.getArchetype(), backend) && sg.getArchetype().isPostsynapticOutputRequired()) {
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
