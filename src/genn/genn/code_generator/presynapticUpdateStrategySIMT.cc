#include "code_generator/presynapticUpdateStrategySIMT.h"

// Standard C++ includes
#include <numeric>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/modelSpecMerged.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
bool isSmallSharedMemoryPop(const GeNN::CodeGenerator::PresynapticUpdateGroupMerged &sg,
                            const GeNN::CodeGenerator::BackendSIMT &backend)
{
    // If shared memory atomics are slow
    const size_t blockSize = backend.getKernelBlockSize(GeNN::CodeGenerator::KernelPresynapticUpdate);
    if(backend.areSharedMemAtomicsSlow()) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.getArchetype().isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if all neuron groups targetted
    // by synapse groups within merged group are small enough that input to then can be stored in a shared memory array
    else if(std::all_of(sg.getGroups().cbegin(), sg.getGroups().cend(),
                        [blockSize](const GeNN::SynapseGroupInternal &sg)
                        {
                            return (sg.getTrgNeuronGroup()->getNumNeurons() <= blockSize);
                        }))
    {
        return true;
    }
    else {
        return false;
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
size_t PreSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    return 0;
}
//----------------------------------------------------------------------------
void PreSpan::genPreamble(EnvironmentExternalBase&, PresynapticUpdateGroupMerged&, 
                          const BackendSIMT&) const
{
}
//----------------------------------------------------------------------------
void PreSpan::genUpdate(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                        PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, bool trueSpike) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
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
        env.getStream() << "scalar lrevInSyn= 0.0;" << std::endl;
    }
    
    env.print("if (spike < $(_src_spk_cnt" + eventSuffix + ")[" + sg.getPreSlot(batchSize) + "])");
    {
        CodeStream::Scope b(env.getStream());

        /*if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
            os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
        }*/

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

            // **TODO** pretty sure __ldg will boost performance here - basically will bring whole row into cache
            env.printLine("const unsigned int ipost = $(_ind)[synAddress];");

            // Create substitution stack for presynaptic simulation code
            EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> synEnv(env, sg);
            synEnv.add(Type::Uint32.addConst(), "id_pre", "preInd");
            synEnv.add(Type::Uint32.addConst(), "id_post", "ipost");
            synEnv.add(Type::Uint32.addConst(), "id_syn", "synAddress");

            synEnv.add(Type::AddToPostDenDelay, "addToPostDelay",
                       backend.getAtomic(model.getPrecision()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "ipost", "$(1)") + "], $(0))");
            synEnv.add(Type::AddToPost, "addToPost",
                       backend.getAtomic(model.getPrecision()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "ipost") + "], $(0))");
            synEnv.add(Type::AddToPre, "addToPre", "lrevInSyn += $(0)");
            
            if(trueSpike) {
                sg.generateSpikeUpdate(backend, synEnv, modelMerged);
            }
            else {
                sg.generateSpikeEventUpdate(backend, synEnv, modelMerged);
            }
            
        }

        /*if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }*/
        
        // Should this be in the Postamble?
        if(sg.getArchetype().isPresynapticOutputRequired()) {
            // write lrevInSyn to global memory if not 0
            env.getStream() << "if(lrevInSyn != 0.0) " << backend.getAtomic(model.getPrecision()) + "(&" + env["_out_pre"] + "[" + sg.getPreISynIndex(batchSize, "preInd") + "], lrevInSyn);" << std::endl;
        }
        
    }
}
//----------------------------------------------------------------------------
void PreSpan::genPostamble(EnvironmentExternalBase &, const ModelSpecMerged&, 
                           PresynapticUpdateGroupMerged&, const BackendSIMT&) const
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
    // If data structure is dense, we can accumulate output directly into register
    if(shouldAccumulateInRegister(sg)) {
        env.getStream() << "scalar linSyn = 0;" << std::endl;
    }
    else if(isSmallSharedMemoryPop(sg, backend)) {
        env.getStream() << "if(" << backend.getThreadID() << " < group->numTrgNeurons)";
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            env.getStream() << "shLg[" << backend.getThreadID() << "] = 0;" << std::endl;
        }
        backend.genSharedMemBarrier(env.getStream());
    }
}
//----------------------------------------------------------------------------
size_t PostSpan::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // One element is required per thread if small shared memory optimization should be used for sg
    return isSmallSharedMemoryPop(sg, backend) ? 1 : 0;
}
//----------------------------------------------------------------------------
void PostSpan::genUpdate(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                         PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, bool trueSpike) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
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
                    synEnv.printLine("const unsigned int ipost = $(_ind)[$(id_syn)];");

                    synEnv.add(Type::Uint32.addConst(), "id_post", "ipost");
                }
                else { // DENSE
                    synEnv.add(Type::Uint32.addConst(), "id_post", "$(id)");
                }
       
                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                synEnv.add(Type::AddToPostDenDelay, "addToPostDelay",
                           backend.getAtomic(model.getPrecision()) + "(&$(_den_delay)[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                
                // If we should accumulate in register, add parameter to register
                if(shouldAccumulateInRegister(sg)) {
                    synEnv.add(Type::AddToPost, "addToPost", "linSyn += $(0)");
                }
                // Otherwise, if we should use shared memory, add to shared memory
                // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                else if(isSmallSharedMemoryPop(sg, backend)) {
                    synEnv.add(Type::AddToPost, "addToPost", "shLg[$(id_post)] += $(0)");
                }
                // Otherwise, use global memory atomic
                else {
                    synEnv.add(Type::AddToPost, "addToPost",
                               backend.getAtomic(model.getPrecision()) + "(&$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                }

                if(sg.getArchetype().isPresynapticOutputRequired()) {
                    synEnv.add(Type::AddToPre, "addToPre",
                               backend.getAtomic(model.getPrecision()) + "(&$(_out_pre)([" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");
                }
                
                if(trueSpike) {
                    sg.generateSpikeUpdate(backend, synEnv, modelMerged);
                }
                else {
                    sg.generateSpikeEventUpdate(backend, synEnv, modelMerged);
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
void PostSpan::genPostamble(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                            PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // If we should accumulate output directly into register
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    if(shouldAccumulateInRegister(sg)) {
        env.getStream() << "// only do this for existing neurons" << std::endl;
        env.print("if ($(id) < $(num_post))");
        {
            CodeStream::Scope b(env.getStream());
            const std::string inSyn = printSubs("$(_out_post)[" + sg.getPostISynIndex(batchSize, "$(id)") + "]", env);
            if(sg.getArchetype().isPSModelFused()) {
                env.getStream() << backend.getAtomic(model.getPrecision()) << "(&" << inSyn << ", linSyn);" << std::endl;
            }
            else {
                env.getStream() << inSyn << " += linSyn;" << std::endl;
            }
        }
    }
    // Otherwise, if we should accumulate into shared memory
    else if(isSmallSharedMemoryPop(sg, backend)) {
        backend.genSharedMemBarrier(env.getStream());
        env.getStream() << "if(" << backend.getThreadID() << " < " << env["num_post"] << ")";
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            const std::string inSyn = printSubs("$(_out_post)[" + sg.getPostISynIndex(batchSize, backend.getThreadID()) + "]", env);
            env.getStream() << backend.getAtomic(model.getPrecision()) << "(&" << inSyn << "], shLg[" << backend.getThreadID() << "]); " << std::endl;
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
void PreSpanProcedural::genUpdate(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                  PresynapticUpdateGroupMerged &sg, const BackendSIMT&, bool trueSpike) const
{
    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
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

        /*if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
            os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
        }*/

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
void PreSpanProcedural::genPostamble(EnvironmentExternalBase&, const ModelSpecMerged&, 
                                     PresynapticUpdateGroupMerged&, const BackendSIMT&) const
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
void PostSpanBitmask::genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &, 
                                  const BackendSIMT &backend) const
{
    // Loop through bits written by this thread
    for(size_t i = 0; i < 32; i++) {
        // Zero entries in this thread's shared memory array
        // **NOTE** this is ordered to prevent bank conflicts
        const std::string index = std::to_string(i * backend.getKernelBlockSize(KernelPresynapticUpdate)) + " + " + backend.getThreadID();
        env.getStream() << "shLg[" << index << "] = 0;" << std::endl;
    }
    backend.genSharedMemBarrier(env.getStream());
}
//----------------------------------------------------------------------------
size_t PostSpanBitmask::getSharedMemoryPerThread(const PresynapticUpdateGroupMerged&, const BackendSIMT&) const
{
    // Each thread sums up the input to 32 postsynaptic neurons
    return 32;
}
//----------------------------------------------------------------------------
void PostSpanBitmask::genUpdate(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, bool trueSpike) const
{
    // Get suffix based on type of events
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
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

                /*if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }
                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
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

                    // Calculate postsynaptic index
                    synEnv.printLine("const unsigned int ipost = ibit + ($(id) * 32);");

                    synEnv.add(Type::Uint32.addConst(), "id_pre", "$(_sh_spk" + eventSuffix + ")[j]");
                    synEnv.add(Type::Uint32.addConst(), "id_post", "ipost");


                    synEnv.add(Type::AddToPost, "addToPost",
                       "shLg[(ibit * " + std::to_string(blockSize) + ") + " + backend.getThreadID() + "] += $(0)");
                    synEnv.add(Type::AddToPre, "addToPre",
                               backend.getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_out_pre)[" + sg.getPreISynIndex(batchSize, "$(id_pre)") + "], $(0))");

                    if(trueSpike) {
                        sg.generateSpikeUpdate(backend, synEnv, modelMerged);
                    }
                    else {
                        sg.generateSpikeEventUpdate(backend, synEnv, modelMerged);
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
void PostSpanBitmask::genPostamble(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                   PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
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
            const std::string inSyn = "$(_out_post)[" + sg.getPostISynIndex(modelMerged.getModel().getBatchSize(), "glbIdx") +"]";
            if(sg.getArchetype().isPSModelFused()) {
                env.printLine(backend.getAtomic(modelMerged.getModel().getPrecision()) + "(&" + inSyn + ", shLg[shIdx]);");
            }
            else {
                env.printLine(inSyn + " += shLg[shIdx];");
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
    if(isSmallSharedMemoryPop(sg, backend)) {
        env.print("if(" + backend.getThreadID() + " < $(num_post))");
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            env.getStream() << "shLg[" << backend.getThreadID() << "] = 0;" << std::endl;
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
void PostSpanToeplitz::genUpdate(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                 PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, bool trueSpike) const
{
    assert(false);
    /*const auto &connectInit = sg.getArchetype().getToeplitzConnectivityInitialiser();

    // Get suffix based on type of events
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const std::string eventSuffix = trueSpike ? "" : "_evnt";
    
    // Create substitution stack for generating Toeplitz connectivity code
    Substitutions connSubs(&popSubs);
    connSubs.addVarSubstitution("id_diag", connSubs["id"]);

    
    // Add substitutions
    connSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                       [&sg](const std::string &p) { return sg.isToeplitzConnectivityInitParamHeterogeneous(p);  },
                                       "", "group->");
    connSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                     [&sg](const std::string &p) { return sg.isToeplitzConnectivityInitDerivedParamHeterogeneous(p);  },
                                     "", "group->");
    connSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group->");
    connSubs.addVarNameSubstitution(connectInit.getSnippet()->getDiagonalBuildStateVars());

    // Initialise any diagonal build state variables defined
    for (const auto &d : connectInit.getSnippet()->getDiagonalBuildStateVars()) {
        // Apply substitutions to value
        std::string value = d.value;
        connSubs.applyCheckUnreplaced(value, "toeplitz diagonal build state var : merged" + std::to_string(sg.getIndex()));
        //value = ensureFtype(value, modelMerged.getModel().getPrecision());

        os << d.type.resolve(sg.getTypeContext()).getName() << " " << d.name << " = " << value << ";" << std::endl;
    }

    os << "const unsigned int numSpikes = group->srcSpkCnt" << eventSuffix << "[" << sg.getPreSlot(batchSize) << "];" << std::endl;
    os << "const unsigned int numSpikeBlocks = (numSpikes + " << backend.getKernelBlockSize(KernelPresynapticUpdate) << " - 1) / " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

    const auto *wu = sg.getArchetype().getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeBlocks; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ") + 1 : " << backend.getKernelBlockSize(KernelPresynapticUpdate) << ";" << std::endl;

        backend.genSharedMemBarrier(os);
        os << "if (" << backend.getThreadID() << " < numSpikesInBlock)";
        {
            CodeStream::Scope b(os);
            const std::string index = "(r * " + std::to_string(backend.getKernelBlockSize(KernelPresynapticUpdate)) + ") + " + backend.getThreadID();
            os << "const unsigned int spk = group->srcSpk" << eventSuffix << "[" << sg.getPreVarIndex(batchSize, VarAccessDuplication::DUPLICATE, index) << "];" << std::endl;
            os << "shSpk" << eventSuffix << "[" << backend.getThreadID() << "] = spk;" << std::endl;
        }
        backend.genSharedMemBarrier(os);

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < numSpikesInBlock; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs["id"] << " < group->rowStride)";
            {
                CodeStream::Scope b(os);
              
                // Create another substitution stack for generating presynaptic simulation code
                Substitutions presynapticUpdateSubs(&popSubs);

                connSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                presynapticUpdateSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");

                if(backend.supportsNamespace() && !wu->getSimSupportCode().empty()) {
                    os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(wu->getSimSupportCode()) << ";" << std::endl;
                }
                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << "if("; 

                    // Generate weight update threshold condition
                    sg.generateSpikeEventThreshold(backend, os, modelMerged, presynapticUpdateSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }


                // Replace $(id_post) with first 'function' parameter as simulation code is
                // going to be, in turn, substituted into procedural connectivity generation code
                presynapticUpdateSubs.addVarSubstitution("id_post", "$(0)");

                // Replace kernel indices with the subsequent 'function' parameters
                for(size_t i = 0; i < sg.getArchetype().getKernelSize().size(); i++) {
                    presynapticUpdateSubs.addVarSubstitution("id_kernel_" + std::to_string(i),
                                                              "$(" + std::to_string(i + 1) + ")");
                }

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.getArchetype().isDendriticDelayRequired()) {
                    presynapticUpdateSubs.addFuncSubstitution("addToInSynDelay", 2, 
                                                              backend.getAtomic(model.getPrecision()) + "(&group->denDelay[" + sg.getPostDenDelayIndex(batchSize, "$(id_post)", "$(1)") + "], $(0))");
                }
                // Otherwise
                else {
                    // If we should use shared memory, add to shared memory
                    if(isSmallSharedMemoryPop(sg, backend)) {
                        presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, "shLg[$(id_post)] += $(0)");
                    }
                    // Otherwise, use global memory atomic
                    else {
                        presynapticUpdateSubs.addFuncSubstitution("addToInSyn", 1, 
                                                                   backend.getAtomic(model.getPrecision()) + "(&group->inSyn[" + sg.getPostISynIndex(batchSize, "$(id_post)") + "], $(0))");
                    }
                }

                if(sg.getArchetype().isPresynapticOutputRequired()) {
                    presynapticUpdateSubs.addFuncSubstitution("addToPre", 1,
                                                              backend.getAtomic(model.getPrecision()) + "(&group->revInSyn[" + sg.getPreISynIndex(batchSize, presynapticUpdateSubs["id_pre"]) + "], $(0))");
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

                // Generate toeplitz connectivity code
                sg.generateToeplitzConnectivity(backend, os, connSubs);

                if(!trueSpike && sg.getArchetype().isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
            }
        }
    }*/
}
//----------------------------------------------------------------------------
void PostSpanToeplitz::genPostamble(EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged, 
                                    PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const
{
    // If we should accumulate into shared memory
    if(isSmallSharedMemoryPop(sg, backend)) {
        backend.genSharedMemBarrier(env.getStream());
        env.print("if(" + backend.getThreadID() + " < $(num_post))");
        {
            CodeGenerator::CodeStream::Scope b(env.getStream());
            const std::string idx = sg.getPostISynIndex(modelMerged.getModel().getBatchSize(), backend.getThreadID());
            env.printLine(backend.getAtomic(modelMerged.getModel().getPrecision()) + "(&$(_out_post)[" + idx + "], shLg[" + backend.getThreadID() + "]);");
        }
    }
}
}   // namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
