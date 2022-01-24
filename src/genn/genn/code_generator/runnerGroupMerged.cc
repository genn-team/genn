#include "code_generator/runnerGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronRunnerGroupMerged::name = "NeuronRunner";
//----------------------------------------------------------------------------
NeuronRunnerGroupMerged::NeuronRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                             const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   RunnerGroupMergedBase<NeuronGroupInternal, NeuronRunnerGroupMerged>(index, precision, groups, backend)
{
    
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    // **TODO** bespoke runner hashes
    orderGroupChildren(m_SortedMergedInSyns, &NeuronGroupInternal::getFusedPSMInSyn,
                       &SynapseGroupInternal::getPSInitHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    // **TODO** bespoke runner hashes
    orderGroupChildren(m_SortedMergedPreOutputOutSyns, &NeuronGroupInternal::getFusedPreOutputOutSyn,
                       &SynapseGroupInternal::getPreOutputInitHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    // **TODO** bespoke runner hashes
    orderGroupChildren(m_SortedCurrentSources, &NeuronGroupInternal::getCurrentSources,
                       &CurrentSourceInternal::getInitHashDigest);

    // Build vector of vectors containing each child group's incoming 
    // synapse groups, ordered to match those of the archetype group
    // **TODO** bespoke runner hashes
    orderGroupChildren(m_SortedInSynWithPostVars, &NeuronGroupInternal::getFusedInSynWithPostVars,
                       &SynapseGroupInternal::getWUPostInitHashDigest);

    // Build vector of vectors containing each child group's outgoing 
    // synapse groups, ordered to match those of the archetype group
    // **TODO** bespoke runner hashes
    orderGroupChildren(m_SortedOutSynWithPreVars, &NeuronGroupInternal::getFusedOutSynWithPreVars,
                       &SynapseGroupInternal::getWUPreInitHashDigest);

    addField("unsigned int", "numNeurons",
              [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
    if(getArchetype().isDelayRequired()) {
        addField("unsigned int", "numDelaySlots",
                 [](const NeuronGroupInternal &ng, size_t) { return std::to_string(ng.getNumDelaySlots()); });
    }
    
    const bool delaySpikes = (getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired());
    addField("unsigned int", "spkCnt", getArchetype().getSpikeLocation(), 0,
             delaySpikes ? "group->numDelaySlots" : "1");
    addField("unsigned int", "spk", getArchetype().getSpikeLocation(), 0,
             delaySpikes ? "group->numDelaySlots * group->numNeurons" : "group->numNeurons");
    
    if(getArchetype().isSpikeRecordingEnabled()) {
        addField("uint32_t", "recordSpk", VarLocation::HOST_DEVICE, POINTER_FIELD_MANUAL_ALLOC);
    }

    const std::string numNeurons = "group->numNeurons";
    const std::string numNeuronsDelayed = getArchetype().isDelayRequired() ? "group->numDelaySlots * group->numNeurons" : numNeurons;
    if(getArchetype().isSpikeEventRequired()) {
        addField("unsigned int", "spkCntEvnt", getArchetype().getSpikeEventLocation(), 0,
                 getArchetype().isDelayRequired() ? "group->numDelaySlots" : "1");
        addField("unsigned int", "spkEvnt", getArchetype().getSpikeEventLocation(), 0,
                 numNeuronsDelayed);

        if(getArchetype().isSpikeEventRecordingEnabled()) {
            addField("uint32_t", "recordSpkEvent", VarLocation::HOST_DEVICE, POINTER_FIELD_MANUAL_ALLOC);
        }
    }

    if(getArchetype().isDelayRequired()) {
        assert(false);
        //addNullPointerField("unsigned int", "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    }

    if(getArchetype().isSpikeTimeRequired()) {
        addField(timePrecision, "sT", getArchetype().getSpikeTimeLocation(), 0, numNeuronsDelayed);
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        addField(timePrecision, "seT", getArchetype().getSpikeTimeLocation(), 0, numNeuronsDelayed);
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addField(timePrecision, "prevST", getArchetype().getPrevSpikeTimeLocation(), 0, numNeuronsDelayed);
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addField(timePrecision, "prevSET", getArchetype().getPrevSpikeTimeLocation(), 0, numNeuronsDelayed);
    }

    // If this backend initialises population RNGs on device and this group requires on for simulation
    if(backend.isPopulationRNGRequired() && getArchetype().isSimRNGRequired() ) 
    {
        assert(false);
        //addNullPointerField(backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rng");
    }

    // Loop through variables
    const auto &varInit = getArchetype().getVarInitialisers();
    for(const auto &var : getArchetype().getNeuronModel()->getVars()) {
        const std::string count = getArchetype().isVarQueueRequired(var.name) ? numNeuronsDelayed : numNeurons;
        addField(var.type, var.name, getArchetype().getVarLocation(var.name), 
                 POINTER_FIELD_PUSH_PULL_GET, count, getVarAccessDuplication(var.access));

        // If we're initializing, add any var init EGPs to structure
        for(const auto &egp : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
            addField(egp.type, egp.name + var.name, 
                     VarLocation::HOST_DEVICE, POINTER_FIELD_PUSH_PULL_GET);
        }
    }

    // Add extra global parmeters
    for(const auto &egp : getArchetype().getNeuronModel()->getExtraGlobalParams()) {
        addField(egp.type, egp.name, getArchetype().getExtraGlobalParamLocation(egp.name),
                 POINTER_FIELD_PUSH_PULL_GET);
    }

    // Loop through merged synaptic inputs to archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
        const SynapseGroupInternal *sg = getSortedArchetypeMergedInSyns().at(i);

        // Add pointer to insyn
        const unsigned int flags = sg->isPSModelFused() ? 0 : POINTER_FIELD_PUSH_PULL_GET;
        addChildField(precision, "inSynInSyn", i, sg->getInSynLocation(), flags, numNeurons);

        // Add pointer to dendritic delay buffer if required
        if(sg->isDendriticDelayRequired()) {
            addChildField("unsigned int", "maxDendriticDelayTimestepsInSyn", i, 
                          [i, this](const NeuronGroupInternal&, size_t groupIndex) 
                          { 
                              const auto *sg = m_SortedMergedInSyns.at(groupIndex).at(i);
                              return std::to_string(sg->getMaxDendriticDelayTimesteps());
                          });

            addChildField(precision, "denDelayInSyn", i, sg->getDendriticDelayLocation(), 0,
                          "group->numNeurons * group->maxDendriticDelayTimestepsInSyn" + std::to_string(i));
            //addChildField("unsigned int", "denDelayPtrInSyn", i, sg->getDendriticDelayLocation(), VarAccessDuplication::DUPLICATE);
        }

        // Add extra global parmeters
        for(const auto &egp : sg->getPSModel()->getExtraGlobalParams()) {
            addChildField(egp.type, egp.name + "InSyn", i, sg->getPSExtraGlobalParamLocation(egp.name), 
                          POINTER_FIELD_PUSH_PULL_GET);
        }

        // Loop through variables
        const auto &varInit = sg->getPSVarInitialisers();
        for(const auto &var : sg->getPSModel()->getVars()) {
            addChildField(var.type, var.name + "InSyn", i, sg->getPSVarLocation(var.name), 
                          flags, numNeurons, getVarAccessDuplication(var.access));
            
            // If we're initializing, add any var init EGPs to structure
            for(const auto &egp : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
                addChildField(egp.type, egp.name + var.name+ "InSyn", i,
                              VarLocation::HOST_DEVICE, POINTER_FIELD_PUSH_PULL_GET);
            }
        }
    }

    // Loop through merged output synapses with presynaptic output of archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeMergedPreOutputOutSyns().size(); i++) {
        // Add pointer to revInSyn
        const auto loc = getSortedArchetypeMergedPreOutputOutSyns().at(i)->getInSynLocation();
        addChildField(precision, "revInSynOutSyn", i, loc, 0, numNeurons);
    }

    // Loop through current sources of archetypical neuron group (0) in sorted order
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const CurrentSourceInternal *cs = getSortedArchetypeCurrentSources().at(i);

        // Add extra global parmeters
        for(const auto &egp : cs->getCurrentSourceModel()->getExtraGlobalParams()) {
            addChildField(egp.type, egp.name + "CS", i, cs->getExtraGlobalParamLocation(egp.name),
                          POINTER_FIELD_PUSH_PULL_GET);
        }

        // Loop through variables
        const auto &varInit = cs->getVarInitialisers();
        for(const auto &var : cs->getCurrentSourceModel()->getVars()) {
            addChildField(var.type, var.name + "CS", i, cs->getVarLocation(var.name), 
                          POINTER_FIELD_PUSH_PULL_GET, numNeurons, getVarAccessDuplication(var.access));
            
             // If we're initializing, add any var init EGPs to structure
            for(const auto &egp : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
                addChildField(egp.type, egp.name + var.name+ "CS", i,
                              VarLocation::HOST_DEVICE, POINTER_FIELD_PUSH_PULL_GET);
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronRunnerGroupMerged::genRecordingBufferAlloc(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged) const
{
    CodeStream::Scope b(runner);
    runner << "// merged neuron runner group " << getIndex() << std::endl;
    runner << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(runner);

        // Get reference to group
        runner << "auto *group = &mergedNeuronRunnerGroup" << getIndex() << "[g]; " << std::endl;

        // Calculate number of words required for spike/spike event buffers
        if(getArchetype().isSpikeRecordingEnabled() || getArchetype().isSpikeEventRecordingEnabled()) {
            runner << "const unsigned int numWords = ((group->numNeurons + 31) / 32) * " << modelMerged.getModel().getBatchSize() << "timesteps;" << std::endl;
        }

        // Allocate spike array if required
        // **YUCK** maybe this should be renamed genDynamicArray
        if(getArchetype().isSpikeRecordingEnabled()) {
            CodeStream::Scope b(runner);
            backend.genExtraGlobalParamAllocation(runner, "uint32_t*", "recordSpk", VarLocation::HOST_DEVICE, "numWords", "group->");

            // Get destinations in merged structures, this EGP 
            // needs to be copied to and call push function
            /*const auto &mergedDestinations = modelMerged.getMergedEGPDestinations("recordSpk" + n.first, backend);
            for(const auto &v : mergedDestinations) {
                runner << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                runner << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << "recordSpk" + n.first << ");" << std::endl;
            }*/
        }

        // Allocate spike event array if required
        // **YUCK** maybe this should be renamed genDynamicArray
        if(getArchetype().isSpikeEventRecordingEnabled()) {
            CodeStream::Scope b(runner);
            backend.genExtraGlobalParamAllocation(runner, "uint32_t*", "recordSpkEvent", VarLocation::HOST_DEVICE, "numWords", "group->");

            // Get destinations in merged structures, this EGP 
            // needs to be copied to and call push function
            /*const auto &mergedDestinations = modelMerged.getMergedEGPDestinations("recordSpkEvent" + n.first, backend);
            for(const auto &v : mergedDestinations) {
                runner << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                runner << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << "recordSpkEvent" + n.first << ");" << std::endl;
            }*/
        }
    }
}
//----------------------------------------------------------------------------
void NeuronRunnerGroupMerged::genRecordingBufferPull(const BackendBase &backend, CodeStream &runner, const ModelSpecMerged &modelMerged) const
{
    CodeStream::Scope b(runner);
    runner << "// merged neuron runner group " << getIndex() << std::endl;
    runner << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(runner);

        // Get reference to group
        runner << "auto *group = &mergedNeuronRunnerGroup" << getIndex() << "[g]; " << std::endl;

        // Calculate number of words required for spike/spike event buffers
        if(getArchetype().isSpikeRecordingEnabled() || getArchetype().isSpikeEventRecordingEnabled()) {
            runner << "const unsigned int numWords = ((group->numNeurons + 31) / 32) * " << modelMerged.getModel().getBatchSize() << "timesteps;" << std::endl;
        }

        // Pull spike array if required
        // **YUCK** maybe this should be renamed pullDynamicArray
        if(getArchetype().isSpikeRecordingEnabled()) {
            CodeStream::Scope b(runner);
            backend.genExtraGlobalParamPull(runner, "uint32_t*", "recordSpk", VarLocation::HOST_DEVICE, "numWords", "group->");
        }
        // AllocaPullte spike event array if required
        // **YUCK** maybe this should be renamed pullDynamicArray
        if(getArchetype().isSpikeEventRecordingEnabled()) {
            CodeStream::Scope b(runner);
            backend.genExtraGlobalParamPull(runner, "uint32_t*", "recordSpkEvent", VarLocation::HOST_DEVICE, "numWords", "group->");
        }
    }
}
//----------------------------------------------------------------------------
// CodeGenerator::SynapseRunnerGroupMerged
//----------------------------------------------------------------------------
SynapseRunnerGroupMerged::SynapseRunnerGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                                   const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   RunnerGroupMergedBase<SynapseGroupInternal, SynapseRunnerGroupMerged>(index, precision, groups, backend)
{
    addField("unsigned int", "numSrcNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "rowStride",
             [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
    addField("unsigned int", "colStride",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });

    // Add pointers to connectivity data
    if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        addField("unsigned int", "rowLength", getArchetype().getSparseConnectivityLocation(), 
                 0, "group->numSrcNeurons", VarAccessDuplication::SHARED);
        addField(getArchetype().getSparseIndType(), "ind", getArchetype().getSparseConnectivityLocation(),
                 0, "group->numSrcNeurons * group->rowStride", VarAccessDuplication::SHARED);

        // Add additional structure for postsynaptic access
        if(backend.isPostsynapticRemapRequired() && !getArchetype().getWUModel()->getLearnPostCode().empty()) {
            addField("unsigned int", "colLength", VarLocation::DEVICE, 
                     0, "group->numTrgNeurons", VarAccessDuplication::SHARED);
            addField("unsigned int", "remap", VarLocation::DEVICE,
                     0, "group->numTrgNeurons * group->colStride", VarAccessDuplication::SHARED);
        }

        // Add additional structure for synapse dynamics access if required
        if(backend.isSynRemapRequired(getArchetype())) {
            addField("unsigned int", "synRemap", VarLocation::DEVICE,
                     0, "(group->numSrcNeurons * group->rowStride) + 1", VarAccessDuplication::SHARED);
        }
    }
    else if(getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        addField("uint32_t", "gp", getArchetype().getSparseConnectivityLocation(), 
                 0, "(((group->numSrcNeurons * group->rowStride) + 31) / 32)", VarAccessDuplication::SHARED);
    }
    
    /*addEGPs(getArchetype().getSparseConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
            backend.getDeviceVarPrefix());
    addEGPs(getArchetype().getToeplitzConnectivityInitialiser().getSnippet()->getExtraGlobalParams(),
            backend.getDeviceVarPrefix());*/

    if((getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)
       || (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL)) 
    {

    }

}
