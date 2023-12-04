#include "code_generator/modelSpecMerged.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::ModelSpecMerged
//----------------------------------------------------------------------------
 ModelSpecMerged::ModelSpecMerged(const BackendBase &backend, const ModelSpecInternal &model)
:   m_Model(model)
{
    createMergedGroups(getModel().getNeuronGroups(), m_MergedNeuronUpdateGroups,
                       [](const NeuronGroupInternal&){ return true; },
                       &NeuronGroupInternal::getHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg) { return (sg.isPreSpikeEventRequired() || sg.isPreSpikeRequired()); },
                       &SynapseGroupInternal::getWUHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg){ return (sg.isPostSpikeEventRequired() || sg.isPostSpikeRequired()); },
                       &SynapseGroupInternal::getWUHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                       [](const SynapseGroupInternal &sg){ return !Utils::areTokensEmpty(sg.getWUInitialiser().getSynapseDynamicsCodeTokens()); },
                       &SynapseGroupInternal::getWUHashDigest);

    createMergedGroups(getModel().getCustomUpdates(), m_MergedCustomUpdateGroups,
                       [](const CustomUpdateInternal &cg) { return !Utils::areTokensEmpty(cg.getUpdateCodeTokens()); },
                       &CustomUpdateInternal::getHashDigest);

    createMergedGroups(getModel().getCustomWUUpdates(), m_MergedCustomUpdateWUGroups,
                       [](const CustomUpdateWUInternal &cg) { return !Utils::areTokensEmpty(cg.getUpdateCodeTokens()) && !cg.isTransposeOperation(); },
                       &CustomUpdateWUInternal::getHashDigest);

    createMergedGroups(getModel().getCustomWUUpdates(), m_MergedCustomUpdateTransposeWUGroups,
                       [](const CustomUpdateWUInternal &cg){ return cg.isTransposeOperation(); },
                       &CustomUpdateWUInternal::getHashDigest);

    if(backend.isHostReductionRequired()) {
        createMergedGroups(getModel().getCustomUpdates(), m_MergedCustomUpdateHostReductionGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isBatchReduction(); },
                           &CustomUpdateInternal::getHashDigest);

        createMergedGroups(getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateHostReductionGroups,
                           [](const CustomUpdateWUInternal &cg){ return cg.isBatchReduction(); },
                           &CustomUpdateWUInternal::getHashDigest);
    }

    createMergedGroups(getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateGroups,
                       [](const CustomConnectivityUpdateInternal &cg){ return !Utils::areTokensEmpty(cg.getRowUpdateCodeTokens()); },
                       &CustomConnectivityUpdateInternal::getHashDigest);

    createMergedGroups(getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityHostUpdateGroups,
                       [](const CustomConnectivityUpdateInternal &cg) { return !Utils::areTokensEmpty(cg.getHostUpdateCodeTokens()); },
                       &CustomConnectivityUpdateInternal::getHashDigest);

    createMergedGroups(getModel().getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                       [](const NeuronGroupInternal &g){ return (g.isDelayRequired() || g.isTrueSpikeRequired() || g.isSpikeEventRequired()); },
                       &NeuronGroupInternal::getSpikeQueueUpdateHashDigest);

    createMergedGroups(getModel().getNeuronGroups(), m_MergedNeuronPrevSpikeTimeUpdateGroups,
                       [](const NeuronGroupInternal &ng){ return (ng.isPrevSpikeTimeRequired() || ng.isPrevSpikeEventTimeRequired()); },
                       &NeuronGroupInternal::getPrevSpikeTimeUpdateHashDigest);

    std::vector<std::reference_wrapper<const SynapseGroupInternal>> synapseGroupsWithDendriticDelay;
    for(const auto &n : getModel().getNeuronGroups()) {
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            if(sg->isDendriticDelayRequired()) {
                synapseGroupsWithDendriticDelay.push_back(std::cref(*sg));
            }
        }
    }
    createMergedGroups(synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                        &SynapseGroupInternal::getDendriticDelayUpdateHashDigest);

    createMergedGroups(getModel().getNeuronGroups(), m_MergedNeuronInitGroups,
                       [](const NeuronGroupInternal &ng)
                       { 
                           return (ng.isSpikeEventRequired() || ng.isTrueSpikeRequired()
                                   || ng.isSpikeTimeRequired() || ng.isPrevSpikeTimeRequired()
                                   || ng.isSpikeEventTimeRequired() || ng.isPrevSpikeEventTimeRequired()
                                   || ng.isVarInitRequired());
                       },
                       &NeuronGroupInternal::getInitHashDigest);

    createMergedGroups(getModel().getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                       [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                       &CustomUpdateInternal::getInitHashDigest);

    createMergedGroups(getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateInitGroups,
                       [](const CustomUpdateWUInternal &cg) 
                       {
                           return (((cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                   || (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL))
                                   && cg.isVarInitRequired());
                       },
                       &CustomUpdateWUInternal::getInitHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedSynapseInitGroups,
                       [](const SynapseGroupInternal &sg)
                       {
                           return (((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                   || (sg.getMatrixType() & SynapseMatrixWeight::KERNEL))
                                   && sg.isWUVarInitRequired());
                       },
                       &SynapseGroupInternal::getWUInitHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                       [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                       &SynapseGroupInternal::getConnectivityInitHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                       [&backend](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                   (sg.isWUVarInitRequired()
                                   || (backend.isPostsynapticRemapRequired() && (sg.isPostSpikeRequired() || sg.isPostSpikeEventRequired()))));
                       },
                       &SynapseGroupInternal::getWUInitHashDigest);

    createMergedGroups(getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                       [](const CustomUpdateWUInternal &cg) 
                       {
                           return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); 
                       },
                       &CustomUpdateWUInternal::getInitHashDigest);

    createMergedGroups(getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePreInitGroups,
                       [&backend](const CustomConnectivityUpdateInternal &cg) 
                       {
                           return (cg.isPreVarInitRequired() || (backend.isPopulationRNGInitialisedOnDevice() && Utils::isRNGRequired(cg.getRowUpdateCodeTokens())));     
                       },
                       &CustomConnectivityUpdateInternal::getInitHashDigest);

    createMergedGroups(getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePostInitGroups,
                       [](const CustomConnectivityUpdateInternal &cg) { return cg.isPostVarInitRequired(); },
                       &CustomConnectivityUpdateInternal::getInitHashDigest);

    createMergedGroups(getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateSparseInitGroups,
                       [](const CustomConnectivityUpdateInternal &cg) { return cg.isVarInitRequired(); },
                       &CustomConnectivityUpdateInternal::getInitHashDigest);

    createMergedGroups(getModel().getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                       [](const SynapseGroupInternal &sg)
                       { 
                           return !Utils::areTokensEmpty(sg.getConnectivityInitialiser().getHostInitCodeTokens());
                       },
                       &SynapseGroupInternal::getConnectivityHostInitHashDigest);
}


//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                  GenMergedGroupFn<NeuronUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedNeuronUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedPresynapticUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                       GenMergedGroupFn<PresynapticUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedPresynapticUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedPostsynapticUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                        GenMergedGroupFn<PostsynapticUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedPostsynapticUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseDynamicsGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                     GenMergedGroupFn<SynapseDynamicsGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseDynamicsGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                  const std::string &updateGroupName, GenMergedGroupFn<CustomUpdateGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomUpdateGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateWUGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                    const std::string &updateGroupName, GenMergedGroupFn<CustomUpdateWUGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomUpdateWUGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateTransposeWUGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                             const std::string &updateGroupName, GenMergedGroupFn<CustomUpdateTransposeWUGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomUpdateTransposeWUGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateHostReductionGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                               const std::string &updateGroupName, GenMergedGroupFn<CustomUpdateHostReductionGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomUpdateHostReductionGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateHostReductionGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                 const std::string &updateGroupName, GenMergedGroupFn<CustomWUUpdateHostReductionGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomWUUpdateHostReductionGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                              const std::string &updateGroupName, GenMergedGroupFn<CustomConnectivityUpdateGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomConnectivityUpdateGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityHostUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                  const std::string &updateGroupName, GenMergedGroupFn<CustomConnectivityHostUpdateGroupMerged> generateGroup)
{
    genMergedCustomUpdateGroups(backend, memorySpaces, m_MergedCustomConnectivityHostUpdateGroups, 
                                updateGroupName, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronSpikeQueueUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                            GenMergedGroupFn<NeuronSpikeQueueUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedNeuronSpikeQueueUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronPrevSpikeTimeUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                               GenMergedGroupFn<NeuronPrevSpikeTimeUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedNeuronPrevSpikeTimeUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseDendriticDelayUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                 GenMergedGroupFn<SynapseDendriticDelayUpdateGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseDendriticDelayUpdateGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                GenMergedGroupFn<NeuronInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedNeuronInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                      GenMergedGroupFn<CustomUpdateInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomUpdateInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                        GenMergedGroupFn<CustomWUUpdateInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomWUUpdateInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                 GenMergedGroupFn<SynapseInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseConnectivityInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                             GenMergedGroupFn<SynapseConnectivityInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseConnectivityInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                       GenMergedGroupFn<SynapseSparseInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseSparseInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                              GenMergedGroupFn<CustomWUUpdateSparseInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomWUUpdateSparseInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdatePreInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                     GenMergedGroupFn<CustomConnectivityUpdatePreInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomConnectivityUpdatePreInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdatePostInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                      GenMergedGroupFn<CustomConnectivityUpdatePostInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomConnectivityUpdatePostInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdateSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                        GenMergedGroupFn<CustomConnectivityUpdateSparseInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedCustomConnectivityUpdateSparseInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseConnectivityHostInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                                 GenMergedGroupFn<SynapseConnectivityHostInitGroupMerged> generateGroup)
{
    genMergedGroups(backend, memorySpaces, m_MergedSynapseConnectivityHostInitGroups, generateGroup);
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getHashDigest(const BackendBase &backend) const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash digest of model properties
    Utils::updateHash(getModel().getHashDigest(), hash);

    // Concatenate hash digest of backend properties
    Utils::updateHash(backend.getHashDigest(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of neuron update groups
    for(const auto &g : m_MergedNeuronUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of presynaptic update groups
    for(const auto &g : m_MergedPresynapticUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of postsynaptic update groups
    for(const auto &g : m_MergedPostsynapticUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse dynamics groups
    for(const auto &g : m_MergedSynapseDynamicsGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom update groups
    for(const auto &g : m_MergedCustomUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom WU update groups
    for(const auto &g : m_MergedCustomUpdateWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom connectivity update groups
    for(const auto &g : m_MergedCustomConnectivityUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom transpose WU update groups
    for(const auto &g : m_MergedCustomUpdateTransposeWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of neuron init groups
    for(const auto &g : m_MergedNeuronInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse init groups
    for(const auto &g : m_MergedSynapseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Update hash with hash digest of synapse sparse init groups
    for(const auto &g : m_MergedSynapseSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse connectivity init groups
    for(const auto &g : m_MergedSynapseConnectivityInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom update init groups
    for(const auto &g : m_MergedCustomUpdateInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom sparse WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity pre update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePreInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity post update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePostInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity sparse update init group
    for (const auto &g : m_MergedCustomConnectivityUpdateSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Update hash with each group's variable locations
    // **NOTE** these only effects the runner - doesn't matter for modules so this is done he
    for(const auto &g : getModel().getNeuronGroups()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getSynapseGroups()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getLocalCurrentSources()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }
    
    for(const auto &g : getModel().getCustomUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getCustomWUUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getCustomConnectivityUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getSynapseGroup()->getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getNeuronUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype neuron update group
    for(const auto &g : m_MergedNeuronUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // **NOTE** all properties of neuron spike queue and previous spike time
    // updates are also included in neuron update groups so no need to hash
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getSynapseUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype presynaptic update group
    for(const auto &g : m_MergedPresynapticUpdateGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // Concatenate hash digest of archetype postsynaptic update group
    for(const auto &g : m_MergedPostsynapticUpdateGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse dynamics group
    for(const auto &g : m_MergedSynapseDynamicsGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // **NOTE** all properties of synapse dendritic delay updates
    //  are also included in synapse update groups so no need to hash

    return hash.get_digest();
    
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getCustomUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);
    
    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype custom update group
    for(const auto &g : m_MergedCustomUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom WU update group
    for(const auto &g : m_MergedCustomUpdateWUGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom transpose WU update group
    for(const auto &g : m_MergedCustomUpdateTransposeWUGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivityupdate group
    for(const auto &g : m_MergedCustomConnectivityUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getInitArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);
    
    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype neuron init group
    for(const auto &g : m_MergedNeuronInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse dense init group
    for(const auto &g : m_MergedSynapseInitGroups) {
        Utils::updateHash(g.getArchetype().getWUInitHashDigest(), hash);
    }

    // Update hash with hash digest of archetype synapse sparse init group
    for(const auto &g : m_MergedSynapseSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getWUInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse connectivity init group
    for(const auto &g : m_MergedSynapseConnectivityInitGroups) {
        Utils::updateHash(g.getArchetype().getConnectivityInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom update init group
    for(const auto &g : m_MergedCustomUpdateInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom WU update init group
    for(const auto &g : m_MergedCustomWUUpdateInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom sparse WU update init group
    for(const auto &g : m_MergedCustomWUUpdateSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity pre update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePreInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity post update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePostInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity sparse update init group
    for (const auto &g : m_MergedCustomConnectivityUpdateSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
/*bool ModelSpecMerged::anyPointerEGPs() const
{
    // Loop through grouped merged EGPs
    for(const auto &e : m_MergedEGPs) {
        // If there's any pointer EGPs, return true
        // **TODO** without scalar EGPS, all EGPS are pointer EGPS!
        if(std::any_of(e.second.cbegin(), e.second.cend(),
                       [](const MergedEGPDestinations::value_type &g){ return g.second.type.isPointer(); }))
        {
            return true;
        }
    }

    return false;
}*/
