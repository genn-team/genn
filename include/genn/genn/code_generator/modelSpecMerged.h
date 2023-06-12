#pragma once

// Standard C++ includes
#include <unordered_map>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"
#include "code_generator/customUpdateGroupMerged.h"
#include "code_generator/customConnectivityUpdateGroupMerged.h"
#include "code_generator/initGroupMerged.h"
#include "code_generator/neuronUpdateGroupMerged.h"
#include "code_generator/synapseUpdateGroupMerged.h"
#include "code_generator/supportCodeMerged.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class BackendBase;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ModelSpecMerged
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT ModelSpecMerged
{
public:
    ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend);

    //--------------------------------------------------------------------------
    // CodeGenerator::ModelSpecMerged::EGPField
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking fields of merged group structure containing EGPs
    struct EGPField
    {
        EGPField(size_t m, const Type::ResolvedType &t, const std::string &f, bool h)
        :   mergedGroupIndex(m), type(t), fieldName(f), hostGroup(h) {}

        size_t mergedGroupIndex;
        Type::ResolvedType type;
        std::string fieldName;
        bool hostGroup;

        //! Less than operator (used for std::set::insert), 
        //! lexicographically compares all three struct members
        bool operator < (const EGPField &other) const
        {
            return (std::make_tuple(mergedGroupIndex, type, fieldName, hostGroup) 
                    < std::make_tuple(other.mergedGroupIndex, other.type, other.fieldName, other.hostGroup));
        }
    };
    
    //--------------------------------------------------------------------------
    // CodeGenerator::ModelSpecMerged::MergedEGP
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking where an extra global variable ends up after merging
    struct MergedEGP : public EGPField
    {
        MergedEGP(size_t m, size_t g, const Type::ResolvedType &t, const std::string &f, bool h)
        :   EGPField(m, t, f, h), groupIndex(g) {}

        const size_t groupIndex;
    };

    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    //! Map of original extra global param names to their locations within merged structures
    // **THINK** why is this a multimap? A variable is only going to be in one merged group of each type....right?
    typedef std::unordered_multimap<std::string, MergedEGP> MergedEGPDestinations;
    typedef std::map<std::string, MergedEGPDestinations> MergedEGPMap;

    template<typename G>
    using GenerateMergedGroupFn = std::function<void(typename G::GroupInternal &)>;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Get underlying, unmerged model
    const ModelSpecInternal &getModel() const{ return m_Model; }
    
    //! Get type context used to resolve all types used in model
    const Type::TypeContext &getTypeContext() const{ return m_TypeContext; }

    //! Get merged neuron groups which require updating
    const std::vector<NeuronUpdateGroupMerged> &getMergedNeuronUpdateGroups() const{ return m_MergedNeuronUpdateGroups; }

    //! Get merged synapse groups which require presynaptic updates
    const std::vector<PresynapticUpdateGroupMerged> &getMergedPresynapticUpdateGroups() const{ return m_MergedPresynapticUpdateGroups; }

    //! Get merged synapse groups which require postsynaptic updates
    const std::vector<PostsynapticUpdateGroupMerged> &getMergedPostsynapticUpdateGroups() const{ return m_MergedPostsynapticUpdateGroups; }

    //! Get merged synapse groups which require synapse dynamics
    const std::vector<SynapseDynamicsGroupMerged> &getMergedSynapseDynamicsGroups() const{ return m_MergedSynapseDynamicsGroups; }

    //! Get merged neuron groups which require initialisation
    const std::vector<NeuronInitGroupMerged> &getMergedNeuronInitGroups() const{ return m_MergedNeuronInitGroups; }

    //! Get merged custom update groups which require initialisation
    const std::vector<CustomUpdateInitGroupMerged> &getMergedCustomUpdateInitGroups() const { return m_MergedCustomUpdateInitGroups; }

    //! Get merged custom updategroups with dense connectivity which require initialisation
    const std::vector<CustomWUUpdateInitGroupMerged> &getMergedCustomWUUpdateInitGroups() const { return m_MergedCustomWUUpdateInitGroups; }

    //! Get merged synapse groups with dense connectivity which require initialisation
    const std::vector<SynapseInitGroupMerged> &getMergedSynapseInitGroups() const{ return m_MergedSynapseInitGroups; }

    //! Get merged synapse groups which require connectivity initialisation
    const std::vector<SynapseConnectivityInitGroupMerged> &getMergedSynapseConnectivityInitGroups() const{ return m_MergedSynapseConnectivityInitGroups; }

    //! Get merged synapse groups with sparse connectivity which require initialisation
    const std::vector<SynapseSparseInitGroupMerged> &getMergedSynapseSparseInitGroups() const{ return m_MergedSynapseSparseInitGroups; }

    //! Get merged custom update groups with sparse connectivity which require initialisation
    const std::vector<CustomWUUpdateSparseInitGroupMerged> &getMergedCustomWUUpdateSparseInitGroups() const { return m_MergedCustomWUUpdateSparseInitGroups; }

    //! Get merged custom connectivity update groups with postsynaptic variables which require initialisation
    const std::vector<CustomConnectivityUpdatePreInitGroupMerged> &getMergedCustomConnectivityUpdatePreInitGroups() const { return m_MergedCustomConnectivityUpdatePreInitGroups; }

    //! Get merged custom connectivity update groups with postsynaptic variables which require initialisation
    const std::vector<CustomConnectivityUpdatePostInitGroupMerged> &getMergedCustomConnectivityUpdatePostInitGroups() const { return m_MergedCustomConnectivityUpdatePostInitGroups; }

    //! Get merged custom connectivity update groups with sparse synaptic variables which require initialisation
    const std::vector<CustomConnectivityUpdateSparseInitGroupMerged> &getMergedCustomConnectivityUpdateSparseInitGroups() const { return m_MergedCustomConnectivityUpdateSparseInitGroups; }

    //! Get merged neuron groups which require their spike queues updating
    const std::vector<NeuronSpikeQueueUpdateGroupMerged> &getMergedNeuronSpikeQueueUpdateGroups() const { return m_MergedNeuronSpikeQueueUpdateGroups; }

    //! Get merged neuron groups which require their previous spike times updating
    const std::vector<NeuronPrevSpikeTimeUpdateGroupMerged> &getMergedNeuronPrevSpikeTimeUpdateGroups() const{ return m_MergedNeuronPrevSpikeTimeUpdateGroups; }

    //! Get merged synapse groups which require their dendritic delay updating
    const std::vector<SynapseDendriticDelayUpdateGroupMerged> &getMergedSynapseDendriticDelayUpdateGroups() const { return m_MergedSynapseDendriticDelayUpdateGroups; }

    //! Get merged synapse groups which require host code to initialise their synaptic connectivity
    const std::vector<SynapseConnectivityHostInitGroupMerged> &getMergedSynapseConnectivityHostInitGroups() const{ return m_MergedSynapseConnectivityHostInitGroups; }

    //! Get merged custom updates of variables
    const std::vector<CustomUpdateGroupMerged> &getMergedCustomUpdateGroups() const { return m_MergedCustomUpdateGroups; }

    //! Get merged custom updates of weight update model variables
    const std::vector<CustomUpdateWUGroupMerged> &getMergedCustomUpdateWUGroups() const { return m_MergedCustomUpdateWUGroups; }

    //! Get merged custom weight update groups where transpose needs to be calculated
    const std::vector<CustomUpdateTransposeWUGroupMerged> &getMergedCustomUpdateTransposeWUGroups() const { return m_MergedCustomUpdateTransposeWUGroups; }

    //! Get merged custom update groups where host reduction needs to be performed
    const std::vector<CustomUpdateHostReductionGroupMerged> &getMergedCustomUpdateHostReductionGroups() const { return m_MergedCustomUpdateHostReductionGroups; }

    //! Get merged custom weight update groups where host reduction needs to be performed
    const std::vector<CustomWUUpdateHostReductionGroupMerged> &getMergedCustomWUUpdateHostReductionGroups() const { return m_MergedCustomWUUpdateHostReductionGroups; }

    //! Get merged custom connectivity update groups
    const std::vector<CustomConnectivityUpdateGroupMerged> &getMergedCustomConnectivityUpdateGroups() const { return m_MergedCustomConnectivityUpdateGroups; }

    //! Get merged custom connectivity update groups where host processing needs to be performed
    const std::vector<CustomConnectivityHostUpdateGroupMerged> &getMergedCustomConnectivityHostUpdateGroups() const { return m_MergedCustomConnectivityHostUpdateGroups; }

    template<typename G>
    void genMergedNeuronUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronUpdateGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getHashDigest, generateGroup);
    }
    
    template<typename G>
    void genMergedPresynapticUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg) { return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                           &SynapseGroupInternal::getWUHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedPostsynapticUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                           &SynapseGroupInternal::getWUHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseDynamicsGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                           &SynapseGroupInternal::getWUHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateGroups,
                           [](const CustomUpdateInternal &) { return true; },
                           &CustomUpdateInternal::getHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomUpdateWUGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomUpdateWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return !cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::getHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomUpdateTransposeWUGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomUpdateTransposeWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::getHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomUpdateHostReductionGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateHostReductionGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isBatchReduction(); },
                           &CustomUpdateInternal::getHashDigest, generateGroup, true);
    }

    template<typename G>
    void genMergedCustomWUUpdateHostReductionGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateHostReductionGroups,
                           [](const CustomUpdateWUInternal &cg) { return cg.isBatchReduction(); },
                           &CustomUpdateWUInternal::getHashDigest, generateGroup, true);
    }

    template<typename G>
    void genMergedCustomConnectivityUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return !cg.getCustomConnectivityUpdateModel()->getRowUpdateCode().empty(); },
                           &CustomConnectivityUpdateInternal::getHashDigest, genereateGroup);
    }

    template<typename G>
    void genMergedCustomConnectivityHostUpdateGroups(BackendBase &backend, G generateGroup)
    {
         createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityHostUpdateGroups,
                            [](const CustomConnectivityUpdateInternal &cg) { return !cg.getCustomConnectivityUpdateModel()->getHostUpdateCode().empty(); },
                            &CustomConnectivityUpdateInternal::getHashDigest, generateGroup, true);
    }

    template<typename G>
    void genMergedNeuronSpikeQueueUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getSpikeQueueUpdateHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedNeuronPrevSpikeTimeUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronPrevSpikeTimeUpdateGroups,
                           [](const NeuronGroupInternal &ng){ return (ng.isPrevSpikeTimeRequired() || ng.isPrevSpikeEventTimeRequired()); },
                           &NeuronGroupInternal::getPrevSpikeTimeUpdateHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseDendriticDelayUpdateGroups(const BackendBase &backend, G generateGroup)
    {
        std::vector<std::reference_wrapper<const SynapseGroupInternal>> synapseGroupsWithDendriticDelay;
        for(const auto &n : getModel().getNeuronGroups()) {
            for(const auto *sg : n.second.getFusedPSMInSyn()) {
                if(sg->isDendriticDelayRequired()) {
                    synapseGroupsWithDendriticDelay.push_back(std::cref(*sg));
                }
            }
        }
        createMergedGroups(backend, synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                           &SynapseGroupInternal::getDendriticDelayUpdateHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedNeuronInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronInitGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomUpdateInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomUpdateInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomWUUpdateInitGroups(const BackendBase &backend, G generateGroup)
    {
         createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateInitGroups,
                            [](const CustomUpdateWUInternal &cg) 
                            {
                                return (((cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                         || (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL))
                                         && cg.isVarInitRequired());
                            },
                            &CustomUpdateWUInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseInitGroups(const BackendBase &backend, G generateGroup)
    {
         createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseInitGroups,
                            [](const SynapseGroupInternal &sg)
                            {
                                return (((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                         || (sg.getMatrixType() & SynapseMatrixWeight::KERNEL))
                                         && sg.isWUVarInitRequired());
                            },
                            &SynapseGroupInternal::getWUInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseConnectivityInitGroups(const BackendBase &backend, G generateGroup)
    {
         createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                           [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                           &SynapseGroupInternal::getConnectivityInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseSparseInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                           [&backend](const SynapseGroupInternal &sg)
                           {
                               return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                       (sg.isWUVarInitRequired()
                                        || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                           },
                           &SynapseGroupInternal::getWUInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomWUUpdateSparseInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                           [](const CustomUpdateWUInternal &cg) 
                           {
                               return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); 
                           },
                           &CustomUpdateWUInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomConnectivityUpdatePreInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePreInitGroups,
                           [&backend](const CustomConnectivityUpdateInternal &cg) 
                           {
                               return (cg.isPreVarInitRequired() || (backend.isPopulationRNGInitialisedOnDevice() && cg.isRowSimRNGRequired()));     
                           },
                           &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomConnectivityUpdatePostInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePostInitGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return cg.isPostVarInitRequired(); },
                           &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedCustomConnectivityUpdateSparseInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateSparseInitGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
    }

    template<typename G>
    void genMergedSynapseConnectivityHostInitGroups(const BackendBase &backend, G generateGroup)
    {
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                           [](const SynapseGroupInternal &sg)
                           { 
                               return !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty();
                           },
                           &SynapseGroupInternal::getConnectivityHostInitHashDigest, generateGroup, true);
    }

    void genMergedNeuronUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedNeuronUpdateGroups); }
    void genMergedPresynapticUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedPresynapticUpdateGroups); }
    void genMergedPostsynapticUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedPostsynapticUpdateGroups); }
    void genMergedSynapseDynamicsGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseDynamicsGroups); }
    void genMergedNeuronInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedNeuronInitGroups); }
    void genMergedCustomUpdateInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomUpdateInitGroups); }
    void genMergedCustomWUUpdateInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomWUUpdateInitGroups); }
    void genMergedSynapseInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseInitGroups); }
    void genMergedSynapseConnectivityInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseConnectivityInitGroups); }
    void genMergedSynapseSparseInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseSparseInitGroups); }
    void genMergedCustomWUUpdateSparseInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomWUUpdateSparseInitGroups); }
    void genMergedCustomConnectivityUpdatePreInitStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityUpdatePreInitGroups); }
    void genMergedCustomConnectivityUpdatePostInitStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityUpdatePostInitGroups); }
    void genMergedCustomConnectivityUpdateSparseInitStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityUpdateSparseInitGroups); }
    void genMergedNeuronSpikeQueueUpdateStructs(CodeStream &os, const BackendBase &backend) const{ genMergedStructures(os, backend, m_MergedNeuronSpikeQueueUpdateGroups); }
    void genMergedNeuronPrevSpikeTimeUpdateStructs(CodeStream &os, const BackendBase &backend) const{ genMergedStructures(os, backend, m_MergedNeuronPrevSpikeTimeUpdateGroups); }
    void genMergedSynapseDendriticDelayUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseDendriticDelayUpdateGroups); }
    void genMergedSynapseConnectivityHostInitStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseConnectivityHostInitGroups); }
    void genMergedCustomUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomUpdateGroups); }
    void genMergedCustomUpdateWUStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomUpdateWUGroups); }
    void genMergedCustomUpdateTransposeWUStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomUpdateTransposeWUGroups); }
    void genMergedCustomUpdateHostReductionStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomUpdateHostReductionGroups); }
    void genMergedCustomWUUpdateHostReductionStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomWUUpdateHostReductionGroups); }
    void genMergedCustomConnectivityUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityUpdateGroups); }
    void genMergedCustomConnectivityHostUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityHostUpdateGroups); }

    void genNeuronUpdateGroupSupportCode(CodeStream &os, bool supportsNamespace = true) const{ m_NeuronUpdateSupportCode.gen(os, getModel().getPrecision(), supportsNamespace); }
    void genPostsynapticDynamicsSupportCode(CodeStream &os, bool supportsNamespace = true) const{ m_PostsynapticDynamicsSupportCode.gen(os, getModel().getPrecision(), supportsNamespace); }
    void genPresynapticUpdateSupportCode(CodeStream &os, bool supportsNamespace = true) const{ m_PresynapticUpdateSupportCode.gen(os, getModel().getPrecision(), supportsNamespace); }
    void genPostsynapticUpdateSupportCode(CodeStream &os, bool supportsNamespace = true) const{ m_PostsynapticUpdateSupportCode.gen(os, getModel().getPrecision(), supportsNamespace); }
    void genSynapseDynamicsSupportCode(CodeStream &os, bool supportsNamespace = true) const{ m_SynapseDynamicsSupportCode.gen(os, getModel().getPrecision(), supportsNamespace); }

    const std::string &getNeuronUpdateSupportCodeNamespace(const std::string &code) const{ return m_NeuronUpdateSupportCode.getSupportCodeNamespace(code); }
    const std::string &getPostsynapticDynamicsSupportCodeNamespace(const std::string &code) const{ return m_PostsynapticDynamicsSupportCode.getSupportCodeNamespace(code); }
    const std::string &getPresynapticUpdateSupportCodeNamespace(const std::string &code) const{ return m_PresynapticUpdateSupportCode.getSupportCodeNamespace(code); }
    const std::string &getPostsynapticUpdateSupportCodeNamespace(const std::string &code) const{ return m_PostsynapticUpdateSupportCode.getSupportCodeNamespace(code); }
    const std::string &getSynapseDynamicsSupportCodeNamespace(const std::string &code) const{ return m_SynapseDynamicsSupportCode.getSupportCodeNamespace(code); }

    //! Get hash digest of entire model
    boost::uuids::detail::sha1::digest_type getHashDigest(const BackendBase &backend) const;

    //! Get hash digest of neuron update module
    boost::uuids::detail::sha1::digest_type getNeuronUpdateArchetypeHashDigest() const;
    
    //! Get hash digest of synapse update module
    boost::uuids::detail::sha1::digest_type getSynapseUpdateArchetypeHashDigest() const;
    
    //! Get hash digest of custom update module
    boost::uuids::detail::sha1::digest_type getCustomUpdateArchetypeHashDigest() const;

    //! Get hash digest of init module
    boost::uuids::detail::sha1::digest_type getInitArchetypeHashDigest() const;

    //! Get the string literal that should be used to represent a value in scalar type
    std::string scalarExpr(double value) const;
    
    //! Does model have any EGPs?
    bool anyPointerEGPs() const;

    //! Are there any destinations within the merged data structures for a particular extra global parameter?
    bool anyMergedEGPDestinations(const std::string &name) const
    {
        return (m_MergedEGPs.find(name) != m_MergedEGPs.cend());
    }
    
    //! Get the map of destinations within the merged data structures for a particular extra global parameter
    const MergedEGPDestinations &getMergedEGPDestinations(const std::string &name) const
    {
        return m_MergedEGPs.at(name);
    }

    // Get set of unique fields referenced in a merged group
    template<typename T>
    std::set<EGPField> getMergedGroupFields() const
    {
        // Loop through all EGPs
        std::set<EGPField> mergedGroupFields;
        for(const auto &e : m_MergedEGPs) {
            // Get all destinations in this type of group
            const auto groupEGPs = e.second.equal_range(T::name);

            // Copy them all into set
            std::transform(groupEGPs.first, groupEGPs.second, std::inserter(mergedGroupFields, mergedGroupFields.end()),
                           [](const MergedEGPMap::value_type::second_type::value_type &g)
                           {
                               return EGPField{g.second.mergedGroupIndex, g.second.type, g.second.fieldName, g.second.hostGroup};
                           });
        }

        // Return set
        return mergedGroupFields;
    }

    template<typename T>
    void genMergedGroupPush(CodeStream &os, const std::vector<T> &groups, const BackendBase &backend) const
    {

        if(!groups.empty()) {
            // Get set of unique fields referenced in a merged group
            const auto mergedGroupFields = getMergedGroupFields<T>();
            
            os << "// ------------------------------------------------------------------------" << std::endl;
            os << "// merged extra global parameter functions" << std::endl;
            os << "// ------------------------------------------------------------------------" << std::endl;
            // Loop through resultant fields and generate function to push updated pointers into group merged
            for(auto f : mergedGroupFields) {
                os << "void pushMerged" << T::name << f.mergedGroupIndex << f.fieldName << "ToDevice(unsigned int idx, " << backend.getMergedGroupFieldHostTypeName(f.type) << " value)";
                {
                    CodeStream::Scope b(os);
                    backend.genMergedDynamicVariablePush(os, T::name, f.mergedGroupIndex, "idx", f.fieldName, "value");
                }
                os << std::endl;
            }
        }
    }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    template<typename T>
    void genMergedStructures(CodeStream &os, const BackendBase &backend, const std::vector<T> &mergedGroups) const
    {
        // Loop through all merged groups and generate struct
        for(const auto &g : mergedGroups) {
            g.generateStruct(os, backend, T::name);
        }
    }

    template<typename Group, typename MergedGroup, typename D, typename G>
    void createMergedGroups(const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const Group>> &unmergedGroups,
                            std::vector<MergedGroup> &mergedGroups, D getHashDigest, G generateGroup, bool host = false)
    {
        // Create a hash map to group together groups with the same SHA1 digest
        std::unordered_map<boost::uuids::detail::sha1::digest_type, 
                           std::vector<std::reference_wrapper<const Group>>, 
                           Utils::SHA1Hash> protoMergedGroups;

        // Add unmerged groups to correct vector
        for(const auto &g : unmergedGroups) {
            protoMergedGroups[std::invoke(g.get(), getHashDigest)].push_back(g);
        }

        // Reserve final merged groups vector
        mergedGroups.reserve(protoMergedGroups.size());

        // Loop through resultant merged groups
        size_t i = 0;
        for(const auto &p : protoMergedGroups) {
            // Add group to vector
            mergedGroups.emplace_back(i, m_TypeContext, backend, p.second);
            generateGroup(mergedGroups.back());

            // Loop through fields
            for(const auto &f : mergedGroups.back().getFields()) {
                // If field is dynamic, add record to merged EGPS
                if((std::get<3>(f) & GroupMergedFieldType::DYNAMIC)) {
                    // Loop through groups within newly-created merged group
                    for(size_t groupIndex = 0; groupIndex < mergedGroups.back().getGroups().size(); groupIndex++) {
                        const auto &g = mergedGroups.back().getGroups()[groupIndex];

                        // Add reference to this group's variable to data structure
                        assert(std::get<0>(f).isPointer());
                        m_MergedEGPs[std::get<2>(f)(g, groupIndex)].emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(MergedGroup::name),
                            std::forward_as_tuple(i, groupIndex, std::get<0>(f), std::get<1>(f), host));
                    }
                }
            }

            i++;
        }
    }

    template<typename Group, typename MergedGroup, typename F, typename U, typename G>
    void createMergedGroups(const BackendBase &backend,
                            const std::map<std::string, Group> &groups, std::vector<MergedGroup> &mergedGroups,
                            F filter, U updateHash, G generateGroup, bool host = false)
    {
        // Build temporary vector of references to groups that pass filter
        std::vector<std::reference_wrapper<const Group>> unmergedGroups;
        for(const auto &g : groups) {
            if(filter(g.second)) {
                unmergedGroups.emplace_back(std::cref(g.second));
            }
        }

        // Merge filtered vector
        createMergedGroups(backend, unmergedGroups, mergedGroups, updateHash, generateGroup, host);
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! Underlying, unmerged model
    const ModelSpecInternal &m_Model;

    //! Merged neuron groups which require updating
    std::vector<NeuronUpdateGroupMerged> m_MergedNeuronUpdateGroups;

    //! Merged synapse groups which require presynaptic updates
    std::vector<PresynapticUpdateGroupMerged> m_MergedPresynapticUpdateGroups;

    //! Merged synapse groups which require postsynaptic updates
    std::vector<PostsynapticUpdateGroupMerged> m_MergedPostsynapticUpdateGroups;

    //! Merged synapse groups which require synapse dynamics update
    std::vector<SynapseDynamicsGroupMerged> m_MergedSynapseDynamicsGroups;

    //! Merged neuron groups which require initialisation
    std::vector<NeuronInitGroupMerged> m_MergedNeuronInitGroups;

    //! Merged custom update groups which require initialisation
    std::vector<CustomUpdateInitGroupMerged> m_MergedCustomUpdateInitGroups;

    //! Merged custom update weight update groups which require initialisation
    std::vector<CustomWUUpdateInitGroupMerged> m_MergedCustomWUUpdateInitGroups;

    //! Merged synapse groups with dense connectivity which require initialisation
    std::vector<SynapseInitGroupMerged> m_MergedSynapseInitGroups;

    //! Merged synapse groups which require connectivity initialisation
    std::vector<SynapseConnectivityInitGroupMerged> m_MergedSynapseConnectivityInitGroups;

    //! Merged synapse groups with sparse connectivity which require initialisation
    std::vector<SynapseSparseInitGroupMerged> m_MergedSynapseSparseInitGroups;

    //! Merged custom update groups with sparse connectivity which require initialisation
    std::vector<CustomWUUpdateSparseInitGroupMerged> m_MergedCustomWUUpdateSparseInitGroups;

    //! Merged custom connectivity update groups with presynaptic variables which require initialisation
    std::vector<CustomConnectivityUpdatePreInitGroupMerged> m_MergedCustomConnectivityUpdatePreInitGroups;

    //! Merged custom connectivity update groups with postsynaptic variables which require initialisation
    std::vector<CustomConnectivityUpdatePostInitGroupMerged> m_MergedCustomConnectivityUpdatePostInitGroups;

    //! Merged custom connectivity update groups with sparse synaptic variables which require initialisation
    std::vector<CustomConnectivityUpdateSparseInitGroupMerged> m_MergedCustomConnectivityUpdateSparseInitGroups;

    //! Merged neuron groups which require their spike queues updating
    std::vector<NeuronSpikeQueueUpdateGroupMerged> m_MergedNeuronSpikeQueueUpdateGroups;

    //! Merged neuron groups which require their previous spike times updating
    std::vector<NeuronPrevSpikeTimeUpdateGroupMerged> m_MergedNeuronPrevSpikeTimeUpdateGroups;

    //! Merged synapse groups which require their dendritic delay updating
    std::vector<SynapseDendriticDelayUpdateGroupMerged> m_MergedSynapseDendriticDelayUpdateGroups;

    //! Merged synapse groups which require host code to initialise their synaptic connectivity
    std::vector<SynapseConnectivityHostInitGroupMerged> m_MergedSynapseConnectivityHostInitGroups;

    //! Merged custom update groups
    std::vector<CustomUpdateGroupMerged> m_MergedCustomUpdateGroups;

    //! Merged custom weight update groups
    std::vector<CustomUpdateWUGroupMerged> m_MergedCustomUpdateWUGroups;

    //! Merged custom weight update groups where transpose needs to be calculated
    std::vector<CustomUpdateTransposeWUGroupMerged> m_MergedCustomUpdateTransposeWUGroups;

    //! Merged custom update groups where host reduction needs to be performed
    std::vector<CustomUpdateHostReductionGroupMerged> m_MergedCustomUpdateHostReductionGroups;

    //! Merged custom weight update groups where host reduction needs to be performed
    std::vector<CustomWUUpdateHostReductionGroupMerged> m_MergedCustomWUUpdateHostReductionGroups;

    //! Merged custom connectivity update groups
    std::vector<CustomConnectivityUpdateGroupMerged> m_MergedCustomConnectivityUpdateGroups;

    //! Merged custom connectivity update groups where host processing needs to be performed
    std::vector<CustomConnectivityHostUpdateGroupMerged> m_MergedCustomConnectivityHostUpdateGroups;

    //! Unique support code strings for neuron update
    SupportCodeMerged m_NeuronUpdateSupportCode;

    //! Unique support code strings for postsynaptic model
    SupportCodeMerged m_PostsynapticDynamicsSupportCode;

    //! Unique support code strings for presynaptic update
    SupportCodeMerged m_PresynapticUpdateSupportCode;

    //! Unique support code strings for postsynaptic update
    SupportCodeMerged m_PostsynapticUpdateSupportCode;

    //! Unique support code strings for synapse dynamics
    SupportCodeMerged m_SynapseDynamicsSupportCode;

    //! Map containing mapping of original extra global param names to their locations within merged groups
    MergedEGPMap m_MergedEGPs;
    
    //! Type context used to resolve all types used in model
    Type::TypeContext m_TypeContext;
};
}   // namespace GeNN::CodeGenerator
