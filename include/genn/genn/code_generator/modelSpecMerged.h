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
#include "code_generator/initGroupMerged.h"
#include "code_generator/neuronUpdateGroupMerged.h"
#include "code_generator/runnerGroupMerged.h"
#include "code_generator/synapseUpdateGroupMerged.h"
#include "code_generator/supportCodeMerged.h"

// Forward declarations
namespace CodeGenerator
{
class BackendBase;
}

//--------------------------------------------------------------------------
// CodeGenerator::ModelSpecMerged
//--------------------------------------------------------------------------
namespace CodeGenerator
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
        EGPField(size_t m, const std::string &t, const std::string &f)
        :   mergedGroupIndex(m), type(t), fieldName(f) {}

        const size_t mergedGroupIndex;
        const std::string type;
        const std::string fieldName;

        //! Less than operator (used for std::set::insert), 
        //! lexicographically compares all three struct members
        bool operator < (const EGPField &other) const
        {
            if(other.mergedGroupIndex < mergedGroupIndex) {
                return false;
            }
            else if(mergedGroupIndex < other.mergedGroupIndex) {
                return true;
            }

            if(other.type < type) {
                return false;
            }
            else if(type < other.type) {
                return true;
            }

            if(other.fieldName < fieldName) {
                return false;
            }
            else if(fieldName < other.fieldName) {
                return true;
            }

            return false;
        }
    };
    
    //--------------------------------------------------------------------------
    // CodeGenerator::ModelSpecMerged::MergedEGP
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking where an extra global variable ends up after merging
    struct MergedEGP : public EGPField
    {
        MergedEGP(size_t m, size_t g, const std::string &t, const std::string &f)
        :   EGPField(m, t, f), groupIndex(g) {}

        const size_t groupIndex;
    };

    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    //! Map of original extra global param names to their locations within merged structures
    typedef std::unordered_multimap<std::string, MergedEGP> MergedEGPDestinations;
    typedef std::map<std::string, MergedEGPDestinations> MergedEGPMap;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Get underlying, unmerged model
    const ModelSpecInternal &getModel() const{ return m_Model; }

    //! Get merged neuron groups used for generating runner
    const std::vector<NeuronRunnerGroupMerged> &getMergedNeuronRunnerGroups() const { return m_MergedNeuronRunnerGroups;  }

    //! Get merged synapse groups used for generating runner
    const std::vector<SynapseRunnerGroupMerged> &getMergedSynapseRunnerGroups() const { return m_MergedSynapseRunnerGroups;  }

    //! Get merged current sources used for generating runner
    const std::vector<CurrentSourceRunnerGroupMerged> &getMergedCurrentSourceRunnerGroups() const { return m_MergedCurrentSourceRunnerGroups;  }

    //! Get merged custom updates used for generating runner
    const std::vector<CustomUpdateRunnerGroupMerged> &getMergedCustomUpdateRunnerGroups() const { return m_MergedCustomUpdateRunnerGroups; }

    //! Get merged custom WU updates used for generating runner
    const std::vector<CustomUpdateWURunnerGroupMerged> &getMergedCustomUpdateWURunnerGroups() const { return m_MergedCustomUpdateWURunnerGroups; }

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
    const std::vector<CustomWUUpdateDenseInitGroupMerged> &getMergedCustomWUUpdateDenseInitGroups() const { return m_MergedCustomWUUpdateDenseInitGroups; }

    //! Get merged synapse groups with dense connectivity which require initialisation
    const std::vector<SynapseDenseInitGroupMerged> &getMergedSynapseDenseInitGroups() const{ return m_MergedSynapseDenseInitGroups; }

    //! Get merged synapse groups which require connectivity initialisation
    const std::vector<SynapseConnectivityInitGroupMerged> &getMergedSynapseConnectivityInitGroups() const{ return m_MergedSynapseConnectivityInitGroups; }

    //! Get merged synapse groups with sparse connectivity which require initialisation
    const std::vector<SynapseSparseInitGroupMerged> &getMergedSynapseSparseInitGroups() const{ return m_MergedSynapseSparseInitGroups; }
    
    //! Get merged synapse groups with kernel connectivity which require initialisation
    const std::vector<SynapseKernelInitGroupMerged> &getMergedSynapseKernelInitGroups() const{ return m_MergedSynapseKernelInitGroups; }

    //! Get merged custom update groups with sparse connectivity which require initialisation
    const std::vector<CustomWUUpdateSparseInitGroupMerged> &getMergedCustomWUUpdateSparseInitGroups() const { return m_MergedCustomWUUpdateSparseInitGroups; }

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

    //! Find the name of the merged runner group associated with neuron group e.g. mergedNeuronRunnerGroup0[6]
    std::string findMergedRunnerGroup(const NeuronGroupInternal &ng) const { return findMergedRunnerGroup<NeuronRunnerGroupMerged>(ng); }

    //! Find the name of the merged runner group associated with synapse group e.g. mergedSynapseRunnerGroup0[6]
    std::string findMergedRunnerGroup(const SynapseGroupInternal &sg) const { return findMergedRunnerGroup<SynapseRunnerGroupMerged>(sg); }

    //! Find the name of the merged runner group associated with current source e.g. mergedCurrentSourceRunnerGroup0[6]
    std::string findMergedRunnerGroup(const CurrentSourceInternal &cs) const { return findMergedRunnerGroup<CurrentSourceRunnerGroupMerged>(cs); }

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateRunnerGroup0[6]
    std::string findMergedRunnerGroup(const CustomUpdateInternal &cu) const { return findMergedRunnerGroup<CustomUpdateRunnerGroupMerged>(cu); }

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateWURunnerGroup0[6]
    std::string findMergedRunnerGroup(const CustomUpdateWUInternal &cu) const { return findMergedRunnerGroup<CustomUpdateWURunnerGroupMerged>(cu); }

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

    //! Does model have any EGPs?
    bool anyPointerEGPs() const;

    //! Get the map of destinations within the merged data structures for a particular extra global parameter
    const MergedEGPDestinations &getMergedEGPDestinations(const std::string &name, const BackendBase &backend) const
    {
        return m_MergedEGPs.at(backend.getDeviceVarPrefix() + name);
    }

    //! Generate calls to update all target merged groups
    template<typename T>
    void genScalarEGPPush(CodeStream &os, const BackendBase &backend) const
    {
        // Loop through all merged EGPs
        for(const auto &e : m_MergedEGPs) {
            // Loop through all destination structures with this suffix
            const auto groupEGPs = e.second.equal_range(T::name);
            for(auto g = groupEGPs.first; g != groupEGPs.second; ++g) {
                // If EGP is scalar, generate code to copy
                if(!Utils::isTypePointer(g->second.type)) {
                    backend.genMergedExtraGlobalParamPush(os, T::name, g->second.mergedGroupIndex,
                                                          std::to_string(g->second.groupIndex),
                                                          g->second.fieldName, e.first);
                }

            }
        }
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
                               return EGPField{g.second.mergedGroupIndex, g.second.type, g.second.fieldName};
                           });
        }

        // Return set
        return mergedGroupFields;
    }

    template<typename T>
    void genMergedStructs(CodeStream &os, const BackendBase &backend, const std::vector<T> &mergedGroups) const
    {
        // Loop through all merged groups and generate struct
        for(const auto &g : mergedGroups) {
            g.generateStruct(os, backend, T::name);
        }
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
            // Loop through resultant fields and generate push function for pointer extra global parameters
            for(auto f : mergedGroupFields) {
                // If EGP is a pointer
                // **NOTE** this is common to all references!
                if(Utils::isTypePointer(f.type)) {
                    os << "void pushMerged" << T::name << f.mergedGroupIndex << f.fieldName << "ToDevice(unsigned int idx, " << backend.getMergedGroupFieldHostType(f.type) << " value)";
                    {
                        CodeStream::Scope b(os);
                        backend.genMergedExtraGlobalParamPush(os, T::name, f.mergedGroupIndex, "idx", f.fieldName, "value");
                    }
                    os << std::endl;
                }
            }
        }
    }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Helper to find merged runner group
    template<typename MergedGroup>
    std::string findMergedRunnerGroup(const typename MergedGroup::GroupInternal &g) const
    {
        // Find group by name
        const auto m = m_MergedRunnerGroups.at(g.getName());

        // Return structure
        return "merged" + MergedGroup::name + "Group" + std::to_string(std::get<0>(m)) + "[" + std::to_string(std::get<1>(m)) + "]";
    }

    template<typename MergedGroup>
    void addMergedRunnerGroups(const std::vector<MergedGroup> &mergedGroups)
    {
        // Loop through merged groups
        for(const auto &g : mergedGroups) {
            // Loop through individual groups inside and add to map
            for(size_t i = 0; i < g.getGroups().size(); i++) {
                auto result = m_MergedRunnerGroups.emplace(std::piecewise_construct,
                                                           std::forward_as_tuple(g.getGroups().at(i).get().getName()),
                                                           std::forward_as_tuple(g.getIndex(), i));
                assert(result.second);
            }
        }
    }

    template<typename Group, typename MergedGroup, typename D>
    void createMergedRunnerGroupsHash(const ModelSpecInternal &model, const BackendBase &backend,
                                      const std::map<std::string, Group> &unmergedGroups, std::vector<MergedGroup> &mergedGroups, 
                                      D getHashDigest)
    {
        // Create a hash map to group together groups with the same SHA1 digest
        std::unordered_map<boost::uuids::detail::sha1::digest_type, 
                           std::vector<std::reference_wrapper<const Group>>, 
                           Utils::SHA1Hash> protoMergedGroups;

        // Add unmerged groups to correct vector
        for(const auto &g : unmergedGroups) {
            protoMergedGroups[std::invoke(getHashDigest, g.second)].push_back(g.second);
        }

        // Reserve final merged groups vector
        mergedGroups.reserve(protoMergedGroups.size());

        // Loop through resultant merged groups
        size_t i = 0;
        for(const auto &p : protoMergedGroups) {
            // Add group to vector
            mergedGroups.emplace_back(i, model.getPrecision(), model.getTimePrecision(), backend, p.second);

            i++;
        }
    }

    template<typename Group, typename MergedGroup, typename D>
    void createMergedRuntimeGroupsHash(const ModelSpecInternal &model, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const Group>> &unmergedGroups,
                                       std::vector<MergedGroup> &mergedGroups, D getHashDigest)
    {
        // Create a hash map to group together groups with the same SHA1 digest
        std::unordered_map<boost::uuids::detail::sha1::digest_type, 
                           std::vector<std::reference_wrapper<const Group>>, 
                           Utils::SHA1Hash> protoMergedGroups;

        // Add unmerged groups to correct vector
        for(const auto &g : unmergedGroups) {
            protoMergedGroups[std::invoke(getHashDigest, g.get())].push_back(g);
        }

        // Reserve final merged groups vector
        mergedGroups.reserve(protoMergedGroups.size());

        // Loop through resultant merged groups
        size_t i = 0;
        for(const auto &p : protoMergedGroups) {
            // Add group to vector
            mergedGroups.emplace_back(i, model.getPrecision(), model.getTimePrecision(), backend, p.second);

            // Loop through fields
            for(const auto &f : mergedGroups.back().getFields()) {
                // If field is an EGP, add record to merged EGPS
                if(std::get<3>(f) == MergedGroup::FieldType::PointerEGP || std::get<3>(f) == MergedGroup::FieldType::ScalarEGP) {
                    // Loop through groups within newly-created merged group
                    for(size_t groupIndex = 0; groupIndex < mergedGroups.back().getGroups().size(); groupIndex++) {
                        const auto &g = mergedGroups.back().getGroups()[groupIndex];

                        // Add reference to this group's variable to data structure
                        m_MergedEGPs[std::get<2>(f)(g, groupIndex)].emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(MergedGroup::name),
                            std::forward_as_tuple(i, groupIndex, std::get<0>(f), std::get<1>(f)));
                    }
                }
            }

            i++;
        }
    }

    template<typename Group, typename MergedGroup, typename F, typename U>
    void createMergedRuntimeGroupsHash(const ModelSpecInternal &model, const BackendBase &backend,
                                       const std::map<std::string, Group> &groups, std::vector<MergedGroup> &mergedGroups,
                                       F filter, U updateHash)
    {
        // Build temporary vector of references to groups that pass filter
        std::vector<std::reference_wrapper<const Group>> unmergedGroups;
        for(const auto &g : groups) {
            if(filter(g.second)) {
                unmergedGroups.emplace_back(std::cref(g.second));
            }
        }

        // Merge filtered vector
        createMergedRuntimeGroupsHash(model, backend, unmergedGroups, mergedGroups, updateHash);
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! Underlying, unmerged model
    const ModelSpecInternal &m_Model;

    //! Merged neuron groups used for generating runner
    std::vector<NeuronRunnerGroupMerged> m_MergedNeuronRunnerGroups;

    //! Merged synapse groups used for generating runner
    std::vector<SynapseRunnerGroupMerged> m_MergedSynapseRunnerGroups; 

    //! Merged current sources used for generating runner
    std::vector<CurrentSourceRunnerGroupMerged> m_MergedCurrentSourceRunnerGroups;

    //! Merged custom updates used for generating runner
    std::vector<CustomUpdateRunnerGroupMerged> m_MergedCustomUpdateRunnerGroups;

    //! Merged custom WU updates used for generating runner
    std::vector<CustomUpdateWURunnerGroupMerged> m_MergedCustomUpdateWURunnerGroups;

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

    //! Merged custom update groups with dense connectivity which require initialisation
    std::vector<CustomWUUpdateDenseInitGroupMerged> m_MergedCustomWUUpdateDenseInitGroups;

    //! Merged synapse groups with dense connectivity which require initialisation
    std::vector<SynapseDenseInitGroupMerged> m_MergedSynapseDenseInitGroups;

    //! Merged synapse groups which require connectivity initialisation
    std::vector<SynapseConnectivityInitGroupMerged> m_MergedSynapseConnectivityInitGroups;

    //! Merged synapse groups with sparse connectivity which require initialisation
    std::vector<SynapseSparseInitGroupMerged> m_MergedSynapseSparseInitGroups;
    
    //! Merged synapse groups with kernel connectivity which require initialisation
    std::vector<SynapseKernelInitGroupMerged> m_MergedSynapseKernelInitGroups;

    //! Merged custom update groups with sparse connectivity which require initialisation
    std::vector<CustomWUUpdateSparseInitGroupMerged> m_MergedCustomWUUpdateSparseInitGroups;

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

    //! Get merged custom update groups where host reduction needs to be performed
    std::vector<CustomUpdateHostReductionGroupMerged> m_MergedCustomUpdateHostReductionGroups;

    //! Get merged custom weight update groups where host reduction needs to be performed
    std::vector<CustomWUUpdateHostReductionGroupMerged> m_MergedCustomWUUpdateHostReductionGroups;

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

    //! Map of group names to index of merged group and index of group within that
    std::unordered_map<std::string, std::tuple<unsigned int, unsigned int>> m_MergedRunnerGroups;
};
}   // namespace CodeGenerator
