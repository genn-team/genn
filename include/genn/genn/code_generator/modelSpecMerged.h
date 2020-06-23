#pragma once

// Standard C++ includes
#include <vector>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"
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
class ModelSpecMerged
{
public:
    ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend);

    //--------------------------------------------------------------------------
    // CodeGenerator::ModelSpecMerged::MergedEGP
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking where an extra global variable ends up after merging
    struct MergedEGP
    {
        MergedEGP(size_t m, size_t g, const std::string &t, const std::string &f)
            : mergedGroupIndex(m), groupIndex(g), type(t), fieldName(f) {}

        const size_t mergedGroupIndex;
        const size_t groupIndex;
        const std::string type;
        const std::string fieldName;
    };

    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    //! Map of original extra global param names to their locations within merged structures
    typedef std::map<std::string, std::unordered_multimap<std::string, MergedEGP>> MergedEGPMap;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Get underlying, unmerged model
    const ModelSpecInternal &getModel() const{ return m_Model; }

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

    //! Get merged synapse groups with dense connectivity which require initialisation
    const std::vector<SynapseDenseInitGroupMerged> &getMergedSynapseDenseInitGroups() const{ return m_MergedSynapseDenseInitGroups; }

    //! Get merged synapse groups which require connectivity initialisation
    const std::vector<SynapseConnectivityInitGroupMerged> &getMergedSynapseConnectivityInitGroups() const{ return m_MergedSynapseConnectivityInitGroups; }

    //! Get merged synapse groups with sparse connectivity which require initialisation
    const std::vector<SynapseSparseInitGroupMerged> &getMergedSynapseSparseInitGroups() const{ return m_MergedSynapseSparseInitGroups; }

    //! Get merged neuron groups which require their spike queues updating
    const std::vector<NeuronSpikeQueueUpdateGroupMerged> &getMergedNeuronSpikeQueueUpdateGroups() const { return m_MergedNeuronSpikeQueueUpdateGroups; }

    //! Get merged synapse groups which require their dendritic delay updating
    const std::vector<SynapseDendriticDelayUpdateGroupMerged> &getMergedSynapseDendriticDelayUpdateGroups() const { return m_MergedSynapseDendriticDelayUpdateGroups; }

    //! Merged synapse groups which require host code to initialise their synaptic connectivity
    const std::vector<SynapseConnectivityHostInitGroupMerged> &getMergedSynapseConnectivityHostInitGroups() const{ return m_MergedSynapseConnectivityHostInitGroups; }

    void genMergedNeuronUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedNeuronUpdateGroups); }
    void genMergedPresynapticUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedPresynapticUpdateGroups); }
    void genMergedPostsynapticUpdateGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedPostsynapticUpdateGroups); }
    void genMergedSynapseDynamicsGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseDynamicsGroups); }
    void genMergedNeuronInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedNeuronInitGroups); }
    void genMergedSynapseDenseInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseDenseInitGroups); }
    void genMergedSynapseConnectivityInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseConnectivityInitGroups); }
    void genMergedSynapseSparseInitGroupStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseSparseInitGroups); }
    void genMergedNeuronSpikeQueueUpdateStructs(CodeStream &os, const BackendBase &backend) const{ genMergedStructures(os, backend, m_MergedNeuronSpikeQueueUpdateGroups); }
    void genMergedSynapseDendriticDelayUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseDendriticDelayUpdateGroups); }
    void genMergedSynapseConnectivityHostInitStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedSynapseConnectivityHostInitGroups); }


    void genNeuronUpdateGroupSupportCode(CodeStream &os) const{ m_NeuronUpdateSupportCode.gen(os, getModel().getPrecision()); }

    void genPostsynapticDynamicsSupportCode(CodeStream &os) const{ m_PostsynapticDynamicsSupportCode.gen(os, getModel().getPrecision()); }

    void genPresynapticUpdateSupportCode(CodeStream &os) const{ m_PresynapticUpdateSupportCode.gen(os, getModel().getPrecision()); }

    void genPostsynapticUpdateSupportCode(CodeStream &os) const{ m_PostsynapticUpdateSupportCode.gen(os, getModel().getPrecision()); }

    void genSynapseDynamicsSupportCode(CodeStream &os) const{ m_SynapseDynamicsSupportCode.gen(os, getModel().getPrecision()); }

    const std::string &getNeuronUpdateSupportCodeNamespace(const std::string &code) const{ return m_NeuronUpdateSupportCode.getSupportCodeNamespace(code); }

    const std::string &getPostsynapticDynamicsSupportCodeNamespace(const std::string &code) const{ return m_PostsynapticDynamicsSupportCode.getSupportCodeNamespace(code); }

    const std::string &getPresynapticUpdateSupportCodeNamespace(const std::string &code) const{ return m_PresynapticUpdateSupportCode.getSupportCodeNamespace(code); }

    const std::string &getPostsynapticUpdateSupportCodeNamespace(const std::string &code) const{ return m_PostsynapticUpdateSupportCode.getSupportCodeNamespace(code); }

    const std::string &getSynapseDynamicsSupportCodeNamespace(const std::string &code) const{ return m_SynapseDynamicsSupportCode.getSupportCodeNamespace(code); }

    const MergedEGPMap &getMergedEGPs() const { return m_MergedEGPs; }

    //! Generate calls to update all target merged groups
    //! **DEPRECATE** 'scalar' EGPs are innefficient and can now be replaced by 'mutable parameters' which can be explicitely set in merged structures
    void genScalarEGPPush(CodeStream &os, const std::string &suffix, const BackendBase &backend) const;

    template<typename T>
    void genMergedGroupPush(CodeStream &os, const std::vector<T> &groups, const BackendBase &backend) const
    {

        if(!groups.empty()) {
            // Loop through all extra global parameters to build a set of unique filename, group index pairs
            // **YUCK** it would be much nicer if this were part of the original data structure
            // **NOTE** tuple would be nicer but doesn't define std::hash overload
            std::set<std::pair<size_t, std::pair<std::string, std::string>>> mergedGroupFields;
            for(const auto &e : m_MergedEGPs) {
                const auto groupEGPs = e.second.equal_range(T::name);
                std::transform(groupEGPs.first, groupEGPs.second, std::inserter(mergedGroupFields, mergedGroupFields.end()),
                               [](const MergedEGPMap::value_type::second_type::value_type &g)
                               {
                                   return std::make_pair(g.second.mergedGroupIndex,
                                                         std::make_pair(g.second.type, g.second.fieldName));
                               });
            }

            os << "// ------------------------------------------------------------------------" << std::endl;
            os << "// merged extra global parameter functions" << std::endl;
            os << "// ------------------------------------------------------------------------" << std::endl;
            // Loop through resultant fields and generate push function for pointer extra global parameters
            for(auto f : mergedGroupFields) {
                // If EGP is a pointer
                // **NOTE** this is common to all references!
                if(Utils::isTypePointer(f.second.first)) {
                    os << "void pushMerged" << T::name << f.first << f.second.second << "ToDevice(unsigned int idx, " << backend.getMergedGroupFieldHostType(f.second.first) << " value)";
                    {
                        CodeStream::Scope b(os);
                        backend.genMergedExtraGlobalParamPush(os, T::name, f.first, "idx", f.second.second, "value");
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
    template<typename T>
    void genMergedStructures(CodeStream &os, const BackendBase &backend, const std::vector<T> &mergedGroups) const
    {
        // Loop through all merged groups and generate struct
        for(const auto &g : mergedGroups) {
            g.generateStruct(os, backend, T::name);
        }
    }

    template<typename Group, typename MergedGroup, typename M>
    void createMergedGroups(const ModelSpecInternal &model, const BackendBase &backend,
                            std::vector<std::reference_wrapper<const Group>> &unmergedGroups,
                            std::vector<MergedGroup> &mergedGroups, M canMerge)
    {
        // Loop through un-merged  groups
        std::vector<std::vector<std::reference_wrapper<const Group>>> protoMergedGroups;
        while(!unmergedGroups.empty()) {
            // Remove last group from vector
            const Group &group = unmergedGroups.back().get();
            unmergedGroups.pop_back();

            // Loop through existing proto-merged groups
            bool existingMergedGroupFound = false;
            for(auto &p : protoMergedGroups) {
                assert(!p.empty());

                // If our group can be merged with this proto-merged group
                if(canMerge(p.front().get(), group)) {
                    // Add group to vector
                    p.emplace_back(group);

                    // Set flag and stop searching
                    existingMergedGroupFound = true;
                    break;
                }
            }

            // If no existing merged groups were found, 
            // create a new proto-merged group containing just this group
            if(!existingMergedGroupFound) {
                protoMergedGroups.emplace_back();
                protoMergedGroups.back().emplace_back(group);
            }
        }

        // Reserve final merged groups vector
        mergedGroups.reserve(protoMergedGroups.size());

        // Loop through resultant merged groups
        for(size_t i = 0; i < protoMergedGroups.size(); i++) {
            // Add group to vector, moving vectors of groups into data structure to avoid copying
            mergedGroups.emplace_back(i, model.getPrecision(), model.getTimePrecision(), backend,
                                      std::move(protoMergedGroups[i]));
     
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
        }
    }
    
    template<typename Group, typename MergedGroup, typename F, typename M>
    void createMergedGroups(const ModelSpecInternal &model, const BackendBase &backend,
                            const std::map<std::string, Group> &groups, std::vector<MergedGroup> &mergedGroups,
                            F filter, M canMerge)
    {
        // Build temporary vector of references to groups that pass filter
        std::vector<std::reference_wrapper<const Group>> unmergedGroups;
        for(const auto &g : groups) {
            if(filter(g.second)) {
                unmergedGroups.emplace_back(std::cref(g.second));
            }
        }

        // Merge filtered vector
        createMergedGroups(model, backend, unmergedGroups, mergedGroups, canMerge);
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

    //! Merged synapse groups with dense connectivity which require initialisation
    std::vector<SynapseDenseInitGroupMerged> m_MergedSynapseDenseInitGroups;

    //! Merged synapse groups which require connectivity initialisation
    std::vector<SynapseConnectivityInitGroupMerged> m_MergedSynapseConnectivityInitGroups;

    //! Merged synapse groups with sparse connectivity which require initialisation
    std::vector<SynapseSparseInitGroupMerged> m_MergedSynapseSparseInitGroups;

    //! Merged neuron groups which require their spike queues updating
    std::vector<NeuronSpikeQueueUpdateGroupMerged> m_MergedNeuronSpikeQueueUpdateGroups;

    //! Merged synapse groups which require their dendritic delay updating
    std::vector<SynapseDendriticDelayUpdateGroupMerged> m_MergedSynapseDendriticDelayUpdateGroups;

    //! Merged synapse groups which require host code to initialise their synaptic connectivity
    std::vector<SynapseConnectivityHostInitGroupMerged> m_MergedSynapseConnectivityHostInitGroups;

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

    // Map containing mapping of original extra global param names to their locations within merged groups
    MergedEGPMap m_MergedEGPs;

};
}   // namespace CodeGenerator
