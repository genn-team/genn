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
    ModelSpecMerged(const BackendBase &backend, const ModelSpecInternal &model);
    ModelSpecMerged(const ModelSpecMerged&) = delete;
    ModelSpecMerged &operator=(const ModelSpecMerged &) = delete;

    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    template<typename G>
    using GenMergedGroupFn = std::function<void(G &)>;

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

    //! Get merged custom connectivity update groups which require regeneration of remap structure
    const std::vector<CustomConnectivityRemapUpdateGroupMerged> &getMergedCustomConnectivityRemapUpdateGroups() const { return m_MergedCustomConnectivityRemapUpdateGroups; }

    //! Get merged custom connectivity update groups where host processing needs to be performed
    const std::vector<CustomConnectivityHostUpdateGroupMerged> &getMergedCustomConnectivityHostUpdateGroups() const { return m_MergedCustomConnectivityHostUpdateGroups; }

    void genMergedNeuronUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                     GenMergedGroupFn<NeuronUpdateGroupMerged> generateGroup);
    void genMergedPresynapticUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                          GenMergedGroupFn<PresynapticUpdateGroupMerged> generateGroup);
    void genMergedPostsynapticUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                           GenMergedGroupFn<PostsynapticUpdateGroupMerged> generateGroup);
    void genMergedSynapseDynamicsGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                        GenMergedGroupFn<SynapseDynamicsGroupMerged> generateGroup);
    void genMergedCustomUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                     GenMergedGroupFn<CustomUpdateGroupMerged> generateGroup);
    void genMergedCustomUpdateWUGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                       GenMergedGroupFn<CustomUpdateWUGroupMerged> generateGroup);
    void genMergedCustomUpdateTransposeWUGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                GenMergedGroupFn<CustomUpdateTransposeWUGroupMerged> generateGroup);
    void genMergedCustomUpdateHostReductionGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                  GenMergedGroupFn<CustomUpdateHostReductionGroupMerged> generateGroup);
    void genMergedCustomWUUpdateHostReductionGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                    GenMergedGroupFn<CustomWUUpdateHostReductionGroupMerged> generateGroup);
    void genMergedCustomConnectivityUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                 GenMergedGroupFn<CustomConnectivityUpdateGroupMerged> generateGroup);
    void genMergedCustomConnectivityRemapUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                     GenMergedGroupFn<CustomConnectivityRemapUpdateGroupMerged> generateGroup);
    void genMergedCustomConnectivityHostUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, const std::string &updateGroupName, 
                                                     GenMergedGroupFn<CustomConnectivityHostUpdateGroupMerged> generateGroup);
    void genMergedNeuronSpikeQueueUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                               GenMergedGroupFn<NeuronSpikeQueueUpdateGroupMerged> generateGroup);
    void genMergedNeuronPrevSpikeTimeUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                  GenMergedGroupFn<NeuronPrevSpikeTimeUpdateGroupMerged> generateGroup);
    void genMergedSynapseDendriticDelayUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                    GenMergedGroupFn<SynapseDendriticDelayUpdateGroupMerged> generateGroup);
    void genMergedNeuronInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                   GenMergedGroupFn<NeuronInitGroupMerged> generateGroup);
    void genMergedCustomUpdateInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                         GenMergedGroupFn<CustomUpdateInitGroupMerged> generateGroup);
    void genMergedCustomWUUpdateInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                           GenMergedGroupFn<CustomWUUpdateInitGroupMerged> generateGroup);
    void genMergedSynapseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                    GenMergedGroupFn<SynapseInitGroupMerged> generateGroup);
    void genMergedSynapseConnectivityInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                GenMergedGroupFn<SynapseConnectivityInitGroupMerged> generateGroup);
    void genMergedSynapseSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                          GenMergedGroupFn<SynapseSparseInitGroupMerged> generateGroup);
    void genMergedCustomWUUpdateSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                 GenMergedGroupFn<CustomWUUpdateSparseInitGroupMerged> generateGroup);
    void genMergedCustomConnectivityUpdatePreInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                        GenMergedGroupFn<CustomConnectivityUpdatePreInitGroupMerged> generateGroup);
    void genMergedCustomConnectivityUpdatePostInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                         GenMergedGroupFn<CustomConnectivityUpdatePostInitGroupMerged> generateGroup);
    void genMergedCustomConnectivityUpdateSparseInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                           GenMergedGroupFn<CustomConnectivityUpdateSparseInitGroupMerged> generateGroup);
    void genMergedSynapseConnectivityHostInitGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces, 
                                                    GenMergedGroupFn<SynapseConnectivityHostInitGroupMerged> generateGroup);


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
    void genMergedCustomConnectivityRemapUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityRemapUpdateGroups); }
    void genMergedCustomConnectivityHostUpdateStructs(CodeStream &os, const BackendBase &backend) const { genMergedStructures(os, backend, m_MergedCustomConnectivityHostUpdateGroups); }


    void genMergedNeuronUpdateGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedNeuronUpdateGroups); }
    void genMergedPresynapticUpdateGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedPresynapticUpdateGroups); }
    void genMergedPostsynapticUpdateGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedPostsynapticUpdateGroups); }
    void genMergedSynapseDynamicsGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseDynamicsGroups); }
    void genMergedNeuronInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedNeuronInitGroups); }
    void genMergedCustomUpdateInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomUpdateInitGroups); }
    void genMergedCustomWUUpdateInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomWUUpdateInitGroups); }
    void genMergedSynapseInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseInitGroups); }
    void genMergedSynapseConnectivityInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseConnectivityInitGroups); }
    void genMergedSynapseSparseInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseSparseInitGroups); }
    void genMergedCustomWUUpdateSparseInitGroupHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomWUUpdateSparseInitGroups); }
    void genMergedCustomConnectivityUpdatePreInitHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityUpdatePreInitGroups); }
    void genMergedCustomConnectivityUpdatePostInitHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityUpdatePostInitGroups); }
    void genMergedCustomConnectivityUpdateSparseInitHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityUpdateSparseInitGroups); }
    void genMergedNeuronSpikeQueueUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const{ genHostMergedStructArrayPush(os, backend, m_MergedNeuronSpikeQueueUpdateGroups); }
    void genMergedNeuronPrevSpikeTimeUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const{ genHostMergedStructArrayPush(os, backend, m_MergedNeuronPrevSpikeTimeUpdateGroups); }
    void genMergedSynapseDendriticDelayUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseDendriticDelayUpdateGroups); }
    void genMergedSynapseConnectivityHostInitStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedSynapseConnectivityHostInitGroups); }
    void genMergedCustomUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomUpdateGroups); }
    void genMergedCustomUpdateWUHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomUpdateWUGroups); }
    void genMergedCustomUpdateTransposeWUHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomUpdateTransposeWUGroups); }
    void genMergedCustomUpdateHostReductionHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomUpdateHostReductionGroups); }
    void genMergedCustomWUUpdateHostReductionHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomWUUpdateHostReductionGroups); }
    void genMergedCustomConnectivityUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityUpdateGroups); }
    void genMergedCustomConnectivityRemapUpdateHostStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityRemapUpdateGroups); }
    void genMergedCustomConnectivityHostUpdateStructArrayPush(CodeStream &os, const BackendBase &backend) const { genHostMergedStructArrayPush(os, backend, m_MergedCustomConnectivityHostUpdateGroups); }


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

    //! Are there any destinations within the merged data structures for a particular extra global parameter?
    /*// Get set of unique fields referenced in a merged group
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
    }*/

    template<typename T>
    void genDynamicFieldPush(CodeStream &os, const std::vector<T> &groups, 
                             const BackendBase &backend, bool host = false) const
    {
        // Loop through merged groups
        for(size_t g = 0; g < groups.size(); g++) {
            // Loop through fields
            const auto &mergedGroup = groups[g];
            for(const auto &f : mergedGroup.getFields()) {
                // If field is dynamic, add record to merged EGPS
                if((f.fieldType & GroupMergedFieldType::DYNAMIC)) {
                    // Add reference to this group's variable to data structure
                    // **NOTE** this works fine with EGP references because the function to
                    // get their value will just return the name of the referenced EGP
                    os << "void pushMerged" << T::name << g << f.name << "ToDevice(unsigned int idx, " << backend.getMergedGroupFieldHostTypeName(f.type) << " value)";
                    {
                        CodeStream::Scope b(os);
                        if(host) {
                            os << "merged" << T::name << "Group" << g << "[idx]." << f.name << " = value;" << std::endl;
                        }
                        else {
                            backend.genMergedDynamicVariablePush(os, T::name, g, "idx", f.name, "value");
                        }
                    }
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

     template<typename T>
    void genHostMergedStructArrayPush(CodeStream &os, const BackendBase &backend, const std::vector<T> &mergedGroups) const
    {
        // Loop through all merged groups and generate host arrays and push functions
        for(const auto &g : mergedGroups) {
            g.genHostMergedStructArrayPush(os, backend, T::name);
        }
    }

    template<typename MergedGroup, typename D>
    void createMergedGroups(const std::vector<std::reference_wrapper<const typename MergedGroup::GroupInternal>> &unmergedGroups,
                            std::vector<MergedGroup> &mergedGroups, D getHashDigest)
    {
        // Create a hash map to group together groups with the same SHA1 digest
        std::unordered_map<boost::uuids::detail::sha1::digest_type, 
                           std::vector<std::reference_wrapper<const typename MergedGroup::GroupInternal>>, 
                           Utils::SHA1Hash> protoMergedGroups;

        // Add unmerged groups to correct vector
        for(const auto &g : unmergedGroups) {
            protoMergedGroups[std::invoke(getHashDigest, g.get())].push_back(g);
        }

        // Reserve final merged groups vector
        assert(mergedGroups.empty());
        mergedGroups.reserve(protoMergedGroups.size());

        // Construct merged groups
        size_t i = 0;
        for(auto &p : protoMergedGroups) {
            mergedGroups.emplace_back(i++, m_Model.getTypeContext(), p.second);
        }
    }

    template<typename MergedGroup, typename F, typename D>
    void createMergedGroups(const std::map<std::string, typename MergedGroup::GroupInternal> &groups, 
                            std::vector<MergedGroup> &mergedGroups, F filter, D getHashDigest)
    {
        // Build temporary vector of references to groups that pass filter
        std::vector<std::reference_wrapper<const typename MergedGroup::GroupInternal>> unmergedGroups;
        for(const auto &g : groups) {
            if(filter(g.second)) {
                unmergedGroups.emplace_back(std::cref(g.second));
            }
        }

        // Merge filtered vector
        createMergedGroups(unmergedGroups, mergedGroups, getHashDigest);
    }

    template<typename MergedGroup>
    void genMergedGroup(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces,
                        MergedGroup &mergedGroup, GenMergedGroupFn<MergedGroup> generateGroup)
    {
        // Call generate function
        generateGroup(mergedGroup);

        // Assign memory spaces
        mergedGroup.assignMemorySpaces(backend, memorySpaces);
    }

    template<typename MergedGroup>
    void genMergedGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces,
                         std::vector<MergedGroup> &mergedGroups, GenMergedGroupFn<MergedGroup> generateGroup)
    {
        // Loop through merged groups and generate
        for(auto &g : mergedGroups) {
            genMergedGroup(backend, memorySpaces, g, generateGroup);
        }
    }

    template<typename MergedGroup>
    void genMergedCustomUpdateGroups(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces,
                                     std::vector<MergedGroup> &mergedGroups, const std::string &updateGroupName,
                                     GenMergedGroupFn<MergedGroup> generateGroup)
    {
        // Loop through merged groups and generate if they are in specified update group
        for(auto &g : mergedGroups) {
            if(g.getArchetype().getUpdateGroupName() == updateGroupName) {
                genMergedGroup(backend, memorySpaces, g, generateGroup);
            }
        }
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

    //! Merged custom connectivity update groups which require remap data structure regenerating
    std::vector<CustomConnectivityRemapUpdateGroupMerged> m_MergedCustomConnectivityRemapUpdateGroups;

    //! Merged custom connectivity update groups where host processing needs to be performed
    std::vector<CustomConnectivityHostUpdateGroupMerged> m_MergedCustomConnectivityHostUpdateGroups;
};
}   // namespace GeNN::CodeGenerator
