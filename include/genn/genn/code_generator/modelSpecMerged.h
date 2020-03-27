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
    // Public API
    //--------------------------------------------------------------------------
    //! Get underlying, unmerged model
    const ModelSpecInternal &getModel() const{ return m_Model; }

    //! Get merged neuron groups which require updating
    const std::vector<NeuronGroupMerged> &getMergedNeuronUpdateGroups() const{ return m_MergedNeuronUpdateGroups; }

    //! Get merged synapse groups which require presynaptic updates
    const std::vector<SynapseGroupMerged> &getMergedPresynapticUpdateGroups() const{ return m_MergedPresynapticUpdateGroups; }

    //! Get merged synapse groups which require postsynaptic updates
    const std::vector<SynapseGroupMerged> &getMergedPostsynapticUpdateGroups() const{ return m_MergedPostsynapticUpdateGroups; }

    //! Get merged synapse groups which require synapse dynamics
    const std::vector<SynapseGroupMerged> &getMergedSynapseDynamicsGroups() const{ return m_MergedSynapseDynamicsGroups; }

    //! Get merged neuron groups which require initialisation
    const std::vector<NeuronGroupMerged> &getMergedNeuronInitGroups() const{ return m_MergedNeuronInitGroups; }

    //! Get merged synapse groups with dense connectivity which require initialisation
    const std::vector<SynapseGroupMerged> &getMergedSynapseDenseInitGroups() const{ return m_MergedSynapseDenseInitGroups; }

    //! Get merged synapse groups which require connectivity initialisation
    const std::vector<SynapseGroupMerged> &getMergedSynapseConnectivityInitGroups() const{ return m_MergedSynapseConnectivityInitGroups; }

    //! Get merged synapse groups with sparse connectivity which require initialisation
    const std::vector<SynapseGroupMerged> &getMergedSynapseSparseInitGroups() const{ return m_MergedSynapseSparseInitGroups; }

    //! Get merged neuron groups which require their spike queues updating
    const std::vector<NeuronSpikeQueueUpdateMergedGroup> &getMergedNeuronSpikeQueueUpdateGroups() const { return m_MergedNeuronSpikeQueueUpdateGroups; }

    //! Get merged synapse groups which require their dendritic delay updating
    const std::vector<SynapseGroupMerged> &getMergedSynapseDendriticDelayUpdateGroups() const { return m_MergedSynapseDendriticDelayUpdateGroups; }

    //! Merged synapse groups which require host code to initialise their synaptic connectivity
    const std::vector<SynapseGroupMerged> &getMergedSynapseConnectivityHostInitGroups() const{ return m_MergedSynapseConnectivityHostInitGroups; }

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

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! Underlying, unmerged model
    const ModelSpecInternal &m_Model;

    //! Merged neuron groups which require updating
    std::vector<NeuronGroupMerged> m_MergedNeuronUpdateGroups;

    //! Merged synapse groups which require presynaptic updates
    std::vector<SynapseGroupMerged> m_MergedPresynapticUpdateGroups;

    //! Merged synapse groups which require postsynaptic updates
    std::vector<SynapseGroupMerged> m_MergedPostsynapticUpdateGroups;

    //! Merged synapse groups which require synapse dynamics update
    std::vector<SynapseGroupMerged> m_MergedSynapseDynamicsGroups;

    //! Merged neuron groups which require initialisation
    std::vector<NeuronGroupMerged> m_MergedNeuronInitGroups;

    //! Merged synapse groups with dense connectivity which require initialisation
    std::vector<SynapseGroupMerged> m_MergedSynapseDenseInitGroups;

    //! Merged synapse groups which require connectivity initialisation
    std::vector<SynapseGroupMerged> m_MergedSynapseConnectivityInitGroups;

    //! Merged synapse groups with sparse connectivity which require initialisation
    std::vector<SynapseGroupMerged> m_MergedSynapseSparseInitGroups;

    //! Merged neuron groups which require their spike queues updating
    std::vector<NeuronSpikeQueueUpdateMergedGroup> m_MergedNeuronSpikeQueueUpdateGroups;

    //! Merged synapse groups which require their dendritic delay updating
    std::vector<SynapseGroupMerged> m_MergedSynapseDendriticDelayUpdateGroups;

    //! Merged synapse groups which require host code to initialise their synaptic connectivity
    std::vector<SynapseGroupMerged> m_MergedSynapseConnectivityHostInitGroups;

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

};
}   // namespace CodeGenerator
