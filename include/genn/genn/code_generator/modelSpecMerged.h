#pragma once

// Standard C++ includes
#include <vector>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"

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
};
}   // namespace CodeGenerator
