#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "neuronGroup.h"
#include "newPostsynapticModels.h"
#include "newWeightUpdateModels.h"
#include "synapseMatrixType.h"

//------------------------------------------------------------------------
// SynapseGroup
//------------------------------------------------------------------------
class SynapseGroup
{
public:
    SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                 const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<double> &wuInitVals,
                 const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<double> &psInitVals,
                 NeuronGroup *srcNeuronGroup, NeuronGroup *trgNeuronGroup) :
        m_PaddedKernelIDRange(0, 0), m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_DelaySteps(delaySteps), m_MaxConnections(trgNeuronGroup->getNumNeurons()), m_MatrixType(matrixType),
        m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup),
        m_TrueSpikeRequired(false), m_SpikeEventRequired(false), m_EventThresholdReTestRequired(false),
        m_WUModel(wu), m_WUParams(wuParams), m_WUInitVals(wuInitVals), m_PSModel(ps), m_PSParams(psParams), m_PSInitVals(psInitVals),
        m_HostID(0), m_DeviceID(0)
    {
    }

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class SpanType
    {
        POSTSYNAPTIC,
        PRESYNAPTIC
    };

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    NeuronGroup *getSrcNeuronGroup(){ return m_SrcNeuronGroup; }
    NeuronGroup *getTrgNeuronGroup(){ return m_TrgNeuronGroup; }

    void setTrueSpikeRequired(bool req){ m_TrueSpikeRequired = req; }
    void setSpikeEventRequired(bool req){ m_SpikeEventRequired = req; }
    void setEventThresholdReTestRequired(bool req){ m_EventThresholdReTestRequired = req; }

    //!< Function to enable the use of zero-copied memory for a particular weight update model state variable:
    //!< May improve IO performance at the expense of kernel performance
    void setWUVarZeroCopyEnabled(const std::string &varName, bool enabled);

    //!< Function to enable the use zero-copied memory for a particular postsynaptic model state variable
    //!< May improve IO performance at the expense of kernel performance
    void setPSVarZeroCopyEnabled(const std::string &varName, bool enabled);
    int getClusterHostID(){ return m_TrgNeuronGroup->getClusterHostID(); }
    int getClusterDeviceID(){ return m_TrgNeuronGroup->getClusterDeviceID(); }

    void setMaxConnections(unsigned int maxConnections);
    void setSpanType(SpanType spanType);

    void initDerivedParams(double dt);
    void calcKernelSizes(unsigned int blockSize, unsigned int &paddedKernelIDStart);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    std::pair<unsigned int, unsigned int> getPaddedKernelIDRange() const{ return m_PaddedKernelIDRange; }

    const std::string &getName() const{ return m_Name; }

    SpanType getSpanType() const{ return m_SpanType; }
    unsigned int getDelaySteps() const{ return m_DelaySteps; }
    unsigned int getMaxConnections() const{ return m_MaxConnections; }
    SynapseMatrixType getMatrixType() const{ return m_MatrixType; }

    unsigned int getPaddedDynKernelSize(unsigned int blockSize) const;
    unsigned int getPaddedPostLearnKernelSize(unsigned int blockSize) const;

    const NeuronGroup *getSrcNeuronGroup() const{ return m_SrcNeuronGroup; }
    const NeuronGroup *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    bool isTrueSpikeRequired() const{ return m_TrueSpikeRequired; }
    bool isSpikeEventRequired() const{ return m_SpikeEventRequired; }
    bool isEventThresholdReTestRequired() const{ return m_EventThresholdReTestRequired; }

    const WeightUpdateModels::Base *getWUModel() const{ return m_WUModel; }

    const std::vector<double> &getWUParams() const{ return m_WUParams; }
    const std::vector<double> &getWUDerivedParams() const{ return m_WUDerivedParams; }
    const std::vector<double> &getWUInitVals() const{ return m_WUInitVals; }

    const PostsynapticModels::Base *getPSModel() const{ return m_PSModel; }

    const std::vector<double> &getPSParams() const{ return m_PSParams; }
    const std::vector<double> &getPSDerivedParams() const{ return m_PSDerivedParams; }
    const std::vector<double> &getPSInitVals() const{ return m_PSInitVals; }

    bool isZeroCopyEnabled() const;
    bool isWUVarZeroCopyEnabled(const std::string &var) const;
    bool isPSVarZeroCopyEnabled(const std::string &var) const;

    //!< Is this synapse group too large to use shared memory for combining postsynaptic output
    // **THINK** this is very cuda-specific
    bool isPSAtomicAddRequired(unsigned int blockSize) const;

    void addExtraGlobalSynapseParams(std::map<string, string> &kernelParameters) const;
    void addExtraGlobalNeuronParams(std::map<string, string> &kernelParameters) const;

    // **THINK** do these really belong here - they are very code-generation specific
    std::string getOffsetPre() const;
    std::string getOffsetPost(const std::string &devPrefix) const;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void addExtraGlobalSimParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                 std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalPostLearnParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                       std::map<std::string, std::string> &kernelParameters) const;
    void addExtraGlobalSynapseDynamicsParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                             std::map<std::string, std::string> &kernelParameters) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Range of indices of this synapse group in synapse kernel
    std::pair<unsigned int, unsigned int> m_PaddedKernelIDRange;

    //!< Name of the synapse group
    std::string m_Name;

    //!< Execution order of synapses in the kernel. It determines whether synapses are executed in parallel for every postsynaptic neuron, or for every presynaptic neuron.
    SpanType m_SpanType;

    //!< Global synaptic conductance delay for the group (in time steps)
    unsigned int m_DelaySteps;

    //!< Padded summed maximum number of connections for a neuron in the neuron groups
    unsigned int m_MaxConnections;

    //!< Connectivity type of synapses
    SynapseMatrixType m_MatrixType;

    //!< Pointer to presynaptic neuron group
    NeuronGroup *m_SrcNeuronGroup;

    //!< Pointer to postsynaptic neuron group
    NeuronGroup *m_TrgNeuronGroup;

    //!< Defines if synapse update is done after detection of real spikes (only one point after threshold)
    bool m_TrueSpikeRequired;

    //!< Defines if synapse update is done after detection of spike events (every point above threshold)
    bool m_SpikeEventRequired;

    //!< Defines whether the Evnt Threshold needs to be retested in the synapse kernel due to multiple non-identical events in the pre-synaptic neuron population
    bool m_EventThresholdReTestRequired;

    //!< Weight update model type
    const WeightUpdateModels::Base *m_WUModel;

    //!< Parameters of weight update model
    std::vector<double> m_WUParams;

    //!< Derived parameters for weight update model
    std::vector<double> m_WUDerivedParams;

    //!< Initial values for weight update model
    std::vector<double> m_WUInitVals;

    //!< Post synapse update model type
    const PostsynapticModels::Base *m_PSModel;

    //!< Parameters of post synapse model
    std::vector<double> m_PSParams;

    //!< Derived parameters for post synapse model
    std::vector<double> m_PSDerivedParams;

    //!< Initial values for post synapse model
    std::vector<double> m_PSInitVals;

    //!< Whether indidividual state variables of weight update model should use zero-copied memory
    std::set<string> m_WUVarZeroCopyEnabled;

    //!< Whether indidividual state variables of post synapse should use zero-copied memory
    std::set<string> m_PSVarZeroCopyEnabled;

    //!< The ID of the cluster node which the synapse group is computed on
    int m_HostID;

    //!< The ID of the CUDA device which the synapse group is comnputed on
    int m_DeviceID;
};