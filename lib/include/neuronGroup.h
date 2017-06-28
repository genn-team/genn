#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "newNeuronModels.h"

//------------------------------------------------------------------------
// NeuronGroup
//------------------------------------------------------------------------
class NeuronGroup
{
public:
    NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                const std::vector<double> &params, const std::vector<double> &initVals) :
        m_Name(name), m_NumNeurons(numNeurons), m_IDRange(0, 0), m_PaddedIDRange(0, 0),
        m_NeuronModel(neuronModel), m_Params(params), m_InitVals(initVals),
        m_SpikeTimeRequired(false), m_TrueSpikeRequired(false), m_SpikeEventRequired(false), m_QueueRequired(false),
        m_NumDelaySlots(1),
#ifdef MPI_ENABLE
        m_SpikeZeroCopyEnabled(false), m_SpikeEventZeroCopyEnabled(false), m_SpikeTimeZeroCopyEnabled(false),
        m_HostID(0), m_DeviceID(0)
#else
        m_SpikeZeroCopyEnabled(false), m_SpikeEventZeroCopyEnabled(false), m_SpikeTimeZeroCopyEnabled(false)
#endif
    {
    }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //!< Checks delay slots currently provided by the neuron group against a required delay and extends if required
    void checkNumDelaySlots(unsigned int requiredDelay);

    // Update which variables require queues based on piece of code
    void updateVarQueues(const std::string &code);

    void setSpikeTimeRequired(bool req){ m_SpikeTimeRequired = req; }
    void setTrueSpikeRequired(bool req){ m_TrueSpikeRequired = req; }
    void setSpikeEventRequired(bool req){ m_SpikeEventRequired = req; }

    //!< Function to enable the use of zero-copied memory for spikes:
    //!< May improve IO performance at the expense of kernel performance
    void setSpikeZeroCopyEnabled(bool enabled){ m_SpikeZeroCopyEnabled = enabled; }

    //!< Function to enable the use of zero-copied memory for spike-like events:
    //!< May improve IO performance at the expense of kernel performance
    void setSpikeEventZeroCopyEnabled(bool enabled){ m_SpikeEventZeroCopyEnabled = enabled; }

    //!< Function to enable the use of zero-copied memory for spike times:
    //!< May improve IO performance at the expense of kernel performance
    void setSpikeTimeZeroCopyEnabled(bool enabled){ m_SpikeTimeZeroCopyEnabled = enabled; }

     //!< Function to enable the use zero-copied memory for a particular state variable:
     //!< May improve IO performance at the expense of kernel performance
    void setVarZeroCopyEnabled(const std::string &varName, bool enabled);

#ifdef MPI_ENABLE
    void setClusterIndex(int hostID, int deviceID){ m_HostID = hostID; m_DeviceID = deviceID; }

    int getClusterHostID(){ return m_HostID; }

    int getClusterDeviceID(){ return m_DeviceID; }
#endif

    void addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace);

    void addInSyn(SynapseGroup *synapseGroup){ m_InSyn.push_back(synapseGroup); }
    void addOutSyn(SynapseGroup *synapseGroup){ m_OutSyn.push_back(synapseGroup); }

    void initDerivedParams(double dt);
    void calcSizes(unsigned int blockSize, unsigned int &idStart, unsigned int &paddedIDStart);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    unsigned int getNumNeurons() const{ return m_NumNeurons; }
    const std::pair<unsigned int, unsigned int> &getPaddedIDRange() const{ return m_PaddedIDRange; }
    const std::pair<unsigned int, unsigned int> &getIDRange() const{ return m_IDRange; }
    const NeuronModels::Base *getNeuronModel() const{ return m_NeuronModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }
    const std::vector<double> &getInitVals() const{ return m_InitVals; }

    const std::vector<SynapseGroup*> &getInSyn() const{ return m_InSyn; }
    const std::vector<SynapseGroup*> &getOutSyn() const{ return m_OutSyn; }

    bool isSpikeTimeRequired() const{ return m_SpikeTimeRequired; }
    bool isTrueSpikeRequired() const{ return m_TrueSpikeRequired; }
    bool isSpikeEventRequired() const{ return m_SpikeEventRequired; }
    bool isQueueRequired() const{ return m_QueueRequired; }

    bool isVarQueueRequired(const std::string &var) const;
    bool isVarQueueRequired() const{ return !m_VarQueueRequired.empty(); }

    const std::set<std::pair<std::string, std::string>> &getSpikeEventCondition() const{ return m_SpikeEventCondition; }

    unsigned int getNumDelaySlots() const{ return m_NumDelaySlots; }
    bool isDelayRequired() const{ return (m_NumDelaySlots > 1); }

    bool isSpikeZeroCopyEnabled() const{ return m_SpikeZeroCopyEnabled; }
    bool isSpikeEventZeroCopyEnabled() const{ return m_SpikeEventZeroCopyEnabled; }
    bool isSpikeTimeZeroCopyEnabled() const{ return m_SpikeTimeZeroCopyEnabled; }
    bool isZeroCopyEnabled() const;
    bool isVarZeroCopyEnabled(const std::string &var) const;

    bool isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const;

    void addExtraGlobalParams(std::map<std::string, std::string> &kernelParameters) const;

    // **THINK** do this really belong here - it is very code-generation specific
    std::string getQueueOffset(const std::string &devPrefix) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    unsigned int m_NumNeurons;
    std::pair<unsigned int, unsigned int> m_IDRange;
    std::pair<unsigned int, unsigned int> m_PaddedIDRange;

    const NeuronModels::Base *m_NeuronModel;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<double> m_InitVals;
    std::vector<SynapseGroup*> m_InSyn;
    std::vector<SynapseGroup*> m_OutSyn;
    bool m_SpikeTimeRequired;
    bool m_TrueSpikeRequired;
    bool m_SpikeEventRequired;
    bool m_QueueRequired;
    std::set<std::pair<std::string, std::string>> m_SpikeEventCondition;
    unsigned int m_NumDelaySlots;

    //!< Vector specifying which variables require queues
    std::set<string> m_VarQueueRequired;

    //!< Whether spikes from neuron group should use zero-copied memory
    bool m_SpikeZeroCopyEnabled;

    //!< Whether spike-like events from neuron group should use zero-copied memory
    bool m_SpikeEventZeroCopyEnabled;

    //!< Whether spike times from neuron group should use zero-copied memory
    bool m_SpikeTimeZeroCopyEnabled;

    //!< Whether indidividual state variables of a neuron group should use zero-copied memory
    std::set<string> m_VarZeroCopyEnabled;

#ifdef MPI_ENABLE
    //!< The ID of the cluster node which the neuron groups are computed on
    int m_HostID;

    //!< The ID of the CUDA device which the neuron groups are comnputed on
    int m_DeviceID;
#endif
};
