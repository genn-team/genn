#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "global.h"
#include "newNeuronModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// NeuronGroup
//------------------------------------------------------------------------
class NeuronGroup
{
public:
    NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                const std::vector<double> &params, const std::vector<NewModels::VarInit> &varInitialisers) :
        m_Name(name), m_NumNeurons(numNeurons), m_IDRange(0, 0), m_PaddedIDRange(0, 0),
        m_NeuronModel(neuronModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_SpikeTimeRequired(false), m_TrueSpikeRequired(false), m_SpikeEventRequired(false), m_QueueRequired(false),
        m_NumDelaySlots(1), m_AnyVarQueuesRequired(false), m_VarQueueRequired(varInitialisers.size(), false),
        m_SpikeVarMode(GENN_PREFERENCES::defaultVarMode), m_SpikeEventVarMode(GENN_PREFERENCES::defaultVarMode),
        m_SpikeTimeVarMode(GENN_PREFERENCES::defaultVarMode), m_VarMode(varInitialisers.size(), GENN_PREFERENCES::defaultVarMode),
        m_HostID(0), m_DeviceID(0)
    {
    }
    NeuronGroup(const NeuronGroup&) = delete;
    NeuronGroup() = delete;

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
    void setSpikeZeroCopyEnabled(bool enabled)
    {
        m_SpikeVarMode = enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST;
    }

    //!< Function to enable the use of zero-copied memory for spike-like events:
    //!< May improve IO performance at the expense of kernel performance
    void setSpikeEventZeroCopyEnabled(bool enabled)
    {
        m_SpikeEventVarMode = enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST;
    }

    //!< Function to enable the use of zero-copied memory for spike times:
    //!< May improve IO performance at the expense of kernel performance
    void setSpikeTimeZeroCopyEnabled(bool enabled)
    {
        m_SpikeTimeVarMode = enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST;
    }

     //!< Function to enable the use zero-copied memory for a particular state variable:
     //!< May improve IO performance at the expense of kernel performance
    void setVarZeroCopyEnabled(const std::string &varName, bool enabled)
    {
        setVarMode(varName, enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST);
    }

    void setSpikeVarMode(VarMode mode) { m_SpikeVarMode = mode; }
    void setSpikeEventVarMode(VarMode mode) { m_SpikeEventVarMode = mode; }
    void setSpikeTimeVarMode(VarMode mode) { m_SpikeTimeVarMode = mode; }

    //!< Function to set variable 'mode' - how variable is implemented in GPU simulation
    //!< This is ignored for CPU simulations
    void setVarMode(const std::string &varName, VarMode mode);

    void setClusterIndex(int hostID, int deviceID){ m_HostID = hostID; m_DeviceID = deviceID; }

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
    const std::vector<NewModels::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    const std::vector<SynapseGroup*> &getInSyn() const{ return m_InSyn; }
    const std::vector<SynapseGroup*> &getOutSyn() const{ return m_OutSyn; }

    bool isSpikeTimeRequired() const{ return m_SpikeTimeRequired; }
    bool isTrueSpikeRequired() const{ return m_TrueSpikeRequired; }
    bool isSpikeEventRequired() const{ return m_SpikeEventRequired; }
    bool isQueueRequired() const{ return m_QueueRequired; }

    bool isVarQueueRequired(const std::string &var) const;
    bool isVarQueueRequired(size_t index) const{ return m_VarQueueRequired[index]; }
    bool isVarQueueRequired() const{ return m_AnyVarQueuesRequired; }

    const std::set<std::pair<std::string, std::string>> &getSpikeEventCondition() const{ return m_SpikeEventCondition; }

    unsigned int getNumDelaySlots() const{ return m_NumDelaySlots; }
    bool isDelayRequired() const{ return (m_NumDelaySlots > 1); }

    bool isSpikeZeroCopyEnabled() const{ return (m_SpikeVarMode & VarLocation::ZERO_COPY); }
    bool isSpikeEventZeroCopyEnabled() const{ return (m_SpikeEventVarMode & VarLocation::ZERO_COPY); }
    bool isSpikeTimeZeroCopyEnabled() const{ return (m_SpikeTimeVarMode & VarLocation::ZERO_COPY); }
    bool isZeroCopyEnabled() const;
    bool isVarZeroCopyEnabled(const std::string &var) const{ return (getVarMode(var) & VarLocation::ZERO_COPY); }

    VarMode getSpikeVarMode() const{ return m_SpikeVarMode; }
    VarMode getSpikeEventVarMode() const{ return m_SpikeEventVarMode; }
    VarMode getSpikeTimeVarMode() const{ return m_SpikeTimeVarMode; }
    VarMode getVarMode(const std::string &varName) const;
    VarMode getVarMode(size_t index) const{ return m_VarMode[index]; }

    bool isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const;

    void addExtraGlobalParams(std::map<std::string, std::string> &kernelParameters) const;

    //!< Does this neuron group require any init code to be run
    bool isInitCodeRequired() const;

    //!< Does this neuron group require an RNG to simulate
    bool isSimRNGRequired() const;

    //!< Does this neuron group require an RNG for it's init code
    bool isInitRNGRequired(VarInit varInitMode) const;

    //!< Is device var init code required for any variables in this neuron group
    bool isDeviceVarInitRequired() const;

    //! Can this neuron group run on the CPU? If we are running in CPU_ONLY mode this is always true,
    //! but some GPU functionality will prevent models being run on both CPU and GPU.
    bool canRunOnCPU() const;

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
    std::vector<NewModels::VarInit> m_VarInitialisers;
    std::vector<SynapseGroup*> m_InSyn;
    std::vector<SynapseGroup*> m_OutSyn;
    bool m_SpikeTimeRequired;
    bool m_TrueSpikeRequired;
    bool m_SpikeEventRequired;
    bool m_QueueRequired;
    std::set<std::pair<std::string, std::string>> m_SpikeEventCondition;
    unsigned int m_NumDelaySlots;

    //!< Vector specifying which variables require queues
    bool m_AnyVarQueuesRequired;
    std::vector<bool> m_VarQueueRequired;

    //!< Whether spikes from neuron group should use zero-copied memory
    VarMode m_SpikeVarMode;

    //!< Whether spike-like events from neuron group should use zero-copied memory
    VarMode m_SpikeEventVarMode;

    //!< Whether spike times from neuron group should use zero-copied memory
    VarMode m_SpikeTimeVarMode;

    //!< Whether indidividual state variables of a neuron group should use zero-copied memory
    std::vector<VarMode> m_VarMode;

    //!< The ID of the cluster node which the neuron groups are computed on
    int m_HostID;

    //!< The ID of the CUDA device which the neuron groups are comnputed on
    int m_DeviceID;
};
