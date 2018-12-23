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

class CurrentSource;

//------------------------------------------------------------------------
// NeuronGroup
//------------------------------------------------------------------------
class NeuronGroup
{
public:
    NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                const std::vector<double> &params, const std::vector<NewModels::VarInit> &varInitialisers, int hostID, int deviceID) :
        m_Name(name), m_NumNeurons(numNeurons), m_NeuronModel(neuronModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_SpikeTimeRequired(false), m_TrueSpikeRequired(false), m_SpikeEventRequired(false),
        m_NumDelaySlots(1), m_VarQueueRequired(varInitialisers.size(), false),
        m_SpikeVarMode(GENN_PREFERENCES::defaultVarMode), m_SpikeEventVarMode(GENN_PREFERENCES::defaultVarMode),
        m_SpikeTimeVarMode(GENN_PREFERENCES::defaultVarMode), m_VarMode(varInitialisers.size(), GENN_PREFERENCES::defaultVarMode),
        m_HostID(hostID), m_DeviceID(deviceID)
    {
    }
    NeuronGroup(const NeuronGroup&) = delete;
    NeuronGroup() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Checks delay slots currently provided by the neuron group against a required delay and extends if required
    void checkNumDelaySlots(unsigned int requiredDelay);

    //! Update which presynaptic variables require queues based on piece of code
    void updatePreVarQueues(const std::string &code);

    //! Update which postsynaptic variables  require queues based on piece of code
    void updatePostVarQueues(const std::string &code);

    void setSpikeTimeRequired(bool req){ m_SpikeTimeRequired = req; }
    void setTrueSpikeRequired(bool req){ m_TrueSpikeRequired = req; }
    void setSpikeEventRequired(bool req){ m_SpikeEventRequired = req; }

    //! Set variable mode used for variables containing this neuron group's output spikes
    /*! This is ignored for CPU simulations */
    void setSpikeVarMode(VarMode mode) { m_SpikeVarMode = mode; }

     //! Set variable mode used for variables containing this neuron group's output spike events
     /*! This is ignored for CPU simulations */
    void setSpikeEventVarMode(VarMode mode) { m_SpikeEventVarMode = mode; }

    //! Set variable mode used for variables containing this neuron group's output spike times
    /*! This is ignored for CPU simulations */
    void setSpikeTimeVarMode(VarMode mode) { m_SpikeTimeVarMode = mode; }

    //! Set variable mode of neuron model state variable
    /*! This is ignored for CPU simulations */
    void setVarMode(const std::string &varName, VarMode mode);

    void addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace);

    void addInSyn(SynapseGroup *synapseGroup){ m_InSyn.push_back(synapseGroup); }
    void addOutSyn(SynapseGroup *synapseGroup){ m_OutSyn.push_back(synapseGroup); }

    void initDerivedParams(double dt);
 
    //! Merge incoming postsynaptic models
    void mergeIncomingPSM();

    //! add input current source
    void injectCurrent(CurrentSource *source);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets number of neurons in group
    unsigned int getNumNeurons() const{ return m_NumNeurons; }

    //! Gets the neuron model used by this group
    const NeuronModels::Base *getNeuronModel() const{ return m_NeuronModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }
    const std::vector<NewModels::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    //! Gets pointers to all synapse groups which provide input to this neuron group
    const std::vector<SynapseGroup*> &getInSyn() const{ return m_InSyn; }
    const std::vector<std::pair<SynapseGroup*, std::vector<SynapseGroup*>>> &getMergedInSyn() const{ return m_MergedInSyn; }

    //! Gets pointers to all synapse groups emanating from this neuron group
    const std::vector<SynapseGroup*> &getOutSyn() const{ return m_OutSyn; }

    //! Gets pointers to all current sources which provide input to this neuron group
    const std::vector<CurrentSource*> &getCurrentSources() const { return m_CurrentSources; }

    int getClusterHostID() const{ return m_HostID; }

    int getClusterDeviceID() const{ return m_DeviceID; }

    bool isSpikeTimeRequired() const{ return m_SpikeTimeRequired; }
    bool isTrueSpikeRequired() const{ return m_TrueSpikeRequired; }
    bool isSpikeEventRequired() const{ return m_SpikeEventRequired; }

    bool isVarQueueRequired(const std::string &var) const;
    bool isVarQueueRequired(size_t index) const{ return m_VarQueueRequired[index]; }

    const std::set<std::pair<std::string, std::string>> &getSpikeEventCondition() const{ return m_SpikeEventCondition; }

    unsigned int getNumDelaySlots() const{ return m_NumDelaySlots; }
    bool isDelayRequired() const{ return (m_NumDelaySlots > 1); }
    bool isZeroCopyEnabled() const;

    //! Get variable mode used for variables containing this neuron group's output spikes
    VarMode getSpikeVarMode() const{ return m_SpikeVarMode; }

    //! Get variable mode used for variables containing this neuron group's output spike events
    VarMode getSpikeEventVarMode() const{ return m_SpikeEventVarMode; }

    //! Get variable mode used for variables containing this neuron group's output spike times
    VarMode getSpikeTimeVarMode() const{ return m_SpikeTimeVarMode; }

    //! Get variable mode used by neuron model state variable
    VarMode getVarMode(const std::string &varName) const;

    //! Get variable mode used by neuron model state variable
    VarMode getVarMode(size_t index) const{ return m_VarMode[index]; }

    //! Do any of the spike event conditions tested by this neuron require specified parameter?
    bool isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const;

    void addExtraGlobalParams(std::map<std::string, std::string> &kernelParameters) const;

    //! Does this neuron group require any initialisation code to be run?
    bool isInitCodeRequired() const;

    //! Does this neuron group require an RNG to simulate?
    bool isSimRNGRequired() const;

    //! Does this neuron group require an RNG for it's init code?
    bool isInitRNGRequired(VarInit varInitMode) const;

    //! Is device var init code required for any variables in this neuron group?
    bool isDeviceVarInitRequired() const;

    //! Is any form of device initialisation required?
    bool isDeviceInitRequired() const;

    //! Does this neuron group have outgoing connections specified host id?
    bool hasOutputToHost(int targetHostID) const;

    //! Get the expression to calculate the queue offset for accessing state of variables this timestep
    std::string getCurrentQueueOffset(const std::string &devPrefix) const;

    //! Get the expression to calculate the queue offset for accessing state of variables in previous timestep
    std::string getPrevQueueOffset(const std::string &devPrefix) const;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Update which variables require queues based on piece of code
    void updateVarQueues(const std::string &code, const std::string &suffix);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    unsigned int m_NumNeurons;

    const NeuronModels::Base *m_NeuronModel;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<NewModels::VarInit> m_VarInitialisers;
    std::vector<SynapseGroup*> m_InSyn;
    std::vector<SynapseGroup*> m_OutSyn;
    std::vector<std::pair<SynapseGroup*, std::vector<SynapseGroup*>>> m_MergedInSyn;
    bool m_SpikeTimeRequired;
    bool m_TrueSpikeRequired;
    bool m_SpikeEventRequired;
    std::set<std::pair<std::string, std::string>> m_SpikeEventCondition;
    unsigned int m_NumDelaySlots;
    std::vector<CurrentSource*> m_CurrentSources;

    //!< Vector specifying which variables require queues
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
