#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "neuronGroup.h"
#include "neuronModels.h"
#include "variableMode.h"

// Forward declarations
class CurrentSource;
class SynapseGroup;

//------------------------------------------------------------------------
// NeuronGroupInternal
//------------------------------------------------------------------------
class NeuronGroupInternal : public NeuronGroup
{
public:
    NeuronGroupInternal(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                        const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                        VarLocation defaultVarLocation, int hostID, int deviceID)
    :   NeuronGroup(name, numNeurons, neuronModel, params, varInitialisers, defaultVarLocation, hostID, deviceID),
        m_NumDelaySlots(1), m_VarQueueRequired(varInitialisers.size(), false)
    {
    }
    
    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Checks delay slots currently provided by the neuron group against a required delay and extends if required
    void checkNumDelaySlots(unsigned int requiredDelay);

    //! Update which presynaptic variables require queues based on piece of code
    void updatePreVarQueues(const std::string &code);

    //! Update which postsynaptic variables  require queues based on piece of code
    void updatePostVarQueues(const std::string &code);
    
     //! Do any of the spike event conditions tested by this neuron require specified parameter?
    bool isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const;

    void addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace);

    void addInSyn(SynapseGroup *synapseGroup){ m_InSyn.push_back(synapseGroup); }
    void addOutSyn(SynapseGroup *synapseGroup){ m_OutSyn.push_back(synapseGroup); }

    void initDerivedParams(double dt);
 
    //! Merge incoming postsynaptic models
    void mergeIncomingPSM(bool merge);

    //! add input current source
    void injectCurrent(CurrentSource *source);
    
    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }
    
    //! Gets pointers to all synapse groups which provide input to this neuron group
    const std::vector<SynapseGroup*> &getInSyn() const{ return m_InSyn; }
    const std::vector<std::pair<SynapseGroup*, std::vector<SynapseGroup*>>> &getMergedInSyn() const{ return m_MergedInSyn; }

    //! Gets pointers to all current sources which provide input to this neuron group
    const std::vector<CurrentSource*> &getCurrentSources() const { return m_CurrentSources; }

    //! Gets pointers to all synapse groups emanating from this neuron group
    const std::vector<SynapseGroup*> &getOutSyn() const{ return m_OutSyn; }

    bool isSpikeTimeRequired() const;
    bool isTrueSpikeRequired() const;
    bool isSpikeEventRequired() const;

    bool isVarQueueRequired(const std::string &var) const;
    bool isVarQueueRequired(size_t index) const{ return m_VarQueueRequired[index]; }

    const std::set<std::pair<std::string, std::string>> &getSpikeEventCondition() const{ return m_SpikeEventCondition; }

    unsigned int getNumDelaySlots() const{ return m_NumDelaySlots; }
    bool isDelayRequired() const{ return (m_NumDelaySlots > 1); }
    
    //! Does this neuron group require an RNG to simulate?
    bool isSimRNGRequired() const;

    //! Does this neuron group require an RNG for it's init code?
    bool isInitRNGRequired() const;
    
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
    std::vector<CurrentSource*> m_CurrentSources;

    std::vector<SynapseGroup*> m_InSyn;
    std::vector<SynapseGroup*> m_OutSyn;
    std::vector<std::pair<SynapseGroup*, std::vector<SynapseGroup*>>> m_MergedInSyn;
    std::set<std::pair<std::string, std::string>> m_SpikeEventCondition;
    unsigned int m_NumDelaySlots;
    
    std::vector<double> m_DerivedParams;
    
    //!< Vector specifying which variables require queues
    std::vector<bool> m_VarQueueRequired;

};