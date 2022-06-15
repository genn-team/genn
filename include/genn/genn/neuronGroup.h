#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "neuronModels.h"
#include "variableMode.h"

// Forward declarations
class CurrentSourceInternal;
class SynapseGroupInternal;

//------------------------------------------------------------------------
// NeuronGroup
//------------------------------------------------------------------------
class GENN_EXPORT NeuronGroup
{
public:
    //------------------------------------------------------------------------
    // SpikeEventThreshold
    //------------------------------------------------------------------------
    //! Structure used for storing spike event data
    struct SpikeEventThreshold
    {
        SpikeEventThreshold(const std::string &e, const std::string &s, bool egp, SynapseGroupInternal *sg)
            : eventThresholdCode(e), supportCode(s), synapseStateInThresholdCode(egp), synapseGroup(sg)
        {
        }

        const std::string eventThresholdCode;
        const std::string supportCode;
        const bool synapseStateInThresholdCode;
        SynapseGroupInternal *synapseGroup;

        //! Less than operator (used for std::set::insert), lexicographically compares all three struct
        //! members - meaning that event thresholds featuring extra global parameters or presynaptic
        //! state variables from different synapse groups will not get combined together in neuron update
        bool operator < (const SpikeEventThreshold &other) const
        {
            if(other.eventThresholdCode < eventThresholdCode) {
                return false;
            }
            else if(eventThresholdCode < other.eventThresholdCode) {
                return true;
            }

            if(other.supportCode < supportCode) {
                return false;
            }
            else if(supportCode < other.supportCode) {
                return true;
            }

            if(synapseStateInThresholdCode) {
                if(other.synapseGroup < synapseGroup) {
                    return false;
                }
                else if(synapseGroup < other.synapseGroup) {
                    return true;
                }
            }

            return false;
        }

        //! Equality operator (used for set::set equality used when testing neuron groups mergability),
        //! Compares only the two code strings as neuron groups with threshold conditions 
        //! featuring extra global parameters from different synapse groups can still be merged
        bool operator == (const SpikeEventThreshold &other) const
        {
            return ((eventThresholdCode == other.eventThresholdCode)
                    && (supportCode == other.supportCode));
        }
    };

    NeuronGroup(const NeuronGroup&) = delete;
    NeuronGroup() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of this neuron group's output spikes
    /*! This is ignored for simulations on hardware with a single memory space */
    void setSpikeLocation(VarLocation loc) { m_SpikeLocation = loc; }

     //! Set location of this neuron group's output spike events
     /*! This is ignored for simulations on hardware with a single memory space */
    void setSpikeEventLocation(VarLocation loc) { m_SpikeEventLocation = loc; }

    //! Set location of this neuron group's output spike times
    /*! This is ignored for simulations on hardware with a single memory space */
    void setSpikeTimeLocation(VarLocation loc) { m_SpikeTimeLocation = loc; }
    
    //! Set location of this neuron group's previous output spike times
    /*! This is ignored for simulations on hardware with a single memory space */
    void setPrevSpikeTimeLocation(VarLocation loc) { m_PrevSpikeTimeLocation = loc; }
    
    //! Set location of this neuron group's output spike-like-event times
    /*! This is ignored for simulations on hardware with a single memory space */
    void setSpikeEventTimeLocation(VarLocation loc) { m_SpikeEventTimeLocation = loc; }
    
    //! Set location of this neuron group's previous output spike-like-event times
    /*! This is ignored for simulations on hardware with a single memory space */
    void setPrevSpikeEventTimeLocation(VarLocation loc) { m_PrevSpikeEventTimeLocation = loc; }

    //! Set variable location of neuron model state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of neuron model extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space
        and only applies to extra global parameters which are pointers. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Enables and disable spike recording for this population
    void setSpikeRecordingEnabled(bool enabled) { m_SpikeRecordingEnabled = enabled; }
    
    //! Enables and disable spike event recording for this population
    void setSpikeEventRecordingEnabled(bool enabled) { m_SpikeEventRecordingEnabled = enabled; }

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets number of neurons in group
    unsigned int getNumNeurons() const{ return m_NumNeurons; }

    //! Gets the neuron model used by this group
    const NeuronModels::Base *getNeuronModel() const{ return m_NeuronModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    bool isSpikeTimeRequired() const;
    bool isPrevSpikeTimeRequired() const;
    bool isSpikeEventTimeRequired() const;
    bool isPrevSpikeEventTimeRequired() const;
    bool isTrueSpikeRequired() const;
    bool isSpikeEventRequired() const;

    unsigned int getNumDelaySlots() const{ return m_NumDelaySlots; }
    bool isDelayRequired() const{ return (m_NumDelaySlots > 1); }
    bool isZeroCopyEnabled() const;

    //! Get location of this neuron group's output spikes
    VarLocation getSpikeLocation() const{ return m_SpikeLocation; }

    //! Get location of this neuron group's output spike events
    VarLocation getSpikeEventLocation() const{ return m_SpikeEventLocation; }

    //! Get location of this neuron group's output spike times
    VarLocation getSpikeTimeLocation() const{ return m_SpikeTimeLocation; }

    //! Get location of this neuron group's previous output spike times
    VarLocation getPrevSpikeTimeLocation() const { return m_PrevSpikeTimeLocation; }

    //! Get location of this neuron group's output spike-like-event times
    VarLocation getSpikeEventTimeLocation() const { return m_SpikeEventTimeLocation;  }

    //! Get location of this neuron group's previous output spike-like-event times
    VarLocation getPrevSpikeEventTimeLocation() const { return m_PrevSpikeEventTimeLocation; }

    //! Get location of neuron model state variable by name
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get location of neuron model state variable by index
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

    //! Get location of neuron model extra global parameter by name
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const;

    //! Get location of neuron model extra global parameter by omdex
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(size_t index) const{ return m_ExtraGlobalParamLocation.at(index); }

    //! Is spike recording enabled for this population?
    bool isSpikeRecordingEnabled() const { return m_SpikeRecordingEnabled; }

    //! Is spike event recording enabled for this population?
    bool isSpikeEventRecordingEnabled() const { return m_SpikeEventRecordingEnabled; }

    //! Does this neuron group require an RNG to simulate?
    bool isSimRNGRequired() const;

    //! Does this neuron group require an RNG for it's init code?
    bool isInitRNGRequired() const;

    //! Does this neuron group require any sort of recording?
    bool isRecordingEnabled() const;

protected:
    NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    //! Checks delay slots currently provided by the neuron group against a required delay and extends if required
    void checkNumDelaySlots(unsigned int requiredDelay);

    //! Update which variables are accessed by piece of code running on incoming synapse
    void updateInSynVarAccess(const std::string &code);

    //! Update which variables are accessed by piece of code running on outgoing synapse
    void updateOutSynVarAccess(const std::string &code);

    void addSpkEventCondition(const std::string &code, SynapseGroupInternal *synapseGroup);

    void addInSyn(SynapseGroupInternal *synapseGroup){ m_InSyn.push_back(synapseGroup); }
    void addOutSyn(SynapseGroupInternal *synapseGroup){ m_OutSyn.push_back(synapseGroup); }

    void initDerivedParams(double dt);

    //! Fuse incoming postsynaptic models
    void fusePrePostSynapses(bool fusePSM, bool fusePrePostWUM);

    //! add input current source
    void injectCurrent(CurrentSourceInternal *source);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    //! Gets pointers to all synapse groups which provide input to this neuron group
    const std::vector<SynapseGroupInternal*> &getInSyn() const{ return m_InSyn; }
    const std::vector<SynapseGroupInternal*> &getFusedPSMInSyn() const{ return m_FusedPSMInSyn; }
    const std::vector<SynapseGroupInternal *> &getFusedWUPostInSyn() const { return m_FusedWUPostInSyn; }
    
    //! Gets pointers to all synapse groups emanating from this neuron group
    const std::vector<SynapseGroupInternal*> &getOutSyn() const{ return m_OutSyn; }
    const std::vector<SynapseGroupInternal *> &getFusedWUPreOutSyn() const { return m_FusedWUPreOutSyn; }

     const std::vector<SynapseGroupInternal *> &getFusedPreOutputOutSyn() const { return m_FusedPreOutputOutSyn; }   

    //! Gets pointers to all current sources which provide input to this neuron group
    const std::vector<CurrentSourceInternal*> &getCurrentSources() const { return m_CurrentSources; }

    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }

    const std::set<SpikeEventThreshold> &getSpikeEventCondition() const{ return m_SpikeEventCondition; }

    //! Helper to get vector of incoming synapse groups which have postsynaptic update code
    std::vector<SynapseGroupInternal*> getFusedInSynWithPostCode() const;

    //! Helper to get vector of outgoing synapse groups which have presynaptic update code
    std::vector<SynapseGroupInternal*> getFusedOutSynWithPreCode() const;

    //! Helper to get vector of incoming synapse groups which have postsynaptic variables
    std::vector<SynapseGroupInternal *> getFusedInSynWithPostVars() const;

    //! Helper to get vector of outgoing synapse groups which have presynaptic variables
    std::vector<SynapseGroupInternal *> getFusedOutSynWithPreVars() const;

    //! Is variable accessed by incoming synapse code
    bool isVarInSynAccessRequired(const std::string &var) const;
    bool isVarInSynAccessRequired(size_t index) const{ return m_VarInSynAccessRequired.at(index); }
    
    //! Is variable accessed by outgoing synapse code
    bool isVarOutSynAccessRequired(const std::string &var) const;
    bool isVarOutSynAccessRequired(size_t index) const{ return m_VarOutSynAccessRequired.at(index); }

    //! Is variable accessed by any kind of synapse code
    bool isVarSynAccessRequired(const std::string &var) const{ return (isVarInSynAccessRequired(var) || isVarOutSynAccessRequired(var)); }
    bool isVarSynAccessRequired(size_t index) const{ return (isVarInSynAccessRequired(index) || isVarOutSynAccessRequired(index)); }
    
    //! Updates hash with neuron group
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with neuron group initialisation
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

    boost::uuids::detail::sha1::digest_type getSpikeQueueUpdateHashDigest() const;

    boost::uuids::detail::sha1::digest_type getPrevSpikeTimeUpdateHashDigest() const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Update which variables are accessed by piece of code
    void updateVarAccess(const std::string &code, const std::string &suffix, std::vector<bool> &varAccessRequired);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;

    const unsigned int m_NumNeurons;

    const NeuronModels::Base *m_NeuronModel;
    const std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<Models::VarInit> m_VarInitialisers;
    std::vector<SynapseGroupInternal*> m_InSyn;
    std::vector<SynapseGroupInternal*> m_OutSyn;
    std::vector<SynapseGroupInternal*> m_FusedPSMInSyn;
    std::vector<SynapseGroupInternal *> m_FusedWUPostInSyn;
    std::vector<SynapseGroupInternal *> m_FusedWUPreOutSyn;
    std::vector<SynapseGroupInternal *> m_FusedPreOutputOutSyn;
    std::set<SpikeEventThreshold> m_SpikeEventCondition;
    unsigned int m_NumDelaySlots;
    std::vector<CurrentSourceInternal*> m_CurrentSources;

    //! Vector specifying which variables are accessed from incoming synapse groups
    std::vector<bool> m_VarInSynAccessRequired;
    
    //! Vector specifying which variables are accessed from outgoing synapse groups
    std::vector<bool> m_VarOutSynAccessRequired;

    //! Location of spikes from neuron group
    VarLocation m_SpikeLocation;

    //! Location of spike-like events from neuron group
    VarLocation m_SpikeEventLocation;

    //! Location of spike times from neuron group
    VarLocation m_SpikeTimeLocation;

    //! Location of previous spike times
    VarLocation m_PrevSpikeTimeLocation;

    //! Location of spike-like-event times
    VarLocation m_SpikeEventTimeLocation;

    //! Location of previous spike-like-event times
    VarLocation m_PrevSpikeEventTimeLocation;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    //! Is spike recording enabled for this population?
    bool m_SpikeRecordingEnabled;

    //! Is spike event recording enabled?
    bool m_SpikeEventRecordingEnabled;
};
