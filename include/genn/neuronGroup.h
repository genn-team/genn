#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "neuronModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// NeuronGroup
//------------------------------------------------------------------------
class NeuronGroup
{
public:
    NeuronGroup(const NeuronGroup&) = delete;
    NeuronGroup() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set variable mode used for variables containing this neuron group's output spikes
    void setSpikeLocation(VarLocation loc) { m_SpikeLocation = loc; }

     //! Set variable mode used for variables containing this neuron group's output spike events
    void setSpikeEventLocation(VarLocation loc) { m_SpikeEventLocation = loc; }

    //! Set variable mode used for variables containing this neuron group's output spike times
    void setSpikeTimeLocation(VarLocation loc) { m_SpikeTimeLocation = loc; }

    //! Set variable mode of neuron model state variable
    void setVarLocation(const std::string &varName, VarLocation loc);

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

    int getClusterHostID() const{ return m_HostID; }

    int getClusterDeviceID() const{ return m_DeviceID; }

    bool isZeroCopyEnabled() const;

    //! Get variable mode used for variables containing this neuron group's output spikes
    VarLocation getSpikeLocation() const{ return m_SpikeLocation; }

    //! Get variable mode used for variables containing this neuron group's output spike events
    VarLocation getSpikeEventLocation() const{ return m_SpikeEventLocation; }

    //! Get variable mode used for variables containing this neuron group's output spike times
    VarLocation getSpikeTimeLocation() const{ return m_SpikeTimeLocation; }

    //! Get variable mode used by neuron model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable mode used by neuron model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }
    
protected:
    NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                VarLocation defaultVarLocation, int hostID, int deviceID) :
        m_Name(name), m_NumNeurons(numNeurons), m_NeuronModel(neuronModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_SpikeLocation(defaultVarLocation), m_SpikeEventLocation(defaultVarLocation),
        m_SpikeTimeLocation(defaultVarLocation), m_VarLocation(varInitialisers.size(), defaultVarLocation),
        m_HostID(hostID), m_DeviceID(deviceID)
    {
    }

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void initInitialiserDerivedParams(double dt);
   
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    unsigned int m_NumNeurons;

    const NeuronModels::Base *m_NeuronModel;
    std::vector<double> m_Params;
    std::vector<Models::VarInit> m_VarInitialisers;
    
    //!< Whether spikes from neuron group should use zero-copied memory
    VarLocation m_SpikeLocation;

    //!< Whether spike-like events from neuron group should use zero-copied memory
    VarLocation m_SpikeEventLocation;

    //!< Whether spike times from neuron group should use zero-copied memory
    VarLocation m_SpikeTimeLocation;

    //!< Whether indidividual state variables of a neuron group should use zero-copied memory
    std::vector<VarLocation> m_VarLocation;

    //!< The ID of the cluster node which the neuron groups are computed on
    int m_HostID;

    //!< The ID of the CUDA device which the neuron groups are comnputed on
    int m_DeviceID;
};
