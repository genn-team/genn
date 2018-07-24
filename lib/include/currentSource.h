#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "global.h"
#include "currentSourceModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// CurrentSource
//------------------------------------------------------------------------
class CurrentSource
{
public:
    CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                const std::vector<double> &params, const std::vector<NewModels::VarInit> &varInitialisers, int hostID, int deviceID) :
        m_Name(name),
        m_CurrentSourceModel(currentSourceModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_VarMode(varInitialisers.size(), GENN_PREFERENCES::defaultVarMode),
        m_HostID(hostID), m_DeviceID(deviceID)
    {
    }
    CurrentSource(const CurrentSource&) = delete;
    CurrentSource() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Function to enable the use zero-copied memory for a particular state variable (deprecated use NeuronGroup::setVarMode):
     /*! May improve IO performance at the expense of kernel performance */
    void setVarZeroCopyEnabled(const std::string &varName, bool enabled)
    {
        setVarMode(varName, enabled ? VarMode::LOC_ZERO_COPY_INIT_HOST : VarMode::LOC_HOST_DEVICE_INIT_HOST);
    }

    //! Set variable mode of neuron model state variable
    /*! This is ignored for CPU simulations */
    void setVarMode(const std::string &varName, VarMode mode);

    void initDerivedParams(double dt);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the current source model used by this group
    const CurrentSourceModels::Base *getCurrentSourceModel() const{ return m_CurrentSourceModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }
    const std::vector<NewModels::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    int getClusterHostID() const{ return m_HostID; }

    int getClusterDeviceID() const{ return m_DeviceID; }

    bool isZeroCopyEnabled() const;
    bool isVarZeroCopyEnabled(const std::string &var) const{ return (getVarMode(var) & VarLocation::ZERO_COPY); }

    //! Get variable mode used by current source model state variable
    VarMode getVarMode(const std::string &varName) const;

    //! Get variable mode used by current source model state variable
    VarMode getVarMode(size_t index) const{ return m_VarMode[index]; }

    void addExtraGlobalParams(std::map<std::string, std::string> &kernelParameters) const;

    //! Does this current source require any initialisation code to be run
    bool isInitCodeRequired() const;

    //! Does this current source require an RNG to simulate
    bool isSimRNGRequired() const;

    //! Does this current source group require an RNG for it's init code
    bool isInitRNGRequired(VarInit varInitMode) const;

    //! Is device var init code required for any variables in this current source
    bool isDeviceVarInitRequired() const;

    //! Can this current source run on the CPU?
    /*! If we are running in CPU_ONLY mode this is always true,
        but some GPU functionality will prevent models being run on both CPU and GPU. */
    bool canRunOnCPU() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    const CurrentSourceModels::Base *m_CurrentSourceModel;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<NewModels::VarInit> m_VarInitialisers;

    //!< Whether indidividual state variables of a neuron group should use zero-copied memory
    std::vector<VarMode> m_VarMode;

    //!< The ID of the cluster node which the neuron groups are computed on
    int m_HostID;

    //!< The ID of the CUDA device which the neuron groups are comnputed on
    int m_DeviceID;
};
