#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceModels.h"
#include "gennExport.h"
#include "variableMode.h"

// Forward declarations
class NeuronGroupInternal;

//------------------------------------------------------------------------
// CurrentSource
//------------------------------------------------------------------------
class GENN_EXPORT CurrentSource
{
public:
    CurrentSource(const CurrentSource&) = delete;
    CurrentSource() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of current source state variable
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space
        and only applies to extra global parameters which are pointers. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the current source model used by this group
    const CurrentSourceModels::Base *getCurrentSourceModel() const{ return m_CurrentSourceModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for current source model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

    //! Get location of neuron model extra global parameter by name
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const;

    //! Get location of neuron model extra global parameter by omdex
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(size_t index) const{ return m_ExtraGlobalParamLocation.at(index); }

protected:
    CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                  const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                  const NeuronGroupInternal *trgNeuronGroup, VarLocation defaultVarLocation, 
                  VarLocation defaultExtraGlobalParamLocation)
    :   m_Name(name), m_CurrentSourceModel(currentSourceModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_TrgNeuronGroup(trgNeuronGroup), m_VarLocation(varInitialisers.size(), defaultVarLocation),
        m_ExtraGlobalParamLocation(currentSourceModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation)
    {
    }

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void initDerivedParams(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const NeuronGroupInternal *getTrgNeuronGroup() const{ return m_TrgNeuronGroup; }

    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }

    //! Does this current source require an RNG to simulate
    bool isSimRNGRequired() const;

    //! Does this current source group require an RNG for it's init code
    bool isInitRNGRequired() const;

    bool isZeroCopyEnabled() const;

    //! Can this current source be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CurrentSource &other) const;

    //! Can the initialisation of these current sources be merged together? i.e. can they be initialised using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canInitBeMerged(const CurrentSource &other) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    const CurrentSourceModels::Base *m_CurrentSourceModel;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<Models::VarInit> m_VarInitialisers;

    const NeuronGroupInternal *m_TrgNeuronGroup;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;
};
