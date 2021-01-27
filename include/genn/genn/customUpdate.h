#pragma once

// Standard includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "customUpdateModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// CustomUpdate
//------------------------------------------------------------------------
class GENN_EXPORT CustomUpdate
{
public:
    CustomUpdate(const CustomUpdate &) = delete;
    CustomUpdate() = delete;

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Operation
    {
        UPDATE,
        UPDATE_TRANSPOSE,
    };

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of extra global parameter
    /*! This is ignored for simulations on hardware with a single memory space
        and only applies to extra global parameters which are pointers. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }

    //! Gets the custom update model used by this group
    const CustomUpdateModels::Base *getCustomUpdateModel() const{ return m_CustomUpdateModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }
    const std::vector<Models::VarReference> &getVarReferences() const{ return m_VarReferences;  }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

    //! Get location of custom update model extra global parameter by name
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const;

    //! Get location of custom update model extra global parameter by omdex
    /*! This is only used by extra global parameters which are pointers*/
    VarLocation getExtraGlobalParamLocation(size_t index) const{ return m_ExtraGlobalParamLocation.at(index); }


protected:
    CustomUpdate(const std::string &name, const CustomUpdateModels::Base *customUpdateModel,
                 const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                 const std::vector<Models::VarReference> &varReferences, VarLocation defaultVarLocation, 
                 VarLocation defaultExtraGlobalParamLocation)
    :   m_Name(name), m_CustomUpdateModel(customUpdateModel), m_Params(params), m_VarInitialisers(varInitialisers),
        m_VarReferences(varReferences), m_VarLocation(varInitialisers.size(), defaultVarLocation),
        m_ExtraGlobalParamLocation(customUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation)
    {
    }

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void initDerivedParams(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const std::vector<double> &getDerivedParams() const{ return m_DerivedParams; }

    //! Does this current source group require an RNG for it's init code
    bool isInitRNGRequired() const;

    //! Can this custom update be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CustomUpdate &other) const;

    //! Can the initialisation of these custom update be merged together? i.e. can they be initialised using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canInitBeMerged(const CustomUpdate &other) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;

    const CustomUpdateModels::Base *m_CustomUpdateModel;
    std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<Models::VarInit> m_VarInitialisers;
    std::vector<Models::VarReference> m_VarReferences;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;
};