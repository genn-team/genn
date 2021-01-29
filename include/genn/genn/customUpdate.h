#pragma once

// Standard includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "customUpdateModels.h"
#include "variableMode.h"
#include "varReference.h"

//------------------------------------------------------------------------
// CustomUpdateBase
//------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateBase
{
public:
    CustomUpdateBase(const CustomUpdateBase &) = delete;
    CustomUpdateBase() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }
    const std::string &getUpdateGroupName() const { return m_Name; }

    //! Gets the custom update model used by this group
    const CustomUpdateModels::Base *getCustomUpdateModel() const{ return m_CustomUpdateModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

protected:
    CustomUpdateBase(const std::string &name, const std::string &updateGroupName,
                 const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                 const std::vector<Models::VarInit> &varInitialisers,
                 VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomUpdateModel(customUpdateModel), m_Params(params), 
        m_VarInitialisers(varInitialisers), m_VarLocation(varInitialisers.size(), defaultVarLocation),
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

    bool isZeroCopyEnabled() const;

    //! Can this custom update be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CustomUpdateBase &other) const;

    //! Can the initialisation of these custom update be merged together? i.e. can they be initialised using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canInitBeMerged(const CustomUpdateBase &other) const;

    //! Helper function to check if variable reference types match those specified in model
    template<typename V>
    void checkVarReferenceTypes(const std::vector<V> &varReferences) const
    {
        // Loop through all variable references
        const auto varRefs = getCustomUpdateModel()->getVarRefs();
        for(size_t i = 0; i < varReferences.size(); i++) {
            const auto varRef = varReferences.at(i);

            // Check types of variable references against those specified in model
            // **THINK** due to GeNN's current string-based type system this is rather conservative
            if(varRef.getVar().type != varRefs.at(i).type) {
                throw std::runtime_error("Incompatible type for variable reference '" + getCustomUpdateModel()->getVarRefs().at(i).name + "'");
            }
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;
    const std::string m_UpdateGroupName;

    const CustomUpdateModels::Base *m_CustomUpdateModel;
    const std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<Models::VarInit> m_VarInitialisers;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;
};

//------------------------------------------------------------------------
// CustomUpdate
//------------------------------------------------------------------------
template<typename V>
class CustomUpdate : public CustomUpdateBase
{
public:
    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::vector<V> &getVarReferences() const{ return m_VarReferences;  }
    unsigned int getSize() const { return m_Size; }

protected:
    CustomUpdate(const std::string &name, const std::string &updateGroupName,
                 const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params, 
                 const std::vector<Models::VarInit> &varInitialisers, const std::vector<V> &varReferences, 
                 VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
        m_VarReferences(varReferences), m_Size(varReferences.empty() ? 0 : varReferences.front().getSize())
    {
        if(varReferences.empty()) {
            throw std::runtime_error("Custom update models must reference variables.");
        }

        // Check variable reference types
        checkVarReferenceTypes(m_VarReferences);

        // Give error if any sizes differ
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [this](const V &v) { return v.getSize() != m_Size; }))
        {
            throw std::runtime_error("All referenced variables must have the same size.");
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::vector<V> m_VarReferences;
    const unsigned int m_Size;
};

//------------------------------------------------------------------------
// CustomUpdateWU
//------------------------------------------------------------------------
class CustomUpdateWU : public CustomUpdateBase
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Operation
    {
        UPDATE,
        UPDATE_TRANSPOSE,
    };

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    Operation getOperation() const { return m_Operation; }
    const std::vector<WUVarReference> &getVarReferences() const{ return m_VarReferences;  }

protected:
    CustomUpdateWU(const std::string &name, const std::string &updateGroupName, Operation operation,
                   const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                   const std::vector<Models::VarInit> &varInitialisers, const std::vector<WUVarReference> &varReferences,
                   VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::vector<WUVarReference> m_VarReferences;
    const Operation m_Operation;
    const SynapseGroupInternal *m_SynapseGroup;
};