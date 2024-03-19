#pragma once

// Standard includes
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "customUpdateModels.h"
#include "varLocation.h"

//------------------------------------------------------------------------
// GeNN::CustomUpdateBase
//------------------------------------------------------------------------
namespace GeNN
{
class GENN_EXPORT CustomUpdateBase
{
public:
    CustomUpdateBase(const CustomUpdateBase &) = delete;
    CustomUpdateBase() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of state variable.
    /*! This is ignored for simulations on hardware with a single memory space */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of extra global parameter.
    /*! This is ignored for simulations on hardware with a single memory space. */
    void setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc);

    //! Set whether parameter is dynamic or not i.e. it can be changed at runtime
    void setParamDynamic(const std::string &paramName, bool dynamic = true);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const{ return m_Name; }
    const std::string &getUpdateGroupName() const { return m_UpdateGroupName; }

    //! Gets the custom update model used by this group
    const CustomUpdateModels::Base *getModel() const{ return m_Model; }

    const auto  &getParams() const{ return m_Params; }
    const auto &getVarInitialisers() const{ return m_VarInitialisers; }

    const auto &getEGPReferences() const{ return m_EGPReferences;  }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const{ return m_VarLocation.get(varName); }

    //! Get location of neuron model extra global parameter by name
    VarLocation getExtraGlobalParamLocation(const std::string &paramName) const{ return m_ExtraGlobalParamLocation.get(paramName); }

    //! Is parameter dynamic i.e. it can be changed at runtime
    bool isParamDynamic(const std::string &paramName) const{ return m_DynamicParams.get(paramName); }

    //! Is var init code required for any variables in this custom update group's custom update model?
    bool isVarInitRequired() const;

    //! Get dimensions of this custom update
    VarAccessDim getDims() const{ return m_Dims; }

protected:
    CustomUpdateBase(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *customUpdateModel, 
                     const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                     const std::map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const auto &getDerivedParams() const{ return m_DerivedParams; }

    //! Does this current source group require an RNG for it's init code
    bool isInitRNGRequired() const;

    bool isZeroCopyEnabled() const;

    bool isModelReduction() const;

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    void updateHash(boost::uuids::detail::sha1 &hash) const;

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    void updateInitHash(boost::uuids::detail::sha1 &hash) const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    const auto &getUpdateCodeTokens() const{ return m_UpdateCodeTokens; }

    template<typename V>
    bool isReduction(const std::map<std::string, V> &varRefs, 
                     VarAccessDim reduceDim) const
    {
        // Return true if any variables have REDUCE flag in their access mode and have reduction dimension 
        // **NOTE** this is correct because custom update variable access types are defined subtractively
        const auto vars = getModel()->getVars();
        if(std::any_of(vars.cbegin(), vars.cend(),
                       [reduceDim](const Models::Base::CustomUpdateVar &v)
                       { 
                           return ((v.access & VarAccessModeAttribute::REDUCE) 
                                   && (static_cast<unsigned int>(v.access) & static_cast<unsigned int>(reduceDim)));
                       }))
        {
            return true;
        }

        // Loop through all variable references
        for(const auto &modelVarRef : getModel()->getVarRefs()) {
            // If custom update model reduces into this variable reference 
            // and the variable it targets doesn't have reduction dimension
            const auto &varRef = varRefs.at(modelVarRef.name);
            if ((modelVarRef.access & VarAccessModeAttribute::REDUCE) 
                && !(varRef.getVarDims() & reduceDim)) 
            {
                return true;
            }
        }

        return false;
    }

    //! Helper function to check if variable reference types match those specified in model
    template<typename V>
    void checkVarReferenceDims(const std::map<std::string, V>& varRefs, unsigned int batchSize)
    {
        // Loop through variable references and or together their dimensions to get dimensionality of update
        m_Dims = VarAccessDim{0};
        for(const auto &v : varRefs) {
            m_Dims = m_Dims | v.second.getVarDims();
        }

        // Loop through all variable references
        for(const auto &modelVarRef : getModel()->getVarRefs()) {
            const auto varRef = varRefs.at(modelVarRef.name);

            // Determine what dimensions are 'missing' from this variable compared to update dimensionality
            const auto missingDims = clearVarAccessDim(m_Dims, varRef.getVarDims());

            // If any dimensions are missing (unless missing dimensions is BATCH and
            // model isn't actually batched), check variable isn't accessed read-write
            if(((missingDims != VarAccessDim{0}) && (missingDims != VarAccessDim::BATCH || batchSize > 1))
               && (modelVarRef.access == VarAccessMode::READ_WRITE))
            {
                throw std::runtime_error("Variable reference '" + modelVarRef.name + "' in custom update '" + getName() + 
                                         "' to lower-dimensional variables cannot be read-write.");
            }
        }
    }

    template<typename G, typename V>
    std::vector<G*> getReferencedCustomUpdates(const std::map<std::string, V>& varRefs) const
    {
        // Loop through variable references
        std::vector<G*> references;
        for(const auto &v : varRefs) {
            // If a custom update is referenced, add to set
            auto *refCU = v.second.getReferencedCustomUpdate();
            if(refCU != nullptr) {
                references.push_back(refCU);
            }
        }

        // Return set
        return references;
    }


private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //! Unique name of custom update
    std::string m_Name;

    //! Name of the update group this custom connectivity update is part of
    std::string m_UpdateGroupName;

    //! Custom update model used for this update
    const CustomUpdateModels::Base *m_Model;

    //! Values of custom connectivity update parameters
    std::map<std::string, Type::NumericValue> m_Params;
    std::map<std::string, Type::NumericValue> m_DerivedParams;
    std::map<std::string, InitVarSnippet::Init> m_VarInitialisers;

    std::map<std::string, Models::EGPReference> m_EGPReferences;

    //! Location of individual state variables
    LocationContainer m_VarLocation;

    //! Location of extra global parameters
    LocationContainer m_ExtraGlobalParamLocation;

    //! Data structure tracking whether parameters are dynamic or not
    Snippet::DynamicParameterContainer m_DynamicParams;

    //! Tokens produced by scanner from update code
    std::vector<Transpiler::Token> m_UpdateCodeTokens;

    //! Dimensions of this custom update
    VarAccessDim m_Dims;
};

//----------------------------------------------------------------------------
// CustomUpdateVarAdapter
//----------------------------------------------------------------------------
class CustomUpdateVarAdapter
{
public:
    CustomUpdateVarAdapter(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getVarLocation(varName); }

    auto getDefs() const{ return m_CU.getModel()->getVars(); }

    const auto &getInitialisers() const{ return m_CU.getVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const CustomUpdateBase &getTarget() const{ return m_CU; }

    VarAccessDim getVarDims(const Models::Base::CustomUpdateVar &var) const
    { 
        return getVarAccessDim(var.access, m_CU.getDims());
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomUpdateEGPAdapter
{
public:
    CustomUpdateEGPAdapter(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getExtraGlobalParamLocation(varName); }

    Snippet::Base::EGPVec getDefs() const{ return m_CU.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//------------------------------------------------------------------------
// CustomUpdate
//------------------------------------------------------------------------
class GENN_EXPORT CustomUpdate : public CustomUpdateBase
{
public:
    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const auto &getVarReferences() const{ return m_VarReferences;  }

    //! Get number of neurons custom update operates over
    /*! This must be the same for all groups whose variables are referenced */
    unsigned int getNumNeurons() const { return m_NumNeurons; }

protected:
    CustomUpdate(const std::string &name, const std::string &updateGroupName,
                 const CustomUpdateModels::Base *customUpdateModel, const std::map<std::string, Type::NumericValue> &params,
                 const std::map<std::string, InitVarSnippet::Init> &varInitialisers, const std::map<std::string, Models::VarReference> &varReferences,
                 const std::map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt, unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction(getVarReferences(), VarAccessDim::BATCH); }
    bool isNeuronReduction() const { return isReduction(getVarReferences(), VarAccessDim::ELEMENT); }

    const NeuronGroup *getDelayNeuronGroup() const { return m_DelayNeuronGroup; }

    //! Get vector of other custom updates referenced by this custom update
    std::vector<CustomUpdate*> getReferencedCustomUpdates() const
    { 
        return CustomUpdateBase::getReferencedCustomUpdates<CustomUpdate>(m_VarReferences);
    }

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::map<std::string, Models::VarReference> m_VarReferences;

    //! Number of neurons custom update operates over.
    /*! This must be the same for all groups whose variables are referenced */
    unsigned int m_NumNeurons;

    const NeuronGroup *m_DelayNeuronGroup;
};

//------------------------------------------------------------------------
// CustomUpdateWU
//------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWU : public CustomUpdateBase
{
public:
    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const auto &getVarReferences() const{ return m_VarReferences;  }

protected:
    CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                   const CustomUpdateModels::Base *customUpdateModel, const std::map<std::string, Type::NumericValue> &params,
                   const std::map<std::string, InitVarSnippet::Init> &varInitialisers, const std::map<std::string, Models::WUVarReference> &varReferences,
                   const std::map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt, unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction(getVarReferences(), VarAccessDim::BATCH); }
    bool isTransposeOperation() const;

    SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

    const std::vector<unsigned int> &getKernelSize() const;

    //! Get vector of other custom updates referenced by this custom update
    std::vector<CustomUpdateWU*> getReferencedCustomUpdates() const
    { 
        return CustomUpdateBase::getReferencedCustomUpdates<CustomUpdateWU>(m_VarReferences); 
    }

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! \note this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::map<std::string, Models::WUVarReference> m_VarReferences;

    //! Synapse group all variables referenced by custom update are associated with
    SynapseGroupInternal *m_SynapseGroup;
};
}   // namespace GeNN
