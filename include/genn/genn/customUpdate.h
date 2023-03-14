#pragma once

// Standard includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "customUpdateModels.h"
#include "variableMode.h"

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
    const std::string &getUpdateGroupName() const { return m_UpdateGroupName; }

    //! Gets the custom update model used by this group
    const CustomUpdateModels::Base *getCustomUpdateModel() const{ return m_CustomUpdateModel; }

    const std::vector<double> &getParams() const{ return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const{ return m_VarInitialisers; }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(size_t index) const{ return m_VarLocation.at(index); }

    //! Is var init code required for any variables in this custom update group's custom update model?
    bool isVarInitRequired() const;

protected:
    CustomUpdateBase(const std::string &name, const std::string &updateGroupName,
                     const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                     const std::vector<Models::VarInit> &varInitialisers,
                     VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomUpdateModel(customUpdateModel), m_Params(params), 
        m_VarInitialisers(varInitialisers), m_VarLocation(varInitialisers.size(), defaultVarLocation),
        m_ExtraGlobalParamLocation(customUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_Batched(false)
    {
        // Validate names
        Utils::validatePopName(name, "Custom update");
        Utils::validatePopName(updateGroupName, "Custom update group name");
        getCustomUpdateModel()->validate();
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

    //! Is this custom update batched i.e. run in parallel across model batches
    bool isBatched() const { return m_Batched; }

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    void updateHash(boost::uuids::detail::sha1 &hash) const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    void updateInitHash(boost::uuids::detail::sha1 &hash) const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    template<typename V>
    bool isReduction(const std::vector<V> &varRefs, VarAccessDuplication duplication) const
    {
        // Return true if any variables have REDUCE flag in their access mode and have correct duplication flag
        const auto vars = getCustomUpdateModel()->getVars();
        if(std::any_of(vars.cbegin(), vars.cend(),
                       [duplication](const Models::Base::Var &v)
                       { 
                           return (v.access & VarAccessModeAttribute::REDUCE) && (v.access & duplication);
                       }))
        {
            return true;
        }

        // Loop through all variable references
        const auto modelVarRefs = getCustomUpdateModel()->getVarRefs();
        for (size_t i = 0; i < varRefs.size(); i++) {
            const auto varRef = varRefs.at(i);
            const auto modelVarRef = modelVarRefs.at(i);

            // If custom update model reduces into this variable reference and the variable it targets has correct duplication flag
            if ((modelVarRef.access & VarAccessModeAttribute::REDUCE) & (varRef.getVar().access & duplication)) {
                return true;
            }
        }

        return false;
    }

    //! Helper function to check if variable reference types match those specified in model
    template<typename V>
    void checkVarReferences(const std::vector<V> &varRefs)
    {
        // Loop through all variable references
        const auto modelVarRefs = getCustomUpdateModel()->getVarRefs();
        for(size_t i = 0; i < varRefs.size(); i++) {
            const auto varRef = varRefs.at(i);
            const auto modelVarRef = modelVarRefs.at(i);

            // Check types of variable references against those specified in model
            // **THINK** due to GeNN's current string-based type system this is rather conservative
            if(varRef.getVar().type != modelVarRef.type) {
                throw std::runtime_error("Incompatible type for variable reference '" + modelVarRef.name + "'");
            }

            // Check that no reduction targets reference duplicated variables
            if((varRef.getVar().access & VarAccessDuplication::DUPLICATE) 
                && (modelVarRef.access & VarAccessModeAttribute::REDUCE))
            {
                throw std::runtime_error("Reduction target variable reference must be to SHARED or SHARED_NEURON variables.");
            }
        }
    }

    //! Helper function to check if variable reference types match those specified in model
    template<typename V>
    void checkVarReferenceBatching(const std::vector<V>& varRefs, unsigned int batchSize)
    {
        // If target of any variable references is not shared across batches, custom update should be batched
        if(batchSize > 1) {
            m_Batched = std::any_of(varRefs.cbegin(), varRefs.cend(),
                                    [](const V& v) 
                                    {
                                        return (v.isBatched() && !(v.getVar().access & VarAccessDuplication::SHARED)); 
                                    });
        }
        else {
            m_Batched = false;
        }

        // Loop through all variable references
        const auto modelVarRefs = getCustomUpdateModel()->getVarRefs();
        for (size_t i = 0; i < varRefs.size(); i++) {
            const auto varRef = varRefs.at(i);
            const auto modelVarRef = modelVarRefs.at(i);

             // If custom update is batched, check that any variable references to shared variables are read-only
            // **NOTE** if custom update isn't batched, it's totally fine to write to shared variables
            if(m_Batched && (varRef.getVar().access & VarAccessDuplication::SHARED)
               && (modelVarRef.access == VarAccessMode::READ_WRITE))
            {
                throw std::runtime_error("Variable references to SHARED variables in batched custom updates cannot be read-write.");
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

    //! Is this custom update batched i.e. run in parallel across model batches
    bool m_Batched;
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
    const std::vector<Models::VarReference> &getVarReferences() const{ return m_VarReferences;  }
    unsigned int getSize() const { return m_Size; }

protected:
    CustomUpdate(const std::string &name, const std::string &updateGroupName,
                 const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                 const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::VarReference> &varReferences,
                 VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalize(unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction(getVarReferences(), VarAccessDuplication::SHARED); }
    bool isNeuronReduction() const { return isReduction(getVarReferences(), VarAccessDuplication::SHARED_NEURON); }

     //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

    const NeuronGroup *getDelayNeuronGroup() const { return m_DelayNeuronGroup; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::vector<Models::VarReference> m_VarReferences;
    const unsigned int m_Size;
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
    const std::vector<Models::WUVarReference> &getVarReferences() const{ return m_VarReferences;  }

protected:
    CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                   const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                   const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::WUVarReference> &varReferences,
                   VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalize(unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction(getVarReferences(), VarAccessDuplication::SHARED); }
    bool isTransposeOperation() const;

    const SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::vector<Models::WUVarReference> m_VarReferences;
    const SynapseGroupInternal *m_SynapseGroup;
};
