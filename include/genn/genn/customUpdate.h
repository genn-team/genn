#pragma once

// Standard includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
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

    //! Can this custom update be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CustomUpdateBase &other) const;

    //! Can the initialisation of these custom update be merged together? i.e. can they be initialised using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canInitBeMerged(const CustomUpdateBase &other) const;

    //! Helper function to determine whether a custom update should be batched
    template<typename R>
    void finalizeBatched(unsigned int batchSize, const std::vector<R> &varRefs)
    {
        // If model has batching at all, custom update should be batched 
        // if targets of any variable references are duplicated
        if(batchSize > 1) {
            m_Batched = std::any_of(varRefs.cbegin(), varRefs.cend(),
                                    [](const R &v) { return (v.getVar().access & VarAccessDuplication::DUPLICATE); });
            
            // If custom update is batched, check that any variable references to shared variables are read-only
            if(m_Batched) {
                const auto modelVarRefs = getCustomUpdateModel()->getVarRefs();
                for(size_t i = 0; i < modelVarRefs.size(); i++) {
                    if((varRefs.at(i).getVar().access & VarAccessDuplication::SHARED) 
                       && (modelVarRefs.at(i).access != VarAccessMode::READ_ONLY))
                    {
                        throw std::runtime_error("Variable references to SHARED variables in batched models must be read-only.");
                    }
                }
            }
        }
        // Otherwise, update should not be batched
        else {
            m_Batched = false;
        }
    }

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
    //! Can this custom update be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CustomUpdate &other) const;

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
    bool isTransposeOperation() const;

    const SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

    //! Can this custom update be merged with other? i.e. can they be simulated using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canBeMerged(const CustomUpdateWU &other) const;

    //! Can the initialisation of these custom update be merged together? i.e. can they be initialised using same generated code
    /*! NOTE: this can only be called after model is finalized */
    bool canInitBeMerged(const CustomUpdateWU &other) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::vector<Models::WUVarReference> m_VarReferences;
    const SynapseGroupInternal *m_SynapseGroup;
};