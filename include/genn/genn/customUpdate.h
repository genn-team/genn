#pragma once

// Standard includes
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "customUpdateModels.h"
#include "variableMode.h"

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

    const std::unordered_map<std::string, double> &getParams() const{ return m_Params; }
    const std::unordered_map<std::string, InitVarSnippet::Init> &getVarInitialisers() const{ return m_VarInitialisers; }

    const std::unordered_map<std::string, Models::EGPReference> &getEGPReferences() const{ return m_EGPReferences;  }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Is var init code required for any variables in this custom update group's custom update model?
    bool isVarInitRequired() const;

protected:
    CustomUpdateBase(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *customUpdateModel, 
                     const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                     const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const std::unordered_map<std::string, double> &getDerivedParams() const{ return m_DerivedParams; }

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

    const std::vector<Transpiler::Token> getUpdateCodeTokens() const{ return m_UpdateCodeTokens; }

    template<typename A, typename V>
    bool isReduction(const std::unordered_map<std::string, V> &varRefs, 
                     VarAccessDim reduceDim) const
    {
        // Return true if any variables have REDUCE flag in their access mode and don't have reduction dimension
        const auto vars = getCustomUpdateModel()->getVars();
        if(std::any_of(vars.cbegin(), vars.cend(),
                       [reduceDim](const Models::Base::Var &v)
                       { 
                           return ((v.access & VarAccessModeAttribute::REDUCE) 
                                   && !(v.access.getDims<A>() & reduceDim));
                       }))
        {
            return true;
        }

        // Loop through all variable references
        for(const auto &modelVarRef : getCustomUpdateModel()->getVarRefs()) {
            // If custom update model reduces into this variable reference 
            // and the variable it targets doesn't have reduction dimension
            const auto &varRef = varRefs.at(modelVarRef.name);
            if ((modelVarRef.access & VarAccessModeAttribute::REDUCE) 
                && !(varRef.getVar().access.getDims<A>() & reduceDim)) 
            {
                return true;
            }
        }

        return false;
    }

    //! Helper function to check if variable reference types match those specified in model
    template<typename A, typename V>
    void checkVarReferenceBatching(const std::unordered_map<std::string, V>& varRefs, unsigned int batchSize)
    {
        // If target of any variable references is duplicated, custom update should be batched
        if(batchSize > 1) {
            m_Batched = std::any_of(varRefs.cbegin(), varRefs.cend(),
                                    [](const auto &v) { return v.second.isDuplicated(); });
        }
        else {
            m_Batched = false;
        }

        // Loop through all variable references
        for(const auto &modelVarRef : getCustomUpdateModel()->getVarRefs()) {
            const auto varRef = varRefs.at(modelVarRef.name);

            // If custom update is batched, check that any variable references to variables that aren't batched are read-only
            // **NOTE** if custom update isn't batched, it's totally fine to write to shared variables
            if(m_Batched && !(varRef.getVar().access.getDims<A>() & VarAccessDim::BATCH)
               && (modelVarRef.access == VarAccessMode::READ_WRITE))
            {
                throw std::runtime_error("Variable references to non-batched variables in batched custom updates cannot be read-write.");
            }
        }
    }

    template<typename G, typename V>
    std::vector<G*> getReferencedCustomUpdates(const std::unordered_map<std::string, V>& varRefs) const
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
    const std::string m_Name;
    const std::string m_UpdateGroupName;

    const CustomUpdateModels::Base *m_CustomUpdateModel;
    const std::unordered_map<std::string, double> m_Params;
    std::unordered_map<std::string, double> m_DerivedParams;
    std::unordered_map<std::string, InitVarSnippet::Init> m_VarInitialisers;

    std::unordered_map<std::string, Models::EGPReference> m_EGPReferences;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    //! Tokens produced by scanner from update code
    std::vector<Transpiler::Token> m_UpdateCodeTokens;

    //! Is this custom update batched i.e. run in parallel across model batches
    bool m_Batched;
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

    Models::Base::VarVec getDefs() const{ return m_CU.getCustomUpdateModel()->getVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CU.getVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

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
    VarLocation getLoc(const std::string&) const{ return VarLocation::HOST_DEVICE; }

    Snippet::Base::EGPVec getDefs() const{ return m_CU.getCustomUpdateModel()->getExtraGlobalParams(); }

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
    const std::unordered_map<std::string, Models::VarReference> &getVarReferences() const{ return m_VarReferences;  }
    unsigned int getSize() const { return m_Size; }

protected:
    CustomUpdate(const std::string &name, const std::string &updateGroupName,
                 const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, double> &params,
                 const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::VarReference> &varReferences,
                 const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt, unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction<NeuronVarAccess>(getVarReferences(), VarAccessDim::BATCH); }
    bool isNeuronReduction() const { return isReduction<NeuronVarAccess>(getVarReferences(), VarAccessDim::NEURON); }
    bool isPerNeuron() const{ return m_PerNeuron; }

    const NeuronGroup *getDelayNeuronGroup() const { return m_DelayNeuronGroup; }

    //! Get vector of other custom updates referenced by this custom update
    std::vector<CustomUpdate*> getReferencedCustomUpdates() const
    { 
        return CustomUpdateBase::getReferencedCustomUpdates<CustomUpdate>(m_VarReferences);
    }

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
    const std::unordered_map<std::string, Models::VarReference> m_VarReferences;
    const unsigned int m_Size;
    const NeuronGroup *m_DelayNeuronGroup;

    //! Is this custom update per-neuron i.e. run in parallel across all neurons
    bool m_PerNeuron;
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
    const std::unordered_map<std::string, Models::WUVarReference> &getVarReferences() const{ return m_VarReferences;  }

protected:
    CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                   const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, double> &params,
                   const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::WUVarReference> &varReferences,
                   const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt, unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    bool isBatchReduction() const { return isReduction<SynapseVarAccess>(getVarReferences(), VarAccessDim::BATCH); }
    bool isTransposeOperation() const;

    SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

    const std::vector<unsigned int> &getKernelSize() const;

    //! Get vector of other custom updates referenced by this custom update
    std::vector<CustomUpdateWU*> getReferencedCustomUpdates() const
    { 
        return CustomUpdateBase::getReferencedCustomUpdates<CustomUpdateWU>(m_VarReferences); 
    }

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
    const std::unordered_map<std::string, Models::WUVarReference> m_VarReferences;
    SynapseGroupInternal *m_SynapseGroup;
};
}   // namespace GeNN
