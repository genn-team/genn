#pragma once

// Standard includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "gennUtils.h"
#include "customConnectivityUpdateModels.h"
#include "variableMode.h"

//------------------------------------------------------------------------
// CustomConnectivityUpdate
//------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityUpdate
{
public:
    CustomConnectivityUpdate(const CustomConnectivityUpdate &) = delete;
    CustomConnectivityUpdate() = delete;

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Set location of synaptic state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of presynaptic state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setPreVarLocation(const std::string &varName, VarLocation loc);

    //! Set location of postsynaptic state variable
    /*! This is ignored for simulations on hardware with a single memory space */
    void setPostVarLocation(const std::string &varName, VarLocation loc);

    //------------------------------------------------------------------------
    // Public const methods
    //------------------------------------------------------------------------
    const std::string &getName() const { return m_Name; }
    const std::string &getUpdateGroupName() const { return m_UpdateGroupName; }

    //! Gets the custom connectivity update model used by this group
    const CustomConnectivityUpdateModels::Base *getCustomConnectivityUpdateModel() const { return m_CustomConnectivityUpdateModel; }

    const std::vector<double> &getParams() const { return m_Params; }
    const std::vector<Models::VarInit> &getVarInitialisers() const { return m_VarInitialisers; }
    const std::vector<Models::VarInit> &getPreVarInitialisers() const { return m_PreVarInitialisers; }
    const std::vector<Models::VarInit> &getPostVarInitialisers() const { return m_PostVarInitialisers; }

    //! Get variable location for synaptic state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for presynaptic state variable
    VarLocation getPreVarLocation(const std::string &varName) const;
    
    //! Get variable location for postsynaptic state variable
    VarLocation getPostVarLocation(const std::string &varName) const;

    //! Get variable location for synaptic state variable
    VarLocation getVarLocation(size_t index) const { return m_VarLocation.at(index); }

    //! Get variable location for presynaptic state variable
    VarLocation getPreVarLocation(size_t index) const { return m_PreVarLocation.at(index); }

    //! Get variable location for postsynaptic state variable
    VarLocation getPostVarLocation(size_t index) const { return m_PostVarLocation.at(index); }

    //! Is var init code required for any variables in this custom update group's custom update model?
    bool isVarInitRequired() const;

protected:
    CustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName,
                             const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel, 
                             const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                             const std::vector<Models::VarInit> &preVarInitialisers, const std::vector<Models::VarInit> &postVarInitialisers,
                             const std::vector<Models::WUVarReference> &varReferences, const std::vector<Models::VarReference> &preVarReferences,
                             const std::vector<Models::VarReference> &postReferences, VarLocation defaultVarLocation, 
                             VarLocation defaultExtraGlobalParamLocation)
        : m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomConnectivityUpdateModel(customConnectivityUpdateModel), 
        m_Params(params), m_VarInitialisers(varInitialisers), m_PreVarInitialisers(preVarInitialisers), m_PostVarInitialisers(postVarInitialisers),
        m_VarLocation(varInitialisers.size(), defaultVarLocation), m_PreVarLocation(preVarInitialisers.size()), m_PostVarLocation(postVarInitialisers.size()),
        m_VarReferences(varReferences), m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<const SynapseGroupInternal *>(m_VarReferences.front().getSynapseGroup()))
        m_ExtraGlobalParamLocation(customUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_Batched(false)
    {
        // Validate names
        Utils::validatePopName(name, "Custom update");
        Utils::validatePopName(updateGroupName, "Custom update group name");
        getCustomConnectivityUpdateModel()->validate();
    }

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void initDerivedParams(double dt);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const std::vector<double> &getDerivedParams() const { return m_DerivedParams; }

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

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;
    const std::string m_UpdateGroupName;

    const CustomConnectivityUpdateModels::Base *m_CustomConnectivityUpdateModel;
    const std::vector<double> m_Params;
    std::vector<double> m_DerivedParams;
    std::vector<Models::VarInit> m_VarInitialisers;
    std::vector<Models::VarInit> m_PreVarInitialisers;
    std::vector<Models::VarInit> m_PostVarInitialisers;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;
    std::vector<VarLocation> m_PreVarLocation;
    std::vector<VarLocation> m_PostVarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    const std::vector<Models::WUVarReference> m_VarReferences;
    const std::vector<Models::VarReference> m_PreVarReferences;
    const std::vector<Models::VarReference> m_PostVarReferences;

    const SynapseGroupInternal *m_SynapseGroup;

    //! Is this custom update batched i.e. run in parallel across model batches
    bool m_Batched;
};