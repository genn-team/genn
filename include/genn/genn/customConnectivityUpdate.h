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

    const std::unordered_map<std::string, double> &getParams() const { return m_Params; }
    const std::unordered_map<std::string, Models::VarInit> &getVarInitialisers() const { return m_VarInitialisers; }
    const std::unordered_map<std::string, Models::VarInit> &getPreVarInitialisers() const { return m_PreVarInitialisers; }
    const std::unordered_map<std::string, Models::VarInit> &getPostVarInitialisers() const { return m_PostVarInitialisers; }

    const std::unordered_map<std::string, Models::WUVarReference> &getVarReferences() const{ return m_VarReferences;  }
    const std::unordered_map<std::string, Models::VarReference> &getPreVarReferences() const{ return m_PreVarReferences;  }
    const std::unordered_map<std::string, Models::VarReference> &getPostVarReferences() const{ return m_PostVarReferences;  }

    //! Get variable location for synaptic state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Get variable location for presynaptic state variable
    VarLocation getPreVarLocation(const std::string &varName) const;
    
    //! Get variable location for postsynaptic state variable
    VarLocation getPostVarLocation(const std::string &varName) const;

    //! Is var init code required for any synaptic variables in this custom connectivity update group?
    bool isVarInitRequired() const;

    //! Is var init code required for any presynaptic variables in this custom connectivity update group?
    bool isPreVarInitRequired() const;

    //! Is var init code required for any postsynaptic variables in this custom connectivity update group?
    bool isPostVarInitRequired() const;

    //! Is a per-row RNG required for this custom connectivity update group
    bool isRowSimRNGRequired() const;

    //! Is a host RNG required for this custom connectivity update group
    bool isHostRNGRequired() const;
    
protected:
    CustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, SynapseGroupInternal *synapseGroup,
                             const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel,
                             const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, Models::VarInit> &varInitialisers,
                             const std::unordered_map<std::string, Models::VarInit> &preVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &postVarInitialisers,
                             const std::unordered_map<std::string, Models::WUVarReference> &varReferences, const std::unordered_map<std::string, Models::VarReference> &preVarReferences,
                             const std::unordered_map<std::string, Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                             VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void initDerivedParams(double dt);

    void finalize(unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const std::unordered_map<std::string, double> &getDerivedParams() const { return m_DerivedParams; }

    //! Does this current source group require an RNG for initialising its presynaptic variables
    bool isPreVarInitRNGRequired() const;

    //! Does this current source group require an RNG for initialising its postsynaptic variables
    bool isPostVarInitRNGRequired() const;

    //! Does this current source group require an RNG for initialising its synaptic variables
    bool isVarInitRNGRequired() const;

    bool isZeroCopyEnabled() const;

    SynapseGroupInternal *getSynapseGroup() const { return m_SynapseGroup; }

    //! Get vector of group names and variables in synapse groups, custom updates and other 
    //! custom connectivity updates which are attached to the same sparse connectivity this 
    //! custom connectivty update will update and thus will need modifying when we add and remove synapses
    std::vector<Models::WUVarReference> getDependentVariables() const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    const NeuronGroup *getPreDelayNeuronGroup() const { return m_PreDelayNeuronGroup; }
    
    const NeuronGroup *getPostDelayNeuronGroup() const { return m_PostDelayNeuronGroup; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    NeuronGroup *getVarRefDelayGroup(const std::unordered_map<std::string, Models::VarReference> &varRefs, 
                                     const std::string &errorContext) const;
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_Name;
    const std::string m_UpdateGroupName;
    SynapseGroupInternal *m_SynapseGroup;

    const CustomConnectivityUpdateModels::Base *m_CustomConnectivityUpdateModel;
    const std::unordered_map<std::string, double> m_Params;
    std::unordered_map<std::string, double> m_DerivedParams;
    std::unordered_map<std::string, Models::VarInit> m_VarInitialisers;
    std::unordered_map<std::string, Models::VarInit> m_PreVarInitialisers;
    std::unordered_map<std::string, Models::VarInit> m_PostVarInitialisers;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;
    std::vector<VarLocation> m_PreVarLocation;
    std::vector<VarLocation> m_PostVarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    const std::unordered_map<std::string, Models::WUVarReference> m_VarReferences;
    const std::unordered_map<std::string, Models::VarReference> m_PreVarReferences;
    const std::unordered_map<std::string, Models::VarReference> m_PostVarReferences;
    
    const NeuronGroup *m_PreDelayNeuronGroup;
    const NeuronGroup *m_PostDelayNeuronGroup;
};
