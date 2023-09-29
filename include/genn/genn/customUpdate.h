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
class GENN_EXPORT CustomUpdate
{
public:
    CustomUpdate(const CustomUpdate &) = delete;
    CustomUpdate() = delete;

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
    const std::unordered_map<std::string, Models::VarReference> &getVarReferences() const{ return m_VarReferences;  }
    const std::unordered_map<std::string, Models::EGPReference> &getEGPReferences() const{ return m_EGPReferences;  }

    //! Get variable location for custom update model state variable
    VarLocation getVarLocation(const std::string &varName) const;

    //! Is var init code required for any variables in this custom update group's custom update model?
    bool isVarInitRequired() const;

protected:
    CustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *customUpdateModel, 
                 const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                 const std::unordered_map<std::string, Models::VarReference> &varReferences, const std::unordered_map<std::string, Models::EGPReference> &egpReferences, 
                 VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation);

    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void finalise(double dt, unsigned int batchSize);

    //------------------------------------------------------------------------
    // Protected const methods
    //------------------------------------------------------------------------
    const std::unordered_map<std::string, double> &getDerivedParams() const{ return m_DerivedParams; }

    //! Does this current source group require an RNG for it's init code
    bool isInitRNGRequired() const;

    bool isZeroCopyEnabled() const;

    bool isModelReduction() const;

    bool isTransposeOperation() const;

    //! Get dimensions of this custom update
    VarAccessDim getDims() const{ return m_Dims; }

    //! Is this custom update synaptic? i.e. has pre and postsynaptic neuron dimensions
    bool isSynaptic() const{ return ((getDims() & VarAccessDim::PRE_NEURON) && (getDims() & VarAccessDim::POST_NEURON)); }

    //! Get size of this custom update
    /*! Only non-synaptic custom updates have size */
    std::optional<unsigned int> getSize() const { return m_Size; }

     //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Updates hash with custom update
    /*! NOTE: this can only be called after model is finalized */
    boost::uuids::detail::sha1::digest_type getInitHashDigest() const;

    boost::uuids::detail::sha1::digest_type getVarLocationHashDigest() const;

    const std::vector<Transpiler::Token> getUpdateCodeTokens() const{ return m_UpdateCodeTokens; }

    bool isReduction(VarAccessDim reduceDim) const;

    std::vector<CustomUpdate*> getReferencedCustomUpdates() const;

    const NeuronGroup *getDelayNeuronGroup() const{ return m_DelayNeuronGroup; }
    const SynapseGroupInternal *getSynapseGroup() const{ return m_SynapseGroup; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_Name;
    std::string m_UpdateGroupName;

    const CustomUpdateModels::Base *m_CustomUpdateModel;
    std::unordered_map<std::string, double> m_Params;
    std::unordered_map<std::string, double> m_DerivedParams;
    std::unordered_map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    const std::unordered_map<std::string, Models::VarReference> m_VarReferences;

    std::unordered_map<std::string, Models::EGPReference> m_EGPReferences;

    //! Location of individual state variables
    std::vector<VarLocation> m_VarLocation;

    //! Location of extra global parameters
    std::vector<VarLocation> m_ExtraGlobalParamLocation;

    //! Tokens produced by scanner from update code
    std::vector<Transpiler::Token> m_UpdateCodeTokens;

    //! Dimensions of this custom update
    VarAccessDim m_Dims;

    std::optional<unsigned int> m_Size;
    const NeuronGroup *m_DelayNeuronGroup;
    const SynapseGroupInternal *m_SynapseGroup;
};

}   // namespace GeNN
