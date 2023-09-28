#pragma once

// GeNN includes
#include "customUpdate.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// GeNN::CustomUpdateInternal
//------------------------------------------------------------------------
namespace GeNN
{
class CustomUpdateInternal : public CustomUpdate
{
public:
    using GroupExternal = CustomUpdate;

    CustomUpdateInternal(const std::string &name, const std::string &updateGroupName,
                         const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, double> &params, 
                         const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::VarReference> &varReferences, 
                         const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CustomUpdate(name, updateGroupName, customUpdateModel, params, varInitialisers, varReferences, 
                     egpReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CustomUpdateBase::getDerivedParams;
    using CustomUpdateBase::isInitRNGRequired;
    using CustomUpdateBase::isZeroCopyEnabled;
    using CustomUpdateBase::getDims;
    using CustomUpdateBase::getVarLocationHashDigest;
    using CustomUpdateBase::getUpdateCodeTokens;

    using CustomUpdate::finalise;
    using CustomUpdate::getHashDigest;
    using CustomUpdate::getInitHashDigest;
    using CustomUpdate::getDelayNeuronGroup;
    using CustomUpdate::getReferencedCustomUpdates;
    using CustomUpdate::isBatchReduction;
    using CustomUpdate::isNeuronReduction;
};


//----------------------------------------------------------------------------
// CustomUpdateVarAdapter
//----------------------------------------------------------------------------
class CustomUpdateVarAdapter
{
public:
    CustomUpdateVarAdapter(const CustomUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CU.getVarLocation(varName); }

    std::vector<Models::Base::CustomUpdateVar> getDefs() const{ return m_CU.getCustomUpdateModel()->getVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CU.getVarInitialisers(); }

    bool isVarDelayed(const std::string &) const { return false; }

    const std::string &getNameSuffix() const{ return m_CU.getName(); }

    VarAccessDim getVarDims(const Models::Base::CustomUpdateVar &var) const
    { 
        return getVarAccessDim(var.access, m_CU.getDims());
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomUpdateEGPAdapter
{
public:
    CustomUpdateEGPAdapter(const CustomUpdateInternal &cu) : m_CU(cu)
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
    const CustomUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateVarRefAdapter
//----------------------------------------------------------------------------
class CustomUpdateVarRefAdapter
{
public:
    CustomUpdateVarRefAdapter(const CustomUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    Models::Base::VarRefVec getDefs() const{ return m_CU.getCustomUpdateModel()->getVarRefs(); }

    const std::unordered_map<std::string, Models::VarReference> &getInitialisers() const{ return m_CU.getVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateInternal &m_CU;
};

//------------------------------------------------------------------------
// CustomUpdateInternal
//------------------------------------------------------------------------
class CustomUpdateWUInternal : public CustomUpdateWU
{
public:
    using GroupExternal = CustomUpdateWU;

    CustomUpdateWUInternal(const std::string &name, const std::string &updateGroupName,
                           const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, double> &params, 
                           const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::WUVarReference> &varReferences, 
                           const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CustomUpdateWU(name, updateGroupName, customUpdateModel, params, varInitialisers, varReferences, 
                       egpReferences, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
        getSynapseGroup()->addCustomUpdateReference(this);
    }

    using CustomUpdateBase::getDerivedParams;
    using CustomUpdateBase::isInitRNGRequired;
    using CustomUpdateBase::isZeroCopyEnabled;
    using CustomUpdateBase::getDims;
    using CustomUpdateBase::isReduction;
    using CustomUpdateBase::getVarLocationHashDigest;
    using CustomUpdateBase::getUpdateCodeTokens;
    
    using CustomUpdateWU::finalise;
    using CustomUpdateWU::getHashDigest;
    using CustomUpdateWU::getInitHashDigest;
    using CustomUpdateWU::getSynapseGroup;
    using CustomUpdateWU::getKernelSize;
    using CustomUpdateWU::getReferencedCustomUpdates;
    using CustomUpdateWU::isBatchReduction;
    using CustomUpdateWU::isTransposeOperation;
};

//----------------------------------------------------------------------------
// CustomUpdateWUVarRefAdapter
//----------------------------------------------------------------------------
class CustomUpdateWUVarRefAdapter
{
public:
    CustomUpdateWUVarRefAdapter(const CustomUpdateWUInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::WUVarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    Models::Base::VarRefVec getDefs() const{ return m_CU.getCustomUpdateModel()->getVarRefs(); }

    const std::unordered_map<std::string, Models::WUVarReference> &getInitialisers() const{ return m_CU.getVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateWUInternal &m_CU;
};
}   // namespace GeNN
