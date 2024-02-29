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
                         const CustomUpdateModels::Base *customUpdateModel, const std::map<std::string, Type::NumericValue> &params, 
                         const std::map<std::string, InitVarSnippet::Init> &varInitialisers, const std::map<std::string, Models::VarReference> &varReferences, 
                         const std::map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
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
    auto getDefs() const{ return m_CU.getModel()->getVarRefs(); }

    const auto &getInitialisers() const{ return m_CU.getVarReferences(); }

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
                           const CustomUpdateModels::Base *customUpdateModel, const std::map<std::string, Type::NumericValue> &params, 
                           const std::map<std::string, InitVarSnippet::Init> &varInitialisers, const std::map<std::string, Models::WUVarReference> &varReferences, 
                           const std::map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
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
    auto getDefs() const{ return m_CU.getModel()->getVarRefs(); }

    const auto &getInitialisers() const{ return m_CU.getVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateWUInternal &m_CU;
};
}   // namespace GeNN
