#include "customUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomUpdateBase
//------------------------------------------------------------------------
void CustomUpdateBase::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
void CustomUpdateBase::initDerivedParams(double dt)
{
    auto derivedParams = getCustomUpdateModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(getParams(), dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canBeMerged(const CustomUpdateBase &other) const
{
    return (getCustomUpdateModel()->canBeMerged(other.getCustomUpdateModel())
            && (getUpdateGroupName() == other.getUpdateGroupName()));
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canInitBeMerged(const CustomUpdateBase &other) const
{
     // If both groups have the same number of variables
    if(getVarInitialisers().size() == other.getVarInitialisers().size()) {
        // if any of the variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getVarInitialisers().size(); i++) {
            if(!getVarInitialisers()[i].canBeMerged(other.getVarInitialisers()[i])) {
                return false;
            }
        }
        
        return true;
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
// CustomUpdate
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName,
                           const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params, 
                           const std::vector<Models::VarInit> &varInitialisers, const std::vector<VarReference> &varReferences, 
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
                    [this](const VarReference &v) { return v.getSize() != m_Size; }))
    {
        throw std::runtime_error("All referenced variables must have the same size.");
    }
}

//----------------------------------------------------------------------------
// CustomUpdateWU
//----------------------------------------------------------------------------
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName, Operation operation,
                               const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                               const std::vector<Models::VarInit> &varInitialisers, const std::vector<WUVarReference> &varReferences,
                               VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_Operation(operation), m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<const SynapseGroupInternal*>(m_VarReferences.front().getSynapseGroup()))
{
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    checkVarReferenceTypes(m_VarReferences);

    // Give error if references point to different synapse groups
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const WUVarReference &v) 
                    { 
                        return (v.getSynapseGroup() != m_VarReferences.front().getSynapseGroup()); 
                    }))
    {
        throw std::runtime_error("All referenced variables must belong to the same synapse group.");
    }

    // If this is a transpose operation
    if(m_Operation == Operation::UPDATE_TRANSPOSE) {
        // Give error if any of the variable references aren't dense
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const WUVarReference &v) 
                       {
                           return !(v.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE); 
                       }))
        {
            throw std::runtime_error("Custom updates that perform a transpose operation can currently only be used on DENSE synaptic matrices.");
        }
    }
}