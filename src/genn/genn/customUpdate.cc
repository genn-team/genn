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
void CustomUpdateBase::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getCustomUpdateModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getCustomUpdateModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getExtraGlobalParamLocation(const std::string &varName) const
{
    return m_ExtraGlobalParamLocation[getCustomUpdateModel()->getExtraGlobalParamIndex(varName)];
}
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
    return getCustomUpdateModel()->canBeMerged(other.getCustomUpdateModel());
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
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName, Operation operation,
                               const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                               const std::vector<Models::VarInit> &varInitialisers, const std::vector<WUVarReference> &varReferences,
                               VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_Operation(operation), m_VarReferences(varReferences)
{
    // Check variable reference types
    checkVarReferenceTypes(m_VarReferences);

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

    // Give error if any sizes differ
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const WUVarReference &v) 
                    { 
                        return ((v.getPreSize() != m_VarReferences.front().getPreSize())
                                || (v.getMaxRowLength() != m_VarReferences.front().getMaxRowLength())); 
                    }))
    {
        throw std::runtime_error("All referenced variables must have the same size.");
    }
}