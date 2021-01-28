#include "customUpdate.h"


// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"

//------------------------------------------------------------------------
// CustomUpdate
//------------------------------------------------------------------------
void CustomUpdate::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void CustomUpdate::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getCustomUpdateModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getCustomUpdateModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdate::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation CustomUpdate::getExtraGlobalParamLocation(const std::string &varName) const
{
    return m_ExtraGlobalParamLocation[getCustomUpdateModel()->getExtraGlobalParamIndex(varName)];
}
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName, Operation operation, 
                           const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params, 
                           const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::VarReference> &varReferences, 
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_Operation(operation), m_CustomUpdateModel(customUpdateModel), m_Params(params), 
    m_VarInitialisers(varInitialisers), m_VarReferences(varReferences), m_VarLocation(varInitialisers.size(), defaultVarLocation),
    m_ExtraGlobalParamLocation(customUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation)
{
    // Loop through all variable references and check types of variable references against those specified in model
    // **THINK** due to GeNN's current string-based type system this is rather conservative
    for(size_t i = 0; i < varReferences.size(); i++) {
        if(m_VarReferences.at(i).getVar().type != getCustomUpdateModel()->getVarRefs().at(i).type) {
            throw std::runtime_error("Incompatible type for variable reference '" + getCustomUpdateModel()->getVarRefs().at(i).name + "'");
        }
    }

    // If this is a transpose operation
    if(m_Operation == Operation::UPDATE_TRANSPOSE) {
        // Give error if any of the variable references aren't to weight update variables
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const Models::VarReference &v) { return v.getType() != Models::VarReference::Type::WU; }))
        {
            throw std::runtime_error("Custom updates that perform a transpose operation can only operate on weight update model variables.");
        }
    }
}
//----------------------------------------------------------------------------
void CustomUpdate::initDerivedParams(double dt)
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
bool CustomUpdate::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CustomUpdate::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
}
//----------------------------------------------------------------------------
bool CustomUpdate::canBeMerged(const CustomUpdate &other) const
{
    return getCustomUpdateModel()->canBeMerged(other.getCustomUpdateModel());
}
//----------------------------------------------------------------------------
bool CustomUpdate::canInitBeMerged(const CustomUpdate &other) const
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
