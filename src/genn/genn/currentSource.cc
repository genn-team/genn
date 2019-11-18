#include "currentSource.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"

//------------------------------------------------------------------------
// CurrentSource
//------------------------------------------------------------------------
void CurrentSource::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCurrentSourceModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void CurrentSource::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getCurrentSourceModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getCurrentSourceModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
VarLocation CurrentSource::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCurrentSourceModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation CurrentSource::getExtraGlobalParamLocation(const std::string &varName) const
{
    return m_ExtraGlobalParamLocation[getCurrentSourceModel()->getExtraGlobalParamIndex(varName)];
}
//----------------------------------------------------------------------------
void CurrentSource::initDerivedParams(double dt)
{
    auto derivedParams = getCurrentSourceModel()->getDerivedParams();

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
bool CurrentSource::isSimRNGRequired() const
{
    // Returns true if any parts of the current source code require an RNG
    if(Utils::isRNGRequired(getCurrentSourceModel()->getInjectionCode())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CurrentSource::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CurrentSource::canBeMerged(const CurrentSource &other) const
{
    return (getCurrentSourceModel()->canBeMerged(other.getCurrentSourceModel())
            && (getParams() == other.getParams())
            && (getDerivedParams() == other.getDerivedParams()));
}
//----------------------------------------------------------------------------
bool CurrentSource::canInitBeMerged(const CurrentSource &other) const
{
     // If both groups have the same number of variables
    if(getVarInitialisers().size() == other.getVarInitialisers().size()) {
        auto otherVarInitialisers = other.getVarInitialisers();
        for(const auto &var : getVarInitialisers()) {
            // If a compatible var can be found amongst the other neuron group's vars, remove it
            const auto otherVar = std::find_if(otherVarInitialisers.cbegin(), otherVarInitialisers.cend(),
                                                [var](const Models::VarInit &v)
                                                {
                                                    return var.canBeMerged(v);
                                                });
            if(otherVar != otherVarInitialisers.cend()) {
                otherVarInitialisers.erase(otherVar);
            }
            // Otherwise, these can't be merged - return false
            else {
                return false;
            }
        }
    }
    else {
        return false;
    }

    return true;
}
