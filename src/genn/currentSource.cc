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

VarLocation CurrentSource::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCurrentSourceModel()->getVarIndex(varName)];
}

void CurrentSource::initDerivedParams(double dt)
{
    auto derivedParams = getCurrentSourceModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.second(m_Params, dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}

bool CurrentSource::isInitCodeRequired() const
{
    // Return true if any of the variables initialisers have any code
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const Models::VarInit &v)
                       {
                           return !v.getSnippet()->getCode().empty();
                       });
}

bool CurrentSource::isSimRNGRequired() const
{
    // Returns true if any parts of the current source code require an RNG
    if(Utils::isRNGRequired(getCurrentSourceModel()->getInjectionCode())) {
        return true;
    }

    return false;
}

bool CurrentSource::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isInitRNGRequired(m_VarInitialisers)) {
        return true;
    }

    return false;
}