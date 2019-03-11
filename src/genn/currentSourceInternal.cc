#include "currentSourceInternal.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"

//------------------------------------------------------------------------
// CurrentSourceInternal
//------------------------------------------------------------------------
void CurrentSourceInternal::initDerivedParams(double dt)
{
    auto derivedParams = getCurrentSourceModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.second(getParams(), dt));
    }

    // Initialise derived parameters for variable initialisers
    initInitialiserDerivedParams(dt);
}

bool CurrentSourceInternal::isInitCodeRequired() const
{
    // Return true if any of the variables initialisers have any code
    return std::any_of(getVarInitialisers().cbegin(), getVarInitialisers().cend(),
                       [](const Models::VarInit &v)
                       {
                           return !v.getSnippet()->getCode().empty();
                       });
}

bool CurrentSourceInternal::isSimRNGRequired() const
{
    // Returns true if any parts of the current source code require an RNG
    if(Utils::isRNGRequired(getCurrentSourceModel()->getInjectionCode())) {
        return true;
    }

    return false;
}

bool CurrentSourceInternal::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isInitRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
