#include "currentSource.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
#include "synapseGroup.h"

//------------------------------------------------------------------------
// CurrentSource
//------------------------------------------------------------------------
void CurrentSource::setVarMode(const std::string &varName, VarMode mode)
{
    m_VarMode[getCurrentSourceModel()->getVarIndex(varName)] = mode;
}

VarMode CurrentSource::getVarMode(const std::string &varName) const
{
    return m_VarMode[getCurrentSourceModel()->getVarIndex(varName)];
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

void CurrentSource::addExtraGlobalParams(std::map<std::string, std::string> &kernelParameters) const
{
    for(auto const &p : getCurrentSourceModel()->getExtraGlobalParams()) {
        std::string pnamefull = p.first + getName();
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getCurrentSourceModel()->getInjectionCode().find("$(" + p.first + ")") != std::string::npos) {
                kernelParameters.emplace(pnamefull, p.second);
            }
        }
    }
}

bool CurrentSource::isInitCodeRequired() const
{
    // Return true if any of the variables initialisers have any code
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const NewModels::VarInit &v)
                       {
                           return !v.getSnippet()->getCode().empty();
                       });
}

bool CurrentSource::isSimRNGRequired() const
{
    // Returns true if any parts of the current source code require an RNG
    if(::isRNGRequired(getCurrentSourceModel()->getInjectionCode()))
    {
        return true;
    }

    return false;
}

bool CurrentSource::isInitRNGRequired(VarInit varInitMode) const
{
    // If initialising the neuron variables require an RNG, return true
    if(::isInitRNGRequired(m_VarInitialisers, m_VarMode, varInitMode)) {
        return true;
    }

    return false;
}

bool CurrentSource::isDeviceVarInitRequired() const
{
    // Return true if any of the variables are initialised on the device
    if(std::any_of(m_VarMode.cbegin(), m_VarMode.cend(),
                   [](const VarMode mode){ return (mode & VarInit::DEVICE); }))
    {
        return true;
    }
    return false;
}

bool CurrentSource::canRunOnCPU() const
{
    // Return true if all of the variables are present on the host
    return std::all_of(m_VarMode.cbegin(), m_VarMode.cend(),
                       [](const VarMode mode){ return (mode & VarLocation::HOST); });
}
