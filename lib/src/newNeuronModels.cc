#include "newNeuronModels.h"

// Standard includes
#include <cassert>

// GeNN includes
#include "neuronModels.h"

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetSimCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.simCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetThresholdConditionCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.thresholdConditionCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetResetCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.resetCode;
}
//----------------------------------------------------------------------------
std::string NeuronModels::LegacyWrapper::GetSupportCode() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.supportCode;
}
//----------------------------------------------------------------------------
std::vector<std::string>  NeuronModels::LegacyWrapper::GetParamNames() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return nm.pNames;
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, NeuronModels::Base::DerivedParamFunc>> NeuronModels::LegacyWrapper::GetDerivedParams() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];

    // Reserve vector to hold derived parameters
    std::vector<std::pair<std::string, DerivedParamFunc>> derivedParams;
    derivedParams.reserve(nm.dpNames.size());

    // Loop through derived parameters
    for(size_t p = 0; p < nm.dpNames.size(); p++)
    {
        // Add pair consisting of parameter name and lambda function which calls
        // through to the DPS object associated with the legacy model
        derivedParams.push_back(std::pair<std::string, DerivedParamFunc>(
          nm.dpNames[p],
          [this, p](const vector<double> &pars, double dt)
          {
              return nModels[m_LegacyTypeIndex].dps->calculateDerivedParameter(p, pars, dt);
          }
        ));
    }

    return derivedParams;
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> NeuronModels::LegacyWrapper::GetInitVals() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return ZipStringVectors(nm.varNames, nm.varTypes);
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> NeuronModels::LegacyWrapper::GetExtraGlobalParams() const
{
    const auto &nm = nModels[m_LegacyTypeIndex];
    return ZipStringVectors(nm.extraGlobalNeuronKernelParameters, nm.extraGlobalNeuronKernelParameterTypes);
}
//----------------------------------------------------------------------------
bool NeuronModels::LegacyWrapper::IsPoisson() const
{
    return (m_LegacyTypeIndex == POISSONNEURON);
}
//----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> NeuronModels::LegacyWrapper::ZipStringVectors(const std::vector<std::string> &a,
                                                                                               const std::vector<std::string> &b)
{
    assert(a.size() == b.size());

    // Reserve vector to hold initial values
    std::vector<std::pair<std::string, std::string>> zip;
    zip.reserve(a.size());

    // Build vector from legacy neuron model
    for(size_t v = 0; v < a.size(); v++)
    {
        zip.push_back(std::pair<std::string, std::string>(a[v], b[v]));
    }

    return zip;
}