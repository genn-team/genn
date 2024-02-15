#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_INJECTION_CODE(INJECTION_CODE) virtual std::string getInjectionCode() const override{ return INJECTION_CODE; }
#define SET_NEURON_VAR_REFS(...) virtual VarRefVec getNeuronVarRefs() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// GeNN::CurrentSourceModels::Base
//----------------------------------------------------------------------------
namespace GeNN::CurrentSourceModels
{
//! Base class for all current source models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets the code that defines current injected each timestep 
    virtual std::string getInjectionCode() const{ return ""; }

    //! Gets model variables
    virtual std::vector<Var> getVars() const{ return {}; }

    //! Gets names and types of model variable references
    virtual VarRefVec getNeuronVarRefs() const{ return {}; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Find the named variable
    std::optional<Var> getVar(const std::string &varName) const
    {
        return getNamed(varName, getVars());
    }

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                  const std::string &description) const;
};

//----------------------------------------------------------------------------
// CurrentSourceModels::DC
//----------------------------------------------------------------------------
//! DC source
/*! It has a single parameter:

    - \c amp    - amplitude of the current [nA]
*/
class DC : public Base
{
    DECLARE_SNIPPET(DC);

    SET_INJECTION_CODE("injectCurrent(amp);\n");

    SET_PARAMS({"amp"});
};

//----------------------------------------------------------------------------
// CurrentSourceModels::GaussianNoise
//----------------------------------------------------------------------------
//! Noisy current source with noise drawn from normal distribution
/*! It has 2 parameters:

    - \c mean   - mean of the normal distribution [nA]
    - \c sd     - standard deviation of the normal distribution [nA]
*/
class GaussianNoise : public Base
{
    DECLARE_SNIPPET(GaussianNoise);

    SET_INJECTION_CODE("injectCurrent(mean + (gennrand_normal() * sd));\n");

    SET_PARAMS({"mean", "sd"} );
};

//----------------------------------------------------------------------------
// CurrentSourceModels::PoissonExp
//----------------------------------------------------------------------------
//! Current source for injecting a current equivalent to a population of
//! Poisson spike sources, one-to-one connected with exponential synapses
/*! It has 3 parameters:

    - \c weight - synaptic weight of the Poisson spikes [nA]
    - \c tauSyn - decay time constant [ms]
    - \c rate   - mean firing rate [Hz]
*/
class PoissonExp : public Base
{
    DECLARE_SNIPPET(PoissonExp);

    SET_INJECTION_CODE(
        "scalar p = 1.0;\n"
        "unsigned int numSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numSpikes++;\n"
        "    p *= gennrand_uniform();\n"
        "} while (p > ExpMinusLambda);\n"
        "current += Init * (scalar)(numSpikes - 1);\n"
        "injectCurrent(current);\n"
        "current *= ExpDecay;\n");

    SET_PARAMS({"weight", "tauSyn", "rate"});
    SET_VARS({{"current", "scalar"}});
    SET_DERIVED_PARAMS({
        {"ExpDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauSyn").cast<double>()); }},
        {"Init", [](const ParamValues &pars, double dt){ return pars.at("weight").cast<double>() * (1.0 - std::exp(-dt / pars.at("tauSyn").cast<double>())) * (pars.at("tauSyn").cast<double>() / dt); }},
        {"ExpMinusLambda", [](const ParamValues &pars, double dt){ return std::exp(-(pars.at("rate").cast<double>() / 1000.0) * dt); }}});
};
} // GeNN::CurrentSourceModels
