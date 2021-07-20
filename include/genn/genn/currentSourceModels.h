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

//----------------------------------------------------------------------------
// CurrentSourceModels::Base
//----------------------------------------------------------------------------
namespace CurrentSourceModels
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

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    using Models::Base::validate;
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
    DECLARE_MODEL(DC, 1, 0);

    SET_INJECTION_CODE("$(injectCurrent, $(amp));\n");

    SET_PARAM_NAMES({"amp"});
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
    DECLARE_MODEL(GaussianNoise, 2, 0);

    SET_INJECTION_CODE("$(injectCurrent, $(mean) + $(gennrand_normal) * $(sd));\n");

    SET_PARAM_NAMES({"mean", "sd"} );
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
    DECLARE_MODEL(PoissonExp, 3, 1);

    SET_INJECTION_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(ExpMinusLambda));\n"
        "$(current) += $(Init) * (scalar)(numSpikes - 1);\n"
        "$(injectCurrent, $(current));\n"
        "$(current) *= $(ExpDecay);\n");

    SET_PARAM_NAMES({"weight", "tauSyn", "rate"});
    SET_VARS({{"current", "scalar"}});
    SET_DERIVED_PARAMS({
        {"ExpDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Init", [](const std::vector<double> &pars, double dt){ return pars[0] * (1.0 - std::exp(-dt / pars[1])) * (pars[1] / dt); }},
        {"ExpMinusLambda", [](const std::vector<double> &pars, double dt){ return std::exp(-(pars[2] / 1000.0) * dt); }}});
};
} // CurrentSourceModels
