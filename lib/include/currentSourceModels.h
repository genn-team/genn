#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
// #include "neuronModels.h"
#include "newModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_INJECTION_CODE(INJECTION_CODE) virtual std::string getInjectionCode() const{ return INJECTION_CODE; }
#define SET_TIME_CONDITION_CODE(TIME_CONDITION_CODE) virtual std::string getTimeConditionCode() const{ return TIME_CONDITION_CODE; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual StringPairVec getExtraGlobalParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// CurrentSourceModels::Base
//----------------------------------------------------------------------------
namespace CurrentSourceModels
{
//! Base class for all current source models
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets the code that defines current injected each timestep 
    virtual std::string getInjectionCode() const{ return ""; }

    //! Gets code which defines the condition when the should be applied described current source model.
    /*! This evaluates to a bool (e.g. "t > 20 && t < 40"). */
    virtual std::string getTimeConditionCode() const{ return ""; }

    //! Gets names and types (as strings) of additional parameters
    virtual NewModels::Base::StringPairVec getExtraGlobalParams() const{ return {}; }
};

//----------------------------------------------------------------------------
// CurrentSourceModels::ACSource
//----------------------------------------------------------------------------
//! AC source
/*! It has 6 parameters:

    - \c tStart - time at which the current starts [ms]
    - \c tStop  - time at which the current stops [ms]
    - \c amp    - amplitude of the sine wave [nA]
    - \c freq   - frequency of the sine wave [Hz]
    - \c phase  - phase of the sine wave [rad]
    - \c offset - offset of the sine wave [nA]
*/
class ACSource : public Base
{
    DECLARE_MODEL(ACSource, 6, 0);

    SET_TIME_CONDITION_CODE("t > $(tStart) && t < $(tStop)");

    SET_INJECTION_CODE("$(amp) * sin($(omega)*t + $(phase)) + $(offset)");

    SET_PARAM_NAMES({"tStart", "tStop", "amp", "freq", "phase", "offset"} );

    SET_DERIVED_PARAMS({{"omega", [](const vector<double> &pars, double dt){ return std::acos(-1.0) * 2.0 * pars[3]; }}});
};

//----------------------------------------------------------------------------
// CurrentSourceModels::DCSource
//----------------------------------------------------------------------------
//! DC source
/*! It has 3 parameters:

    - \c tStart - time at which the current starts [ms]
    - \c tStop  - time at which the current stops [ms]
    - \c amp    - amplitude of the current [nA]
*/
class DCSource : public Base
{
    DECLARE_MODEL(DCSource, 3, 0);

    SET_TIME_CONDITION_CODE("t > $(tStart) && t < $(tStop)");

    SET_INJECTION_CODE("$(amp)");

    SET_PARAM_NAMES({"tStart", "tStop", "amp"});
};

//----------------------------------------------------------------------------
// CurrentSourceModels::NoisyNormalCurrentSource
//----------------------------------------------------------------------------
//! Noisy current source with noise drawn from normal distribution
/*! It has 4 parameters:

    - \c tStart - time at which the current starts [ms]
    - \c tStop  - time at which the current stops [ms]
    - \c mean   - mean of the normal distribution [nA]
    - \c sd     - standard deviation of the normal distribution [nA]
*/
class NoisyNormalCurrentSource : public Base
{
    DECLARE_MODEL(NoisyNormalCurrentSource, 3, 0);

    SET_TIME_CONDITION_CODE("t > $(tStart) && t < $(tStop)");

    SET_INJECTION_CODE("$(mean) + $(gennrand_normal) * $(sd)");

    SET_PARAM_NAMES({"tStart", "tStop", "mean", "sd"} );
};
} // CurrentSourceModels
