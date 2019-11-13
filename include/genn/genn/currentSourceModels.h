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
class Base : public Models::Base
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
    //! Can this neuron model be merged with other? i.e. can they be simulated using same generated code
    bool canBeMerged(const Base *other) const;
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
} // CurrentSourceModels
