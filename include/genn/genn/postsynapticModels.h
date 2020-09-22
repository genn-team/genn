#pragma once

// Standard C includes
#include <cmath>

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_DECAY_CODE(DECAY_CODE) virtual std::string getDecayCode() const override{ return DECAY_CODE; }
#define SET_CURRENT_CONVERTER_CODE(CURRENT_CONVERTER_CODE) virtual std::string getApplyInputCode() const override{ return "$(Isyn) += " CURRENT_CONVERTER_CODE ";"; }
#define SET_APPLY_INPUT_CODE(APPLY_INPUT_CODE) virtual std::string getApplyInputCode() const override{ return APPLY_INPUT_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string getSupportCode() const override{ return SUPPORT_CODE; }

//----------------------------------------------------------------------------
// PostsynapticModels::Base
//----------------------------------------------------------------------------
namespace PostsynapticModels
{
//! Base class for all postsynaptic models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getDecayCode() const{ return ""; }
    virtual std::string getApplyInputCode() const{ return ""; }
    virtual std::string getSupportCode() const{ return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Can this postsynaptic model be merged with other? i.e. can they be simulated using same generated code
    bool canBeMerged(const Base *other) const;
};

//----------------------------------------------------------------------------
// PostsynapticModels::ExpCurr
//----------------------------------------------------------------------------
//! Exponential decay with synaptic input treated as a current value.
/*! This model has no variables and a single parameter:
  - \c tau : Decay time constant*/
class ExpCurr : public Base
{
public:
    DECLARE_MODEL(ExpCurr, 1, 0);

    SET_DECAY_CODE("$(inSyn) *= $(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(init) * $(inSyn)");

    SET_PARAM_NAMES({"tau"});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"init", [](const std::vector<double> &pars, double dt){ return (pars[0] * (1.0 - std::exp(-dt / pars[0]))) * (1.0 / dt); }}});
};

//----------------------------------------------------------------------------
// PostsynapticModels::ExpCond
//----------------------------------------------------------------------------
//! Exponential decay with synaptic input treated as a conductance value.
/*! This model has no variables and two parameters:
  - \c tau : Decay time constant
  - \c E : Reversal potential

  \c tau is used by the derived parameter \c expdecay which returns expf(-dt/tau). */
class ExpCond : public Base
{
public:
    DECLARE_MODEL(ExpCond, 2, 0);

    SET_DECAY_CODE("$(inSyn)*=$(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(inSyn) * ($(E) - $(V))");

    SET_PARAM_NAMES({"tau", "E"});

    SET_DERIVED_PARAMS({{"expDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
};

//----------------------------------------------------------------------------
// PostsynapticModels::DeltaCurr
//----------------------------------------------------------------------------
//! Simple delta current synapse.
/*! Synaptic input provides a direct inject of instantaneous current*/
class DeltaCurr : public Base
{
public:
    DECLARE_MODEL(DeltaCurr, 0, 0);

    SET_CURRENT_CONVERTER_CODE("$(inSyn); $(inSyn) = 0");
};
}
