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
#define SET_NEURON_VAR_REFS(...) virtual VarRefVec getNeuronVarRefs() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::Base
//----------------------------------------------------------------------------
namespace GeNN::PostsynapticModels
{
//! Base class for all postsynaptic models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets model variables
    virtual std::vector<Var> getVars() const{ return {}; }

    //! Gets names and types of model variable references
    virtual VarRefVec getNeuronVarRefs() const{ return {}; }
    
    virtual std::string getDecayCode() const{ return ""; }
    virtual std::string getApplyInputCode() const{ return ""; }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Find the index of a named variable
    size_t getVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getVars());
    }

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, double> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                  const std::string &description) const;
};

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::ExpCurr
//----------------------------------------------------------------------------
//! Exponential decay with synaptic input treated as a current value.
/*! This model has no variables and a single parameter:
  - \c tau : Decay time constant*/
class ExpCurr : public Base
{
public:
    DECLARE_SNIPPET(ExpCurr);

    SET_DECAY_CODE("$(inSyn) *= $(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(init) * $(inSyn)");

    SET_PARAM_NAMES({"tau"});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::unordered_map<std::string, double> &pars, double dt){ return std::exp(-dt / pars.at("tau")); }},
        {"init", [](const std::unordered_map<std::string, double> &pars, double dt){ return (pars.at("tau") * (1.0 - std::exp(-dt / pars.at("tau")))) * (1.0 / dt); }}});
};

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::ExpCond
//----------------------------------------------------------------------------
//! Exponential decay with synaptic input treated as a conductance value.
/*! This model has no variables, two parameters and a variable reference
  - \c tau : Decay time constant
  - \c E   : Reversal potential
  - \c V   : Is a reference to the neuron's membrane voltage
  \c tau is used by the derived parameter \c expdecay which returns expf(-dt/tau). */
class ExpCond : public Base
{
public:
    DECLARE_SNIPPET(ExpCond);

    SET_DECAY_CODE("$(inSyn)*=$(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(inSyn) * ($(E) - $(V))");

    SET_PARAM_NAMES({"tau", "E"});

    SET_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_DERIVED_PARAMS({{"expDecay", [](const std::unordered_map<std::string, double> &pars, double dt){ return std::exp(-dt / pars.at("tau")); }}});
};

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::DeltaCurr
//----------------------------------------------------------------------------
//! Simple delta current synapse.
/*! Synaptic input provides a direct inject of instantaneous current*/
class DeltaCurr : public Base
{
public:
    DECLARE_SNIPPET(DeltaCurr);

    SET_CURRENT_CONVERTER_CODE("$(inSyn); $(inSyn) = 0");
};
}   // namespace GeNN::PostsynapticModels
