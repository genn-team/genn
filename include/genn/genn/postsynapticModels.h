#pragma once

// Standard C includes
#include <cmath>

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const override{ return SIM_CODE; }
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
    
    virtual std::string getSimCode() const{ return ""; }
    
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
    void validate(const std::map<std::string, Type::NumericValue> &paramValues, 
                  const std::map<std::string, InitVarSnippet::Init> &varValues,
                  const std::map<std::string, Models::VarReference> &varRefTargets) const;
};

//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<Base>
{
public:
    Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params, 
         const std::map<std::string, InitVarSnippet::Init> &varInitialisers, 
         const std::map<std::string, Models::VarReference> &neuronVarReferences);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;
    bool isVarInitRequired() const;

    const auto &getVarInitialisers() const{ return m_VarInitialisers; }
    const auto &getNeuronVarReferences() const{ return m_NeuronVarReferences;  }
    
    const auto &getSimCodeTokens() const{ return m_SimCodeTokens; }

    void finalise(double dt);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Transpiler::Token> m_SimCodeTokens;

    std::map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    std::map<std::string, Models::VarReference> m_NeuronVarReferences;
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

    SET_SIM_CODE(
        "injectCurrent(init * inSyn);\n"
        "inSyn *= expDecay;\n");

    SET_PARAMS({"tau"});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tau").cast<double>()); }},
        {"init", [](const ParamValues &pars, double dt){ return (pars.at("tau").cast<double>() * (1.0 - std::exp(-dt / pars.at("tau").cast<double>()))) * (1.0 / dt); }}});
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

    SET_SIM_CODE(
        "injectCurrent(inSyn * (E - V));\n"
        "inSyn *= expDecay;\n");

    SET_PARAMS({"tau", "E"});

    SET_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_DERIVED_PARAMS({{"expDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tau").cast<double>()); }}});
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

    SET_SIM_CODE(
        "injectCurrent(inSyn);\n"
        "inSyn = 0.0;\n");
};
}   // namespace GeNN::PostsynapticModels
