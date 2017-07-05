#pragma once

// GeNN includes
#include "newModels.h"
#include "postSynapseModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_DECAY_CODE(DECAY_CODE) virtual std::string getDecayCode() const{ return DECAY_CODE; }
#define SET_CURRENT_CONVERTER_CODE(CURRENT_CONVERTER_CODE) virtual std::string getCurrentConverterCode() const{ return CURRENT_CONVERTER_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string getSupportCode() const{ return SUPPORT_CODE; }

//----------------------------------------------------------------------------
// PostsynapticModels::Base
//----------------------------------------------------------------------------
namespace PostsynapticModels
{
//! Base class for all postsynaptic models
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getDecayCode() const{ return ""; }
    virtual std::string getCurrentConverterCode() const{ return ""; }
    virtual std::string getSupportCode() const{ return ""; }
};

//----------------------------------------------------------------------------
// PostsynapticModels::LegacyWrapper
//----------------------------------------------------------------------------
class LegacyWrapper : public NewModels::LegacyWrapper<Base, postSynModel, postSynModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, postSynModel, postSynModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual std::string getDecayCode() const;
    virtual std::string getCurrentConverterCode() const;
    virtual std::string getSupportCode() const;
};

//----------------------------------------------------------------------------
// PostsynapticModels::ExpConductance
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

    SET_DERIVED_PARAMS({{"expDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
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

    SET_PARAM_NAMES({"tau", "E"});
};
}