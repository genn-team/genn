#pragma once

// GeNN includes
#include "newModels.h"
#include "synapseModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string GetSimCode() const{ return SIM_CODE; }
#define SET_EVENT_CODE(EVENT_CODE) virtual std::string GetEventCode() const{ return EVENT_CODE; }
#define SET_LEARN_POST_CODE(LEARN_POST_CODE) virtual std::string GetLearnPostCode() const{ return LEARN_POST_CODE; }
#define SET_SYNAPSE_DYNAMICS_CODE(SYNAPSE_DYNAMICS_CODE) virtual std::string GetSynapseDynamicsCode() const{ return SYNAPSE_DYNAMICS_CODE; }
#define SET_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) virtual std::string GetEventThresholdConditionCode() const{ return EVENT_THRESHOLD_CONDITION_CODE; }

#define SET_SIM_SUPPORT_CODE(SIM_SUPPORT_CODE) virtual std::string GetSimSupportCode() const{ return SIM_SUPPORT_CODE; }
#define SET_LEARN_POST_SUPPORT_CODE(LEARN_POST_SUPPORT_CODE) virtual std::string GetLearnPostSupportCode() const{ return LEARN_POST_SUPPORT_CODE; }
#define SET_SYNAPSE_DYNAMICS_SUPPORT_CODE(SYNAPSE_DYNAMICS_SUPPORT_CODE) virtual std::string GetSynapseDynamicsSuppportCode() const{ return SYNAPSE_DYNAMICS_SUPPORT_CODE; }

#define SET_EXTRA_GLOBAL_PARAMS(...) virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return __VA_ARGS__; }

#define SET_NEEDS_PRE_SPIKE_TIME(NEEDS_PRE_SPIKE_TIME) static const bool NeedsPreSpikeTime = NEEDS_PRE_SPIKE_TIME
#define SET_NEEDS_POST_SPIKE_TIME(NEEDS_POST_SPIKE_TIME) static const bool NeedsPostSpikeTime = NEEDS_POST_SPIKE_TIME

//----------------------------------------------------------------------------
// WeightUpdateModels::Base
//----------------------------------------------------------------------------
namespace WeightUpdateModels
{
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const{ return ""; }
    virtual std::string GetEventCode() const{ return ""; }
    virtual std::string GetLearnPostCode() const{ return ""; }
    virtual std::string GetSynapseDynamicsCode() const{ return ""; }
    virtual std::string GetEventThresholdConditionCode() const{ return ""; }

    virtual std::string GetSimSupportCode() const{ return ""; }
    virtual std::string GetLearnPostSupportCode() const{ return ""; }
    virtual std::string GetSynapseDynamicsSuppportCode() const{ return ""; }

    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const{ return {}; }

    //----------------------------------------------------------------------------
    // Constants
    //----------------------------------------------------------------------------
    static const bool NeedsPreSpikeTime = false;
    static const bool NeedsPostSpikeTime = false;
};

//----------------------------------------------------------------------------
// WeightUpdateModels::LegacyWrapper
//----------------------------------------------------------------------------
class LegacyWrapper : public NewModels::LegacyWrapper<Base, weightUpdateModel, weightUpdateModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, weightUpdateModel, weightUpdateModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    virtual std::string GetSimCode() const;
    virtual std::string GetEventCode() const;
    virtual std::string GetLearnPostCode() const;
    virtual std::string GetSynapseDynamicsCode() const;
    virtual std::string GetEventThresholdConditionCode() const;

    virtual std::string GetSimSupportCode() const;
    virtual std::string GetLearnPostSupportCode() const;
    virtual std::string GetSynapseDynamicsSuppportCode() const;

    virtual std::vector<std::pair<std::string, std::string>> GetExtraGlobalParams() const;
};

//----------------------------------------------------------------------------
// WeightUpdateModels::StaticPulse
//----------------------------------------------------------------------------
class StaticPulse : public Base
{
public:
    DECLARE_MODEL(StaticPulse, 0, 1);

    SET_VARS({{"g","scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        " $(updatelinsyn);\n");
};

//----------------------------------------------------------------------------
// WeightUpdateModels::StaticGraded
//----------------------------------------------------------------------------
class StaticGraded : public Base
{
public:
    DECLARE_MODEL(StaticGraded, 2, 1);

    SET_PARAM_NAMES({"Epre", "Vslope"});
    SET_VARS({{"g","scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g) * tanh(($(V_pre) - $(Epre)) / $(Vslope))* DT;\n"
        "if ($(addtoinSyn) < 0) $(addtoinSyn) = 0.0;\n"
        " $(updatelinsyn);\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > $(Epre)");
};

//----------------------------------------------------------------------------
// Learn1
//----------------------------------------------------------------------------
// **TODO** what is this learning rule actually called
class Learn1 : public Base
{
public:
    DECLARE_MODEL(Learn1, 10, 2);

    SET_PARAM_NAMES({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
        "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);"
        "$(updatelinsyn); \n"
        "scalar dt = $(sT_post) - $(t) - ($(tauShift)); \n"
        "scalar dg = 0;\n"
        "if (dt > $(lim0))  \n"
        "    dg = -($(off0)) ; \n"
        "else if (dt > 0)  \n"
        "    dg = $(slope0) * dt + ($(off1)); \n"
        "else if (dt > $(lim1))  \n"
        "    dg = $(slope1) * dt + ($(off1)); \n"
        "else dg = - ($(off2)) ; \n"
        "$(gRaw) += dg; \n"
        "$(g)=$(gMax)/2 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n");
    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - ($(sT_pre)) - ($(tauShift)); \n"
        "scalar dg =0; \n"
        "if (dt > $(lim0))  \n"
        "    dg = -($(off0)) ; \n"
        "else if (dt > 0)  \n"
        "    dg = $(slope0) * dt + ($(off1)); \n"
        "else if (dt > $(lim1))  \n"
        "    dg = $(slope1) * dt + ($(off1)); \n"
        "else dg = -($(off2)) ; \n"
        "$(gRaw) += dg; \n"
        "$(g)=$(gMax)/2.0 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n");

    SET_DERIVED_PARAMS({
        {"lim0", [](const vector<double> &pars, double){ return (1/pars[4] + 1/pars[1]) * pars[0] / (2/pars[1]); }},
        {"lim1", [](const vector<double> &pars, double){ return  -((1/pars[3] + 1/pars[1]) * pars[0] / (2/pars[1])); }},
        {"slope0", [](const vector<double> &pars, double){ return  -2*pars[5]/(pars[1]*pars[0]); }},
        {"slope1", [](const vector<double> &pars, double){ return  2*pars[5]/(pars[1]*pars[0]); }},
        {"off0", [](const vector<double> &pars, double){ return  pars[5] / pars[4]; }},
        {"off1", [](const vector<double> &pars, double){ return  pars[5] / pars[1]; }},
        {"off2", [](const vector<double> &pars, double){ return  pars[5] / pars[3]; }}});

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
} // WeightUpdateModels