#pragma once

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DECLARE_WEIGHT_UPDATE_MODEL(TYPE, NUM_PARAMS, NUM_VARS, NUM_PRE_VARS, NUM_POST_VARS)    \
    DECLARE_SNIPPET(TYPE, NUM_PARAMS);                                                          \
    typedef Models::VarInitContainerBase<NUM_VARS> VarValues;                                \
    typedef Models::VarInitContainerBase<NUM_PRE_VARS> PreVarValues;                         \
    typedef Models::VarInitContainerBase<NUM_POST_VARS> PostVarValues

#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const override{ return SIM_CODE; }
#define SET_EVENT_CODE(EVENT_CODE) virtual std::string getEventCode() const override{ return EVENT_CODE; }
#define SET_LEARN_POST_CODE(LEARN_POST_CODE) virtual std::string getLearnPostCode() const override{ return LEARN_POST_CODE; }
#define SET_SYNAPSE_DYNAMICS_CODE(SYNAPSE_DYNAMICS_CODE) virtual std::string getSynapseDynamicsCode() const override{ return SYNAPSE_DYNAMICS_CODE; }
#define SET_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) virtual std::string getEventThresholdConditionCode() const override{ return EVENT_THRESHOLD_CONDITION_CODE; }

#define SET_SIM_SUPPORT_CODE(SIM_SUPPORT_CODE) virtual std::string getSimSupportCode() const override{ return SIM_SUPPORT_CODE; }
#define SET_LEARN_POST_SUPPORT_CODE(LEARN_POST_SUPPORT_CODE) virtual std::string getLearnPostSupportCode() const override{ return LEARN_POST_SUPPORT_CODE; }
#define SET_SYNAPSE_DYNAMICS_SUPPORT_CODE(SYNAPSE_DYNAMICS_SUPPORT_CODE) virtual std::string getSynapseDynamicsSuppportCode() const override{ return SYNAPSE_DYNAMICS_SUPPORT_CODE; }
#define SET_PRE_SPIKE_CODE(PRE_SPIKE_CODE) virtual std::string getPreSpikeCode() const override{ return PRE_SPIKE_CODE; }
#define SET_POST_SPIKE_CODE(POST_SPIKE_CODE) virtual std::string getPostSpikeCode() const override{ return POST_SPIKE_CODE; }

#define SET_PRE_VARS(...) virtual VarVec getPreVars() const override{ return __VA_ARGS__; }
#define SET_POST_VARS(...) virtual VarVec getPostVars() const override{ return __VA_ARGS__; }

#define SET_NEEDS_PRE_SPIKE_TIME(PRE_SPIKE_TIME_REQUIRED) virtual bool isPreSpikeTimeRequired() const override{ return PRE_SPIKE_TIME_REQUIRED; }
#define SET_NEEDS_POST_SPIKE_TIME(POST_SPIKE_TIME_REQUIRED) virtual bool isPostSpikeTimeRequired() const override{ return POST_SPIKE_TIME_REQUIRED; }

//----------------------------------------------------------------------------
// WeightUpdateModels::Base
//----------------------------------------------------------------------------
namespace WeightUpdateModels
{
//! Base class for all weight update models
class Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets simulation code run when 'true' spikes are received
    virtual std::string getSimCode() const{ return ""; }

    //! Gets code run when events (all the instances where event threshold condition is met) are received
    virtual std::string getEventCode() const{ return ""; }

    //! Gets code to include in the learnSynapsesPost kernel/function.
    /*! For examples when modelling STDP, this is where the effect of postsynaptic
        spikes which occur _after_ presynaptic spikes are applied. */
    virtual std::string getLearnPostCode() const{ return ""; }

    //! Gets code for synapse dynamics which are independent of spike detection
    virtual std::string getSynapseDynamicsCode() const{ return ""; }

    //! Gets codes to test for events
    virtual std::string getEventThresholdConditionCode() const{ return ""; }

    //! Gets support code to be made available within the synapse kernel/function.
    /*! This is intended to contain user defined device functions that are used in the weight update code.
        Preprocessor defines are also allowed if appropriately safeguarded against multiple
        definition by using ifndef; functions should be declared as "__host__ __device__"
        to be available for both GPU and CPU versions; note that this support code is available to
        sim, event threshold and event code */
    virtual std::string getSimSupportCode() const{ return ""; }

    //! Gets support code to be made available within learnSynapsesPost kernel/function.
    /*! Preprocessor defines are also allowed if appropriately safeguarded against multiple
        definition by using ifndef; functions should be declared as "__host__ __device__"
        to be available for both GPU and CPU versions. */
    virtual std::string getLearnPostSupportCode() const{ return ""; }

    //! Gets support code to be made available within the synapse dynamics kernel/function.
    /*! Preprocessor defines are also allowed if appropriately safeguarded against multiple
        definition by using ifndef; functions should be declared as "__host__ __device__"
        to be available for both GPU and CPU versions. */
    virtual std::string getSynapseDynamicsSuppportCode() const{ return ""; }

    //! Gets code to be run once per spiking presynaptic
    //! neuron before sim code is run on synapses
    /*! This is typically for the code to update presynaptic variables. Postsynaptic
        and synapse variables are not accesible from within this code */
    virtual std::string getPreSpikeCode() const{ return ""; }

    //! Gets code to be run once per spiking postsynaptic
    //! neuron before learn post code is run on synapses
    /*! This is typically for the code to update postsynaptic variables. Presynaptic
        and synapse variables are not accesible from within this code */
    virtual std::string getPostSpikeCode() const{ return ""; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses coming from the same presynaptic neuron
    virtual VarVec getPreVars() const{ return {}; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses going to the same postsynaptic neuron
    virtual VarVec getPostVars() const{ return {}; }

    //! Whether presynaptic spike times are needed or not
    virtual bool isPreSpikeTimeRequired() const{ return false; }

    //! Whether postsynaptic spike times are needed or not
    virtual bool isPostSpikeTimeRequired() const{ return false; }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the index of a named presynaptic variable
    size_t getPreVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getPreVars());
    }

    //! Find the index of a named postsynaptic variable
    size_t getPostVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getPostVars());
    }

};

//----------------------------------------------------------------------------
// WeightUpdateModels::StaticPulse
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse.
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 1 variable:
    - g - conductance of scalar type
    and no other parameters.

    \c sim code is:

    \code
    "$(addToInSyn, $(g));\n"
    \endcode*/
class StaticPulse : public Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulse, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");
};

//----------------------------------------------------------------------------
// WeightUpdateModels::StaticPulseDendriticDelay
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse with heterogenous dendritic delays
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 2 variables:
    - g - conductance of scalar type
    - d - dendritic delay in timesteps
    and no other parameters.

    \c sim code is:

    \code
    " $(addToInSynDelay, $(g), $(d));\n\
    \endcode*/
class StaticPulseDendriticDelay : public Base
{
public:
    DECLARE_MODEL(StaticPulseDendriticDelay, 0, 2);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY},{"d", "uint8_t", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSynDelay, $(g), $(d));\n");
};

//----------------------------------------------------------------------------
// WeightUpdateModels::StaticGraded
//----------------------------------------------------------------------------
//! Graded-potential, static synapse
/*! In a graded synapse, the conductance is updated gradually with the rule:
    \f[ gSyn= g * tanh((V - E_{pre}) / V_{slope} \f]
    whenever the membrane potential \f$V\f$ is larger than the threshold \f$E_{pre}\f$.
    The model has 1 variable:
    - \c g: conductance of \c scalar type

    The parameters are:
    - \c Epre: Presynaptic threshold potential
    - \c Vslope: Activation slope of graded release

    \c event code is:
    \code
    $(addToInSyn, $(g)* tanh(($(V_pre)-($(Epre)))*DT*2/$(Vslope)));
    \endcode

    \c event threshold condition code is:

    \code
    $(V_pre) > $(Epre)
    \endcode
    \note The pre-synaptic variables are referenced with the suffix `_pre` in synapse related code
    such as an the event threshold test. Users can also access post-synaptic neuron variables using the suffix `_post`.*/
class StaticGraded : public Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticGraded, 2, 1, 0, 0);

    SET_PARAM_NAMES({"Epre", "Vslope"});
    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_EVENT_CODE("$(addToInSyn, max(0.0, $(g) * tanh(($(V_pre) - $(Epre)) / $(Vslope))* DT));\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > $(Epre)");
};

//----------------------------------------------------------------------------
// PiecewiseSTDP
//----------------------------------------------------------------------------
//! This is a simple STDP rule including a time delay for the finite transmission speed of the synapse.
/*! The STDP window is defined as a piecewise function:
    \image html LEARN1SYNAPSE_explain_html.png
    \image latex LEARN1SYNAPSE_explain.png width=10cm

    The STDP curve is applied to the raw synaptic conductance `gRaw`, which is then filtered through the sugmoidal filter displayed above to obtain the value of `g`.

    \note
    The STDP curve implies that unpaired pre- and post-synaptic spikes incur a negative increment in `gRaw` (and hence in `g`).

    \note
    The time of the last spike in each neuron, "sTXX", where XX is the name of a neuron population is (somewhat arbitrarily) initialised to -10.0 ms. If neurons never spike, these spike times are used.

    \note
    It is the raw synaptic conductance `gRaw` that is subject to the STDP rule. The resulting synaptic conductance is a sigmoid filter of `gRaw`. This implies that `g` is initialised but not `gRaw`, the synapse will revert to the value that corresponds to `gRaw`.

    An example how to use this synapse correctly is given in `map_classol.cc` (MBody1 userproject):
    \code
    for (int i= 0; i < model.neuronN[1]*model.neuronN[3]; i++) {
            if (gKCDN[i] < 2.0*SCALAR_MIN){
                cnt++;
                fprintf(stdout, "Too low conductance value %e detected and set to 2*SCALAR_MIN= %e, at index %d \n", gKCDN[i], 2*SCALAR_MIN, i);
                gKCDN[i] = 2.0*SCALAR_MIN; //to avoid log(0)/0 below
            }
            scalar tmp = gKCDN[i] / myKCDN_p[5]*2.0 ;
            gRawKCDN[i]=  0.5 * log( tmp / (2.0 - tmp)) /myKCDN_p[7] + myKCDN_p[6];
    }
    cerr << "Total number of low value corrections: " << cnt << endl;
    \endcode

    \note
    One cannot set values of `g` fully to `0`, as this leads to `gRaw`= -infinity and this is not support. I.e., 'g' needs to be some nominal value > 0 (but can be extremely small so that it acts like it's 0).

    <!--
    If no spikes at t: \f$ g_{raw}(t+dt) = g_0 + (g_{raw}(t)-g_0)*\exp(-dt/\tau_{decay}) \f$
    If pre or postsynaptic spike at t: \f$ g_{raw}(t+dt) = g_0 + (g_{raw}(t)-g_0)*\exp(-dt/\tau_{decay})
    +A(t_{post}-t_{pre}-\tau_{decay}) \f$
    -->

    The model has 2 variables:
    - \c g: conductance of \c scalar type
    - \c gRaw: raw conductance of \c scalar type

    Parameters are (compare to the figure above):
    - \c tLrn: Time scale of learning changes
    - \c tChng: Width of learning window
    - \c tDecay: Time scale of synaptic strength decay
    - \c tPunish10: Time window of suppression in response to 1/0
    - \c tPunish01: Time window of suppression in response to 0/1
    - \c gMax: Maximal conductance achievable
    - \c gMid: Midpoint of sigmoid g filter curve
    - \c gSlope: Slope of sigmoid g filter curve
    - \c tauShift: Shift of learning curve
    - \c gSyn0: Value of syn conductance g decays to */
class PiecewiseSTDP : public Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(PiecewiseSTDP, 10, 2, 0, 0);

    SET_PARAM_NAMES({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
        "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
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
        {"lim0", [](const std::vector<double> &pars, double){ return (1/pars[4] + 1/pars[1]) * pars[0] / (2/pars[1]); }},
        {"lim1", [](const std::vector<double> &pars, double){ return  -((1/pars[3] + 1/pars[1]) * pars[0] / (2/pars[1])); }},
        {"slope0", [](const std::vector<double> &pars, double){ return  -2*pars[5]/(pars[1]*pars[0]); }},
        {"slope1", [](const std::vector<double> &pars, double){ return  2*pars[5]/(pars[1]*pars[0]); }},
        {"off0", [](const std::vector<double> &pars, double){ return  pars[5] / pars[4]; }},
        {"off1", [](const std::vector<double> &pars, double){ return  pars[5] / pars[1]; }},
        {"off2", [](const std::vector<double> &pars, double){ return  pars[5] / pars[3]; }}});

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
} // WeightUpdateModels
