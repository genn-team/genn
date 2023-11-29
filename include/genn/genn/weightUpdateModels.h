#pragma once

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const override{ return SIM_CODE; }
#define SET_EVENT_CODE(EVENT_CODE) virtual std::string getEventCode() const override{ return EVENT_CODE; }
#define SET_LEARN_POST_CODE(LEARN_POST_CODE) virtual std::string getLearnPostCode() const override{ return LEARN_POST_CODE; }
#define SET_SYNAPSE_DYNAMICS_CODE(SYNAPSE_DYNAMICS_CODE) virtual std::string getSynapseDynamicsCode() const override{ return SYNAPSE_DYNAMICS_CODE; }
#define SET_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) virtual std::string getEventThresholdConditionCode() const override{ return EVENT_THRESHOLD_CONDITION_CODE; }

#define SET_PRE_SPIKE_CODE(PRE_SPIKE_CODE) virtual std::string getPreSpikeCode() const override{ return PRE_SPIKE_CODE; }
#define SET_POST_SPIKE_CODE(POST_SPIKE_CODE) virtual std::string getPostSpikeCode() const override{ return POST_SPIKE_CODE; }
#define SET_PRE_DYNAMICS_CODE(PRE_DYNAMICS_CODE) virtual std::string getPreDynamicsCode() const override{ return PRE_DYNAMICS_CODE; }
#define SET_POST_DYNAMICS_CODE(POST_DYNAMICS_CODE) virtual std::string getPostDynamicsCode() const override{ return POST_DYNAMICS_CODE; }

#define SET_PRE_VARS(...) virtual std::vector<Var> getPreVars() const override{ return __VA_ARGS__; }
#define SET_POST_VARS(...) virtual std::vector<Var> getPostVars() const override{ return __VA_ARGS__; }

#define SET_PRE_NEURON_VAR_REFS(...) virtual VarRefVec getPreNeuronVarRefs() const override{ return __VA_ARGS__; }
#define SET_POST_NEURON_VAR_REFS(...) virtual VarRefVec getPostNeuronVarRefs() const override{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::Base
//----------------------------------------------------------------------------
namespace GeNN::WeightUpdateModels
{
//! Base class for all weight update models
class GENN_EXPORT Base : public Models::Base
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

    //! Gets code to be run after presynaptic neuron update
    /*! This is typically for the code to update presynaptic variables. Postsynaptic
        and synapse variables are not accesible from within this code */
    virtual std::string getPreDynamicsCode() const{ return ""; }

    //! Gets code to be run after postsynaptic neuron update
    /*! This is typically for the code to update postsynaptic variables. Presynaptic
        and synapse variables are not accesible from within this code */
    virtual std::string getPostDynamicsCode() const{ return ""; }

    //! Gets model variables
    virtual std::vector<Var> getVars() const{ return {}; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses coming from the same presynaptic neuron
    virtual std::vector<Var> getPreVars() const{ return {}; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses going to the same postsynaptic neuron
    virtual std::vector<Var> getPostVars() const{ return {}; }

    //! Gets names and types of variable references to presynaptic neuron
    virtual VarRefVec getPreNeuronVarRefs() const{ return {}; }

    //! Gets names and types of variable references to postsynaptic neuron
    virtual VarRefVec getPostNeuronVarRefs() const{ return {}; }

    //------------------------------------------------------------------------
    // Public methods
    //------------------------------------------------------------------------
    //! Find the named variable
    std::optional<Var> getVar(const std::string &varName) const
    {
        return getNamed(varName, getVars());
    }

    //! Find the named presynaptic variable
    std::optional<Var> getPreVar(const std::string &varName) const
    {
        return getNamed(varName, getPreVars());
    }

    //! Find the named postsynaptic variable
    std::optional<Var> getPostVar(const std::string &varName) const
    {
        return getNamed(varName, getPostVars());
    }

    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Update hash from presynaptic components of model
    boost::uuids::detail::sha1::digest_type getPreHashDigest() const;

    //! Update hash from postsynaptic components of  model
    boost::uuids::detail::sha1::digest_type getPostHashDigest() const;

    //! Update hash from presynaptic event-triggering components of model
    boost::uuids::detail::sha1::digest_type getPreEventHashDigest() const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &preVarValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &postVarValues,
                  const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                  const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets) const;
};


//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<Base>
{
public:
    Init(const Base *snippet, const std::unordered_map<std::string, Type::NumericValue> &params, 
         const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, 
         const std::unordered_map<std::string, InitVarSnippet::Init> &preVarInitialisers, 
         const std::unordered_map<std::string, InitVarSnippet::Init> &postVarInitialisers,
         const std::unordered_map<std::string, Models::VarReference> &preNeuronVarReferences, 
         const std::unordered_map<std::string, Models::VarReference> &postNeuronVarReferences);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;

    const std::unordered_map<std::string, InitVarSnippet::Init> &getVarInitialisers() const{ return m_VarInitialisers; }
    const std::unordered_map<std::string, InitVarSnippet::Init> &getPreVarInitialisers() const{ return m_PreVarInitialisers; }
    const std::unordered_map<std::string, InitVarSnippet::Init> &getPostVarInitialisers() const{ return m_PostVarInitialisers; }
    const std::unordered_map<std::string, Models::VarReference> &getPreNeuronVarReferences() const{ return m_PreNeuronVarReferences;  }
    const std::unordered_map<std::string, Models::VarReference> &getPostNeuronVarReferences() const{ return m_PostNeuronVarReferences;  }
    
    const std::vector<Transpiler::Token> &getSimCodeTokens() const{ return m_SimCodeTokens; }
    const std::vector<Transpiler::Token> &getEventCodeTokens() const{ return m_EventCodeTokens; }
    const std::vector<Transpiler::Token> &getPostLearnCodeTokens() const{ return m_PostLearnCodeTokens; }
    const std::vector<Transpiler::Token> &getSynapseDynamicsCodeTokens() const{ return m_SynapseDynamicsCodeTokens; }
    const std::vector<Transpiler::Token> &getEventThresholdCodeTokens() const{ return m_EventThresholdCodeTokens; }
    const std::vector<Transpiler::Token> &getPreSpikeCodeTokens() const{ return m_PreSpikeCodeTokens; }
    const std::vector<Transpiler::Token> &getPostSpikeCodeTokens() const{ return m_PostSpikeCodeTokens; }
    const std::vector<Transpiler::Token> &getPreDynamicsCodeTokens() const{ return m_PreDynamicsCodeTokens; }
    const std::vector<Transpiler::Token> &getPostDynamicsCodeTokens() const{ return m_PostDynamicsCodeTokens; }

    void finalise(double dt);
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Transpiler::Token> m_SimCodeTokens;
    std::vector<Transpiler::Token> m_EventCodeTokens;
    std::vector<Transpiler::Token> m_PostLearnCodeTokens;
    std::vector<Transpiler::Token> m_SynapseDynamicsCodeTokens;
    std::vector<Transpiler::Token> m_EventThresholdCodeTokens;
    std::vector<Transpiler::Token> m_PreSpikeCodeTokens;
    std::vector<Transpiler::Token> m_PostSpikeCodeTokens;
    std::vector<Transpiler::Token> m_PreDynamicsCodeTokens;
    std::vector<Transpiler::Token> m_PostDynamicsCodeTokens;

    std::unordered_map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    std::unordered_map<std::string, InitVarSnippet::Init> m_PreVarInitialisers;
    std::unordered_map<std::string, InitVarSnippet::Init> m_PostVarInitialisers;
    std::unordered_map<std::string, Models::VarReference> m_PreNeuronVarReferences;
    std::unordered_map<std::string, Models::VarReference> m_PostNeuronVarReferences;
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulse
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
    DECLARE_SNIPPET(StaticPulse);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE("addToPost(g);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulseConstantWeight
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse.
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 1 parameter:
    - g - conductance
    and no other variables.

    \c sim code is:

    \code
    "addToPost(g);"
    \endcode*/
class StaticPulseConstantWeight : public Base
{
public:
    DECLARE_SNIPPET(StaticPulseConstantWeight);

    SET_PARAMS({"g"});

    SET_SIM_CODE("addToPost(g);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulseDendriticDelay
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
    DECLARE_SNIPPET(StaticPulseDendriticDelay);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}, {"d", "uint8_t", VarAccess::READ_ONLY}});

    SET_SIM_CODE("addToPostDelay(g, d);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticGraded
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
    DECLARE_SNIPPET(StaticGraded);

    SET_PARAMS({"Epre", "Vslope"});
    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_EVENT_CODE("addToPost(fmax(0.0, g * tanh((V_pre - Epre) / Vslope) * DT));\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("V_pre > Epre");
};

//----------------------------------------------------------------------------
// GeNN::PiecewiseSTDP
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
    DECLARE_SNIPPET(PiecewiseSTDP);

    SET_PARAMS({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
                     "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_SIM_CODE(
        "addToPost(g);\n"
        "scalar dt = sT_post - t - tauShift; \n"
        "scalar dg = 0;\n"
        "if (dt > lim0)  \n"
        "    dg = -off0 ; \n"
        "else if (dt > 0)  \n"
        "    dg = slope0 * dt + off1; \n"
        "else if (dt > lim1)  \n"
        "    dg = slope1 * dt + ($(off1)); \n"
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
        {"lim0", [](const ParamValues &pars, double){ return (1/pars.at("tPunish01").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>()); }},
        {"lim1", [](const ParamValues &pars, double){ return  -((1/pars.at("tPunish10").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>())); }},
        {"slope0", [](const ParamValues &pars, double){ return  -2*pars.at("gMax").cast<double>() /(pars.at("tChng").cast<double>()*pars.at("tLrn").cast<double>()); }},
        {"slope1", [](const ParamValues &pars, double){ return  2*pars.at("gMax").cast<double>() / (pars.at("tChng").cast<double>() * pars.at("tLrn").cast<double>()); }},
        {"off0", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish01").cast<double>(); }},
        {"off1", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tChng").cast<double>(); }},
        {"off2", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish10").cast<double>(); }}});
};
}   //namespace GeNN::WeightUpdateModels
