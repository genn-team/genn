#pragma once

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_PRE_SPIKE_SYN_CODE(CODE) virtual std::string getPreSpikeSynCode() const override{ return CODE; }
#define SET_PRE_EVENT_SYN_CODE(CODE) virtual std::string getPreEventSynCode() const override{ return CODE; }
#define SET_POST_EVENT_SYN_CODE(CODE) virtual std::string getPostEventSynCode() const override{ return CODE; }
#define SET_POST_SPIKE_SYN_CODE(CODE) virtual std::string getPostSpikeSynCode() const override{ return CODE; }
#define SET_SYNAPSE_DYNAMICS_CODE(SYNAPSE_DYNAMICS_CODE) virtual std::string getSynapseDynamicsCode() const override{ return SYNAPSE_DYNAMICS_CODE; }
#define SET_PRE_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) virtual std::string getPreEventThresholdConditionCode() const override{ return EVENT_THRESHOLD_CONDITION_CODE; }
#define SET_POST_EVENT_THRESHOLD_CONDITION_CODE(EVENT_THRESHOLD_CONDITION_CODE) virtual std::string getPostEventThresholdConditionCode() const override{ return EVENT_THRESHOLD_CONDITION_CODE; }

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
    //! Gets code run when a presynaptic spike is received at the synapse
    virtual std::string getPreSpikeSynCode() const{ return ""; }

    //! Gets code run when a presynaptic spike-like event is received at the synapse
    /*! Presynaptic events are triggered for all presynaptic neurons where 
        the presynaptic event threshold condition is met*/
    virtual std::string getPreEventSynCode() const{ return ""; }

    //! Gets code run when a postsynaptic spike-like event is received at the synapse
    /*! Postsynaptic events are triggered for all postsynaptic neurons where 
        the postsynaptic event threshold condition is met*/
    virtual std::string getPostEventSynCode() const{ return ""; }

    //! Gets code run when a postsynaptic spike is received at the synapse
    /*! For examples when modelling STDP, this is where the effect of postsynaptic
        spikes which occur _after_ presynaptic spikes are applied. */
    virtual std::string getPostSpikeSynCode() const{ return ""; }

    //! Gets code for synapse dynamics which are independent of spike detection
    virtual std::string getSynapseDynamicsCode() const{ return ""; }

    //! Gets codes to test for presynaptic events
    virtual std::string getPreEventThresholdConditionCode() const{ return ""; }

    //! Gets codes to test for postsynaptic events
    virtual std::string getPostEventThresholdConditionCode() const{ return ""; }

    //! Gets code to be run once per spiking presynaptic neuron before sim code is run on synapses
    /*! This is typically for the code to update presynaptic variables. Postsynaptic
        and synapse variables are not accesible from within this code */
    virtual std::string getPreSpikeCode() const{ return ""; }

    //! Gets code to be run once per spiking postsynaptic neuron before learn post code is run on synapses
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

    //! Update hash from postsynaptic event-triggering components of model
    boost::uuids::detail::sha1::digest_type getPostEventHashDigest() const;
    
    //! Validate names of parameters etc
    void validate(const std::map<std::string, Type::NumericValue> &paramValues, 
                  const std::map<std::string, InitVarSnippet::Init> &varValues,
                  const std::map<std::string, InitVarSnippet::Init> &preVarValues,
                  const std::map<std::string, InitVarSnippet::Init> &postVarValues,
                  const std::map<std::string, Models::VarReference> &preVarRefTargets,
                  const std::map<std::string, Models::VarReference> &postVarRefTargets) const;
};


//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
class GENN_EXPORT Init : public Snippet::Init<Base>
{
public:
    Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params, 
         const std::map<std::string, InitVarSnippet::Init> &varInitialisers, 
         const std::map<std::string, InitVarSnippet::Init> &preVarInitialisers, 
         const std::map<std::string, InitVarSnippet::Init> &postVarInitialisers,
         const std::map<std::string, Models::VarReference> &preNeuronVarReferences, 
         const std::map<std::string, Models::VarReference> &postNeuronVarReferences);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool isRNGRequired() const;

    const auto &getVarInitialisers() const{ return m_VarInitialisers; }
    const auto &getPreVarInitialisers() const{ return m_PreVarInitialisers; }
    const auto &getPostVarInitialisers() const{ return m_PostVarInitialisers; }
    const auto &getPreNeuronVarReferences() const{ return m_PreNeuronVarReferences;  }
    const auto &getPostNeuronVarReferences() const{ return m_PostNeuronVarReferences;  }
    
    const auto &getPreSpikeSynCodeTokens() const{ return m_PreSpikeSynCodeTokens; }
    const auto &getPreEventSynCodeTokens() const{ return m_PreEventSynCodeTokens; }
    const auto &getPostEventSynCodeTokens() const{ return m_PostEventSynCodeTokens; }
    const auto &getPostSpikeSynCodeTokens() const{ return m_PostSpikeSynCodeTokens; }
    const auto &getSynapseDynamicsCodeTokens() const{ return m_SynapseDynamicsCodeTokens; }
    const auto &getPreEventThresholdCodeTokens() const{ return m_PreEventThresholdCodeTokens; }
    const auto &getPostEventThresholdCodeTokens() const{ return m_PostEventThresholdCodeTokens; }
    const auto &getPreSpikeCodeTokens() const{ return m_PreSpikeCodeTokens; }
    const auto &getPostSpikeCodeTokens() const{ return m_PostSpikeCodeTokens; }
    const auto &getPreDynamicsCodeTokens() const{ return m_PreDynamicsCodeTokens; }
    const auto &getPostDynamicsCodeTokens() const{ return m_PostDynamicsCodeTokens; }

    void finalise(double dt);
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<Transpiler::Token> m_PreSpikeSynCodeTokens;
    std::vector<Transpiler::Token> m_PreEventSynCodeTokens;
    std::vector<Transpiler::Token> m_PostEventSynCodeTokens;
    std::vector<Transpiler::Token> m_PostSpikeSynCodeTokens;
    std::vector<Transpiler::Token> m_SynapseDynamicsCodeTokens;
    std::vector<Transpiler::Token> m_PreEventThresholdCodeTokens;
    std::vector<Transpiler::Token> m_PostEventThresholdCodeTokens;
    std::vector<Transpiler::Token> m_PreSpikeCodeTokens;
    std::vector<Transpiler::Token> m_PostSpikeCodeTokens;
    std::vector<Transpiler::Token> m_PreDynamicsCodeTokens;
    std::vector<Transpiler::Token> m_PostDynamicsCodeTokens;

    std::map<std::string, InitVarSnippet::Init> m_VarInitialisers;
    std::map<std::string, InitVarSnippet::Init> m_PreVarInitialisers;
    std::map<std::string, InitVarSnippet::Init> m_PostVarInitialisers;
    std::map<std::string, Models::VarReference> m_PreNeuronVarReferences;
    std::map<std::string, Models::VarReference> m_PostNeuronVarReferences;
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulse
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse with heterogeneous weight.
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 1 variable:

    - \c g - synaptic weight

    and no other parameters.*/
class StaticPulse : public Base
{
public:
    DECLARE_SNIPPET(StaticPulse);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulseConstantWeight
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse with homogeneous weight.
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 1 parameter:

    - \c g - synaptic weight

    and no other variables.*/
class StaticPulseConstantWeight : public Base
{
public:
    DECLARE_SNIPPET(StaticPulseConstantWeight);

    SET_PARAMS({"g"});

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticPulseDendriticDelay
//----------------------------------------------------------------------------
//! Pulse-coupled, static synapse with heterogenous weight and dendritic delays
/*! No learning rule is applied to the synapse and for each pre-synaptic spikes,
    the synaptic conductances are simply added to the postsynaptic input variable.
    The model has 2 variables:

    - \c g - synaptic weight
    - \c d - dendritic delay in timesteps

    and no other parameters.*/
class StaticPulseDendriticDelay : public Base
{
public:
    DECLARE_SNIPPET(StaticPulseDendriticDelay);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}, {"d", "uint8_t", VarAccess::READ_ONLY}});

    SET_PRE_SPIKE_SYN_CODE("addToPostDelay(g, d);\n");
};

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::StaticGraded
//----------------------------------------------------------------------------
//! Graded-potential, static synapse
/*! In a graded synapse, the conductance is updated gradually with the rule:
    \f[ gSyn= g * tanh((V - E_{pre}) / V_{slope} \f]
    whenever the membrane potential \f$V\f$ is larger than the threshold \f$E_{pre}\f$.
    The model has 1 variable:

    - \c g: synaptic weight
    
    The model also has 1 presynaptic neuron variable reference:

    - \c V: Presynaptic membrane potential

    The parameters are:

    - \c Epre: Presynaptic threshold potential
    - \c Vslope: Activation slope of graded release*/
class StaticGraded : public Base
{
public:
    DECLARE_SNIPPET(StaticGraded);

    SET_PARAMS({"Epre", "Vslope"});
    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    SET_PRE_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_PRE_EVENT_THRESHOLD_CONDITION_CODE("V > Epre");
    SET_PRE_EVENT_SYN_CODE("addToPost(fmax(0.0, g * tanh((V_pre - Epre) / Vslope) * dt));\n");
};

//----------------------------------------------------------------------------
// GeNN::AdditiveSTDP
//----------------------------------------------------------------------------
//! Simply asymmetrical STDP rule.
/*! This rule makes purely additive weight updates within hard bounds and uses nearest-neighbour spike pairing and the following time-dependence:
    \f[
        \Delta w_{ij} & = \
            \begin{cases}
                A_{+}\exp\left(-\frac{\Delta t}{\tau_{+}}\right) & if\, \Delta t>0\
                A_{-}\exp\left(\frac{\Delta t}{\tau_{-}}\right) & if\, \Delta t\leq0
            \end{cases}  
    \f]
The model has 1 variable:

    - \c g - conductance of scalar type

    and 6 parameters:

    - \c tauPlus - Potentiation time constant (ms)
    - \c tauMinus - Depression time constant (ms)
    - \c Aplus - Rate of potentiation
    - \c Aminus - Rate of depression
    - \c Wmin - Minimum weight
    - \c Wmax - Maximum weight*/
class STDP : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDP);

    SET_PARAMS({"tauPlus", "tauMinus", "Aplus", "Aminus", "Wmin", "Wmax"});

    SET_VARS({{"g", "scalar"}});

    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "scalar dt = t - st_post; \n"
        "if (dt > 0) {\n"
        "    scalar timing = exp(-dt / tauMinus);\n"
        "    scalar newWeight = g - (Aminus * timing);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_POST_SPIKE_SYN_CODE(
        "scalar dt = t - st_pre;\n"
        "if (dt > 0) {\n"
        "    scalar timing = exp(-dt / tauPlus);\n"
        "    scalar newWeight = g + (Aplus * timing);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
};
}   //namespace GeNN::WeightUpdateModels
