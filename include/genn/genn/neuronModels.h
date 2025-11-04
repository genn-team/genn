#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// Standard C includes
#include <cmath>

// GeNN includes
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const override{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string getThresholdConditionCode() const override{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string getResetCode() const override{ return RESET_CODE; }
#define SET_ADDITIONAL_INPUT_VARS(...) virtual ParamValVec getAdditionalInputVars() const override{ return __VA_ARGS__; }
#define SET_NEEDS_AUTO_REFRACTORY(AUTO_REFRACTORY_REQUIRED) virtual bool isAutoRefractoryRequired() const override{ return AUTO_REFRACTORY_REQUIRED; }

//----------------------------------------------------------------------------
// GeNN::NeuronModels::Base
//----------------------------------------------------------------------------
namespace GeNN::NeuronModels
{
//! Base class for all neuron models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets model variables
    virtual std::vector<Var> getVars() const{ return {}; }

    //! Gets the code that defines the execution of one timestep of integration of the neuron model.
    /*! The code will refer to NN for the value of the variable with name "NN".
        It needs to refer to the predefined variable "ISYN", i.e. contain ISYN, if it is to receive input. */
    virtual std::string getSimCode() const{ return ""; }

    //! Gets code which defines the condition for a true spike in the described neuron model.
    /*! This evaluates to a bool (e.g. "V > 20"). */
    virtual std::string getThresholdConditionCode() const{ return ""; }

    //! Gets code that defines the reset action taken after a spike occurred. This can be empty
    virtual std::string getResetCode() const{ return ""; }

    //! Gets names, types (as strings) and initial values of local variables into which
    //! the 'apply input code' of (potentially) multiple postsynaptic input models can apply input
    virtual Models::Base::ParamValVec getAdditionalInputVars() const{ return {}; }

    //! Does this model require auto-refractory logic?
    virtual bool isAutoRefractoryRequired() const{ return false; }

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
                  const std::string &description) const;
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::RulkovMap
//----------------------------------------------------------------------------
//! Rulkov Map neuron
/*! The RulkovMap type is a map based neuron model based on  \cite Rulkov2002 but in
    the 1-dimensional map form used in \cite Nowotny2005:
    \f{eqnarray*}{
    V(t+\Delta t) &=& \left\{ \begin{array}{ll}
    V_{\rm spike} \Big(\frac{\alpha V_{\rm spike}}{V_{\rm spike}-V(t) \beta I_{\rm syn}} + y \Big) & V(t) \leq 0 \\
    V_{\rm spike} \big(\alpha+y\big) & V(t) \leq V_{\rm spike} \big(\alpha + y\big) \; \& \; V(t-\Delta t) \leq 0 \\
    -V_{\rm spike} & {\rm otherwise}
    \end{array}
    \right.
    \f}
    \note
    The `RulkovMap` type only works as intended for the single time step size of `DT`= 0.5.

    The `RulkovMap` type has 2 variables:

    - \c V - the membrane potential
    - \c preV - the membrane potential at the previous time step

    and it has 4 parameters:

    - \c Vspike - determines the amplitude of spikes, typically -60mV
    - \c alpha - determines the shape of the iteration function, typically \f$\alpha \f$= 3
    - \c y - "shift / excitation" parameter, also determines the iteration function,originally, y= -2.468
    - \c beta - roughly speaking equivalent to the input resistance, i.e. it regulates the scale of the input into the neuron, typically \f$\beta\f$= 2.64 \f${\rm M}\Omega\f$.

    \note
    The initial values array for the `RulkovMap` type needs two entries for `V` and `preV` and the
    parameter array needs four entries for `Vspike`, `alpha`, `y` and `beta`,  *in that order*.*/
class RulkovMap : public Base
{
public:
    DECLARE_SNIPPET(NeuronModels::RulkovMap);

    SET_SIM_CODE(
        "if (V <= 0) {\n"
        "   preV= V;\n"
        "   V= ip0/((Vspike) - V - (beta)*Isyn) +(ip1);\n"
        "}\n"
        "else {"
        "   if ((V < ip2) && (preV <= 0)) {\n"
        "       preV= V;\n"
        "       V= ip2;\n"
        "   }\n"
        "   else {\n"
        "       preV= V;\n"
        "       V= -(Vspike);\n"
        "   }\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("V >= ip2");

    SET_PARAMS({"Vspike", "alpha", "y", "beta"});
    SET_VARS({{"V","scalar"}, {"preV", "scalar"}});

    SET_DERIVED_PARAMS({
        {"ip0", [](const ParamValues &pars, double){ return pars.at("Vspike").cast<double>() * pars.at("Vspike").cast<double>() * pars.at("alpha").cast<double>(); }},
        {"ip1", [](const ParamValues &pars, double){ return pars.at("Vspike").cast<double>() * pars.at("y").cast<double>(); }},
        {"ip2", [](const ParamValues &pars, double){ return (pars.at("Vspike").cast<double>() * pars.at("alpha").cast<double>()) + (pars.at("Vspike").cast<double>() * pars.at("y").cast<double>()); }}});
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::Izhikevich
//----------------------------------------------------------------------------
//! Izhikevich neuron with fixed parameters  \cite Izhikevich2003.
/*! It is usually described as
    \f{eqnarray*}
    \frac{dV}{dt} &=& 0.04 V^2 + 5 V + 140 - U + I, \\
    \frac{dU}{dt} &=& a (bV-U),
    \f}
    I is the input current and the voltage V is reset to parameter c and U incremented by parameter d, whenever V >= 30 mV. This is paired with a particular integration procedure of two 0.5 ms Euler time steps for the V equation followed by one 1 ms time step of the U equation. Because of its popularity we provide this model in this form here event though due to the details of the usual implementation it is strictly speaking inconsistent with the displayed equations.

    Variables are:

    - \c V - Membrane potential
    - \c U - Membrane recovery variable

    Parameters are:

    - \c a - time scale of U
    - \c b - sensitivity of U
    - \c c - after-spike reset value of V
    - \c d - after-spike reset value of U*/
class Izhikevich : public Base
{
public:
    DECLARE_SNIPPET(NeuronModels::Izhikevich);

    SET_SIM_CODE(
        "if (V >= 30.0){\n"
        "   V=c;\n"
        "   U+=d;\n"
        "} \n"
        "V+=0.5*(0.04*V*V+5.0*V+140.0-U+Isyn)*dt; //at two times for numerical stability\n"
        "V+=0.5*(0.04*V*V+5.0*V+140.0-U+Isyn)*dt;\n"
        "U+=a*(b*V-U)*dt;\n"
        "if (V > 30.0){   //keep this to not confuse users with unrealistiv voltage values \n"
        "  V=30.0; \n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("V >= 29.99");

    SET_PARAMS({"a", "b", "c", "d"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::IzhikevichVariable
//----------------------------------------------------------------------------
//! Izhikevich neuron with variable parameters  \cite Izhikevich2003.
/*! This is the same model as NeuronModels::Izhikevich but parameters are defined as
    "variables" in order to allow users to provide individual values for each
    individual neuron instead of fixed values for all neurons across the population.

    Accordingly, the model has the variables:

    - \c V - Membrane potential
    - \c U - Membrane recovery variable
    - \c a - time scale of U
    - \c b - sensitivity of U
    - \c c - after-spike reset value of V
    - \c d - after-spike reset value of U

    and no parameters.*/
class IzhikevichVariable : public Izhikevich
{
public:
    DECLARE_SNIPPET(NeuronModels::IzhikevichVariable);

    SET_PARAMS({});
    SET_VARS({{"V","scalar"}, {"U", "scalar"},
              {"a", "scalar", VarAccess::READ_ONLY}, {"b", "scalar", VarAccess::READ_ONLY},
              {"c", "scalar", VarAccess::READ_ONLY}, {"d", "scalar", VarAccess::READ_ONLY}});
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::LIF
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron with a refractory timer.
/*! It is usually described as
    \f{eqnarray*}
    \tau_m \frac{dV}{dt} &=& V_{\text{rest}}-V + R I, \\
    \f}
    I is the input current and the voltage V is reset to parameter Vreset and RefracTime is set to TauRefrac whenever V >= Vthresh mV and RefracTime <= 0.0. 
    This model

    Variables are:

    - \c V - Membrane potential
    - \c RefracTime - Membrane recovery variable

    Parameters are:

    - \c C - Membrane capacitance
    - \c TauM - Membrane time constant [ms]
    - \c Vrest - Resting membrane potential [mV]
    - \c Vreset - Reset voltage [mV]
    - \c Vthresh - after-spike reset value of U
    - \c Ioffset" - Spiking threshold [mV]
    - \c TauRefrac - after-spike reset value of U*/
class LIF : public Base
{
public:
    DECLARE_SNIPPET(LIF);

    SET_SIM_CODE(
        "if (RefracTime <= 0.0) {\n"
        "  scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;\n"
        "  V = alpha - (ExpTC * (alpha - V));\n"
        "}\n"
        "else {\n"
        "  RefracTime -= dt;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= Vthresh");

    SET_RESET_CODE(
        "V = Vreset;\n"
        "RefracTime = TauRefrac;\n");

    SET_PARAMS({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("TauM").cast<double>() / pars.at("C").cast<double>(); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::SpikeSourceArray
//----------------------------------------------------------------------------
//! Spike source array
/*! A neuron which reads spike times from a global spikes array.
    It has 2 variables:

    - \c startSpike - Index of the next spike in the global array
    - \c endSpike   - Index of the spike next to the last in the globel array

    and 1 extra global parameter:

    - \c spikeTimes - Array with all spike times

  */
class SpikeSourceArray : public Base
{
public:
    DECLARE_SNIPPET(NeuronModels::SpikeSourceArray);
    SET_SIM_CODE("")
    SET_THRESHOLD_CONDITION_CODE(
        "startSpike != endSpike && "
        "t >= spikeTimes[startSpike]" );
    SET_RESET_CODE( "startSpike++;\n" );
    SET_VARS({{"startSpike", "unsigned int"}, {"endSpike", "unsigned int", VarAccess::READ_ONLY_DUPLICATE}});
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeTimes", "scalar*"}} );
    SET_NEEDS_AUTO_REFRACTORY(false);
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::Poisson
//----------------------------------------------------------------------------
//! Poisson neurons
/*! This neuron model emits spikes according to the Poisson distribution with a mean firing
  rate as determined by its single parameter. 
  It has 1 state variable:

    - \c timeStepToSpike - Number of timesteps to next spike

    and 1 parameter:

    - \c rate - Mean firing rate (Hz)

    \note Internally this samples from the exponential distribution using
    the C++ 11 \<random\> library on the CPU and by transforming the
    uniform distribution, generated using cuRAND, with a natural log on the GPU.
    \note If you are connecting Poisson neurons one-to-one to another neuron population, 
    it is more efficient to add a CurrentSourceModels::PoissonExp instead. */
class Poisson : public Base
{
public:
    DECLARE_SNIPPET(NeuronModels::Poisson);

    SET_SIM_CODE(
        "if(timeStepToSpike <= 0.0f) {\n"
        "    timeStepToSpike += isi * gennrand_exponential();\n"
        "}\n"
        "timeStepToSpike -= 1.0;\n"
    );

    SET_THRESHOLD_CONDITION_CODE("timeStepToSpike <= 0.0");

    SET_PARAMS({"rate"});
    SET_VARS({{"timeStepToSpike", "scalar"}});
    SET_DERIVED_PARAMS({{"isi", [](const ParamValues &pars, double dt){ return 1000.0 / (pars.at("rate").cast<double>() * dt); }}});
    SET_NEEDS_AUTO_REFRACTORY(false);
};

//----------------------------------------------------------------------------
// GeNN::NeuronModels::TraubMiles
//----------------------------------------------------------------------------
//! Hodgkin-Huxley neurons with Traub & Miles algorithm.
/*! This conductance based model has been taken from  \cite Traub1991 and can be described by the equations:
    \f{eqnarray*}{
    C \frac{d V}{dt}  &=& -I_{{\rm Na}} -I_K-I_{{\rm leak}}-I_M-I_{i,DC}-I_{i,{\rm syn}}-I_i, \\
    I_{{\rm Na}}(t) &=& g_{{\rm Na}} m_i(t)^3 h_i(t)(V_i(t)-E_{{\rm Na}}) \\
    I_{{\rm K}}(t) &=& g_{{\rm K}} n_i(t)^4(V_i(t)-E_{{\rm K}})  \\
    \frac{dy(t)}{dt} &=& \alpha_y (V(t))(1-y(t))-\beta_y(V(t)) y(t), \f}
    where \f$y_i= m, h, n\f$, and
    \f{eqnarray*}{
    \alpha_n&=& 0.032(-50-V)/\big(\exp((-50-V)/5)-1\big)  \\
    \beta_n &=& 0.5\exp((-55-V)/40)  \\
    \alpha_m &=& 0.32(-52-V)/\big(\exp((-52-V)/4)-1\big)  \\
    \beta_m &=& 0.28(25+V)/\big(\exp((25+V)/5)-1\big)  \\
    \alpha_h &=& 0.128\exp((-48-V)/18)  \\
    \beta_h &=& 4/\big(\exp((-25-V)/5)+1\big).
    \f}
    and typical parameters are \f$C=0.143\f$ nF, \f$g_{{\rm leak}}= 0.02672\f$
    \f$\mu\f$S, \f$E_{{\rm leak}}= -63.563\f$ mV, \f$g_{{\rm Na}}=7.15\f$ \f$\mu\f$S,
    \f$E_{{\rm Na}}= 50\f$ mV, \f$g_{{\rm {\rm K}}}=1.43\f$ \f$\mu\f$S,
    \f$E_{{\rm K}}= -95\f$ mV.

    It has 4 variables:

    - \c V - membrane potential E
    - \c m - probability for Na channel activation m
    - \c h - probability for not Na channel blocking h
    - \c n - probability for K channel activation n

    and 7 parameters:

    - \c gNa - Na conductance in 1/(mOhms * cm^2)
    - \c ENa - Na equi potential in mV
    - \c gK - K conductance in 1/(mOhms * cm^2)
    - \c EK - K equi potential in mV
    - \c gl - Leak conductance in 1/(mOhms * cm^2)
    - \c El - Leak equi potential in mV
    - \c C - Membrane capacity density in muF/cm^2

    \note
    Internally, the ordinary differential equations defining the model are integrated with a
    linear Euler algorithm and GeNN integrates 25 internal time steps for each neuron for each
    network time step. I.e., if the network is simulated at `DT= 0.1` ms, then the neurons are
    integrated with a linear Euler algorithm with `lDT= 0.004` ms.
    This variant uses IF statements to check for a value at which a singularity would be hit.
    If so, value calculated by L'Hospital rule is used.*/
class TraubMiles : public Base
{
public:
    DECLARE_SNIPPET(NeuronModels::TraubMiles);

    SET_SIM_CODE(
        "const scalar mdt= dt/25.0;\n"
        "for (unsigned int mt=0; mt < 25; mt++) {\n"
        "   const scalar Imem= -(m*m*m*h*gNa*(V-(ENa))+\n"
        "       n*n*n*n*gK*(V-(EK))+\n"
        "       gl*(V-(El))-Isyn);\n"
        "   scalar a;\n"
        "   if (V == -52.0) {\n"
        "       a= 1.28;\n"
        "   }\n"
        "   else {\n"
        "       a= 0.32*(-52.0-V)/(exp((-52.0-V)/4.0)-1.0);\n"
        "   }\n"
        "   scalar b;\n"
        "   if (V == -25.0) {\n"
        "       b= 1.4;\n"
        "   }\n"
        "   else {\n"
        "       b= 0.28*(V+25.0)/(exp((V+25.0)/5.0)-1.0);\n"
        "   }\n"
        "   m+= (a*(1.0-m)-b*m)*mdt;\n"
        "   a= 0.128*exp((-48.0-V)/18.0);\n"
        "   b= 4.0 / (exp((-25.0-V)/5.0)+1.0);\n"
        "   h+= (a*(1.0-h)-b*h)*mdt;\n"
        "   if (V == -50.0) {\n"
        "       a= 0.16;\n"
        "   }\n"
        "   else {\n"
        "       a= 0.032*(-50.0-V)/(exp((-50.0-V)/5.0)-1.0);\n"
        "   }\n"
        "   b= 0.5*exp((-55.0-V)/40.0);\n"
        "   n+= (a*(1.0-n)-b*n)*mdt;\n"
        "   V+= Imem/C*mdt;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("V >= 0.0");

    SET_PARAMS({"gNa", "ENa", "gK", "EK", "gl", "El", "C"});
    SET_VARS({{"V", "scalar"}, {"m", "scalar"}, {"h", "scalar"}, {"n", "scalar"}});
};
} // GeNN::NeuronModels
