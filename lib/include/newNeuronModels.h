#pragma once

// Standard includes
#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// GeNN includes
#include "codeGenUtils.h"
#include "neuronModels.h"
#include "newModels.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_SIM_CODE(SIM_CODE) virtual std::string getSimCode() const{ return SIM_CODE; }
#define SET_THRESHOLD_CONDITION_CODE(THRESHOLD_CONDITION_CODE) virtual std::string getThresholdConditionCode() const{ return THRESHOLD_CONDITION_CODE; }
#define SET_RESET_CODE(RESET_CODE) virtual std::string getResetCode() const{ return RESET_CODE; }
#define SET_SUPPORT_CODE(SUPPORT_CODE) virtual std::string getSupportCode() const{ return SUPPORT_CODE; }
#define SET_EXTRA_GLOBAL_PARAMS(...) virtual StringPairVec getExtraGlobalParams() const{ return __VA_ARGS__; }

//----------------------------------------------------------------------------
// NeuronModels::Base
//----------------------------------------------------------------------------
namespace NeuronModels
{
//! Base class for all neuron models
class Base : public NewModels::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets the code that defines the execution of one timestep of integration of the neuron model.
    /*! The code will refer to $(NN) for the value of the variable with name "NN".
        It needs to refer to the predefined variable "ISYN", i.e. contain $(ISYN), if it is to receive input. */
    virtual std::string getSimCode() const{ return ""; }

    //! Gets code which defines the condition for a true spike in the described neuron model.
    /*! This evaluates to a bool (e.g. "V > 20"). */
    virtual std::string getThresholdConditionCode() const{ return ""; }

    //! Gets code that defines the reset action taken after a spike occurred. This can be empty
    virtual std::string getResetCode() const{ return ""; }

    //! Gets support code to be made available within the neuron kernel/funcion.
    /*! This is intended to contain user defined device functions that are used in the neuron codes.
        Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef;
        functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions. */
    virtual std::string getSupportCode() const{ return ""; }

    //! Gets names and types (as strings) of additional
    //! per-population parameters for the weight update model.
    virtual std::vector<std::pair<std::string, std::string>> getExtraGlobalParams() const{ return {}; }

    //! Is this neuron model the internal Poisson model (which requires a number of special cases)
    //! \private
    virtual bool isPoisson() const{ return false; }
};

//----------------------------------------------------------------------------
// NeuronModels::LegacyWrapper
//----------------------------------------------------------------------------
//! Wrapper around legacy weight update models stored in #nModels array of neuronModel objects.
class LegacyWrapper : public NewModels::LegacyWrapper<Base, neuronModel, nModels>
{
public:
    LegacyWrapper(unsigned int legacyTypeIndex) : NewModels::LegacyWrapper<Base, neuronModel, nModels>(legacyTypeIndex)
    {
    }

    //----------------------------------------------------------------------------
    // Base virtuals
    //----------------------------------------------------------------------------
    //! \copydoc Base::getSimCode
    virtual std::string getSimCode() const;

    //! \copydoc Base::getThresholdConditionCode
    virtual std::string getThresholdConditionCode() const;

    //! \copydoc Base::getResetCode
    virtual std::string getResetCode() const;

    //! \copydoc Base::getSupportCode
    virtual std::string getSupportCode() const;

    //! \copydoc Base::getExtraGlobalParams
    virtual NewModels::Base::StringPairVec getExtraGlobalParams() const;

    //! \copydoc Base::isPoisson
    virtual bool isPoisson() const;
};

//----------------------------------------------------------------------------
// NeuronModels::RulkovMap
//----------------------------------------------------------------------------
//! Rulkov Map neuron
/*! The RulkovMap type is a map based neuron model based on \cite Rulkov2002 but in
    the 1-dimensional map form used in \cite nowotny2005self :
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
    The initial values array for the `RulkovMap` type needs two entries for `V` and `Vpre` and the
    parameter array needs four entries for `Vspike`, `alpha`, `y` and `beta`,  *in that order*.*/
class RulkovMap : public Base
{
public:
    DECLARE_MODEL(NeuronModels::RulkovMap, 4, 2);

    SET_SIM_CODE(
        "if ($(V) <= 0) {\n"
        "   $(preV)= $(V);\n"
        "   $(V)= $(ip0)/(($(Vspike)) - $(V) - ($(beta))*$(Isyn)) +($(ip1));\n"
        "}\n"
        "else {"
        "   if (($(V) < $(ip2)) && ($(preV) <= 0)) {\n"
        "       $(preV)= $(V);\n"
        "       $(V)= $(ip2);\n"
        "   }\n"
        "   else {\n"
        "       $(preV)= $(V);\n"
        "       $(V)= -($(Vspike));\n"
        "   }\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= $(ip2)");

    SET_PARAM_NAMES({"Vspike", "alpha", "y", "beta"});
    SET_VARS({{"V","scalar"}, {"preV", "scalar"}});

    SET_DERIVED_PARAMS({
        {"ip0", [](const vector<double> &pars, double){ return pars[0] * pars[0] * pars[1]; }},
        {"ip1", [](const vector<double> &pars, double){ return pars[0] * pars[2]; }},
        {"ip2", [](const vector<double> &pars, double){ return (pars[0] * pars[1]) + (pars[0] * pars[2]); }}});
};

//----------------------------------------------------------------------------
// NeuronModels::Izhikevich
//----------------------------------------------------------------------------
//! Izhikevich neuron with fixed parameters \cite izhikevich2003simple.
/*! It is usually described as
    \f{eqnarray*}
    \frac{dV}{dt} &=& 0.04 V^2 + 5 V + 140 - U + I, \\
    \frac{dU}{dt} &=& a (bV-U),
    \f}
    I is an external input current and the voltage V is reset to parameter c and U incremented by parameter d, whenever V >= 30 mV. This is paired with a particular integration procedure of two 0.5 ms Euler time steps for the V equation followed by one 1 ms time step of the U equation. Because of its popularity we provide this model in this form here event though due to the details of the usual implementation it is strictly speaking inconsistent with the displayed equations.

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
    DECLARE_MODEL(NeuronModels::Izhikevich, 4, 2);

    SET_SIM_CODE(
        "if ($(V) >= 30.0){\n"
        "   $(V)=$(c);\n"
        "   $(U)+=$(d);\n"
        "} \n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "//if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
        "//  $(V)=30.0; \n"
        "//}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 29.99");

    SET_PARAM_NAMES({"a", "b", "c", "d"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}});
};

//----------------------------------------------------------------------------
// NeuronModels::IzhikevichVariable
//----------------------------------------------------------------------------
//! Izhikevich neuron with variable parameters \cite izhikevich2003simple.
/*! This is the same model as Izhikevich but parameters are defined as
    "variables" in order to allow users to provide individual values for each
    individual neuron instead of fixed values for all neurons across the population.

    Accordingly, the model has the Variables:
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
    DECLARE_MODEL(NeuronModels::IzhikevichVariable, 0, 6);

    SET_PARAM_NAMES({});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"a", "scalar"},
             {"b", "scalar"}, {"c", "scalar"}, {"d", "scalar"}});
};

//----------------------------------------------------------------------------
// NeuronModels::SpikeSource
//----------------------------------------------------------------------------
//! Empty neuron which allows setting spikes from external sources
/*! This model does not contain any update code and can be used to implement
    the equivalent of a SpikeGeneratorGroup in Brian or a SpikeSourceArray in PyNN. */
class SpikeSource : public Base
{
public:
    DECLARE_MODEL(NeuronModels::SpikeSource, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("0");
};

//----------------------------------------------------------------------------
// NeuronModels::Poisson
//----------------------------------------------------------------------------
//! Poisson neurons
/*! Poisson neurons have constant membrane potential (\c Vrest) unless they are
    activated randomly to the \c Vspike value if (t- \c SpikeTime ) > \c trefract.

    It has 3 variables:

    - \c V - Membrane potential
    - \c Seed - Seed for random number generation
    - \c SpikeTime - Time at which the neuron spiked for the last time

    and 4 parameters:

    - \c therate - Firing rate
    - \c trefract - Refractory period
    - \c Vspike - Membrane potential at spike (mV)
    - \c Vrest - Membrane potential at rest (mV)

    \note The initial values array for the `Poisson` type needs three entries
    for `V`, `Seed` and `SpikeTime` and the parameter array needs four entries for
    `therate`, `trefract`, `Vspike` and `Vrest`,  *in that order*.

    \note Internally, GeNN uses a linear approximation for the probability
    of firing a spike in a given time step of size `DT`, i.e. the
    probability of firing is `therate` times `DT`: \f$ p = \lambda \Delta t
    \f$. This approximation is usually very good, especially for typical,
    quite small time steps and moderate firing rates. However, it is worth
    noting that the approximation becomes poor for very high firing rates
    and large time steps. An unrelated problem may occur with very low
    firing rates and small time steps. In that case it can occur that the
    firing probability is so small that the granularity of the 64 bit
    integer based random number generator begins to show. The effect
    manifests itself in that small changes in the firing rate do not seem
    to have an effect on the behaviour of the Poisson neurons because the
    numbers are so small that only if the random number is identical 0 a
    spike will be triggered.

    \note GeNN uses a separate random number generator for each Poisson neuron.
    The seeds (and later states) of these random number generators are stored in the `seed` variable.
    GeNN allocates memory for these seeds/states in the generated `allocateMem()` function.
    It is, however, currently the responsibility of the user to fill the array of seeds with actual random seeds.
    Not doing so carries the risk that all random number generators are seeded with the same seed ("0")
    and produce the same random numbers across neurons at each given time step.
    When using the GPU, `seed` also must be copied to the GPU after having been initialized.*/
class Poisson : public Base
{
public:
    DECLARE_MODEL(NeuronModels::Poisson, 4, 3);

    SET_SIM_CODE(
        "uint64_t theRnd;\n"
        "if ($(V) > $(Vrest)) {\n"
        "   $(V)= $(Vrest);\n"
        "}"
        "else if ($(t) - $(spikeTime) > ($(trefract))) {\n"
        "   MYRAND($(seed),theRnd);\n"
        "   if (theRnd < *($(rates)+$(offset)+$(id))) {\n"
        "       $(V)= $(Vspike);\n"
        "       $(spikeTime)= $(t);\n"
        "   }\n"
        "}\n");
    SET_THRESHOLD_CONDITION_CODE("$(V) >= $(Vspike)");

    SET_PARAM_NAMES({"therate", "trefract", "Vspike", "Vrest"});
    SET_VARS({{"V", "scalar"}, {"seed", "uint64_t"}, {"spikeTime", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"rates", "uint64_t *"}, {"offset", "unsigned int"}});

    virtual bool isPoisson() const{ return true; }
};

//----------------------------------------------------------------------------
// NeuronModels::TraubMiles
//----------------------------------------------------------------------------
//! Hodgkin-Huxley neurons with Traub & Miles algorithm.
/*! This conductance based model has been taken from \cite Traub1991 and can be described by the equations:
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
    - \c Cmem - Membrane capacity density in muF/cm^2

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
    DECLARE_MODEL(NeuronModels::TraubMiles, 7, 4);

    SET_SIM_CODE(
        "scalar Imem;\n"
        "unsigned int mt;\n"
        "scalar mdt= DT/25.0;\n"
        "for (mt=0; mt < 25; mt++) {\n"
        "   Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n"
        "       $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n"
        "       $(gl)*($(V)-($(El)))-$(Isyn));\n"
        "   scalar _a;\n"
        "   if (lV == -52.0) {\n"
        "       _a= 1.28;\n"
        "   }\n"
        "   else {\n"
        "       _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n"
        "   }\n"
        "   scalar _b;\n"
        "   if (lV == -25.0) {\n"
        "       _b= 1.4;\n"
        "   }\n"
        "   else {\n"
        "       _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n"
        "   }\n"
        "   $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n"
        "   _a= 0.128*exp((-48.0-$(V))/18.0);\n"
        "   _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n"
        "   $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n"
        "   if (lV == -50.0) {\n"
        "       _a= 0.16;\n"
        "   }\n"
        "   else {\n"
        "       _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n"
        "   }\n"
        "   _b= 0.5*exp((-55.0-$(V))/40.0);\n"
        "   $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n"
        "   $(V)+= Imem/$(C)*mdt;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 0.0");

    SET_PARAM_NAMES({"gNa", "ENa", "gK", "EK", "gl", "El", "C"});
    SET_VARS({{"V", "scalar"}, {"m", "scalar"}, {"h", "scalar"}, {"n", "scalar"}});
};

//----------------------------------------------------------------------------
// NeuronModels::TraubMilesFast
//----------------------------------------------------------------------------
//! Hodgkin-Huxley neurons with Traub & Miles algorithm: Original fast implementation, using 25 inner iterations.
/*! There are singularities in this model, which can be  easily hit in float precision*/
class TraubMilesFast : public TraubMiles
{
public:
    DECLARE_MODEL(NeuronModels::TraubMilesFast, 7, 4);

    SET_SIM_CODE(
        "scalar Imem;\n"
        "unsigned int mt;\n"
        "scalar mdt= DT/25.0;\n"
        "for (mt=0; mt < 25; mt++) {\n"
        "   Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n"
        "       $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n"
        "       $(gl)*($(V)-($(El)))-$(Isyn));\n"
        "   scalar _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n"
        "   scalar _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n"
        "   $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n"
        "   _a= 0.128*exp((-48.0-$(V))/18.0);\n"
        "   _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n"
        "   $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n"
        "   _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n"
        "   _b= 0.5*exp((-55.0-$(V))/40.0);\n"
        "   $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n"
        "   $(V)+= Imem/$(C)*mdt;\n"
        "}\n");
};

//----------------------------------------------------------------------------
// NeuronModels::TraubMilesAlt
//----------------------------------------------------------------------------
//! Hodgkin-Huxley neurons with Traub & Miles algorithm
/*! Using a workaround to avoid singularity: adding the munimum numerical value of the floating point precision used.*/
class TraubMilesAlt : public TraubMiles
{
public:
    DECLARE_MODEL(NeuronModels::TraubMilesAlt, 7, 4);

    SET_SIM_CODE(
        "scalar Imem;\n"
        "unsigned int mt;\n"
        "scalar mdt= DT/25.0;\n"
        "for (mt=0; mt < 25; mt++) {\n"
        "   Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n"
        "       $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n"
        "       $(gl)*($(V)-($(El)))-$(Isyn));\n"
        "   scalar volatile _tmp= abs(exp((-52.0-$(V))/4.0)-1.0);\n"
        "   scalar _a= 0.32*abs(-52.0-$(V))/(_tmp+SCALAR_MIN);\n"
        "   _tmp= abs(exp(($(V)+25.0)/5.0)-1.0);\n"
        "   scalar _b= 0.28*abs($(V)+25.0)/(_tmp+SCALAR_MIN);\n"
        "   $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n"
        "   _a= 0.128*exp((-48.0-$(V))/18.0);\n"
        "   _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n"
        "   $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n"
        "   _tmp= abs(exp((-50.0-$(V))/5.0)-1.0);\n"
        "   _a= 0.032*abs(-50.0-$(V))/(_tmp+SCALAR_MIN);\n"
        "   _b= 0.5*exp((-55.0-$(V))/40.0);\n"
        "   $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n"
        "   $(V)+= Imem/$(C)*mdt;\n"
        "}\n");
};

//----------------------------------------------------------------------------
// NeuronModels::TraubMilesNStep
//----------------------------------------------------------------------------
//! Hodgkin-Huxley neurons with Traub & Miles algorithm.
/*! Same as standard TraubMiles model but number of inner loops can be set using a parameter*/
class TraubMilesNStep : public TraubMiles
{
public:
    DECLARE_MODEL(NeuronModels::TraubMilesNStep, 8, 4);

    SET_SIM_CODE(
        "scalar Imem;\n"
        "unsigned int mt;\n"
        "scalar mdt= DT/scalar($(ntimes));\n"
        "for (mt=0; mt < $(ntimes); mt++) {\n"
        "   Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n"
        "       $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n"
        "       $(gl)*($(V)-($(El)))-$(Isyn));\n"
        "   scalar _a;\n"
        "   if (lV == -52.0) {\n"
        "       _a= 1.28;\n"
        "   }\n"
        "   else {\n"
        "       _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n"
        "   }\n"
        "   scalar _b;\n"
        "   if (lV == -25.0) {\n"
        "       _b= 1.4;\n"
        "   }\n"
        "   else {\n"
        "       _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n"
        "   }\n"
        "   $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n"
        "   _a= 0.128*exp((-48.0-$(V))/18.0);\n"
        "   _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n"
        "   $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n"
        "   if (lV == -50.0) {\n"
        "       _a= 0.16;\n"
        "   }\n"
        "   else {\n"
        "       _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n"
        "   }\n"
        "   _b= 0.5*exp((-55.0-$(V))/40.0);\n"
        "   $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n"
        "   $(V)+= Imem/$(C)*mdt;\n"
        "}\n");

    SET_PARAM_NAMES({"gNa", "ENa", "gK", "EK", "gl", "El", "C", "ntimes"});
};
} // NeuronModels