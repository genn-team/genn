/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/MBody1_project/model/MBody1.cc

\brief This file contains the model definition of the mushroom body "MBody1" model. It is used in both the GeNN code generation and the user side simulation code (class classol, file classol_sim).
*/
//--------------------------------------------------------------------------

#include "modelSpec.h"
#include "sizes.h"


// Create variable initialisation snippet to zero all weights aside from those
// that pass a fixed probability test. Sample those from the normal distribution.
class GaussianFixedProbability : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(GaussianFixedProbability, 3);

    SET_CODE(
        "const scalar r = $(gennrand_uniform);\n"
        "if(r < $(pconn)) {\n"
        "   $(value) = $(gsynMean) + ($(gennrand_normal) * $(gsynSD));\n"
        "}\n"
        "else {\n"
        "   $(value) = 0.0;\n"
        "}\n");

    SET_PARAM_NAMES({"pconn", "gsynMean", "gsynSD"});
};
IMPLEMENT_SNIPPET(GaussianFixedProbability);

// Create variable initialisation snippet to zero all weights aside from those
// that pass a fixed probability test. Sample those from the normal distribution.
class PNLHIWeight : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PNLHIWeight, 2);

    SET_CODE("$(value) = $(theta) / ($(minact) + $(id_post));\n");

    SET_PARAM_NAMES({"theta", "minact"});
};
IMPLEMENT_SNIPPET(PNLHIWeight);


class GaussianMin : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(GaussianMin, 3);

    SET_CODE("$(value) = fmax($(min), $(mean) + ($(gennrand_normal) * $(sd)));");

    SET_PARAM_NAMES({"mean", "sd", "min"});
};
IMPLEMENT_SNIPPET(GaussianMin);

//--------------------------------------------------------------------------
/*! \brief This function defines the MBody1 model, and it is a good example of how networks should be defined.
 */
//--------------------------------------------------------------------------
void modelDefinition(ModelSpec &model) 
{

#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG

#ifdef _GPU_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = _GPU_DEVICE;
#endif

    NeuronModels::Poisson::ParamValues myPOI_p(
        10.0,   // 0 - refratory period
        1.0,    // 1 - spike duration
        20.0,   // 2 - Vspike
        -60.0); // 3 - Vrest

    NeuronModels::Poisson::VarValues myPOI_ini(
        -60.0,  // 0 - V
        -10.0); // 2 - SpikeTime

    NeuronModels::TraubMiles::ParamValues stdTM_p(
        7.15,       // 0 - gNa: Na conductance in 1/(mOhms * cm^2)
        50.0,       // 1 - ENa: Na equi potential in mV
        1.43,       // 2 - gK: K conductance in 1/(mOhms * cm^2)
        -95.0,      // 3 - EK: K equi potential in mV
        0.02672,    // 4 - gl: leak conductance in 1/(mOhms * cm^2)
        -63.563,    // 5 - El: leak equi potential in mV
        0.143);     // 6 - Cmem: membr. capacity density in muF/cm^2


    NeuronModels::TraubMiles::VarValues stdTM_ini(
        -60.0,      // 0 - membrane potential E
        0.0529324,  // 1 - prob. for Na channel activation m
        0.3176767,  // 2 - prob. for not Na channel blocking h
        0.5961207); // 3 - prob. for K channel activation n

#ifdef BITMASK
    WeightUpdateModels::StaticPulse::VarValues myPNKC_ini(
        100.0 / _NAL * _GScale);    // 0 - g: initial synaptic conductance

    InitSparseConnectivitySnippet::FixedProbability::ParamValues myPNKC_conn_p(
        0.5);   // 0 - Probability of connection
#else
    GaussianFixedProbability::ParamValues myPNKC_gsyn_p(
        0.5,                            // 0 - Probability of connection
        100.0 / _NAL * _GScale,         // 1 - Weight mean
        100.0 / _NAL * _GScale / 15.0); // 2 - Weight standard deviation

    WeightUpdateModels::StaticPulse::VarValues myPNKC_ini(
        initVar<GaussianFixedProbability>(myPNKC_gsyn_p));    // 0 - g: initial synaptic conductance
#endif

    PostsynapticModels::ExpCond::ParamValues postExpPNKC(
        1.0,    // 0 - tau_S: decay time constant for S [ms]
        0.0);   // 1 - Erev: Reversal potential

    PNLHIWeight::ParamValues myPNLHI_g_p(
        100.0 / _NAL * 14.0 * _GScale,  // 0 - theta
        15.0);                          // 1 - minact

    WeightUpdateModels::StaticPulse::VarValues myPNLHI_ini(
        initVar<PNLHIWeight>(myPNLHI_g_p)); // 0 - g: initial synaptic conductance

    PostsynapticModels::ExpCond::ParamValues postExpPNLHI(
        1.0,    // 0 - tau_S: decay time constant for S [ms]
        0.0);   // 1 - Erev: Reversal potential

    WeightUpdateModels::StaticGraded::ParamValues myLHIKC_p(
        -40.0,  // 0 - Epre: Presynaptic threshold potential
        50.0);  // 1 - Vslope: Activation slope of graded release

    WeightUpdateModels::StaticGraded::VarValues myLHIKC_ini(
        1.0 / _NLHI); // 0 - g: initial synaptic conductance

    PostsynapticModels::ExpCond::ParamValues postExpLHIKC(
        1.5,        // 0 - tau_S: decay time constant for S [ms]
        -92.0);     // 1 - Erev: Reversal potential

    WeightUpdateModels::PiecewiseSTDP::ParamValues myKCDN_p(
        50.0,       // 0 - TLRN: time scale of learning changes
        50.0,       // 1 - TCHNG: width of learning window
        50000.0,    // 2 - TDECAY: time scale of synaptic strength decay
        100000.0,   // 3 - TPUNISH10: Time window of suppression in response to 1/0
        200.0,      // 4 - TPUNISH01: Time window of suppression in response to 0/1
        0.015,      // 5 - GMAX: Maximal conductance achievable
        0.0075,     // 6 - GMID: Midpoint of sigmoid g filter curve
        33.33,      // 7 - GSLOPE: slope of sigmoid g filter curve
        10.0,       // 8 - TAUSHIFT: shift of learning curve
        0.00006);   // 9 - GSYN0: value of syn conductance g decays to

    GaussianMin::ParamValues myKCDN_gsyn_p(
        2500.0 / _NKC * 0.05 * _GScale,                                             // 0 - mean weight
        2500.0 / (sqrt((double) 1000.0) * sqrt((double)_NKC)) * 0.005 * _GScale,    // 1 - weight SD
        1e-20);                                                                     // 2 - min

    WeightUpdateModels::PiecewiseSTDP::VarValues myKCDN_ini(
        initVar<GaussianMin>(myKCDN_gsyn_p),    // 0 - g: synaptic conductance
        uninitialisedVar());                    // 1 - graw: raw synaptic conductance

    PostsynapticModels::ExpCond::ParamValues postExpKCDN(
        5.0,    // 0 - tau_S: decay time constant for S [ms]
        0.0);   // 1 - Erev: Reversal potential

    WeightUpdateModels::StaticGraded::ParamValues myDNDN_p(
        -30.0,  // 0 - Epre: Presynaptic threshold potential
        50.0);  // 1 - Vslope: Activation slope of graded release

    WeightUpdateModels::StaticGraded::VarValues myDNDN_ini(
        10.0 / _NDN);  // 0 - g: synaptic conductance

    PostsynapticModels::ExpCond::ParamValues postExpDNDN(
        2.5,    // 0 - tau_S: decay time constant for S [ms]
        -92.0); // 1 - Erev: Reversal potential

    model.setName("MBody1");
    model.setDT(0.1);
    model.addNeuronPopulation<NeuronModels::Poisson>("PN", _NAL, myPOI_p, myPOI_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("KC", _NKC, stdTM_p, stdTM_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("LHI", _NLHI, stdTM_p, stdTM_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("DN", _NDN, stdTM_p, stdTM_ini);

#ifdef DELAYED_SYNAPSES
    const unsigned int kcDNDelaySteps = 5;
    const unsigned int dnDNDelaySteps = 3;
#else
    const unsigned int kcDNDelaySteps = NO_DELAY;
    const unsigned int dnDNDelaySteps = NO_DELAY;
#endif

#ifdef BITMASK
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>(
        "PNKC", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY,
        "PN", "KC",
        {}, myPNKC_ini,
        postExpPNKC, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(myPNKC_conn_p));
#else
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>(
        "PNKC", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "PN", "KC",
        {}, myPNKC_ini,
        postExpPNKC, {});
#endif

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>(
        "PNLHI", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "PN", "LHI",
        {}, myPNLHI_ini,
        postExpPNLHI, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticGraded, PostsynapticModels::ExpCond>(
        "LHIKC", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
        "LHI", "KC",
        myLHIKC_p, myLHIKC_ini,
        postExpLHIKC, {});
    model.addSynapsePopulation<WeightUpdateModels::PiecewiseSTDP, PostsynapticModels::ExpCond>(
        "KCDN", SynapseMatrixType::DENSE_INDIVIDUALG, kcDNDelaySteps,
        "KC", "DN",
        myKCDN_p, myKCDN_ini,
        postExpKCDN, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticGraded, PostsynapticModels::ExpCond>(
        "DNDN", SynapseMatrixType::DENSE_GLOBALG, dnDNDelaySteps,
        "DN", "DN",
        myDNDN_p, myDNDN_ini,
        postExpDNDN, {});

    model.setSeed(1234);
    model.setPrecision(_FTYPE);
    model.setTiming(_TIMING);
}
