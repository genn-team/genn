/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file MBody_userdef.cc

\brief This file contains the model definition of the mushroom body model.
 tis used in the GeNN code generation and the user side simulation code 
(class classol, file classol_sim). 
*/
//--------------------------------------------------------------------------

#include "modelSpec.h"
#include "global.h"
#include "sizes.h"

//uncomment the following line to turn on timing measures (Linux/MacOS only)
#define TIMING   

/******************************************************************/
// redefine WeightUpdateModels::PiecewiseSTDP as a user-defined syapse type:
class PiecewiseSTDPUserDef : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(PiecewiseSTDPUserDef, 10, 2);

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
IMPLEMENT_MODEL(PiecewiseSTDPUserDef);

/******************************************************************/
// redefine WeightUpdateModels::StaticPulse as a user-defined syapse type
class StaticPulseUserDef : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(StaticPulseUserDef, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        " $(updatelinsyn);\n");
};
IMPLEMENT_MODEL(StaticPulseUserDef);

/******************************************************************/
// redefine WeightUpdateModels::StaticGraded as a user-defined syapse type:
class StaticGradedUserDef : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(StaticGradedUserDef, 2, 1);

    SET_PARAM_NAMES({"Epre", "Vslope"});
    SET_VARS({{"g", "scalar"}});

    SET_EVENT_CODE(
        "$(addtoinSyn) = $(g) * tanh(($(V_pre) - $(Epre)) / $(Vslope))* DT;\n"
        "if ($(addtoinSyn) < 0) $(addtoinSyn) = 0.0;\n"
        " $(updatelinsyn);\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > $(Epre)");
};
IMPLEMENT_MODEL(StaticGradedUserDef);

/******************************************************************/
// redefine PostsynapticModels::ExpCond as a user-defined syapse type:
class ExpCondUserDef : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(ExpCondUserDef, 2, 0);

    SET_DECAY_CODE("$(inSyn)*=$(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(inSyn) * ($(E) - $(V))");

    SET_PARAM_NAMES({"tau", "E"});

    SET_DERIVED_PARAMS({{"expDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
};
IMPLEMENT_MODEL(ExpCondUserDef);

NeuronModels::Poisson::ParamValues myPOI_p(
  0.1,        // 0 - firing rate
  2.5,        // 1 - refratory period
  20.0,       // 2 - Vspike
  -60.0       // 3 - Vrest
);

NeuronModels::Poisson::VarValues myPOI_ini(
 -60.0,        // 0 - V
  0,           // 1 - seed
  -10.0        // 2 - SpikeTime
);

NeuronModels::TraubMiles::ParamValues stdTM_p(
  7.15,          // 0 - gNa: Na conductance in 1/(mOhms * cm^2)
  50.0,          // 1 - ENa: Na equi potential in mV
  1.43,          // 2 - gK: K conductance in 1/(mOhms * cm^2)
  -95.0,         // 3 - EK: K equi potential in mV
  0.02672,       // 4 - gl: leak conductance in 1/(mOhms * cm^2)
  -63.563,       // 5 - El: leak equi potential in mV
  0.143          // 6 - Cmem: membr. capacity density in muF/cm^2
);

NeuronModels::TraubMiles::VarValues stdTM_ini(
  -60.0,                       // 0 - membrane potential E
  0.0529324,                   // 1 - prob. for Na channel activation m
  0.3176767,                   // 2 - prob. for not Na channel blocking h
  0.5961207                    // 3 - prob. for K channel activation n
);


StaticPulseUserDef::VarValues myPNKC_ini(
  0.01            // 0 - g: initial synaptic conductance
);

ExpCondUserDef::ParamValues postExpPNKC(
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
);

StaticPulseUserDef::VarValues myPNLHI_ini(
    0.0          // 0 - g: initial synaptic conductance
);

ExpCondUserDef::ParamValues postExpPNLHI(
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
);

StaticGradedUserDef::ParamValues myLHIKC_p(
  -40.0,          // 0 - Epre: Presynaptic threshold potential
  50.0            // 1 - Vslope: Activation slope of graded release 
);

StaticGradedUserDef::VarValues myLHIKC_ini(
    1.0/_NLHI   // 0 - g: initial synaptic conductance
);

ExpCondUserDef::ParamValues postExpLHIKC(
  1.5,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
);

PiecewiseSTDPUserDef::ParamValues myKCDN_p(
  50.0,          // 0 - TLRN: time scale of learning changes
  50.0,         // 1 - TCHNG: width of learning window
  50000.0,       // 2 - TDECAY: time scale of synaptic strength decay
  100000.0,      // 3 - TPUNISH10: Time window of suppression in response to 1/0
  200.0,         // 4 - TPUNISH01: Time window of suppression in response to 0/1
  0.015,          // 5 - GMAX: Maximal conductance achievable
  0.0075,          // 6 - GMID: Midpoint of sigmoid g filter curve
  33.33,         // 7 - GSLOPE: slope of sigmoid g filter curve
  10.0,          // 8 - TAUSHiFT: shift of learning curve
  0.00006          // 9 - GSYN0: value of syn conductance g decays to
);

PiecewiseSTDPUserDef::VarValues myKCDN_ini(
  0.01,            // 0 - g: synaptic conductance
  0.01		  // 1 - graw: raw synaptic conductance
);

ExpCondUserDef::ParamValues postExpKCDN(
  5.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
);

StaticGradedUserDef::ParamValues myDNDN_p(
  -30.0,         // 0 - Epre: Presynaptic threshold potential 
  50.0           // 1 - Vslope: Activation slope of graded release 
);

StaticGradedUserDef::VarValues myDNDN_ini(
    5.0/_NLB      // 0 - g: synaptic conductance
);

ExpCondUserDef::ParamValues postExpDNDN(
  2.5,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
);

//for sparse only -- we need to set them by hand if we want to do dense to sparse conversion. Sparse connectivity will only create sparse arrays.
scalar * gpPNKC = new scalar[_NAL*_NMB];
scalar * gpKCDN = new scalar[_NMB*_NLB];
//-------------------------------------

//--------------------------------------------------------------------------
/*! \brief This function defines the MBody1 model with user defined synapses. 
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
  initGeNN();

#ifdef DEBUG
  GENN_PREFERENCES::debugCode = true;
#else
  GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG

  model.setPrecision(_FTYPE);

  model.setName("MBody_userdef");
  model.setDT(0.1);
  model.addNeuronPopulation<NeuronModels::Poisson>("PN", _NAL, myPOI_p, myPOI_ini);
  model.addNeuronPopulation<NeuronModels::TraubMiles>("KC", _NMB, stdTM_p, stdTM_ini);
  model.addNeuronPopulation<NeuronModels::TraubMiles>("LHI", _NLHI, stdTM_p, stdTM_ini);
  model.addNeuronPopulation<NeuronModels::TraubMiles>("DN", _NLB, stdTM_p, stdTM_ini);
  
  model.addSynapsePopulation<StaticPulseUserDef, ExpCondUserDef>("PNKC", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                 "PN", "KC",
                                                                 {}, myPNKC_ini,
                                                                 postExpPNKC, {});
  //setting max number of connections to the number of target neurons (used only in default mode (post-span))
  model.setMaxConn("PNKC", _NMB); 
  //set synapse update to pre-span mode 
  //model.setSpanTypeToPre("PNKgpPNKCC");

  model.addSynapsePopulation<StaticPulseUserDef, ExpCondUserDef>("PNLHI", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                 "PN", "LHI",
                                                 {}, myPNLHI_ini,
                                                 postExpPNLHI, {});

  model.addSynapsePopulation<StaticGradedUserDef, ExpCondUserDef>("LHIKC", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                  "LHI", "KC",
                                                                  myLHIKC_p, myLHIKC_ini,
                                                                  postExpLHIKC, {});

  model.addSynapsePopulation<PiecewiseSTDPUserDef, ExpCondUserDef>("KCDN", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                   "KC", "DN",
                                                                   myKCDN_p, myKCDN_ini,
                                                                   postExpKCDN, {});
  //setting max number of connections to the number of target neurons (used only in default mode (post-span))
  //model.setMaxConn("KCDN", _NLB); 
  //set synapse update to pre-span mode 
  model.setSpanTypeToPre("KCDN");

  model.addSynapsePopulation<StaticGradedUserDef, ExpCondUserDef>("DNDN", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                  "DN", "DN",
                                                                  myDNDN_p, myDNDN_ini,
                                                                  postExpDNDN, {});
  
#ifdef nGPU 
  cerr << "nGPU: " << nGPU << endl;
  model.setGPUDevice(nGPU);
#endif 
  model.setSeed(1234);
#ifdef TIMING
    model.setTiming(true);
#else
    model.setTiming(false);
#endif // TIMING
  model.finalize();
}
