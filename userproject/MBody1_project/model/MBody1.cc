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
#include "global.h"
#include "sizes.h"

//uncomment the following line to turn on timing measures
//#define TIMING   


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

WeightUpdateModels::StaticPulse::VarValues myPNKC_ini(
  0.01            // 0 - g: initial synaptic conductance
);

PostsynapticModels::ExpCond::ParamValues postExpPNKC(
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0             // 1 - Erev: Reversal potential
);

WeightUpdateModels::StaticPulse::VarValues myPNLHI_ini(
    0.0          // 0 - g: initial synaptic conductance
);

PostsynapticModels::ExpCond::ParamValues postExpPNLHI(
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
);

WeightUpdateModels::StaticGraded::ParamValues myLHIKC_p(
  -40.0,          // 0 - Epre: Presynaptic threshold potential
  50.0            // 1 - Vslope: Activation slope of graded release 
);

WeightUpdateModels::StaticGraded::VarValues myLHIKC_ini(
    1.0/_NLHI   // 0 - g: initial synaptic conductance
);

PostsynapticModels::ExpCond::ParamValues postExpLHIKC(
    1.5, //3.0,            // 0 - tau_S: decay time constant for S [ms]
  -92.0                    // 1 - Erev: Reversal potential
);

WeightUpdateModels::PiecewiseSTDP::ParamValues myKCDN_p(
  50.0,               // 0 - TLRN: time scale of learning changes
  50.0,               // 1 - TCHNG: width of learning window
  50000.0,            // 2 - TDECAY: time scale of synaptic strength decay
  100000.0,           // 3 - TPUNISH10: Time window of suppression in response to 1/0
  200.0,              // 4 - TPUNISH01: Time window of suppression in response to 0/1
  0.015,              // 5 - GMAX: Maximal conductance achievable
  0.0075,             // 6 - GMID: Midpoint of sigmoid g filter curve
  33.33,              // 7 - GSLOPE: slope of sigmoid g filter curve
  10.0,               // 8 - TAUSHIFT: shift of learning curve
  0.00006             // 9 - GSYN0: value of syn conductance g decays to
);

WeightUpdateModels::PiecewiseSTDP::VarValues myKCDN_ini(
  0.01,            // 0 - g: synaptic conductance
  0.01             // 1 - graw: raw synaptic conductance
);

PostsynapticModels::ExpCond::ParamValues postExpKCDN(
  5.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0             // 1 - Erev: Reversal potential
);

WeightUpdateModels::StaticGraded::ParamValues myDNDN_p(
  -30.0,        // 0 - Epre: Presynaptic threshold potential 
  50.0          // 1 - Vslope: Activation slope of graded release 
);

WeightUpdateModels::StaticGraded::VarValues myDNDN_ini(
    5.0/_NLB            // 0 - g: synaptic conductance
);

PostsynapticModels::ExpCond::ParamValues postExpDNDN(
  2.5,            // 0 - tau_S: decay time constant for S [ms]
  -92.0           // 1 - Erev: Reversal potential
);

//--------------------------------------------------------------------------
/*! \brief This function defines the MBody1 model, and it is a good example of how networks should be defined.
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

    GENN_PREFERENCES::userNvccFlags = " -O3 -use_fast_math";
     GENN_PREFERENCES::optimizeCode = false;
   //GENN_PREFERENCES::autoChooseDevice= 0;
    //GENN_PREFERENCES::optimiseBlockSize= 0;
    //GENN_PREFERENCES::neuronBlockSize= 192; 

    model.setName("MBody1");
    model.setDT(0.1);
    model.addNeuronPopulation<NeuronModels::Poisson>("PN", _NAL, myPOI_p, myPOI_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("KC", _NMB, stdTM_p, stdTM_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("LHI", _NLHI, stdTM_p, stdTM_ini);
    model.addNeuronPopulation<NeuronModels::TraubMiles>("DN", _NLB, stdTM_p, stdTM_ini);
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>("PNKC", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                             "PN", "KC",
                                                                                             {}, myPNKC_ini,
                                                                                             postExpPNKC, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>("PNLHI", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                             "PN", "LHI",
                                                                                             {}, myPNLHI_ini,
                                                                                             postExpPNLHI, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticGraded, PostsynapticModels::ExpCond>("LHIKC", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                              "LHI", "KC",
                                                                                              myLHIKC_p, myLHIKC_ini,
                                                                                              postExpLHIKC, {});
    model.addSynapsePopulation<WeightUpdateModels::PiecewiseSTDP, PostsynapticModels::ExpCond>("KCDN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                               "KC", "DN",
                                                                                               myKCDN_p, myKCDN_ini,
                                                                                               postExpKCDN, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticGraded, PostsynapticModels::ExpCond>("DNDN", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                              "DN", "DN",
                                                                                              myDNDN_p, myDNDN_ini,
                                                                                              postExpDNDN, {});
#ifdef nGPU 
    cerr << "nGPU: " << nGPU << endl;
    model.setGPUDevice(nGPU);
#endif 
    model.setSeed(1234);
    model.setPrecision(_FTYPE);
#ifdef TIMING
    model.setTiming(true);
#else
    model.setTiming(false);
#endif // TIMING
  model.finalize();
}
