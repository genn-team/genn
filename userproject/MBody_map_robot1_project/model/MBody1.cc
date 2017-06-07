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

NeuronModels::RulkovMap::ParamValues stdRMP_p(
  60.0,         // 0 - Vspike
  3.0,           // 1 - alpha
  -2.468,        // 2 - y 
  2.64           // 3 - beta
);

NeuronModels::RulkovMap::VarValues stdRMP_ini(
  -60.0,                       // 0 - membrane potential E
  -60.0                        // 1 - E of prev time step
);

WeightUpdateModels::StaticPulse::VarValues myPNKC_ini(
  1.0            // 0 - g: initial synaptic conductance
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
  100.0,               // 0 - TLRN: time scale of learning changes
  50.0,               // 1 - TCHNG: width of learning window
  50000.0,            // 2 - TDECAY: time scale of synaptic strength decay
  100000.0,           // 3 - TPUNISH10: Time window of suppression in response to 1/0
  200.0,              // 4 - TPUNISH01: Time window of suppression in response to 0/1
  0.0015,              // 5 - GMAX: Maximal conductance achievable
  0.00075,             // 6 - GMID: Midpoint of sigmoid g filter curve
  333.3,              // 7 - GSLOPE: slope of sigmoid g filter curve
  10.0,               // 8 - TAUSHIFT: shift of learning curve
  0.000006             // 9 - GSYN0: value of syn conductance g decays to
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
    0.01            // 0 - g: synaptic conductance
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
    model.setDT(0.5);
    model.addNeuronPopulation<NeuronModels::Poisson>("PN", _NAL, myPOI_p, myPOI_ini);
    model.addNeuronPopulation<NeuronModels::RulkovMap>("KC", _NMB, stdRMP_p, stdRMP_ini);
    model.addNeuronPopulation<NeuronModels::RulkovMap>("LHI", _NLHI, stdRMP_p, stdRMP_ini);
    model.addNeuronPopulation<NeuronModels::RulkovMap>("DN", _NLB, stdRMP_p, stdRMP_ini);
    
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
