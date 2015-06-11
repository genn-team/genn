/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file MBody_delayedSyn.cc

\brief This file contains the model definition of the mushroom body "MBody_delayedSyn" model. It is used in both the GeNN code generation and the user side simulation code (class classol, file classol_sim).
*/
//--------------------------------------------------------------------------

#define DT 0.1  //!< This defines the global time step at which the simulation will run
#include "modelSpec.h"
#include "modelSpec.cc"
#include "sizes.h"

//uncomment the following line to turn on timing measures (Linux/MacOS only)
//#define TIMING   

double myPOI_p[4]= {
  0.1,        // 0 - firing rate
  2.5,        // 1 - refratory period
  20.0,       // 2 - Vspike
  -60.0       // 3 - Vrest
};

double myPOI_ini[3]= {
 -60.0,        // 0 - V
  0,           // 1 - seed
  -10.0        // 2 - SpikeTime
};

double stdTM_p[7]= {
  7.15,          // 0 - gNa: Na conductance in 1/(mOhms * cm^2)
  50.0,          // 1 - ENa: Na equi potential in mV
  1.43,          // 2 - gK: K conductance in 1/(mOhms * cm^2)
  -95.0,         // 3 - EK: K equi potential in mV
  0.02672,       // 4 - gl: leak conductance in 1/(mOhms * cm^2)
  -63.563,       // 5 - El: leak equi potential in mV
  0.143          // 6 - Cmem: membr. capacity density in muF/cm^2
};


double stdTM_ini[4]= {
  -60.0,                       // 0 - membrane potential E
  0.0529324,                   // 1 - prob. for Na channel activation m
  0.3176767,                   // 2 - prob. for not Na channel blocking h
  0.5961207                    // 3 - prob. for K channel activation n
};

double *myPNKC_p= NULL;

double myPNKC_ini[1]= {
  0.01            // 0 - g: initial synaptic conductance
};

double postExpPNKC[2]={
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

double *myPNLHI_p= NULL;

double myPNLHI_ini[1]= {
    0.0          // 0 - g: initial synaptic conductance
};

double postExpPNLHI[2]={
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

double myLHIKC_p[2]= {
  -40.0,          // 0 - Epre: Presynaptic threshold potential
  50.0            // 1 - Vslope: Activation slope of graded release 
};

//double gLHIKC= 0.6;
double myLHIKC_ini[1] = {
    1.0/_NLHI   // 0 - g: initial synaptic conductance
};

double postExpLHIKC[2]={
    1.5, //3.0,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
};

double myKCDN_p[10]= {
  50.0,             // 0 - TLRN: time scale of learning changes
  50.0,             // 1 - TCHNG: width of learning window
  50000.0,          // 2 - TDECAY: time scale of synaptic strength decay
  100000.0,         // 3 - TPUNISH10: Time window of suppression in response to 1/0
  200.0,            // 4 - TPUNISH01: Time window of suppression in response to 0/1
  0.015,            // 5 - GMAX: Maximal conductance achievable
  0.0075,           // 6 - GMID: Midpoint of sigmoid g filter curve
  33.33,            // 7 - GSLOPE: slope of sigmoid g filter curve
  10.0,             // 8 - TAUSHIFT: shift of learning curve
  0.00006           // 9 - GSYN0: value of syn conductance g decays to
};

double myKCDN_ini[2]={
  0.01,            // 0 - g: synaptic conductance
  0.01,		  // 1 - graw: raw synaptic conductance
};

//#define KCDNGSYN0 0.006
double postExpKCDN[2]={
  5.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

double myDNDN_p[2]= {
  -30.0,        // 0 - Epre: Presynaptic threshold potential 
  50.0          // 1 - Vslope: Activation slope of graded release 
};
//double gDNDN= 0.04;

double myDNDN_ini[1]={
    5.0/_NLB            // 0 - g: synaptic conductance
};

double postExpDNDN[2]={
  2.5,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
};

double *postSynV = NULL;


//--------------------------------------------------------------------------
/*! \brief This function defines the MBody_delayedSyn model, and it is a good example of how networks should be defined.
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
    initGeNN();
    model.setName("MBody_delayedSyn");
    model.addNeuronPopulation("PN", _NAL, POISSONNEURON, myPOI_p, myPOI_ini);
    model.addNeuronPopulation("KC", _NMB, TRAUBMILES, stdTM_p, stdTM_ini);
    model.addNeuronPopulation("LHI", _NLHI, TRAUBMILES, stdTM_p, stdTM_ini);
    model.addNeuronPopulation("DN", _NLB, TRAUBMILES, stdTM_p, stdTM_ini);
    
    model.addSynapsePopulation("PNKC", NSYNAPSE, DENSE, INDIVIDUALG, NO_DELAY, EXPDECAY, "PN", "KC", myPNKC_ini, myPNKC_p, postSynV,postExpPNKC);
    model.addSynapsePopulation("PNLHI", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, EXPDECAY, "PN", "LHI",  myPNLHI_ini, myPNLHI_p, postSynV, postExpPNLHI);
    model.addSynapsePopulation("LHIKC", NGRADSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, EXPDECAY, "LHI", "KC",  myLHIKC_ini, myLHIKC_p, postSynV, postExpLHIKC);
    model.addSynapsePopulation("KCDN", LEARN1SYNAPSE, ALLTOALL, INDIVIDUALG, 5, EXPDECAY, "KC", "DN",  myKCDN_ini,  myKCDN_p, postSynV, postExpKCDN);
    model.addSynapsePopulation("DNDN", NGRADSYNAPSE, ALLTOALL, GLOBALG, 3, EXPDECAY, "DN", "DN", myDNDN_ini, myDNDN_p, postSynV, postExpDNDN);     

  #ifdef nGPU 
    cerr << "nGPU: " << nGPU << endl;
    model.setGPUDevice(nGPU);
  #endif 
  
    model.setSeed(1234);
    model.setPrecision(_FTYPE);
#ifdef TIMING
    model.setTiming(TRUE);
#else
    model.setTiming(FALSE);
#endif // TIMING
  model.finalize();
}
