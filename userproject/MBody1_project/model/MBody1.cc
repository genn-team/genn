/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file MBody1.cc

\brief This file contains the model definition of the mushroom body "MBody1" model. It is used in both the GeNN code generation and the user side simulation code (class classol, file classol_sim).
*/
//--------------------------------------------------------------------------

#define DT 0.1  //!< This defines the global time step at which the simulation will run
#include "modelSpec.h"
#include "modelSpec.cc"
#include "../../userproject/include/sizes.h"

int nGPU= 0;

float myPOI_p[4]= {
  0.1,        // 0 - firing rate
  2.5,        // 1 - refratory period
  20.0,       // 2 - Vspike
  -60.0       // 3 - Vrest
};

float myPOI_ini[3]= {
 -60.0,        // 0 - V
  0,           // 1 - seed
  -10.0        // 2 - SpikeTime
};

float stdTM_p[7]= {
  7.15,          // 0 - gNa: Na conductance in 1/(mOhms * cm^2)
  50.0,          // 1 - ENa: Na equi potential in mV
  1.43,          // 2 - gK: K conductance in 1/(mOhms * cm^2)
  -95.0,         // 3 - EK: K equi potential in mV
  0.02672,       // 4 - gl: leak conductance in 1/(mOhms * cm^2)
  -63.563,       // 5 - El: leak equi potential in mV
  0.143          // 6 - Cmem: membr. capacity density in muF/cm^2
};


float stdTM_ini[4]= {
  -60.0,                       // 0 - membrane potential E
  0.0529324,                   // 1 - prob. for Na channel activation m
  0.3176767,                   // 2 - prob. for not Na channel blocking h
  0.5961207                    // 3 - prob. for K channel activation n
};

float *myPNKC_p= NULL;

float myPNKC_ini[1]= {
  0.01            // 0 - g: initial synaptic conductance
};

float postExpPNKC[2]={
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

float *myPNLHI_p= NULL;

float myPNLHI_ini[1]= {
    0.0          // 0 - g: initial synaptic conductance
};

float postExpPNLHI[2]={
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

float myLHIKC_p[2]= {
  -40.0,          // 0 - Epre: Presynaptic threshold potential
  50.0            // 1 - Vslope: Activation slope of graded release 
};

//float gLHIKC= 0.6;
float myLHIKC_ini[1] = {
    0.35/_NLHI   // 0 - g: initial synaptic conductance
};

float postExpLHIKC[2]={
    1.5, //3.0,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
};

float myKCDN_p[11]= {
  -20.0,         // 0 - Epre: Presynaptic threshold potential
  50.0, //25.0,          // 1 - TLRN: time scale of learning changes
  50.0, //100.0        // 2 - TCHNG: width of learning window
  50000.0,       // 3 - TDECAY: time scale of synaptic strength decay
  100000.0,      // 4 - TPUNISH10: Time window of suppression in response to 1/0
  200.0, //100.0,  // 5 - TPUNISH01: Time window of suppression in response to 0/1
  0.015, // 0.06,          // 6 - GMAX: Maximal conductance achievable
  0.0075, // 0.03,          // 7 - GMID: Midpoint of sigmoid g filter curve
  33.33,         // 8 - GSLOPE: slope of sigmoid g filter curve
  10.0,          // 9 - TAUSHIFT: shift of learning curve
  0.00006 // 0.006   // 10 - GSYN0: value of syn conductance g decays to
};

float myKCDN_ini[2]={
  0.01,            // 0 - g: synaptic conductance
  0.01,		  // 1 - graw: raw synaptic conductance
};

//#define KCDNGSYN0 0.006
float postExpKCDN[2]={
  5.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

float myDNDN_p[2]= {
  -30.0,        // 0 - Epre: Presynaptic threshold potential 
  50.0          // 1 - Vslope: Activation slope of graded release 
};
//float gDNDN= 0.04;

float myDNDN_ini[1]={
    1.0/_NLB            // 0 - g: synaptic conductance
};

float postExpDNDN[2]={
  8.0,            // 0 - tau_S: decay time constant for S [ms]
  -92.0		  // 1 - Erev: Reversal potential
};

float *postSynV = NULL;


//--------------------------------------------------------------------------
/*! \brief This function defines the MBody1 model, and it is a good example of how networks should be defined.
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
    initGeNN();
    model.setName("MBody1");
    model.addNeuronPopulation("PN", _NAL, POISSONNEURON, myPOI_p, myPOI_ini);
    model.addNeuronPopulation("KC", _NMB, TRAUBMILES, stdTM_p, stdTM_ini);
    model.addNeuronPopulation("LHI", _NLHI, TRAUBMILES, stdTM_p, stdTM_ini);
    model.addNeuronPopulation("DN", _NLB, TRAUBMILES, stdTM_p, stdTM_ini);
    
    float *init = NULL;
    model.addSynapsePopulation("PNKC", NSYNAPSE, DENSE, INDIVIDUALG, NO_DELAY, EXPDECAY, "PN", "KC", myPNKC_ini, myPNKC_p, postSynV,postExpPNKC);
    model.addSynapsePopulation("PNLHI", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, EXPDECAY, "PN", "LHI",  myPNLHI_ini, myPNLHI_p, postSynV, postExpPNLHI);
    model.addSynapsePopulation("LHIKC", NGRADSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, EXPDECAY, "LHI", "KC",  myLHIKC_ini, myLHIKC_p, postSynV, postExpLHIKC);
    model.addSynapsePopulation("KCDN", LEARN1SYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, EXPDECAY, "KC", "DN",  myKCDN_ini,  myKCDN_p, postSynV, postExpKCDN);
    model.addSynapsePopulation("DNDN", NGRADSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, EXPDECAY, "DN", "DN", myDNDN_ini, myDNDN_p, postSynV, postExpDNDN);     
    model.setGPUDevice(nGPU);
    model.setSeed(1234);
}
