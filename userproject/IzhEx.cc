/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/
#define DT 1.0
#include "modelSpec.h"
#include "modelSpec.cc"


float myPOI_p[4]= {
//POISSON neuron parameters
  0.1,        // 0 - firing rate
  2.5,        // 1 - refractory period
  20.0,       // 2 - Vspike
  -60.0       // 3 - Vrest
};

float myPOI_ini[4]= {
 -60.0,        // 0 - V
  0,           // 1 - seed
  -10.0,       // 2 - SpikeTime
};

float exIzh_p[4]={
//Izhikevich model parameters - phasic bursting
	0.02,	// 0 - a
	0.25,	// 1 - b
	-55,	// 2 - c
	0.05	// 3 - d
};

float exIzh_ini[2]={
//Izhikevich model initial conditions - tonic spiking
	-65,	//0 - V
	-20	//1 - U
};

float myPNIzh1_p[4]= {
  -20.0,           // 0 - Erev: Reversal potential
  -30.0,         // 1 - Epre: Presynaptic threshold potential
  8.0,            // 2 - tau_S: decay time constant for S [ms]
  20.0         // 3 - Vslope: Activation slope of graded release - needed if NGRADSYNAPSE
};

float inpIzh1 = 10.0;

#include "../../userproject/include/sizes.h"

void modelDefinition(NNmodel &model) 
{
  model.setName("IzhEx");
  model.addNeuronPopulation("PN", _NAL, POISSONNEURON, myPOI_p, myPOI_ini);
  model.addNeuronPopulation("Izh1", _NMB, IZHIKEVICH, exIzh_p, exIzh_ini);
  
  printf("const inp=%d ",CONSTINP);
  model.activateDirectInput("Izh1", CONSTINP);
  model.setConstInp("Izh1",inpIzh1);

  model.addSynapsePopulation("PNIzh1", NGRADSYNAPSE, ALLTOALL, INDIVIDUALG, "PN", "Izh1", myPNIzh1_p);
  model.setSynapseG("PNIzh1", inpIzh1);
}
