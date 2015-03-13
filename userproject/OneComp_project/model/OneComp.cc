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

double exIzh_p[4]={
//Izhikevich model parameters - tonic spiking
	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6 	// 3 - d
};

double exIzh_ini[2]={
//Izhikevich model initial conditions - tonic spiking
	-65,	//0 - V
	-20	//1 - U
};


double mySyn_p[3]= {
  0.0,           // 0 - Erev: Reversal potential
  -20.0,         // 1 - Epre: Presynaptic threshold potential
  1.0            // 2 - tau_S: decay time constant for S [ms]
};

double postExp[2]={
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};
double *postSynV = NULL;


#include "../../userproject/include/sizes.h"

float inpIzh1 = 4.0;

void modelDefinition(NNmodel &model) 
{
  initGeNN();
    model.setGPUDevice(0);
  model.setName("OneComp");
  model.addNeuronPopulation("Izh1", _NC1, IZHIKEVICH, exIzh_p, exIzh_ini);        	 
 // model.addSynapsePopulation("IzhIzh", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "Izh1", "Izh1", mySyn_p, postSynV, postExp);
 // model.setSynapseG("IzhIzh", gIzh1);
  
  model.activateDirectInput("Izh1", CONSTINP);
  model.setConstInp("Izh1", inpIzh1);
  model.setPrecision(FLOAT);
  model.finalize();
}
