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

double exIzh_p[5]={
//Izhikevich model parameters - tonic spiking
	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6, 	// 3 - d
	4.0     // 4 - I0 (input current)
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

void modelDefinition(NNmodel &model) 
{
  initGeNN();
#ifndef CPU_ONLY
    model.setGPUDevice(0);
#endif
  model.setName("OneComp");
  neuronModel n= nModels[IZHIKEVICH];
  n.pNames.push_back(tS("I0"));
  n.simCode= tS("    if ($(V) >= 30.0){\n\
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n	\
   //  $(V)=30.0; \n\
   //}\n\
   ");
  unsigned int MYIZHIKEVICH= nModels.size();
  nModels.push_back(n);
  model.addNeuronPopulation("Izh1", _NC1, MYIZHIKEVICH, exIzh_p, exIzh_ini);        	 
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
