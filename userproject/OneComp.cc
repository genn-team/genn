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

float exIzh_p[4]={
//Izhikevich model parameters - tonic spiking
	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6 	// 3 - d
};

float exIzh_ini[2]={
//Izhikevich model initial conditions - tonic spiking
	-65,	//0 - V
	-20	//1 - U
};

#include "../../userproject/include/sizes.h"

void modelDefinition(NNmodel &model) 
{
  model.setName("IzhEx");
  model.addNeuronPopulation("Izh1", _NC1, IZHIKEVICH, exIzh_p, exIzh_ini);
  
  model.activateDirectInput("Izh1", INPRULE);
}
