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
#include <vector>

std::vector<unsigned int> neuronPSize;
std::vector<unsigned int> neuronVSize;
std::vector<unsigned int> synapsePSize; 

float *excIzh_p = NULL;
//Izhikevich model parameters - tonic spiking
/*	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6 	// 3 - d
};*/


float *inhIzh_p = NULL;
//Izhikevich model parameters - tonic spiking
/*	0.02,	// 0 - a
	0.25, 	// 1 - b
	-65, 	// 2 - c
	2 	// 3 - d
};*/

float IzhExc_ini[6]={
//Izhikevich model initial conditions - excitatory population
	-65.0,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 0 - a
	0.2, 	// 1 - b
	-65.0, 	// 2 - c
	 8.0 	// 3 - d
};

float IzhInh_ini[6]={
//Izhikevich model initial conditions - inhibitory population
	-65,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 0 - a
	0.25, 	// 1 - b
	-65.0, 	// 2 - c
	 2.0 	// 3 - d 
};

float SynIzh_p[3]= {
  0.0,           // 0 - Erev: Reversal potential
  30.0,         // 1 - Epre: Presynaptic threshold potential
  1.0            // 2 - tau_S: decay time constant for S [ms]
};

float postExpP[2]={
  0.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

float *postSynV = NULL;


//float inpIzh1 = 4.0;
//float gIzh1= 0.05;
//float * input1, *input2;
#include "../../userproject/include/sizes.h"

void modelDefinition(NNmodel &model) 
{
  model.setName("Izh_sparse");
  model.addNeuronPopulation("PExc", _NExc, IZHIKEVICH_V, excIzh_p, IzhExc_ini);
  neuronPSize.push_back(0);
  neuronVSize.push_back(sizeof IzhExc_ini);
  

  model.addNeuronPopulation("PInh", _NInh, IZHIKEVICH_V, inhIzh_p, IzhInh_ini);
  neuronPSize.push_back(0);
  neuronVSize.push_back(sizeof IzhInh_ini);
  
  model.addSynapsePopulation("Exc_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PExc", SynIzh_p, postSynV, postExpP); 
  //model.setSynapseG("Exc_Exc", gIzh1);
  synapsePSize.push_back(sizeof SynIzh_p);
  
  
  model.addSynapsePopulation("Exc_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PInh", SynIzh_p, postSynV, postExpP); 
 //model.setSynapseG("Exc_Inh", gIzh1);
  synapsePSize.push_back(sizeof SynIzh_p);
  
  
  model.addSynapsePopulation("Inh_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PExc", SynIzh_p, postSynV, postExpP); 
  //model.setSynapseG("Inh_Exc", gIzh1);
  synapsePSize.push_back(sizeof SynIzh_p);
  
  
  model.addSynapsePopulation("Inh_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PInh", SynIzh_p, postSynV, postExpP); 
 //model.setSynapseG("Inh_Inh", gIzh1);
  synapsePSize.push_back(sizeof SynIzh_p);
  fprintf(stderr, "#model created.\n"); 

  model.activateDirectInput("PExc", INPRULE);
  model.activateDirectInput("PInh", INPRULE);
	model.setMaxConn("Exc_Exc", _NMaxConnP0);
	model.setMaxConn("Exc_Inh", _NMaxConnP1);
	model.setMaxConn("Inh_Exc", _NMaxConnP2);
	model.setMaxConn("Inh_Inh", _NMaxConnP3);
  //model.setConstInp("Izh1", input1);
  model.setPrecision(FLOAT);
  
  //model.checkSizes(&neuronPSize[0], &neuronVSize[0], &synapsePSize[0]); //it would be better to call this before adding populations as there is a risk of segfault
}
