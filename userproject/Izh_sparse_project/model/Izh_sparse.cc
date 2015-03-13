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
#include "sizes.h"

std::vector<unsigned int> neuronPSize;
std::vector<unsigned int> neuronVSize;
std::vector<unsigned int> synapsePSize; 

scalar meanInpExc = 5.0; //5.0 for balanced regime
scalar meanInpInh = 2.0; //2.0 for balanced regime

double *excIzh_p = NULL;
//Izhikevich model parameters - tonic spiking
/*	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6 	// 3 - d
};*/


double *inhIzh_p = NULL;
//Izhikevich model parameters - tonic spiking
/*	0.02,	// 0 - a
	0.25, 	// 1 - b
	-65, 	// 2 - c
	2 	// 3 - d
};*/

double IzhExc_ini[6]={
//Izhikevich model initial conditions - excitatory population
	-65.0,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 0 - a
	0.2, 	// 1 - b
	-65.0, 	// 2 - c
	 8.0 	// 3 - d
};

double IzhInh_ini[6]={
//Izhikevich model initial conditions - inhibitory population
	-65,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 0 - a
	0.25, 	// 1 - b
	-65.0, 	// 2 - c
	 2.0 	// 3 - d 
};

double *SynIzh_p= NULL;

double postExpP[2]={
  0.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

double *postSynV = NULL;

double SynIzh_ini[1]= {
    0.0 // default synaptic conductance
};

//double inpIzh1 = 4.0;
//double gIzh1= 0.05;
//double * input1, *input2;

void modelDefinition(NNmodel &model) 
{
  initGeNN();
  model.setName("Izh_sparse");
  model.addNeuronPopulation("PExc", _NExc, IZHIKEVICH_V, excIzh_p, IzhExc_ini);

  model.addNeuronPopulation("PInh", _NInh, IZHIKEVICH_V, inhIzh_p, IzhInh_ini);
  
  model.addSynapsePopulation("Exc_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PExc", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
  //model.setSynapseG("Exc_Exc", gIzh1);
  
  model.addSynapsePopulation("Exc_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PInh", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
 //model.setSynapseG("Exc_Inh", gIzh1);

  model.addSynapsePopulation("Inh_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PExc", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
  //model.setSynapseG("Inh_Exc", gIzh1);
  
  model.addSynapsePopulation("Inh_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PInh", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
 //model.setSynapseG("Inh_Inh", gIzh1);
  fprintf(stderr, "#model created.\n"); 

  model.activateDirectInput("PExc", INPRULE);
  model.activateDirectInput("PInh", INPRULE);
	model.setMaxConn("Exc_Exc", _NMaxConnP0);
	model.setMaxConn("Exc_Inh", _NMaxConnP1);
	model.setMaxConn("Inh_Exc", _NMaxConnP2);
	model.setMaxConn("Inh_Inh", _NMaxConnP3);
  //model.setConstInp("Izh1", input1);
  
  #ifdef nGPU 
    cerr << "nGPU: " << nGPU << endl;
    model.setGPUDevice(nGPU);
  #endif 
  model.setPrecision(_FTYPE);
  model.finalize();
}
