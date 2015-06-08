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

scalar meanInpExc = 5.0*inputFac; //5.0 for balanced regime
scalar meanInpInh = 2.0*inputFac; //2.0 for balanced regime

double *excIzh_p = NULL;

double *inhIzh_p = NULL;

double IzhExc_ini[7]={
//Izhikevich model initial conditions - excitatory population
	-65.0,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 2 - a
	0.2, 	// 3 - b
	-65.0, 	// 4 - c
	8.0, 	// 5 - d
	0.0     // 6 - I0
};

double IzhInh_ini[7]={
//Izhikevich model initial conditions - inhibitory population
	-65,	//0 - V
	 0.0,	//1 - U
	 0.02,	// 2 - a
	0.25, 	// 3 - b
	-65.0, 	// 4 - c
	2.0, 	// 5 - d 
	0.0     // 6 - I0
};

double *SynIzh_p= NULL;

double *postExpP= NULL;

double *postSynV = NULL;

double SynIzh_ini[1]= {
    0.0 // default synaptic conductance
};

void modelDefinition(NNmodel &model) 
{
  initGeNN();
  //we modify the IZHIKEVICH_V neuron to receive external inputs
  neuronModel n= nModels[IZHIKEVICH_V];
  n.varNames.push_back(tS("I0"));
  n.varTypes.push_back(tS("scalar"));
    n.simCode= tS("    if ($(V) >= 30.0){\n\
      $(V)=$(c);\n\
      $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
    //if ($(V) > 30.0){      //keep this only for visualisation -- not really necessaary otherwise \n\
    //  $(V)=30.0; \n\
    //}\n\
    ");
    unsigned int myIZHIKEVICH= nModels.size();
    nModels.push_back(n);
  
  model.setName("Izh_sparse");
  model.addNeuronPopulation("PExc", _NExc, myIZHIKEVICH, excIzh_p, IzhExc_ini);

  model.addNeuronPopulation("PInh", _NInh, myIZHIKEVICH, inhIzh_p, IzhInh_ini);
  
  model.addSynapsePopulation("Exc_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PExc", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
  model.addSynapsePopulation("Exc_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PExc", "PInh", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
  model.addSynapsePopulation("Inh_Exc", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PExc", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
  model.addSynapsePopulation("Inh_Inh", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "PInh", "PInh", SynIzh_ini, SynIzh_p, postSynV, postExpP); 
 
  fprintf(stderr, "#model created.\n"); 

  model.setMaxConn("Exc_Exc", _NMaxConnP0);
  model.setMaxConn("Exc_Inh", _NMaxConnP1);
  model.setMaxConn("Inh_Exc", _NMaxConnP2);
  model.setMaxConn("Inh_Inh", _NMaxConnP3);
  
  #ifdef nGPU 
    cerr << "nGPU: " << nGPU << endl;
    model.setGPUDevice(nGPU);
  #endif 
  model.setPrecision(_FTYPE);
  model.finalize();
}
