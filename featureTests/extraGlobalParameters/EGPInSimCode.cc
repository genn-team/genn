
#define DT 1.0f
//#define BLOCKSZ_DEBUG


#include "modelSpec.h"
#include "modelSpec.cc"


// NEURONS
//==============

double *neuron_p= NULL;
double neuron_ini[2] = { // one neuron variable
    0.0, // 0 - the time
    0.0  // 1 - individual shift
};

// Synapses
//==================================================

double *synapses_p= NULL;

double synapses_ini[1]= {
    0.0 // the copied time value
};

double *postSyn_p= NULL;

double *postSyn_ini = NULL;


void modelDefinition(NNmodel &model) 
{
  initGeNN();
  model.setName("EGPInSimCode");

  neuronModel n;
  n.varNames.push_back(tS("x"));
  n.varTypes.push_back(tS("scalar"));
  n.varNames.push_back(tS("shift"));
  n.varTypes.push_back(tS("scalar"));
  n.simCode= tS("$(x)= $(t)+$(shift)+$(input);");
  n.extraGlobalNeuronKernelParameters.push_back(tS("input"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("scalar"));
  int DUMMYNEURON= nModels.size();
  nModels.push_back(n);
    model.addNeuronPopulation("pre", 10, DUMMYNEURON, neuron_p, neuron_ini);
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
