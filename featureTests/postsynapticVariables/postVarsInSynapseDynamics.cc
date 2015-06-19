
#define DT 1.0f

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
  model.setName("postVarsInSynapseDynamics");

  neuronModel n;
  n.varNames.push_back(tS("x"));
  n.varTypes.push_back(tS("scalar"));
  n.varNames.push_back(tS("shift"));
  n.varTypes.push_back(tS("scalar"));
  n.simCode= tS("$(x)= $(t)+$(shift);");
  int DUMMYNEURON= nModels.size();
  nModels.push_back(n);
  
  weightUpdateModel s;
  s.varNames.push_back(tS("w"));
  s.varTypes.push_back(tS("scalar"));
  s.synapseDynamics= tS("$(w)= $(x_post);");
  int DUMMYSYNAPSE= weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, neuron_p, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, neuron_p, neuron_ini);
  string synName= tS("syn");
  for (int i= 0; i < 10; i++) {
      string theName= synName+tS(i);
      model.addSynapsePopulation(theName, DUMMYSYNAPSE, DENSE, INDIVIDUALG, i,IZHIKEVICH_PS, "pre", "post", synapses_ini, synapses_p, postSyn_ini, postSyn_p);
  }
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
