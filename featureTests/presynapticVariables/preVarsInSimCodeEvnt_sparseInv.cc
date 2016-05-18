
#define DT 0.1

#include "modelSpec.h"
#include "stringUtils.h"


// NEURONS
//==============

double *neuron_p= NULL;
double neuron_ini[2] = { // one neuron variable
    0.0, // 0 - the time
    0.0  // 1 - individual shift
};

// Synapses
//==================================================

double synapses_p[1]= {
    0.0 // the myTrigger parameter
};

double synapses_ini[1]= {
    0.0 // the copied time value
};

double *postSyn_p= NULL;

double *postSyn_ini = NULL;


void modelDefinition(NNmodel &model) 
{
  initGeNN();
  model.setName("preVarsInSimCodeEvnt_sparseInv");

  neuronModel n;
  n.varNames.push_back("x");
  n.varTypes.push_back("scalar");
  n.varNames.push_back("shift");
  n.varTypes.push_back("scalar");
  n.simCode= "$(x)= t+$(shift);";
  int DUMMYNEURON= nModels.size();
  nModels.push_back(n);
  
  weightUpdateModel s;
  s.varNames.push_back("w");
  s.varTypes.push_back("scalar");
  s.pNames.push_back("myTrigger");
  s.evntThreshold= "(fmod($(x_pre),$(myTrigger)) < 1e-4)";
  s.simCodeEvnt= "$(w)= $(x_pre);";
  int DUMMYSYNAPSE= weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, neuron_p, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, neuron_p, neuron_ini);
  string synName= "syn";
  for (int i= 0; i < 10; i++) {
      string theName= synName+tS(i);
      synapses_p[0]= (float) (2*(i+1));
      model.addSynapsePopulation(theName, DUMMYSYNAPSE, SPARSE, INDIVIDUALG, i,IZHIKEVICH_PS, "pre", "post", synapses_ini, synapses_p, postSyn_ini, postSyn_p);
      model.setSpanTypeToPre(theName);
  }
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
