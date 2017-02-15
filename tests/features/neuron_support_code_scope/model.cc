#include "modelSpec.h"

// NEURONS
//==============

double neuron_ini[2] = { // one neuron variable
    0.0, // 0 - the time
    0.0  // 1 - individual shift
};

// Synapses
//==================================================

double synapses_ini[1]= {
    0.0 // the copied time value
};

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(0.1);
  model.setName("neuron_support_code_scope");

  neuronModel n;
  n.varNames = {"x", "shift"};
  n.varTypes = {"scalar", "scalar"};
  n.supportCode = "bool checkThreshold(scalar x, scalar threshold){ return (fmod(x, 1.0) < threshold); }";
  n.simCode=
      "scalar lThreshold = 1e-4;"
      "$(x)= $(t)+$(shift);";
  n.thresholdConditionCode= "checkThreshold($(x), lThreshold)";

  const int DUMMYNEURON= nModels.size();
  nModels.push_back(n);

  weightUpdateModel s;
  s.varNames = {"w"};
  s.varTypes = {"scalar"};
  s.simCode= "$(w)= $(x_pre);";
  const int DUMMYSYNAPSE= weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, NULL, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, NULL, neuron_ini);
  string synName= "syn";
  for (int i= 0; i < 10; i++)
  {
      string theName= synName + std::to_string(i);
      model.addSynapsePopulation(theName, DUMMYSYNAPSE, DENSE, INDIVIDUALG, i,IZHIKEVICH_PS, "pre", "post",
                                 synapses_ini, NULL,
                                 NULL, NULL);
  }
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
