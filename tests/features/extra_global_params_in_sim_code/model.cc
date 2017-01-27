#include "modelSpec.h"

// NEURONS
//==============
double neuron_ini[2] =
{
    0.0, // 0 - the time
    0.0  // 1 - individual shift
};

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(0.1);
  model.setName("extra_global_parameters");

  neuronModel n;

  n.varNames = {"x", "shift"};
  n.varTypes = {"scalar",  "scalar"};
  n.simCode= "$(x)= $(t)+$(shift)+$(input);";

  n.extraGlobalNeuronKernelParameters = {"input"};
  n.extraGlobalNeuronKernelParameterTypes = {"scalar"};

  const int DUMMYNEURON = nModels.size();
  nModels.push_back(n);
  model.addNeuronPopulation("pre", 10, DUMMYNEURON, NULL, neuron_ini);
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
