#include "modelSpec.h"

// NEURONS
//==============
double neuron_ini[2] =
{
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
  model.setName("extra_global_params_in_sim_code_event_sparse_inv");

  neuronModel n;

  n.varNames = {"x", "shift"};
  n.varTypes = {"scalar",  "scalar"};
  n.simCode= "$(x)= $(t)+$(shift);";

  const int DUMMYNEURON = nModels.size();
  nModels.push_back(n);

  weightUpdateModel s;
  s.varNames = {"w"};
  s.varTypes = {"scalar"};
  s.extraGlobalSynapseKernelParameters = {"thresh"};
  s.extraGlobalSynapseKernelParameterTypes = {"scalar"};
  s.evntThreshold= "(fmod($(x_pre),$(thresh)) < 1e-4)";
  s.simCodeEvnt= "$(w)= $(x_pre);";

  const int DUMMYSYNAPSE = weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, NULL, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, NULL, neuron_ini);
  string synName= "syn";
  for (int i= 0; i < 10; i++)
  {
      string theName= synName + std::to_string(i);
      model.addSynapsePopulation(theName, DUMMYSYNAPSE, SPARSE, INDIVIDUALG, i,IZHIKEVICH_PS, "pre", "post",
                                 synapses_ini, NULL,
                                 NULL, NULL);
      model.setSpanTypeToPre(theName);
  }

  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
