#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 1, 2);

    SET_SIM_CODE("$(x)= $(t)+$(shift);\n");

    SET_THRESHOLD_CONDITION_CODE("(fmod($(x),$(ISI)) < 1e-4)");

    SET_PARAM_NAMES({"ISI"});
    SET_INIT_VALS({{"x", "scalar"}, {"shift", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);

double neuron_p[1]={
    1.0   // 0 - ISI for spiking
};

double neuron_p2[1]={
    2.0   // 0 - ISI for spiking
};

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
  model.setName("pre_vars_in_synapse_dynamics");

  neuronModel n;
  n.varNames = {"x", "shift"};
  n.varTypes = {"scalar", "scalar"};

  n.pNames = {"ISI"};

  n.simCode= "$(x)= $(t)+$(shift);";

  const int DUMMYNEURON= nModels.size();
  nModels.push_back(n);

  weightUpdateModel s;
  s.varNames = {"w"};
  s.varTypes = {"scalar"};
  s.synapseDynamics= "$(w)= $(x_pre);";
  const int DUMMYSYNAPSE= weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, neuron_p, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, neuron_p2, neuron_ini);
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
