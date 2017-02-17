#include "modelSpec.h"

// NEURONS
//==============

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

void modelDefinition(NNmodel &model)
{
  initGeNN();
  
  model.setDT(0.1);
  model.setName("synapse_support_code_event_sim_code");

  neuronModel n;
  n.varNames = {"x", "shift"};
  n.varTypes = {"scalar", "scalar"};
  n.simCode= "$(x)= $(t)+$(shift);";
  n.thresholdConditionCode= "(fmod($(x),1.0) < 1e-4)";

  const int DUMMYNEURON= nModels.size();
  nModels.push_back(n);

  weightUpdateModel s;
  s.varNames = {"w"};
  s.varTypes = {"scalar"};
  s.pNames = {"myTrigger"};
  s.simCode_supportCode = "__device__ __host__ scalar getWeight(scalar x){ return x; }";
  s.evntThreshold= "(fmod($(x_pre), $(myTrigger)) < 1e-4)";
  s.simCodeEvnt= "$(w)= getWeight($(x_pre));";
  const int DUMMYSYNAPSE= weightUpdateModels.size();
  weightUpdateModels.push_back(s);

  model.addNeuronPopulation("pre", 10, DUMMYNEURON, NULL, neuron_ini);
  model.addNeuronPopulation("post", 10, DUMMYNEURON, NULL, neuron_ini);
  string synName= "syn";
  for (int i= 0; i < 10; i++)
  {
      string theName= synName + std::to_string(i);
      synapses_p[0]= (float) (2*(i+1));
      model.addSynapsePopulation(theName, DUMMYSYNAPSE, DENSE, INDIVIDUALG, i,IZHIKEVICH_PS, "pre", "post",
                                 synapses_ini, synapses_p,
                                 NULL, NULL);
  }
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
