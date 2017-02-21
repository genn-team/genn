#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_SIM_CODE("$(x)= $(t)+$(shift)+$(input);\n");

    SET_EXTRA_GLOBAL_PARAMS({{"input", "scalar"}});

    SET_INIT_VALS({{"x", "scalar"}, {"shift", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(0.1);
  model.setName("extra_global_params_in_sim_code");

  model.addNeuronPopulation<Neuron>("pre", 10, {}, Neuron::InitValues(0.0, 0.0));
  model.setPrecision(GENN_FLOAT);
  model.finalize();
}
