#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(gennrand_uniform);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);


void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);
    model.setName("neuron_rng_uniform_new");

    model.addNeuronPopulation<Neuron>("Pop", 1000, {}, Neuron::VarValues(0.0));

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
