#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);


void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setDT(0.1);
    model.setName("individualg_sparse");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 4, {}, Neuron::VarValues(0.0));


    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::Izhikevich>(
        "Syn", SPARSE, INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
