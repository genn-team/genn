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

    model.setDT(1.0);
    model.setName("decode_matrix_den_delay_individualg_dense_new");

    // Static synapse parameters
    WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
        uninitialisedVar(),     // 0 - Wij (nA)
        uninitialisedVar());    // 1 - Dij (timestep)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 1, {}, Neuron::VarValues(0.0));

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});
    syn->setMaxDendriticDelayTimesteps(10);

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
