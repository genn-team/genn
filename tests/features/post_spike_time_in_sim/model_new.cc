#include "modelSpec.h"

//----------------------------------------------------------------------------
// PreNeuron
//----------------------------------------------------------------------------
class PreNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PreNeuron, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("true");
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("$(t) >= (scalar)$(id) && fmodf($(t) - (scalar)$(id), 10.0f)< 1e-4");
};

IMPLEMENT_MODEL(PostNeuron);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModel, 0, 1);

    SET_VARS({{"w", "scalar"}});

    SET_SIM_CODE("$(w)= $(sT_post);");
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    // Turn off auto refractory logic so post neuron can spike every timestep
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::autoRefractory = false;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    initGeNN();
    model.setDT(1.0);
    model.setName("post_spike_time_in_sim_new");

    model.addNeuronPopulation<PreNeuron>("pre", 10, {}, {});
    model.addNeuronPopulation<PostNeuron>("post", 10, {}, {});

    auto *syn = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "syn", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY, "pre", "post",
        {}, WeightUpdateModel::VarValues(0.0),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));

    syn->setBackPropDelaySteps(20);

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}