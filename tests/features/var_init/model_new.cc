#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 4);

    SET_VARS({{"constant", "scalar"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}});
};
IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// PostsynapticModel
//----------------------------------------------------------------------------
class PostsynapticModel : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PostsynapticModel, 0, 4);

    SET_VARS({{"pconstant", "scalar"}, {"puniform", "scalar"}, {"pnormal", "scalar"}, {"pexponential", "scalar"}});
};
IMPLEMENT_MODEL(PostsynapticModel);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModel, 0, 4);

    SET_VARS({{"constant", "scalar"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}});
};
IMPLEMENT_MODEL(WeightUpdateModel);

void modelDefinition(NNmodel &model)
{
    initGeNN();

    model.setSeed(12345);
    model.setDT(0.1);
    model.setName("var_init_new");

    GENN_PREFERENCES::autoInitSparseVars = true;

    // Parameters for configuring uniform and normal distributions
    InitVarSnippet::Uniform::ParamValues uniformParams(
        0.0,        // 0 - min
        1.0);       // 1 - max

    InitVarSnippet::Uniform::ParamValues normalParams(
        0.0,        // 0 - mean
        1.0);       // 1 - sd

    InitVarSnippet::Exponential::ParamValues exponentialParams(
        1.0);       // 0 - lambda

    // Neuron parameters
    Neuron::VarValues neuronInit(
        13.0,
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams));

    // PostsynapticModel parameters
    PostsynapticModel::VarValues postsynapticInit(
        13.0,
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams));

    // WeightUpdateModel parameters
    WeightUpdateModel::VarValues weightUpdateInit(
        13.0,
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams));

    // Neuron populations
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 1, {}, {});
    model.addNeuronPopulation<Neuron>("Pop", 10000, {}, neuronInit);

    // Synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Pop",
        {}, weightUpdateInit,
        {}, postsynapticInit);

    // Synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Pop",
        {}, weightUpdateInit,
        {}, postsynapticInit);
    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
