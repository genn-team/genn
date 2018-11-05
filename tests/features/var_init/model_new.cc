//--------------------------------------------------------------------------
/*! \file model_new.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


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
// CurrentSrc
//----------------------------------------------------------------------------
class CurrentSrc : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CurrentSrc, 0, 4);

    SET_VARS({{"constant", "scalar"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}});
};
IMPLEMENT_MODEL(CurrentSrc);

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

    // Current source parameters
    CurrentSrc::VarValues currentSourceInit(
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
    model.addCurrentSource<CurrentSrc>("CurrSource", "Pop", {}, currentSourceInit);
#ifndef CPU_ONLY
    auto *popGPU = model.addNeuronPopulation<Neuron>("PopGPU", 10000, {}, neuronInit);
    popGPU->setVarMode("constant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    popGPU->setVarMode("uniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    popGPU->setVarMode("normal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    popGPU->setVarMode("exponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    auto *currSourceGPU = model.addCurrentSource<CurrentSrc>("CurrSourceGPU", "PopGPU", {}, neuronInit);
    currSourceGPU->setVarMode("constant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    currSourceGPU->setVarMode("uniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    currSourceGPU->setVarMode("normal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    currSourceGPU->setVarMode("exponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
#endif

    // Dense synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Pop",
        {}, weightUpdateInit,
        {}, postsynapticInit);
#ifndef CPU_ONLY
    auto *denseGPU = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "DenseGPU", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "PopGPU",
        {}, weightUpdateInit,
        {}, postsynapticInit);
    denseGPU->setPSVarMode("pconstant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setPSVarMode("puniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setPSVarMode("pnormal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setPSVarMode("pexponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setWUVarMode("constant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setWUVarMode("uniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setWUVarMode("normal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    denseGPU->setWUVarMode("exponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
#endif

    // Sparse synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Pop",
        {}, weightUpdateInit,
        {}, postsynapticInit);
#ifndef CPU_ONLY
    auto *sparseGPU = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "SparseGPU", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "PopGPU",
        {}, weightUpdateInit,
        {}, postsynapticInit);
    sparseGPU->setPSVarMode("pconstant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setPSVarMode("puniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setPSVarMode("pnormal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setPSVarMode("pexponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setWUVarMode("constant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setWUVarMode("uniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setWUVarMode("normal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    sparseGPU->setWUVarMode("exponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
#endif  // CPU_ONLY

    // Ragged synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Ragged", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Pop",
        {}, weightUpdateInit,
        {}, postsynapticInit);
#ifndef CPU_ONLY
    auto *raggedGPU = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "RaggedGPU", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "PopGPU",
        {}, weightUpdateInit,
        {}, postsynapticInit);
    raggedGPU->setPSVarMode("pconstant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setPSVarMode("puniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setPSVarMode("pnormal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setPSVarMode("pexponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setWUVarMode("constant", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setWUVarMode("uniform", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setWUVarMode("normal", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    raggedGPU->setWUVarMode("exponential", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
#endif  // CPU_ONLY

    model.setPrecision(GENN_FLOAT);
    model.finalize();
}
