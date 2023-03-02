//--------------------------------------------------------------------------
/*! \file var_init/model.cc

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
    DECLARE_MODEL(Neuron, 0, 7);

    SET_VARS({{"num", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
};
IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// CurrentSrc
//----------------------------------------------------------------------------
class CurrentSrc : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CurrentSrc, 0, 7);

    SET_VARS({{"num", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
};
IMPLEMENT_MODEL(CurrentSrc);

//----------------------------------------------------------------------------
// PostsynapticModel
//----------------------------------------------------------------------------
class PostsynapticModel : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PostsynapticModel, 0, 7);

    SET_VARS({{"pnum", "unsigned int"}, {"pnum_batch", "unsigned int"}, {"puniform", "scalar"}, {"pnormal", "scalar"}, {"pexponential", "scalar"}, {"pgamma", "scalar"}, {"pbinomial", "unsigned int"}});
};
IMPLEMENT_MODEL(PostsynapticModel);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModel, 0, 8, 7, 7);

    SET_VARS({{"num_pre", "unsigned int"}, {"num_post", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
    SET_PRE_VARS({{"pre_num", "unsigned int"}, {"pre_num_batch", "unsigned int"}, {"pre_uniform", "scalar"}, {"pre_normal", "scalar"}, {"pre_exponential", "scalar"}, {"pre_gamma", "scalar"}, {"pre_binomial", "unsigned int"}});
    SET_POST_VARS({{"post_num", "unsigned int"}, {"post_num_batch", "unsigned int"}, {"post_uniform", "scalar"}, {"post_normal", "scalar"}, {"post_exponential", "scalar"}, {"post_gamma", "scalar"}, {"post_binomial", "unsigned int"}});
};
IMPLEMENT_MODEL(WeightUpdateModel);

//----------------------------------------------------------------------------
// WeightUpdateModelNoPrePost
//----------------------------------------------------------------------------
class WeightUpdateModelNoPrePost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelNoPrePost, 0, 8, 0, 0);

    SET_VARS({{"num_pre", "unsigned int"}, {"num_post", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
};
IMPLEMENT_MODEL(WeightUpdateModelNoPrePost);

//----------------------------------------------------------------------------
// NopCustomUpdateModel
//----------------------------------------------------------------------------
class NopCustomUpdateModel : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(NopCustomUpdateModel, 0, 7, 1);

    SET_VARS({{"num", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
    SET_VAR_REFS({{"R", "unsigned int", VarAccessMode::READ_WRITE}})
};
IMPLEMENT_MODEL(NopCustomUpdateModel);

//----------------------------------------------------------------------------
// NopCustomUpdateModelWU
//----------------------------------------------------------------------------
class NopCustomUpdateModelWU : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(NopCustomUpdateModelWU, 0, 8, 1);

    SET_VARS({{"num_pre", "unsigned int"}, {"num_post", "unsigned int"}, {"num_batch", "unsigned int"}, {"uniform", "scalar"}, {"normal", "scalar"}, {"exponential", "scalar"}, {"gamma", "scalar"}, {"binomial", "unsigned int"}});
    SET_VAR_REFS({{"R", "unsigned int", VarAccessMode::READ_WRITE}})
};
IMPLEMENT_MODEL(NopCustomUpdateModelWU);

//----------------------------------------------------------------------------
// NumBatch
//----------------------------------------------------------------------------
class NumBatch : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(NumBatch, 0);
    
     SET_CODE("$(value) = $(num_batch);");
};
IMPLEMENT_SNIPPET(NumBatch);

//----------------------------------------------------------------------------
// Num
//----------------------------------------------------------------------------
class Num : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Num, 0);
    
     SET_CODE("$(value) = $(num);");
};
IMPLEMENT_SNIPPET(Num);

//----------------------------------------------------------------------------
// NumPre
//----------------------------------------------------------------------------
class NumPre : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(NumPre, 0);
    
     SET_CODE("$(value) = $(num_pre);");
};
IMPLEMENT_SNIPPET(NumPre);

//----------------------------------------------------------------------------
// NumPost
//----------------------------------------------------------------------------
class NumPost : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(NumPost, 0);
    
     SET_CODE("$(value) = $(num_post);");
};
IMPLEMENT_SNIPPET(NumPost);

void modelDefinition(ModelSpec &model)
{
#ifdef CL_HPP_TARGET_OPENCL_VERSION
    if(std::getenv("OPENCL_DEVICE") != nullptr) {
        GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
        GENN_PREFERENCES.manualDeviceID = std::atoi(std::getenv("OPENCL_DEVICE"));
    }
    if(std::getenv("OPENCL_PLATFORM") != nullptr) {
        GENN_PREFERENCES.manualPlatformID = std::atoi(std::getenv("OPENCL_PLATFORM"));
    }
#endif
    model.setSeed(2346679);
    model.setDT(0.1);
    model.setName("var_init");


    // Parameters for configuring uniform and normal distributions
    InitVarSnippet::Uniform::ParamValues uniformParams(
        0.0,        // 0 - min
        1.0);       // 1 - max

    InitVarSnippet::Uniform::ParamValues normalParams(
        0.0,        // 0 - mean
        1.0);       // 1 - sd

    InitVarSnippet::Exponential::ParamValues exponentialParams(
        1.0);       // 0 - lambda

    InitVarSnippet::Gamma::ParamValues gammaParams(
        4.0,        // 0 - a
        1.0);       // 1 - b
    
    InitVarSnippet::Binomial::ParamValues binomialParams(
        20,         // 0 - n
        0.5);       // 1 - p

    // Neuron parameters
    Neuron::VarValues neuronInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));

    // Current source parameters
    CurrentSrc::VarValues currentSourceInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));

    // PostsynapticModel parameters
    PostsynapticModel::VarValues postsynapticInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));

    // WeightUpdateModel parameters
    WeightUpdateModel::VarValues weightUpdateInit(
        initVar<NumPre>(),
        initVar<NumPost>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));
    WeightUpdateModel::PreVarValues weightUpdatePreInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));
    WeightUpdateModel::PostVarValues weightUpdatePostInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));

    // CustomUpdateModel parameters
    NopCustomUpdateModel::VarValues customUpdateInit(
        initVar<Num>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));
    NopCustomUpdateModelWU::VarValues customUpdateWUInit(
        initVar<NumPre>(),
        initVar<NumPost>(),
        initVar<NumBatch>(),
        initVar<InitVarSnippet::Uniform>(uniformParams),
        initVar<InitVarSnippet::Normal>(normalParams),
        initVar<InitVarSnippet::Exponential>(exponentialParams),
        initVar<InitVarSnippet::Gamma>(gammaParams),
        initVar<InitVarSnippet::Binomial>(binomialParams));
    
    InitToeplitzConnectivitySnippet::Conv2D::ParamValues convParams(
        3, 3,       // conv_kh, conv_kw
        100, 100, 5,  // conv_ih, conv_iw, conv_ic
        100, 100, 5); // conv_oh, conv_ow, conv_oc
    
    // Neuron populations
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource1", 1, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource2", 50000, {}, {});
    NeuronGroup *ng = model.addNeuronPopulation<Neuron>("Pop", 50000, {}, neuronInit);
    CurrentSource *cs = model.addCurrentSource<CurrentSrc>("CurrSource", "Pop", {}, currentSourceInit);

    // Dense synapse populations
    SynapseGroup *sgDense = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource1", "Pop",
        {}, weightUpdateInit, weightUpdatePreInit, weightUpdatePostInit,
        {}, postsynapticInit);

    // Sparse synapse populations
    SynapseGroup *sgSparse = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource2", "Pop",
        {}, weightUpdateInit, weightUpdatePreInit, weightUpdatePostInit,
        {}, postsynapticInit,
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
    
    SynapseGroup *sgKernel = model.addSynapsePopulation<WeightUpdateModelNoPrePost, PostsynapticModels::DeltaCurr>(
        "Kernel", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY,
        "SpikeSource2", "Pop",
        {}, weightUpdateInit, {}, {},
        {}, {},
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParams));
        
    // Custom updates
    NopCustomUpdateModel::VarReferences neuronVarReferences(createVarRef(ng, "num")); // R
    model.addCustomUpdate<NopCustomUpdateModel>("NeuronCustomUpdate", "Test",
                                               {}, customUpdateInit, neuronVarReferences);
    
    NopCustomUpdateModel::VarReferences currentSourceVarReferences(createVarRef(cs, "num")); // R
    model.addCustomUpdate<NopCustomUpdateModel>("CurrentSourceCustomUpdate", "Test",
                                               {}, customUpdateInit, currentSourceVarReferences);
                                               
    NopCustomUpdateModel::VarReferences psmVarReferences(createPSMVarRef(sgDense, "pnum")); // R
    model.addCustomUpdate<NopCustomUpdateModel>("PSMCustomUpdate", "Test",
                                               {}, customUpdateInit, neuronVarReferences);
                                    
    NopCustomUpdateModel::VarReferences wuPreVarReferences(createWUPreVarRef(sgSparse, "pre_num")); // R
    model.addCustomUpdate<NopCustomUpdateModel>("WUPreCustomUpdate", "Test",
                                               {}, customUpdateInit, wuPreVarReferences);
                                               
    NopCustomUpdateModel::VarReferences wuPostVarReferences(createWUPostVarRef(sgDense, "post_num")); // R
    model.addCustomUpdate<NopCustomUpdateModel>("WUPostCustomUpdate", "Test",
                                                {}, customUpdateInit, wuPostVarReferences);
    
    NopCustomUpdateModelWU::WUVarReferences wuSparseVarReferences(createWUVarRef(sgSparse, "num_pre")); // R
    model.addCustomUpdate<NopCustomUpdateModelWU>("WUSparseCustomUpdate", "Test",
                                                  {}, customUpdateWUInit, wuSparseVarReferences);
    
    NopCustomUpdateModelWU::WUVarReferences wuDenseVarReferences(createWUVarRef(sgDense, "num_pre")); // R
    model.addCustomUpdate<NopCustomUpdateModelWU>("WUDenseCustomUpdate", "Test",
                                                  {}, customUpdateWUInit, wuDenseVarReferences);
    
    NopCustomUpdateModelWU::WUVarReferences wuKernelVarReferences(createWUVarRef(sgKernel, "num_pre")); // R
    model.addCustomUpdate<NopCustomUpdateModelWU>("WUKernelCustomUpdate", "Test",
                                                  {}, customUpdateWUInit, wuKernelVarReferences);

    model.setPrecision(GENN_FLOAT);
}
