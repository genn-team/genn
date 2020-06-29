//--------------------------------------------------------------------------
/*! \file var_init/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// ScalarEGP
//----------------------------------------------------------------------------
class ScalarEGP : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(ScalarEGP, 0);

    SET_CODE("$(value) = $(val);");

    SET_EXTRA_GLOBAL_PARAMS({{"val", "scalar"}});
};
IMPLEMENT_SNIPPET(ScalarEGP);

//----------------------------------------------------------------------------
// RepeatVal
//----------------------------------------------------------------------------
class RepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(RepeatVal, 0);

    SET_CODE("$(value) = $(values)[$(id) % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(RepeatVal);

//----------------------------------------------------------------------------
// PostRepeatVal
//----------------------------------------------------------------------------
class PostRepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PostRepeatVal, 0);

    SET_CODE("$(value) = $(values)[$(id_post) % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(PostRepeatVal);

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_VARS({{"vconstant", "scalar"}, {"vrepeat", "scalar"}});
};
IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// CurrentSrc
//----------------------------------------------------------------------------
class CurrentSrc : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CurrentSrc, 0, 2);

    SET_VARS({{"vconstant", "scalar"}, {"vrepeat", "scalar"}});
};
IMPLEMENT_MODEL(CurrentSrc);

//----------------------------------------------------------------------------
// PostsynapticModel
//----------------------------------------------------------------------------
class PostsynapticModel : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PostsynapticModel, 0, 2);

    SET_VARS({{"pvconstant", "scalar"}, {"pvrepeat", "scalar"}});
};
IMPLEMENT_MODEL(PostsynapticModel);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModel, 0, 2, 2, 2);

    SET_VARS({{"vconstant", "scalar"}, {"vrepeat", "scalar"}});
    SET_PRE_VARS({{"pre_vconstant", "scalar"}, {"pre_vrepeat", "scalar"}});
    SET_POST_VARS({{"post_vconstant", "scalar"}, {"post_vrepeat", "scalar"}});
};
IMPLEMENT_MODEL(WeightUpdateModel);

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
    model.setDT(0.1);
    model.setName("var_init_egp");
    
    // Parameters
    Neuron::VarValues neuronInit(initVar<ScalarEGP>(), initVar<RepeatVal>());
    CurrentSrc::VarValues currentSourceInit(initVar<ScalarEGP>(), initVar<RepeatVal>());
    PostsynapticModel::VarValues postsynapticInit(initVar<ScalarEGP>(), initVar<RepeatVal>());
    WeightUpdateModel::VarValues weightUpdateInit(initVar<ScalarEGP>(), initVar<PostRepeatVal>());
    WeightUpdateModel::PreVarValues weightUpdatePreInit(initVar<ScalarEGP>(), initVar<RepeatVal>());
    WeightUpdateModel::PostVarValues weightUpdatePostInit(initVar<ScalarEGP>(), initVar<RepeatVal>());
    
    // Neuron populations
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource1", 1, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource2", 100, {}, {});
    model.addNeuronPopulation<Neuron>("Pop", 100, {}, neuronInit);
    model.addCurrentSource<CurrentSrc>("CurrSource", "Pop", {}, currentSourceInit);

    // Dense synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource1", "Pop",
        {}, weightUpdateInit, weightUpdatePreInit, weightUpdatePostInit,
        {}, postsynapticInit);

    // Sparse synapse populations
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModel>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource2", "Pop",
        {}, weightUpdateInit, weightUpdatePreInit, weightUpdatePostInit,
        {}, postsynapticInit,
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>());

    model.setPrecision(GENN_FLOAT);
}
