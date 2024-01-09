//--------------------------------------------------------------------------
/*! \file batch_prev_pre_spike_time_in_sim/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// PreNeuron
//----------------------------------------------------------------------------
class PreNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PreNeuron, 0, 0);

    SET_THRESHOLD_CONDITION_CODE("$(t) >= (scalar)($(id) + ($(batch) * 10)) && fmodf($(t) - (scalar)($(id) + ($(batch) * 10)), 20.0f)< 1e-4");
    SET_NEEDS_AUTO_REFRACTORY(false);
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 0);
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

    SET_SIM_CODE("$(w)= $(prev_sT_pre);");
    SET_NEEDS_PREV_PRE_SPIKE_TIME(true);
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
    model.setDT(1.0);
    model.setName("batch_prev_pre_spike_time_in_sim");
    model.setBatchSize(2);
    
    model.addNeuronPopulation<PreNeuron>("pre", 10, {}, {});
    model.addNeuronPopulation<PreNeuron>("preDelay", 10, {}, {});
    model.addNeuronPopulation<PostNeuron>("post", 10, {}, {});
    model.addNeuronPopulation<PostNeuron>("postDelay", 10, {}, {});

    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "pre", "post",
        {}, WeightUpdateModel::VarValues(-std::numeric_limits<float>::max()),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));
    
    model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "synDelay", SynapseMatrixType::SPARSE_INDIVIDUALG, 20, "preDelay", "postDelay",
        {}, WeightUpdateModel::VarValues(-std::numeric_limits<float>::max()),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));
    model.setPrecision(GENN_FLOAT);
}
