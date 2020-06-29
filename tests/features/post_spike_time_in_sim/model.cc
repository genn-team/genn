//--------------------------------------------------------------------------
/*! \file post_spike_time_in_sim/model.cc

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

    SET_THRESHOLD_CONDITION_CODE("true");
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

    SET_THRESHOLD_CONDITION_CODE("$(t) >= (scalar)$(id) && fmodf($(t) - (scalar)$(id), 10.0f)< 1e-4");
    SET_NEEDS_AUTO_REFRACTORY(false);
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

void modelDefinition(ModelSpec &model)
{
#ifdef OPENCL_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = OPENCL_DEVICE;
#endif
#ifdef OPENCL_PLATFORM
    GENN_PREFERENCES.manualPlatformID = OPENCL_PLATFORM;
#endif
    model.setDT(1.0);
    model.setName("post_spike_time_in_sim");

    model.addNeuronPopulation<PreNeuron>("pre", 10, {}, {});
    model.addNeuronPopulation<PostNeuron>("post", 10, {}, {});

    auto *syn = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "pre", "post",
        {}, WeightUpdateModel::VarValues(0.0),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));

    syn->setBackPropDelaySteps(20);

    model.setPrecision(GENN_FLOAT);
}
