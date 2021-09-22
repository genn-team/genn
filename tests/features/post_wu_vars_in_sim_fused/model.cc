//--------------------------------------------------------------------------
/*! \file post_wu_vars_in_sim_fused/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include <limits>

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
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModel, 0, 1, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_POST_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_POST_SPIKE_CODE("$(s) = $(t);\n");
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
    model.setName("post_wu_vars_in_sim_fused");
    model.setMergePrePostWeightUpdateModels(true);
    
    model.addNeuronPopulation<PreNeuron>("pre1", 10, {}, {});
    model.addNeuronPopulation<PreNeuron>("pre2", 10, {}, {});
    model.addNeuronPopulation<PostNeuron>("post", 10, {}, {});

    auto *syn1 = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "syn1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "pre1", "post",
        {}, WeightUpdateModel::VarValues(0.0), {}, WeightUpdateModel::PostVarValues(std::numeric_limits<float>::lowest()),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));
    auto *syn2 = model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
        "syn2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "pre2", "post",
        {}, WeightUpdateModel::VarValues(0.0), {}, WeightUpdateModel::PostVarValues(std::numeric_limits<float>::lowest()),
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>({}));

    syn1->setBackPropDelaySteps(20);
    syn2->setBackPropDelaySteps(20);

    model.setPrecision(GENN_FLOAT);
}
