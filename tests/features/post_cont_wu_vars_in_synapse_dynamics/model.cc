//--------------------------------------------------------------------------
/*! \file post_cont_wu_vars_in_synapse_dynamics/model.cc

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
    DECLARE_MODEL(Neuron, 0, 1);

    SET_THRESHOLD_CONDITION_CODE("$(t) > 1e-4 && fmod($(t)+$(shift),1.0) < 1e-4");

    SET_VARS({{"shift", "scalar", VarAccess::READ_ONLY}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};

IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// WeightUpdateModel
//----------------------------------------------------------------------------
class WeightUpdateModel : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModel, 0, 1, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_POST_VARS({{"x", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(w)= $(x);");
    SET_POST_DYNAMICS_CODE("$(x) = $(t)+$(shift_post);\n");
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
    model.setName("post_cont_wu_vars_in_synapse_dynamics");

    model.addNeuronPopulation<Neuron>("pre", 10, {}, {uninitialisedVar()});
    model.addNeuronPopulation<Neuron>("post", 10, {}, {uninitialisedVar()});

    std::string synName= "syn";
    for (int i= 0; i < 10; i++)
    {
        std::string theName= synName + std::to_string(i);
        model.addSynapsePopulation<WeightUpdateModel, PostsynapticModels::DeltaCurr>(
            theName, SynapseMatrixType::DENSE_INDIVIDUALG, i, "pre", "post",
            {}, {0.0}, {}, {0.0},
            {}, {});
    }
    model.setPrecision(GENN_FLOAT);
}
