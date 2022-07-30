//--------------------------------------------------------------------------
/*! \file decode_matrix_event_individualg_dense/model.cc

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
    DECLARE_MODEL(PreNeuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 1);

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PostNeuron);

//---------------------------------------------------------------------------
// Synapse
//---------------------------------------------------------------------------
class Synapse : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Synapse, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_EVENT_THRESHOLD_CONDITION_CODE("fabs($(t)/0.2f - (int) ($(t)/0.2f)) < 1e-5"); 

    SET_EVENT_CODE("$(addToPre, $(g) * $(x_post));\n");
};
IMPLEMENT_MODEL(Synapse);


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
    model.setName("pre_output_decode_matrix_event_individualg_dense");

    // synapse parameters
    Synapse::VarValues staticSynapseInit(uninitialisedVar());    // 0 - Wij (nA)

    model.addNeuronPopulation<PreNeuron>("Pre", 4, {}, PreNeuron::VarValues(0.0));
    model.addNeuronPopulation<PostNeuron>("Post", 10, {}, PostNeuron::VarValues(0.0));


    model.addSynapsePopulation<Synapse, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.setPrecision(GENN_FLOAT);
}
