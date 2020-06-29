//--------------------------------------------------------------------------
/*! \file decode_matrix_cont_den_delay_individualg_ragged/model.cc

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

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PreNeuron);

//----------------------------------------------------------------------------
// PostNeuron
//----------------------------------------------------------------------------
class PostNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(PostNeuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(PostNeuron);

//---------------------------------------------------------------------------
// ContinuousDendriticDelay
//---------------------------------------------------------------------------
class ContinuousDendriticDelay : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(ContinuousDendriticDelay, 0, 2);

    SET_VARS({{"g", "scalar"},
              {"d", "uint8_t"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSynDelay, $(g) * $(x_pre), $(d));\n");
};
IMPLEMENT_MODEL(ContinuousDendriticDelay);


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
    model.setName("decode_matrix_cont_den_delay_individualg_ragged");

    // Static synapse parameters
    ContinuousDendriticDelay::VarValues staticSynapseInit(
        1.0,                    // 0 - Wij (nA)
        uninitialisedVar());    // 1 - Dij (timestep)

    model.addNeuronPopulation<PreNeuron>("Pre", 10, {}, PreNeuron::VarValues(0.0));
    model.addNeuronPopulation<PostNeuron>("Post", 1, {}, PostNeuron::VarValues(0.0));


    auto *syn = model.addSynapsePopulation<ContinuousDendriticDelay, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});
    syn->setMaxDendriticDelayTimesteps(10);
    syn->setMaxConnections(1);

    model.setPrecision(GENN_FLOAT);
}
