//--------------------------------------------------------------------------
/*! \file decode_matrix_individualg_dense/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Decoder
//----------------------------------------------------------------------------
class Decoder : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Decoder, 0);

    SET_CODE(
        "const unsigned int j_value = (1 << $(id_post));\n"
        "$(value) = ((($(id_pre) + 1) & j_value) != 0) ? 1.0f : 0.0f;\n")
};
IMPLEMENT_SNIPPET(Decoder);

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};

IMPLEMENT_MODEL(Neuron);


void modelDefinition(ModelSpec &model)
{
#ifdef OPENCL_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = OPENCL_DEVICE;
#endif
#ifdef OPENCL_PLATFORM
    GENN_PREFERENCES.manualPlatformID = OPENCL_PLATFORM;
#endif
    model.setDT(0.1);
    model.setName("decode_matrix_conn_gen_individualg_dense");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(initVar<Decoder>());    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {});

    model.setPrecision(GENN_FLOAT);
}
