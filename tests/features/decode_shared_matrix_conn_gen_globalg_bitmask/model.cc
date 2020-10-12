//--------------------------------------------------------------------------
/*! \file decode_shared_matrix_conn_gen_globalg_bitmask/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Decoder
//----------------------------------------------------------------------------
class Decoder : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Decoder, 0);

    SET_ROW_BUILD_CODE(
        "if(j < $(num_post)) {\n"
        "   const unsigned int jValue = (1 << j);\n"
        "   if((($(id_pre) + 1) & jValue) != 0)\n"
        "   {\n"
        "       $(addSynapse, j);\n"
        "   }\n"
        "}\n"
        "else {\n"
        "   $(endRow);\n"
        "}\n"
        "j++;\n");
    SET_ROW_BUILD_STATE_VARS({{"j", "unsigned int", 0}});
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
    model.setName("decode_shared_matrix_conn_gen_globalg_bitmask");

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(1.0);    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<Neuron>("Post1", 4, {}, Neuron::VarValues(0.0));
    model.addNeuronPopulation<Neuron>("Post2", 4, {}, Neuron::VarValues(0.0));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY, "Pre", "Post1",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<Decoder>({}));
    model.addSlaveSynapsePopulation<PostsynapticModels::DeltaCurr>(
        "Syn2", "Syn1", NO_DELAY, "Pre", "Post2",
        {}, {});
        
    model.setPrecision(GENN_FLOAT);
}
