#include "modelSpec.h"

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
    model.setName("decode_matrix_conn_gen_proceduralg_procedural_kernel");

    // Synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        initVar<InitVarSnippet::Kernel>());    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 64 * 64, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("PrePool", 128 * 128, {}, {});
    model.addNeuronPopulation<PostNeuron>("Post", 62 * 62 * 2, {}, PostNeuron::VarValues(0.0));
    model.addNeuronPopulation<PostNeuron>("PostPool", 62 * 62 * 2, {}, PostNeuron::VarValues(0.0));
   
    InitSparseConnectivitySnippet::Conv2D::ParamValues convParams(
        3, 3,       // conv_kh, conv_kw
        1, 1,       // conv_sh, conv_sw
        0, 0,       // conv_padh, conv_padw
        64, 64, 1,  // conv_ih, conv_iw, conv_ic
        62, 62, 2); // conv_oh, conv_ow, conv_oc
    
    InitSparseConnectivitySnippet::AvgPoolConv2D::ParamValues avgPoolConvParams(
        2, 2,           // pool_kh, pool_kw
        2, 2,           // pool_sh, pool_sw
        0, 0,           // pool_padh, pool_padw
        128, 128, 1,    // pool_ih, pool_iw, pool_ic
        3, 3,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        64, 64,         // conv_ih, conv_iw
        62, 62, 2);     // conv_oh, conv_ow, conv_oc
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY, "Pre", "Post",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>({convParams}));
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynPool", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY, "PrePool", "PostPool",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::AvgPoolConv2D>({avgPoolConvParams}));
}
