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
    model.setName("decode_matrix_conn_gen_kernelg_toeplitz");

    // Synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        uninitialisedVar());    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 64 * 64, {}, {});
    model.addNeuronPopulation<PostNeuron>("PostHorz", 62 * 62, {}, PostNeuron::VarValues(0.0));
    model.addNeuronPopulation<PostNeuron>("PostVert", 62 * 62, {}, PostNeuron::VarValues(0.0));
   
    InitToeplitzConnectivitySnippet::Conv2D::ParamValues convParams(
        3, 3,       // conv_kh, conv_kw
        64, 64, 1,  // conv_ih, conv_iw, conv_ic
        62, 62, 1); // conv_oh, conv_ow, conv_oc
        
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynHorz", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY, "Pre", "PostHorz",
        {}, staticSynapseInit,
        {}, {},
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>({convParams}));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynVert", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY, "Pre", "PostVert",
        {}, staticSynapseInit,
        {}, {},
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>({convParams}));
}
