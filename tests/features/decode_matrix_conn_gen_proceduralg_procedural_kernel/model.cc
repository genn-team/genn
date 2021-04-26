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

class Deconv2D : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Deconv2D, 12);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});

    SET_ROW_BUILD_STATE_VARS({{"outRow", "int", "($(id_pre)/ (int)$(conv_oc)) / (int)$(conv_ow)"},
                              {"outCol", "int", "($(id_pre)/ (int)$(conv_oc)) % (int)$(conv_ow)"},
                              {"outChan", "int", "$(id_pre) % (int)$(conv_oc)"},
                              {"strideRow", "int", "(outRow * (int)$(conv_sh)) - (int)$(conv_padh)"},
                              {"strideCol", "int", "(outCol * (int)$(conv_sw)) - (int)$(conv_padw)"},
                              {"inRow", "int", "min((int)$(conv_ih), max(0, (outRow * (int)$(conv_sh)) - (int)$(conv_padh)))"},
                              {"maxInRow", "int", "min((int)$(conv_ih), max(0, (outRow * (int)$(conv_sh)) + (int)$(conv_kh) - (int)$(conv_padh)))"},
                              {"minInCol", "int", "min((int)$(conv_iw), max(0, (outCol * (int)$(conv_sw)) - (int)$(conv_padw)))"},
                              {"maxInCol", "int", "min((int)$(conv_iw), max(0, (outCol * (int)$(conv_sw)) + (int)$(conv_kw) - (int)$(conv_padw)))"}});

    SET_ROW_BUILD_CODE(
        "if($(inRow) == $(maxInRow)) {\n"
        "   $(endRow);\n"
        "}\n"
        "const int kernRow = $(inRow) - $(strideRow);\n"
        "for(int inCol = $(minInCol); inCol < $(maxInCol); inCol++) {\n"
        "    const int kernCol = inCol - $(strideCol);\n"
        "    for(unsigned int inChan = 0; inChan < (unsigned int)$(conv_ic); inChan++) {\n"
        "        const int idPre = (($(inRow) * (int)$(conv_iw) * (int)$(conv_ic)) +\n"
        "                            (inCol * (int)$(conv_ic)) +\n"
        "                            inChan);\n"
        "        $(addSynapse, idPre, kernRow, kernCol, inChan, $(outChan));\n"
        "    }\n"
        "}\n"
        "$(inRow)++;\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            const unsigned int conv_kh = (unsigned int)pars[0];
            const unsigned int conv_kw = (unsigned int)pars[1];
            const unsigned int conv_sh = (unsigned int)pars[2];
            const unsigned int conv_sw = (unsigned int)pars[3];
            const unsigned int conv_oc = (unsigned int)pars[11];
            return (conv_kh / conv_sh) * (conv_kw / conv_sw) * conv_oc;
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::vector<double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars[0], (unsigned int)pars[1],
                    (unsigned int)pars[8], (unsigned int)pars[11]};
        });
};
IMPLEMENT_MODEL(Deconv2D);

class Graded : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(Graded, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_EVENT_CODE("$(addToInSyn, $(g) * $(x_pre));\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(x_pre) > 0.0");
};
IMPLEMENT_MODEL(Graded);

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
    model.addNeuronPopulation<PostNeuron>("Deconv", 64 * 64, {}, PostNeuron::VarValues(0.0));
    
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
    
    model.addSynapsePopulation<Graded, PostsynapticModels::DeltaCurr>(
        "SynDeconv", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY, "Post", "Deconv",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<Deconv2D>({convParams}));
        
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynPool", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY, "PrePool", "PostPool",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::AvgPoolConv2D>({avgPoolConvParams}));
}
