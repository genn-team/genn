//--------------------------------------------------------------------------
/*! \file batch_var_init/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class Gradient3D : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Gradient3D, 0);

    SET_CODE("$(value) = sqrt((scalar)($(id_kernel_0) * $(id_kernel_0)) + (scalar)($(id_kernel_1) * $(id_kernel_1)) + (scalar)($(id_kernel_3) * $(id_kernel_3)));");
};
IMPLEMENT_SNIPPET(Gradient3D);

class PSM : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PSM, 0, 1);

    SET_CURRENT_CONVERTER_CODE("$(inSyn); $(inSyn) = 0");
    SET_VARS({{"psm", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}});
};
IMPLEMENT_MODEL(PSM);

class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_VARS({{"V", "scalar", VarAccess::READ_ONLY_SHARED_NEURON}});
};
IMPLEMENT_MODEL(Neuron);

class StaticPulseDuplicate : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseDuplicate, 0, 1, 1, 1);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY_DUPLICATE}});
    SET_PRE_VARS({{"pre", "int", VarAccess::READ_ONLY_SHARED_NEURON}});
    SET_POST_VARS({{"post", "int", VarAccess::READ_ONLY_SHARED_NEURON}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");
};
IMPLEMENT_MODEL(StaticPulseDuplicate);

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
    model.setSeed(2346679);
    model.setDT(0.1);
    model.setName("batch_var_init");
    model.setBatchSize(10);

    // Connectivity parameters 
    InitSparseConnectivitySnippet::Conv2D::ParamValues convParams(
        3, 3,       // conv_kh, conv_kw
        1, 1,       // conv_sh, conv_sw
        0, 0,       // conv_padh, conv_padw
        64, 64, 1,  // conv_ih, conv_iw, conv_ic
        62, 62, 4); // conv_oh, conv_ow, conv_oc
    
    StaticPulseDuplicate::VarValues weightUpdateInit(
        initVar<Gradient3D>());
    
    // Neuron populations
    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 64 * 64 * 1, {}, {});
    model.addNeuronPopulation<Neuron>("Post", 62 * 62 * 4, {}, {33});
    
    model.addSynapsePopulation<StaticPulseDuplicate, PSM>(
        "Kernel", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Pre", "Post",
        {}, weightUpdateInit, {13}, {31},
        {}, {33},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>(convParams));

    model.setPrecision(GENN_FLOAT);
}
