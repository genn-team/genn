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

class StaticPulseDuplicate : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseDuplicate, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY_DUPLICATE}});

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
    model.setBatchSize(1);
    
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        0.0,    // 2 - Vrest
        0.0,    // 3 - Vreset
        20.0,   // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);   // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        0.0,     // 0 - V
        0.0);   // 1 - RefracTime
        
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
    model.addNeuronPopulation<NeuronModels::LIF>("Post", 62 * 62 * 4, lifParams, lifInit);
    
    model.addSynapsePopulation<StaticPulseDuplicate, PostsynapticModels::DeltaCurr>(
        "Kernel", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Pre", "Post",
        {}, weightUpdateInit, {}, {},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>(convParams));

    model.setPrecision(GENN_FLOAT);
}
