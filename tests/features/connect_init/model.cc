//--------------------------------------------------------------------------
/*! \file connect_init/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"


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
    model.setName("connect_init");

    NeuronModels::LIF::ParamValues lifParams(
                0.25,   // 0 - C
                10.0,   // 1 - TauM
                -65.0,  // 2 - Vrest
                -65.0,  // 3 - Vreset
                -50.0,  // 4 - Vthresh
                0.0,    // 5 - Ioffset
                2.0);   // 6 - TauRefrac
    NeuronModels::LIF::VarValues lifInit(
        -65.0,  // 0 - V
        0.0);   // 1 - RefracTime
    
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(0.1);
                                
    InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement::ParamValues fixedNumTotalParams(1000);
    InitSparseConnectivitySnippet::FixedNumberPostWithReplacement::ParamValues fixedNumPostParams(10);
    
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 100, {}, {});
    model.addNeuronPopulation<NeuronModels::LIF>("LIF", 100, lifParams, lifInit);
    
    // Fixed number total connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "FixedNumberTotal", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "LIF",
        {}, staticSynapseInit, {}, {},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement>(fixedNumTotalParams));
    
    // Fixed number post connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "FixedNumberPost", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "LIF",
        {}, staticSynapseInit, {}, {},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumPostParams));
}
