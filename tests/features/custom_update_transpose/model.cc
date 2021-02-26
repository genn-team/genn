//--------------------------------------------------------------------------
/*! \file custom_update_transpose/model.cc

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
    model.setDT(1.0);
    model.setName("custom_update_transpose");

    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -70.0,  // 2 - Vrest
        -70.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);   // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -70.0,  // 0 - V
        0.0);   // 1 - RefracTime

    InitVarSnippet::Normal::ParamValues gDist(
        0.0,    // 0 - mean
        1.0);   // 1 - sd

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseVarInit(
        initVar<InitVarSnippet::Normal>(gDist));    // 0 - Wij (nA)

    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 100, {}, {});
    model.addNeuronPopulation<NeuronModels::LIF>("Neuron", 100, lifParams, lifInit);
    auto *denseSG1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Dense1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, staticSynapseVarInit,
        {}, {});
    auto *denseSG2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Dense2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, staticSynapseVarInit,
        {}, {});
    auto *transposeSG1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Transpose1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Neuron", "SpikeSource",
        {}, {0.0},
        {}, {});
    auto *transposeSG2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Transpose2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Neuron", "SpikeSource",
        {}, {0.0},
        {}, {});

    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    CustomUpdateModels::Transpose::WUVarReferences wuDenseVarReferences1(createWUVarRef(denseSG1, "g", transposeSG1, "g")); // R
    model.addCustomUpdate<CustomUpdateModels::Transpose>("WUDenseTranspose1", "Test",
                                                         {}, {}, wuDenseVarReferences1);
    CustomUpdateModels::Transpose::WUVarReferences wuDenseVarReferences2(createWUVarRef(denseSG2, "g", transposeSG2, "g")); // R
    model.addCustomUpdate<CustomUpdateModels::Transpose>("WUDenseTranspose2", "Test",
                                                         {}, {}, wuDenseVarReferences2);
}
