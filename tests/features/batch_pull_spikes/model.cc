//--------------------------------------------------------------------------
/*! \file batch_pull_spikes/model.cc

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
    model.setBatchSize(10);
    model.setDT(1.0);
    model.setName("batch_pull_spikes");
    
    NeuronModels::SpikeSourceArray::VarValues init(
        uninitialisedVar(),     // startSpike
        uninitialisedVar());    // endSpike
    
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
    
    auto *pop = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("Pop", 10, {}, init);
    auto *popDelay = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("PopDelay", 10, {}, init);
    model.addNeuronPopulation<NeuronModels::LIF>("Post", 10, lifParams, lifInit);
    
    pop->setSpikeRecordingEnabled(true);
    popDelay->setSpikeRecordingEnabled(true);
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "PopDelay_Post", SynapseMatrixType::SPARSE_GLOBALG, 5,
        "PopDelay", "Post",
        {}, {1.0},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
}
