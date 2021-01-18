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

    auto *pop = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("Pop", 10, {}, init);
    pop->setSpikeRecordingEnabled(true);
}
