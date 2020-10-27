//--------------------------------------------------------------------------
/*! \file spike_recording/model.cc

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
    model.setName("spike_recording");
    
    NeuronModels::SpikeSourceArray::VarValues varInit(uninitialisedVar(), uninitialisedVar());
    auto *pop = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("Pop", 100, {}, varInit);
    pop->setSpikeRecordingEnabled(true);
}
