//--------------------------------------------------------------------------
/*! \file current_source_rng_normal/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_SIM_CODE("$(x)= $(Isyn);\n");

    SET_VARS({{"x", "scalar"}});
};
IMPLEMENT_MODEL(Neuron);


void modelDefinition(ModelSpec &model)
{
#ifdef OPENCL_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = OPENCL_DEVICE;
#endif
#ifdef OPENCL_PLATFORM
    GENN_PREFERENCES.manualPlatformID = OPENCL_PLATFORM;
#endif
    CurrentSourceModels::GaussianNoise::ParamValues paramVals(
        0.0,        // 2 - mean
        1.0);       // 3 - standard deviation

    model.setDT(0.1);
    model.setName("current_source_rng_normal");

    model.addNeuronPopulation<Neuron>("Pop", 1000, {}, Neuron::VarValues(0.0));

    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("CurrentSource",
                                                               "Pop",
                                                               paramVals, {});

    model.setPrecision(GENN_FLOAT);
}
