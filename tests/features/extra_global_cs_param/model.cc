//--------------------------------------------------------------------------
/*! \file extra_global_cs_param/model.cc

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

//----------------------------------------------------------------------------
// CS
//----------------------------------------------------------------------------
class CS : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CS, 0, 0);

    SET_INJECTION_CODE("$(injectCurrent, ($(id) == (int)$(k)) ? 1.0 : 0.0);\n");

    SET_EXTRA_GLOBAL_PARAMS({{"k", "unsigned int"}});
};
IMPLEMENT_MODEL(CS);

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
  model.setName("extra_global_cs_param");

  model.addNeuronPopulation<Neuron>("pop", 10, {}, Neuron::VarValues(0.0));
  model.addCurrentSource<CS>("cs", "pop", {}, {});
  model.setPrecision(GENN_FLOAT);
}
