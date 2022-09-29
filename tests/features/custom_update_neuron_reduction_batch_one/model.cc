//--------------------------------------------------------------------------
/*! \file custom_update_neuron_reduction_batch_one/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class TestNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TestNeuron, 0, 1);

    SET_VARS({{"V","scalar", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestNeuron);

class Reduce : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(Reduce, 0, 2, 1);
    
    SET_UPDATE_CODE(
        "$(Sum) = $(V);\n"
        "$(Max) = $(V);\n");

    SET_VARS({{"Sum", "scalar", VarAccess::REDUCE_NEURON_SUM}, {"Max", "scalar", VarAccess::REDUCE_NEURON_MAX}});
    SET_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}})
};
IMPLEMENT_MODEL(Reduce);

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
    model.setName("custom_update_neuron_reduction_batch_one");

    auto *ng = model.addNeuronPopulation<TestNeuron>("Neuron", 50, {}, {uninitialisedVar()});

    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    Reduce::VarReferences neuronReduceVarReferences(createVarRef(ng, "V")); // V
    model.addCustomUpdate<Reduce>("NeuronReduce", "Test",
                                  {}, {0.0, 0.0}, neuronReduceVarReferences);
}