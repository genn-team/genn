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
    DECLARE_MODEL(TestNeuron, 0, 2);

    SET_VARS({{"Y","scalar", VarAccess::READ_ONLY}, {"Pi","scalar", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestNeuron);

class Softmax1 : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(Softmax1, 0, 1, 1);
    
     SET_UPDATE_CODE(
        "$(MaxY) = $(Y);\n");
        
    SET_VARS({{"MaxY", "scalar", VarAccess::REDUCE_NEURON_MAX}});
    SET_VAR_REFS({{"Y", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Softmax1);

class Softmax2 : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(Softmax2, 0, 1, 2);
    
     SET_UPDATE_CODE(
        "$(SumExpPi) = exp($(Y) - $(MaxY));\n");
        
    SET_VARS({{"SumExpPi", "scalar", VarAccess::REDUCE_NEURON_SUM}});
    SET_VAR_REFS({{"Y", "scalar", VarAccessMode::READ_ONLY}, 
                  {"MaxY", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Softmax2);

class Softmax3 : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(Softmax3, 0, 0, 4);
    
     SET_UPDATE_CODE(
        "$(Pi) = exp($(Y) - $(MaxY)) / $(SumExpPi);\n");
        
    SET_VAR_REFS({{"Y", "scalar", VarAccessMode::READ_ONLY}, 
                  {"MaxY", "scalar", VarAccessMode::READ_ONLY},
                  {"SumExpPi", "scalar", VarAccessMode::READ_ONLY},
                  {"Pi", "scalar", VarAccessMode::READ_WRITE}});
};
IMPLEMENT_MODEL(Softmax3);

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

    auto *ng = model.addNeuronPopulation<TestNeuron>("Neuron", 50, {}, {uninitialisedVar(), 0.0});

    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    Softmax1::VarReferences softmax1VarReferences(createVarRef(ng, "Y"));   // Y
    auto *softmax1 = model.addCustomUpdate<Softmax1>("Softmax1", "Softmax1",
                                                     {}, {0.0}, softmax1VarReferences);
    
    Softmax2::VarReferences softmax2VarReferences(createVarRef(ng, "Y"),            // Y
                                                  createVarRef(softmax1, "MaxY"));  // MaxY
    auto *softmax2 = model.addCustomUpdate<Softmax2>("Softmax2", "Softmax2",
                                                     {}, {0.0}, softmax2VarReferences);
    
    Softmax3::VarReferences softmax3VarReferences(createVarRef(ng, "Y"),                // Y
                                                  createVarRef(softmax1, "MaxY"),       // MaxY
                                                  createVarRef(softmax2, "SumExpPi"),   // SumExpPi
                                                  createVarRef(ng, "Pi"));              // Pi
    model.addCustomUpdate<Softmax3>("Softmax3", "Softmax3",
                                  {}, {}, softmax3VarReferences);
}