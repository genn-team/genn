//--------------------------------------------------------------------------
/*! \file custom_update_delay/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class TestNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TestNeuron, 0, 2);

    SET_VARS({{"V", "scalar"}, {"U", "scalar"}});
};
IMPLEMENT_MODEL(TestNeuron);

class TestWUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUM, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(V_pre) * $(g));\n");
};
IMPLEMENT_MODEL(TestWUM);

class TestWUMPre : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUMPre, 0, 1, 1, 0);
    
    SET_PRE_VARS({{"pre", "scalar"}});
    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    
    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(pre) * $(g));\n");
    SET_PRE_DYNAMICS_CODE("$(pre) = $(V_pre);\n");
};
IMPLEMENT_MODEL(TestWUMPre);

class SetTime : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(SetTime, 0, 0, 1);
    
    SET_UPDATE_CODE(
        "$(R) = $(t);\n");

    SET_VAR_REFS({{"R", "scalar", VarAccessMode::READ_WRITE}})
};
IMPLEMENT_MODEL(SetTime);

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
    model.setName("custom_update_delay");

    auto *pre = model.addNeuronPopulation<TestNeuron>("Pre", 100, {}, {0.0, 0.0});
    model.addNeuronPopulation<TestNeuron>("Post", 100, {}, {0.0, 0.0});
    model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::DENSE_INDIVIDUALG, 5,
        "Pre", "Post",
        {}, {0.0},
        {}, {});
    auto *sg = model.addSynapsePopulation<TestWUMPre, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::DENSE_INDIVIDUALG, 5,
        "Pre", "Post",
        {}, {0.0}, {0.0}, {},
        {}, {});

    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    SetTime::VarReferences neuronDelayVarReferences(createVarRef(pre, "V")); // R
    model.addCustomUpdate<SetTime>("NeuronDelaySetTime", "Test",
                                   {}, {}, neuronDelayVarReferences);
    
    SetTime::VarReferences wuPreDelayVarReferences(createWUPreVarRef(sg, "pre")); // R
    model.addCustomUpdate<SetTime>("WUPreDelaySetTime", "Test",
                                   {}, {}, wuPreDelayVarReferences);
                                   
    SetTime::VarReferences neuronNoDelayVarReferences(createVarRef(pre, "U")); // R
    model.addCustomUpdate<SetTime>("NeuronNoDelaySetTime", "Test",
                                   {}, {}, neuronNoDelayVarReferences);
}