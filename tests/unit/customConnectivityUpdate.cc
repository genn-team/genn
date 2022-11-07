// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

namespace
{
class Sum : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum, 0, 1, 1);

    SET_UPDATE_CODE("$(sum) += $(a);\n");

    SET_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Sum);

class RemoveSynapse : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapse, 0, 1, 0, 0, 0, 0, 0);
    
    SET_VARS({{"a", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == ($(id_pre) + 1)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapse);

class RemoveSynapseVarRef : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseVarRef, 0, 1, 0, 0, 1, 0, 0);
    
    SET_VARS({{"a", "scalar"}});
    SET_VAR_REFS({{"b", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == ($(id_pre) + 1)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapseVarRef);
}
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, DependentVariables)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom connectivity update
    auto *ccu1 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom connectivity update
    auto *ccu2 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate2", "Test2", "Synapses2",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    auto *sg3 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses3", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom update model
    Sum::WUVarReferences cu3VarRefs(createWUVarRef(sg3, "g"));
    auto *cu3 = model.addCustomUpdate<Sum>("CustomUpdate3", "Test1", {}, {0.0}, cu3VarRefs);

    // Attach custom connectivity update
    auto *ccu3 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate3", "Test2", "Synapses3",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    auto *sg4 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses4", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom connectivity update
    auto *ccu42 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate42", "Test2", "Synapses4",
                                                                   {}, {1.0}, {}, {},
                                                                   {}, {}, {});

    // Attach custom connectivity update
    auto *ccu4 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate4", "Test1", "Synapses4",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    model.finalize();

    // Check no dependencies for CCU1
    auto ccu1DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu1)->getDependentVariables();
    ASSERT_TRUE(ccu1DependentVars.empty());

    // Check dependencies for CCU2
    auto ccu2DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu2)->getDependentVariables();
    ASSERT_EQ(ccu2DependentVars.size(), 1);
    ASSERT_EQ(ccu2DependentVars[0].getTargetName(), "Synapses2");
    ASSERT_EQ(ccu2DependentVars[0].getVar().name, "g");
   
    // Check dependencies for CCU3
    auto ccu3DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu3)->getDependentVariables();
    ASSERT_EQ(ccu3DependentVars.size(), 2);
    ASSERT_TRUE(std::find_if(ccu3DependentVars.cbegin(), ccu3DependentVars.cend(), 
                             [](const Models::WUVarReference &r)
                             { 
                                 return (r.getTargetName() == "Synapses3") && (r.getVar().name == "g");
                             }) != ccu3DependentVars.cend());
    ASSERT_TRUE(std::find_if(ccu3DependentVars.cbegin(), ccu3DependentVars.cend(), 
                             [](const Models::WUVarReference &r)
                             { 
                                 return (r.getTargetName() == "CustomUpdate3") && (r.getVar().name == "sum");
                             }) != ccu3DependentVars.cend());

    // Check dependencies for CCU4
    auto ccu4DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu4)->getDependentVariables();
    ASSERT_EQ(ccu4DependentVars.size(), 2);
    ASSERT_TRUE(std::find_if(ccu4DependentVars.cbegin(), ccu4DependentVars.cend(), 
                             [](const Models::WUVarReference &r)
                             { 
                                 return (r.getTargetName() == "Synapses4") && (r.getVar().name == "g");
                             }) != ccu4DependentVars.cend());
    ASSERT_TRUE(std::find_if(ccu4DependentVars.cbegin(), ccu4DependentVars.cend(), 
                             [](const Models::WUVarReference &r)
                             { 
                                 return (r.getTargetName() == "CustomConnectivityUpdate42") && (r.getVar().name == "a");
                             }) != ccu4DependentVars.cend());
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, DependentVariablesManualReferences)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create three synapse groups, each with an attached custom update and custom connectivity update
    SynapseGroup *synapseGroups[3] = {nullptr, nullptr, nullptr};
    CustomUpdateWU *customUpdates[3] = {nullptr, nullptr, nullptr};
    CustomConnectivityUpdate *customConnectivityUpdates[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; i++) {
        // Create synapse group with individual weights
        synapseGroups[i] = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
            "Synapses" + std::to_string(i), SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
            "Pre", "Post",
            {}, {1.0},
            {}, {});

        // Attach custom update model
        Sum::WUVarReferences cu3VarRefs(createWUVarRef(synapseGroups[i], "g"));
        customUpdates[i] = model.addCustomUpdate<Sum>("CustomUpdate" + std::to_string(i), "Test1", {}, {0.0}, cu3VarRefs);

        // Attach custom connectivity update
        customConnectivityUpdates[i] = model.addCustomConnectivityUpdate<RemoveSynapse>(
            "CustomConnectivityUpdate" + std::to_string(i), "Test1", "Synapses" + std::to_string(i),
            {}, {1.0}, {}, {},
            {}, {}, {});
    }

    // Add another custom connectivity update to first synapse group with a manual reference to synapse group variable
    RemoveSynapseVarRef::VarReferences ccu12VarRefs(createWUVarRef(synapseGroups[0], "g"));
    auto *ccu12 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate12", "Test2", "Synapses0",
                                                                         {}, {1.0}, {}, {},
                                                                         ccu12VarRefs, {}, {});

    // Add another custom connectivity update to second synapse group with a manual reference to custom update variable
    RemoveSynapseVarRef::VarReferences ccu22VarRefs(createWUVarRef(customUpdates[1], "sum"));
    auto *ccu22 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate22", "Test2", "Synapses1",
                                                                         {}, {1.0}, {}, {},
                                                                         ccu22VarRefs, {}, {});

    // Add another custom connectivity update to third synapse group with a manual reference to custom connectivity update variable
    RemoveSynapseVarRef::VarReferences ccu32VarRefs(createWUVarRef(customConnectivityUpdates[2], "a"));
    auto *ccu32 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate32", "Test2", "Synapses2",
                                                                         {}, {1.0}, {}, {},
                                                                         ccu32VarRefs, {}, {});

    model.finalize();

    // Check dependencies for CCU12
    auto ccu12DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu12)->getDependentVariables();
    
    // Check dependencies for CCU12
    auto ccu22DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu22)->getDependentVariables();

    // Check dependencies for CCU12
    auto ccu32DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu32)->getDependentVariables();

    while(false);
    // Create model with 4 synapse groups, each with individualg, custom wu update, custom connectivity update

    // Attach custom connectivity update to each one:
    // 1) No var references
    // 2) Var reference to synapse var
    // 3) Var reference to custom update var
    // 4) Var reference to custom connectivity update var
    
    // Then check that dependent variables contains everything expected
}