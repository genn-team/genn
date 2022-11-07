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

bool hasVarRef(const std::vector<Models::WUVarReference> &varRefs, const std::string &targetName, const std::string &varName)
{
    return std::find_if(varRefs.cbegin(), varRefs.cend(), 
                        [&targetName, &varName](const Models::WUVarReference &r)
                        { 
                            return (r.getTargetName() == targetName) && (r.getVar().name == varName);
                        }) != varRefs.cend();
}
}   // Anonymous namespace
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
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom connectivity update
    auto *ccu1 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
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
    model.addCustomUpdate<Sum>("CustomUpdate3", "Test1", {}, {0.0}, cu3VarRefs);

    // Attach custom connectivity update
    auto *ccu3 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate3", "Test2", "Synapses3",
                                                                  {}, {1.0}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses4", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    // Attach custom connectivity update
    model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate42", "Test2", "Synapses4",
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
    ASSERT_TRUE(hasVarRef(ccu2DependentVars, "Synapses2", "g"));
   
    // Check dependencies for CCU3
    auto ccu3DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu3)->getDependentVariables();
    ASSERT_EQ(ccu3DependentVars.size(), 2);
    ASSERT_TRUE(hasVarRef(ccu3DependentVars, "Synapses3", "g"));
    ASSERT_TRUE(hasVarRef(ccu3DependentVars, "CustomUpdate3", "sum"));

    // Check dependencies for CCU4
    auto ccu4DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu4)->getDependentVariables();
    ASSERT_EQ(ccu4DependentVars.size(), 2);
    ASSERT_TRUE(hasVarRef(ccu4DependentVars, "Synapses4", "g"));
    ASSERT_TRUE(hasVarRef(ccu4DependentVars, "CustomConnectivityUpdate42", "a"));
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

    // Check synapse group variable has been removed from CCU12 dependent variables as it's manually referenced
    auto ccu12DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu12)->getDependentVariables();
    ASSERT_EQ(ccu12DependentVars.size(), 2);
    ASSERT_TRUE(hasVarRef(ccu12DependentVars, "CustomUpdate0", "sum"));
    ASSERT_TRUE(hasVarRef(ccu12DependentVars, "CustomConnectivityUpdate0", "a"));
                
    // Check custom update variable has been removed from CCU22 dependent variables as it's manually referenced
    auto ccu22DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu22)->getDependentVariables();
    ASSERT_EQ(ccu22DependentVars.size(), 2);
    ASSERT_TRUE(hasVarRef(ccu22DependentVars, "Synapses1", "g"));
    ASSERT_TRUE(hasVarRef(ccu22DependentVars, "CustomConnectivityUpdate1", "a"));
    
    // Check custom connectivity update variable has been removed from CCU32 dependent variables as it's manually referenced
    auto ccu32DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu32)->getDependentVariables();
    ASSERT_EQ(ccu32DependentVars.size(), 2);
    ASSERT_TRUE(hasVarRef(ccu32DependentVars, "Synapses2", "g"));
    ASSERT_TRUE(hasVarRef(ccu32DependentVars, "CustomUpdate2", "sum"));
}
