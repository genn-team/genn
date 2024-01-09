// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

using namespace GeNN;

namespace
{
class StaticPulseDendriticDelayReverse : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseDendriticDelayReverse);

    SET_VARS({{"d", "uint8_t", VarAccess::READ_ONLY}, {"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE("addToPostDelay(g, d);\n");
};
IMPLEMENT_SNIPPET(StaticPulseDendriticDelayReverse);

class Sum : public CustomUpdateModels::Base
{
    DECLARE_SNIPPET(Sum);

    SET_UPDATE_CODE("sum += a;\n");

    SET_CUSTOM_UPDATE_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_SNIPPET(Sum);

class RemoveSynapse : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapse);
    
    SET_VARS({{"a", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(id_post == (id_pre + 1)) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapse);

class RemoveSynapseVarRef : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapseVarRef);
    
    SET_VARS({{"a", "scalar"}});
    SET_VAR_REFS({{"b", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(id_post == (id_pre + 1)) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapseVarRef);

class RemoveSynapsePre : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapsePre);
    
    SET_VAR_REFS({{"g", "scalar"}});
    SET_PRE_VAR_REFS({{"threshLow", "scalar"}, {"threshHigh", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(g < threshLow || g > threshHigh) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapsePre);

class RemoveSynapsePost : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapsePost);
    
    SET_VAR_REFS({{"g", "scalar"}});
    SET_POST_VAR_REFS({{"threshLow", "scalar"}, {"threshHigh", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(g < threshLow || g > threshHigh) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapsePost);

class Cont : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(Cont);

    SET_VARS({{"g", "scalar"}});
    SET_PRE_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "addToPost(g * V);\n");
};
IMPLEMENT_SNIPPET(Cont);

class ContPost : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(ContPost);

    SET_VARS({{"g", "scalar"}});
    SET_POST_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "addToPost(g * V);\n");
};
IMPLEMENT_SNIPPET(ContPost);

/*bool hasVarRef(const std::vector<Models::WUVarReference> &varRefs, const std::string &targetName, const std::string &varName)
{
    return std::find_if(varRefs.cbegin(), varRefs.cend(), 
                        [&targetName, &varName](const Models::WUVarReference &r)
                        { 
                            return (r.getTargetName() == targetName) && (r.getVarName() == varName);
                        }) != varRefs.cend();
}*/
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, DependentVariables)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    // Attach custom connectivity update
    auto *ccu1 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    // Attach custom connectivity update
    auto *ccu2 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate2", "Test2", "Synapses2",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    auto *sg3 = model.addSynapsePopulation(
        "Synapses3", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    // Attach custom update model
    WUVarReferences cu3VarRefs{{"a", createWUVarRef(sg3, "g")}};
    model.addCustomUpdate<Sum>("CustomUpdate3", "Test1", {}, {{"sum", 0.0}}, cu3VarRefs);

    // Attach custom connectivity update
    auto *ccu3 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate3", "Test2", "Synapses3",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});

    // Create synapse group with individual weights
    model.addSynapsePopulation(
        "Synapses4", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    // Attach custom connectivity update
    model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate42", "Test2", "Synapses4",
                                                     {}, {{"a", 1.0}}, {}, {},
                                                     {}, {}, {});

    // Attach custom connectivity update
    auto *ccu4 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate4", "Test1", "Synapses4",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});

    model.finalise();

    // Check no dependencies for CCU1
    auto ccu1DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu1)->getDependentVariables();
    ASSERT_TRUE(ccu1DependentVars.empty());

    // Check dependencies for CCU2
    auto ccu2DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu2)->getDependentVariables();
    ASSERT_EQ(ccu2DependentVars.size(), 1);
    //ASSERT_TRUE(hasVarRef(ccu2DependentVars, "Synapses2", "g"));
   
    // Check dependencies for CCU3
    auto ccu3DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu3)->getDependentVariables();
    ASSERT_EQ(ccu3DependentVars.size(), 2);
    //ASSERT_TRUE(hasVarRef(ccu3DependentVars, "Synapses3", "g"));
    //ASSERT_TRUE(hasVarRef(ccu3DependentVars, "CustomUpdate3", "sum"));

    // Check dependencies for CCU4
    auto ccu4DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu4)->getDependentVariables();
    ASSERT_EQ(ccu4DependentVars.size(), 2);
    //ASSERT_TRUE(hasVarRef(ccu4DependentVars, "Synapses4", "g"));
    //ASSERT_TRUE(hasVarRef(ccu4DependentVars, "CustomConnectivityUpdate42", "a"));
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, DependentVariablesManualReferences)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create three synapse groups, each with an attached custom update and custom connectivity update
    SynapseGroup *synapseGroups[3] = {nullptr, nullptr, nullptr};
    CustomUpdateWU *customUpdates[3] = {nullptr, nullptr, nullptr};
    CustomConnectivityUpdate *customConnectivityUpdates[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; i++) {
        // Create synapse group with individual weights
        synapseGroups[i] = model.addSynapsePopulation(
            "Synapses" + std::to_string(i), SynapseMatrixType::SPARSE, NO_DELAY,
            "Pre", "Post",
            initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
            initPostsynaptic<PostsynapticModels::DeltaCurr>());

        // Attach custom update model
        WUVarReferences cu3VarRefs{{"a", createWUVarRef(synapseGroups[i], "g")}};
        customUpdates[i] = model.addCustomUpdate<Sum>("CustomUpdate" + std::to_string(i), "Test1", {}, {{"sum", 0.0}}, cu3VarRefs);

        // Attach custom connectivity update
        customConnectivityUpdates[i] = model.addCustomConnectivityUpdate<RemoveSynapse>(
            "CustomConnectivityUpdate" + std::to_string(i), "Test1", "Synapses" + std::to_string(i),
            {}, {{"a", 1.0}}, {}, {},
            {}, {}, {});
    }

    // Add another custom connectivity update to first synapse group with a manual reference to synapse group variable
    WUVarReferences ccu12VarRefs{{"b", createWUVarRef(synapseGroups[0], "g")}};
    auto *ccu12 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate12", "Test2", "Synapses0",
                                                                         {}, {{"a", 1.0}}, {}, {},
                                                                         ccu12VarRefs, {}, {});

    // Add another custom connectivity update to second synapse group with a manual reference to custom update variable
    WUVarReferences ccu22VarRefs{{"b", createWUVarRef(customUpdates[1], "sum")}};
    auto *ccu22 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate22", "Test2", "Synapses1",
                                                                         {}, {{"a", 1.0}}, {}, {},
                                                                         ccu22VarRefs, {}, {});

    // Add another custom connectivity update to third synapse group with a manual reference to custom connectivity update variable
    WUVarReferences ccu32VarRefs{{"b", createWUVarRef(customConnectivityUpdates[2], "a")}};
    auto *ccu32 = model.addCustomConnectivityUpdate<RemoveSynapseVarRef>("CustomConnectivityUpdate32", "Test2", "Synapses2",
                                                                         {}, {{"a", 1.0}}, {}, {},
                                                                         ccu32VarRefs, {}, {});

    model.finalise();

    // Check synapse group variable has been removed from CCU12 dependent variables as it's manually referenced
    auto ccu12DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu12)->getDependentVariables();
    ASSERT_EQ(ccu12DependentVars.size(), 2);
    //ASSERT_TRUE(hasVarRef(ccu12DependentVars, "CustomUpdate0", "sum"));
    //ASSERT_TRUE(hasVarRef(ccu12DependentVars, "CustomConnectivityUpdate0", "a"));
                
    // Check custom update variable has been removed from CCU22 dependent variables as it's manually referenced
    auto ccu22DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu22)->getDependentVariables();
    ASSERT_EQ(ccu22DependentVars.size(), 2);
    //ASSERT_TRUE(hasVarRef(ccu22DependentVars, "Synapses1", "g"));
    //ASSERT_TRUE(hasVarRef(ccu22DependentVars, "CustomConnectivityUpdate1", "a"));
    
    // Check custom connectivity update variable has been removed from CCU32 dependent variables as it's manually referenced
    auto ccu32DependentVars = static_cast<CustomConnectivityUpdateInternal*>(ccu32)->getDependentVariables();
    ASSERT_EQ(ccu32DependentVars.size(), 2);
    //ASSERT_TRUE(hasVarRef(ccu32DependentVars, "Synapses2", "g"));
    //ASSERT_TRUE(hasVarRef(ccu32DependentVars, "CustomUpdate2", "sum"));
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, CompareDifferentDependentVars)
{
    ModelSpecInternal model;
    
    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseDendriticDelay>({}, {{"g", 1.0}, {"d", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<StaticPulseDendriticDelayReverse>({}, {{"g", 1.0}, {"d", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    model.addSynapsePopulation(
        "Synapses3", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto *ccu1 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});
    auto *ccu2 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate2", "Test2", "Synapses2",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});
    auto *ccu3 = model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate3", "Test2", "Synapses3",
                                                                  {}, {{"a", 1.0}}, {}, {},
                                                                  {}, {}, {});
    model.finalise();
    
    auto *ccu1Internal = static_cast<CustomConnectivityUpdateInternal*>(ccu1);
    auto *ccu2Internal = static_cast<CustomConnectivityUpdateInternal*>(ccu2);
    auto *ccu3Internal = static_cast<CustomConnectivityUpdateInternal*>(ccu3);
    
    ASSERT_EQ(ccu1Internal->getHashDigest(), ccu2Internal->getHashDigest());
    ASSERT_NE(ccu1Internal->getHashDigest(), ccu3Internal->getHashDigest());
    
    ASSERT_EQ(ccu1Internal->getInitHashDigest(), ccu2Internal->getInitHashDigest());
    ASSERT_EQ(ccu1Internal->getInitHashDigest(), ccu3Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check correct groups are merged
    ASSERT_EQ(modelSpecMerged.getMergedCustomConnectivityUpdateGroups().size(), 2);
    ASSERT_EQ(modelSpecMerged.getMergedCustomConnectivityUpdatePreInitGroups().size(), 0);
    ASSERT_EQ(modelSpecMerged.getMergedCustomConnectivityUpdatePostInitGroups().size(), 0);
    ASSERT_EQ(modelSpecMerged.getMergedCustomConnectivityUpdateSparseInitGroups().size(), 1);
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, BitmaskConnectivity)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::BITMASK, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    try {
        model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                         {}, {{"a", 1.0}}, {}, {},
                                                         {}, {}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, WrongPrePostSize)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    auto *other = model.addNeuronPopulation<NeuronModels::Izhikevich>("Other", 10, paramVals, varVals);
    auto *otherWrongSize = model.addNeuronPopulation<NeuronModels::Izhikevich>("WrongSize", 12, paramVals, varVals);

    // Create synapse group with global weights
    auto *syn = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    // Create custom update with presynaptic var reference
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(pre, "V")}, 
                                                                      {"threshHigh", createVarRef(pre, "U")}}, {});
    
    // Create customauto *post = update with postsynaptic var reference
    model.addCustomConnectivityUpdate<RemoveSynapsePost>("CustomConnectivityUpdate2", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                        {}, VarReferences{{"threshLow", createVarRef(post, "V")}, 
                                                                          {"threshHigh", createVarRef(post, "U")}});
    
    // Create custom update with presynaptic var reference to other population
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate3", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(other, "V")}, 
                                                                    {"threshHigh", createVarRef(other, "U")}}, {});
    
    // Create custom update with postsynaptic var reference to other population
    model.addCustomConnectivityUpdate<RemoveSynapsePost>("CustomConnectivityUpdate4", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                        {}, VarReferences{{"threshLow", createVarRef(other, "V")}, 
                                                                          {"threshHigh", createVarRef(other, "U")}});
    
    // Create custom update with presynaptic var reference to other population with wrong size
    try {
        model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate4", "Test2", "Synapses1",
                                                            {}, {}, {}, {},
                                                            WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                            VarReferences{{"threshLow", createVarRef(otherWrongSize, "V")}, 
                                                                          {"threshHigh", createVarRef(otherWrongSize, "U")}}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
        
    // Create custom update with postsynaptic var reference to other population with wrong size
    try {
        model.addCustomConnectivityUpdate<RemoveSynapsePost>("CustomConnectivityUpdate5", "Test2", "Synapses1",
                                                            {}, {}, {}, {},
                                                            WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                            {}, VarReferences{{"threshLow", createVarRef(otherWrongSize, "V")}, 
                                                                              {"threshHigh", createVarRef(otherWrongSize, "U")}});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, WrongSG)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    // Create synapse group with global weights
    auto *syn1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    // Create synapse group with global weights
    auto *syn2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    // Create custom update with var reference on syn1
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn1, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(pre, "V")}, 
                                                                      {"threshHigh", createVarRef(pre, "U")}}, {});
    
    // Create custom update with var reference on syn2
    try {
        model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate2", "Test2", "Synapses1",
                                                            {}, {}, {}, {},
                                                            WUVarReferences{{"g", createWUVarRef(syn2, "g")}}, 
                                                            VarReferences{{"threshLow", createVarRef(pre, "V")}, 
                                                                          {"threshHigh", createVarRef(pre, "U")}}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, DuplicatePrePost)
{
    ModelSpecInternal model;
    model.setBatchSize(5);

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    auto *syn = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
     
    // Create custom update with var reference on syn2
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(pre, "V")}, 
                                                                      {"threshHigh", createVarRef(pre, "U")}}, {});

    try {
        model.finalise();
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
    
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, MixedPreDelayGroups)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre1", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post1", 10, paramVals, varVals);
    auto *pre2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre2", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post2", 10, paramVals, varVals);
    
    // Create synapse group with global weights
    auto *syn1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, 5,
        "Pre1", "Post1",
        initWeightUpdate<Cont>({}, {{"g", 1.0}}, {}, {}, {{"V", createVarRef(pre1, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    // Create synapse group with global weights
    model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE, 10,
        "Pre2", "Post2",
        initWeightUpdate<Cont>({}, {{"g", 1.0}}, {}, {}, {{"V", createVarRef(pre2, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    // Create custom update with both presynaptic var references to same (delay) group
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn1, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(pre1, "V")}, 
                                                                      {"threshHigh", createVarRef(pre1, "V")}}, {});
    
    // Create custom update with both presynaptic var references to different (delay) groups
    model.addCustomConnectivityUpdate<RemoveSynapsePre>("CustomConnectivityUpdate2", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn1, "g")}}, 
                                                        VarReferences{{"threshLow", createVarRef(pre1, "V")}, 
                                                                      {"threshHigh", createVarRef(pre2, "V")}}, {});
    try {
        model.finalise();
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, MixedPostDelayGroups)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre1", 10, paramVals, varVals);
    auto *post1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post1", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre2", 10, paramVals, varVals);
    auto *post2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post2", 10, paramVals, varVals);
    
    // Create synapse group with global weights
    auto *syn1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre1", "Post1",
        initWeightUpdate<ContPost>({}, {{"g", 1.0}}, {}, {}, {}, {{"V", createVarRef(post1, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    syn1->setBackPropDelaySteps(5);
    
    // Create synapse group with global weights
    auto *syn2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre2", "Post2",
        initWeightUpdate<ContPost>({}, {{"g", 1.0}}, {}, {}, {}, {{"V", createVarRef(post2, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    syn2->setBackPropDelaySteps(10);
    
    // Create custom update with both postsynaptic var references to same (delay) group
    model.addCustomConnectivityUpdate<RemoveSynapsePost>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn1, "g")}}, 
                                                        {}, VarReferences{{"threshLow", createVarRef(post1, "V")}, 
                                                                          {"threshHigh", createVarRef(post1, "V")}});
    
    // Create custom update with both postsynaptic var references to different (delay) groups
    model.addCustomConnectivityUpdate<RemoveSynapsePost>("CustomConnectivityUpdate2", "Test2", "Synapses1",
                                                        {}, {}, {}, {},
                                                        WUVarReferences{{"g", createWUVarRef(syn1, "g")}}, 
                                                        {}, VarReferences{{"threshLow", createVarRef(post1, "V")}, 
                                                                          {"threshHigh", createVarRef(post2, "V")}});
    try {
        model.finalise();
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, InvalidName)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    try {
        model.addCustomConnectivityUpdate<RemoveSynapse>("Custom-Connectivity-Update-1", "Test2", "Synapses1",
                                                         {}, {{"a", 1.0}}, {}, {},
                                                         {}, {}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(CustomConnectivityUpdate, InvalidUpdateGroupName)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};   
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    // Create synapse group with global weights
    model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    try {
        model.addCustomConnectivityUpdate<RemoveSynapse>("CustomConnectivityUpdate1", "Test-2", "Synapses1",
                                                         {}, {{"a", 1.0}}, {}, {},
                                                         {}, {}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
