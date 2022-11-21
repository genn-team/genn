//--------------------------------------------------------------------------
/*! \file custom_connectivity_update_delay/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class TestNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TestNeuron, 0, 1);

    SET_SIM_CODE("$(removeIdx) = ($(id) + (int)round($(t) / DT)) % 64;\n");
        
    SET_VARS({{"removeIdx", "int"}});
};
IMPLEMENT_MODEL(TestNeuron);

class TestWUMPre : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUMPre, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    SET_SIM_CODE("$(addToInSyn, $(g) * (float)$(removeIdx_pre));\n");
};
IMPLEMENT_MODEL(TestWUMPre);

class TestWUMPost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUMPost, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    SET_SIM_CODE("$(addToInSyn, $(g) * (float)$(removeIdx_post));\n");
};
IMPLEMENT_MODEL(TestWUMPost);

class Weight : public InitVarSnippet::Base
{
    DECLARE_SNIPPET(Weight, 0);
    
    SET_CODE("$(value) = ($(id_pre) * 64) + $(id_post);");
};
IMPLEMENT_SNIPPET(Weight);

class Dense : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Dense, 0);

    SET_ROW_BUILD_CODE(
        "if(j < $(num_post)) {\n"
        "   $(addSynapse, j);\n"
        "}\n"
        "else {\n"
        "   $(endRow);\n"
        "}\n"
        "j++;\n");
    SET_ROW_BUILD_STATE_VARS({{"j", "unsigned int", 0}});
};
IMPLEMENT_SNIPPET(Dense);

class RemoveSynapseUpdatePre : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseUpdatePre, 0, 0, 0, 0, 0, 1, 0);
    
    SET_PRE_VAR_REFS({{"removeIdx", "int"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == $(removeIdx)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapseUpdatePre);

class RemoveSynapseUpdatePost : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseUpdatePost, 0, 0, 0, 0, 0, 0, 1);
    
    SET_POST_VAR_REFS({{"removeIdx", "int"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == $(removeIdx)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapseUpdatePost);

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
    model.setName("custom_connectivity_update_delay");

    auto *pre = model.addNeuronPopulation<TestNeuron>("Pre", 64, {}, {0.0});
    auto *post = model.addNeuronPopulation<TestNeuron>("Post", 64, {}, {0.0});

    TestWUMPre::VarValues testWUMPreInit(initVar<Weight>());
    model.addSynapsePopulation<TestWUMPre, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::SPARSE_INDIVIDUALG, 5, "Pre", "Post",
        {}, testWUMPreInit,
        {}, {},
        initConnectivity<Dense>());
    
    TestWUMPost::VarValues testWUMPostInit(initVar<Weight>());
    auto *syn2 = model.addSynapsePopulation<TestWUMPost, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::SPARSE_INDIVIDUALG, 0, "Pre", "Post",
        {}, testWUMPostInit,
        {}, {},
        initConnectivity<Dense>());
    syn2->setBackPropDelaySteps(5);

    RemoveSynapseUpdatePre::PreVarReferences removeSynapsePreVarRefInit(createVarRef(pre, "removeIdx"));
    model.addCustomConnectivityUpdate<RemoveSynapseUpdatePre>(
        "RemoveSynapsePre", "RemoveSynapse", "Syn1",
        {}, {}, {}, {},
        {}, removeSynapsePreVarRefInit, {});
    
    RemoveSynapseUpdatePost::PostVarReferences removeSynapsePostVarRefInit(createVarRef(post, "removeIdx"));
    model.addCustomConnectivityUpdate<RemoveSynapseUpdatePost>(
        "RemoveSynapsePost", "RemoveSynapse", "Syn2",
        {}, {}, {}, {},
        {}, {}, removeSynapsePostVarRefInit);

    model.setPrecision(GENN_FLOAT);
}
