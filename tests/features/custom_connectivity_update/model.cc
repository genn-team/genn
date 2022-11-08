//--------------------------------------------------------------------------
/*! \file custom_connectivity_update/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class TestNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TestNeuron, 0, 1);

    SET_VARS({{"V","scalar"}});
};
IMPLEMENT_MODEL(TestNeuron);

class TestWUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUM, 0, 2, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY},
              {"d", "unsigned int", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestWUM);

class Weight : public InitVarSnippet::Base
{
    DECLARE_SNIPPET(Weight, 0);
    
    SET_CODE("$(value) = ($(id_pre) * 64) + $(id_post);");
};
IMPLEMENT_SNIPPET(Weight);

class Delay : public InitVarSnippet::Base
{
    DECLARE_SNIPPET(Delay, 0);
    
    SET_CODE("$(value) = ($(id_post) * 64) + $(id_pre);");
};
IMPLEMENT_SNIPPET(Delay);

class Triangle : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Triangle, 0);

    SET_ROW_BUILD_CODE(
        "if(j < $(num_post)) {\n"
        "   if(j >= $(id_pre)) {\n"
        "       $(addSynapse, j);\n"
        "   }\n"
        "}\n"
        "else {\n"
        "   $(endRow);\n"
        "}\n"
        "j++;\n");
    SET_ROW_BUILD_STATE_VARS({{"j", "unsigned int", 0}});
};
IMPLEMENT_SNIPPET(Triangle);

class RemoveSynapseUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseUpdate, 0, 1, 0, 0, 0, 0, 0);
    
    SET_VARS({{"a", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == $(id_pre)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapseUpdate);

class RemoveSynapseHostEGPUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseHostEGPUpdate, 0, 0, 0, 0, 0, 0, 0);
    
    SET_EXTRA_GLOBAL_PARAMS({{"d", "uint32_t*"}});
    SET_ROW_UPDATE_CODE(
        "const unsigned int wordsPerRow = ($(num_post) + 31) / 32;\n"
        "const uint32_t *dRow = &$(d)[wordsPerRow * $(id_pre)];\n"
        "$(for_each_synapse,\n"
        "{\n"
        "   if(dRow[$(id_post) / 32] & (1 << ($(id_post) % 32))) {\n"
        "       $(remove_synapse);\n"
        "   }\n"
        "});\n");
    SET_HOST_UPDATE_CODE(
        "const unsigned int wordsPerRow = ($(num_post) + 31) / 32;\n"
        "memset($(d), 0, wordsPerRow * $(num_pre) * sizeof(uint32_t));\n"
        "for(unsigned int i = 0; i < $(num_pre); i++) {\n"
        "   uint32_t *dRow = &$(d)[wordsPerRow * i];\n"
        "   dRow[i / 32] |= (1 << (i % 32));\n"
        "}\n"
        "$(pushdToDevice, wordsPerRow * $(num_pre));\n");
};
IMPLEMENT_MODEL(RemoveSynapseHostEGPUpdate);

class RemoveSynapseHostVarUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapseHostVarUpdate, 0, 0, 1, 0, 0, 0, 0);
    
    SET_PRE_VARS({{"postInd", "unsigned int"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(postInd) == $(id_post)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
    SET_HOST_UPDATE_CODE(
        "for(unsigned int i = 0; i < $(num_pre); i++) {\n"
        "   $(postInd)[i] = i;\n"
        "}\n"
        "$(pushpostIndToDevice);\n");
};
IMPLEMENT_MODEL(RemoveSynapseHostVarUpdate);

class AddSynapseUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(AddSynapseUpdate, 0, 0, 0, 0, 3, 0, 0);

    SET_VAR_REFS({{"g", "scalar"}, {"d", "unsigned int"}, {"a", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "const scalar weight = ($(id_pre) * 64) + $(id_pre);\n"
        "const unsigned int delay = ($(id_pre) * 64) + $(id_pre);\n"
        "$(add_synapse, $(id_pre), weight, delay, weight);\n");
        
};
IMPLEMENT_MODEL(AddSynapseUpdate);

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
    model.setName("custom_connectivity_update");
    
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 64, {}, {});
    model.addNeuronPopulation<TestNeuron>("Neuron", 64, {}, {0.0});
    
    TestWUM::VarValues testWUMInit(initVar<Weight>(), initVar<Delay>());
    auto *sg1 = model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, testWUMInit,
        {}, {},
        initConnectivity<Triangle>({}));
    
    model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, testWUMInit,
        {}, {},
        initConnectivity<Triangle>({}));
    
    model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn3", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, testWUMInit,
        {}, {},
        initConnectivity<Triangle>({}));
    
    RemoveSynapseUpdate::VarValues removeSynapseVarInit(initVar<Weight>());
    auto *ccu = model.addCustomConnectivityUpdate<RemoveSynapseUpdate>(
        "RemoveSynapse", "RemoveSynapse", "Syn1",
        {}, removeSynapseVarInit, {}, {},
        {}, {}, {});
    
    AddSynapseUpdate::VarReferences addSynapseVarRefInit(createWUVarRef(sg1, "g"), createWUVarRef(sg1, "d"), createWUVarRef(ccu, "a"));
    model.addCustomConnectivityUpdate<AddSynapseUpdate>(
        "AddSynapse", "AddSynapse", "Syn1",
        {}, {}, {}, {},
        addSynapseVarRefInit, {}, {});
    
    model.addCustomConnectivityUpdate<RemoveSynapseHostEGPUpdate>(
        "RemoveSynapseHostEGPUpdate", "RemoveSynapse", "Syn2",
        {}, {}, {}, {},
        {}, {}, {});
    
    model.addCustomConnectivityUpdate<RemoveSynapseHostVarUpdate>(
        "RemoveSynapseHostVarUpdate", "RemoveSynapse", "Syn3",
        {}, {}, {0}, {},
        {}, {}, {});
    
    model.setPrecision(GENN_FLOAT);
}
