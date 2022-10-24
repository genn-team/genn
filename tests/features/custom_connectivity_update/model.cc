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
        "   if(j > $(id_pre)) {\n"
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

class ConnectUpdate : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(ConnectUpdate, 0, 1, 0, 0, 0, 0, 0);
    
    SET_VARS({{"v", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   $(remove_synapse);\n"
        "   break;\n"
        "});\n");
        
};
IMPLEMENT_MODEL(ConnectUpdate);

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
    auto *sg = model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, testWUMInit,
        {}, {},
        initConnectivity<Triangle>({}));
    
    ConnectUpdate::VarValues connectUpdateInit(initVar<Weight>());
    model.addCustomConnectivityUpdate<ConnectUpdate>(
        "CustomConnectivityUpdate", "Update", "Syn",
        {}, connectUpdateInit, {}, {},
        {}, {}, {});
    model.setPrecision(GENN_FLOAT);
}