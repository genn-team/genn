//--------------------------------------------------------------------------
/*! \file custom_connectivity_update_remove/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"
// 10 pre, 10 post neurons
// Triangle connectivity i.e. pre connected to all subsequent post
// Initialise weights to  postsynaptic index * (presynaptic index + 1)
// Initialise custom connectivity update variable to something else
// connectivity update which removes first synapse in each row

// Initialise model
// Launch custom update
// pull connectivity, state variables et
// check all synapse variables and connectivity match expectations

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
              {"d", "uint8_t", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestWUM);

class Weight : public InitVarSnippet::Base
{
    DECLARE_SNIPPET(Weight, 0);
    
    SET_CODE("$(value) = ($(id_pre) * 100) + $(id_post);");
};
IMPLEMENT_SNIPPET(Weight);

class Delay : public InitVarSnippet::Base
{
    DECLARE_SNIPPET(Delay, 0);
    
    SET_CODE("$(value) = ($(id_post) * 100) + $(id_pre);");
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
    model.setName("custom_connectivity_update_remove");
    
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 100, {}, {});
    model.addNeuronPopulation<TestNeuron>("Neuron", 100, {}, {0.0});
    
    TestWUM::VarValues testWUMInit(initVar<Weight>(), initVar<Delay>());
    model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, testWUMInit,
        {}, {},
        initConnectivity<Triangle>({}));

    model.setPrecision(GENN_FLOAT);
}