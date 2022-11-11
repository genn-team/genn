//--------------------------------------------------------------------------
/*! \file custom_connectivity_update_rng/model.cc

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

class RNGTest : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RNGTest, 0, 0, 0, 0, 0, 0, 0);

    SET_EXTRA_GLOBAL_PARAMS({{"Output", "scalar*"}});
    SET_ROW_UPDATE_CODE(
        "for(int j = 0; j < 1000; j++) {\n"
        "   $(Output)[$(id_pre) + (j * $(num_pre))] = $(gennrand_uniform);\n"
        "}\n");
        
};
IMPLEMENT_MODEL(RNGTest);

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
    model.setName("custom_connectivity_update_rng");
    
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 1000, {}, {});
    model.addNeuronPopulation<TestNeuron>("Neuron", 1000, {}, {0.0});
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY, "SpikeSource", "Neuron",
        {}, {1.0},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.1}));

    model.addCustomConnectivityUpdate<RNGTest>(
        "RNGTest", "RNGTest", "Syn1",
        {}, {}, {}, {},
        {}, {}, {});
    
    model.setPrecision(GENN_FLOAT);
}
