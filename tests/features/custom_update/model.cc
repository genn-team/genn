//--------------------------------------------------------------------------
/*! \file custom_update/model.cc

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

class TestCurrentSource : public CurrentSourceModels::Base
{
    DECLARE_MODEL(TestCurrentSource, 0, 1);

    SET_VARS({{"C", "scalar"}});
};
IMPLEMENT_MODEL(TestCurrentSource);

class TestPSM : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(TestPSM, 0, 1);

    SET_CURRENT_CONVERTER_CODE("$(inSyn); $(inSyn) = 0");
    SET_VARS({{"P", "scalar"}});
};
IMPLEMENT_MODEL(TestPSM);

class TestWUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUM, 0, 1, 1, 1);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});
    SET_PRE_VARS({{"Pre", "scalar"}});
    SET_POST_VARS({{"Post", "scalar"}});
    SET_SIM_CODE("$(addToInSyn, $(g));\n");
};
IMPLEMENT_MODEL(TestWUM);

class SetTime : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(SetTime, 0, 1, 1);
    
    SET_UPDATE_CODE(
        "$(V) = $(t);\n"
        "$(R) = $(t);\n");

    SET_VARS({{"V", "scalar"}});
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
    model.setName("custom_update");

    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 100, {}, {});
    auto *ng = model.addNeuronPopulation<TestNeuron>("Neuron", 100, {}, {0.0});
    auto *cs = model.addCurrentSource<TestCurrentSource>("CurrentSource", "Neuron", {}, {0.0});
    auto *denseSG = model.addSynapsePopulation<TestWUM, TestPSM>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, {0.0}, {0.0}, {0.0},
        {}, {0.0});
    auto *sparseSG = model.addSynapsePopulation<TestWUM, TestPSM>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, {0}, {0.0}, {0.0},
        {}, {0},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.1}));
    
    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    SetTime::VarReferences neuronVarReferences(createVarRef(ng, "V")); // R
    model.addCustomUpdate<SetTime>("NeuronSetTime", "Test",
                                   {}, {0.0}, neuronVarReferences);
    
    SetTime::VarReferences csVarReferences(createVarRef(cs, "C")); // R
    model.addCustomUpdate<SetTime>("CurrentSourceSetTime", "Test",
                                   {}, {0.0}, csVarReferences);
                                   
    SetTime::VarReferences psmVarReferences(createPSMVarRef(denseSG, "P")); // R
    model.addCustomUpdate<SetTime>("PSMSetTime", "Test",
                                   {}, {0.0}, psmVarReferences);
                               
    SetTime::VarReferences wuPreVarReferences(createWUPreVarRef(denseSG, "Pre")); // R
    model.addCustomUpdate<SetTime>("WUPreSetTime", "Test",
                                   {}, {0.0}, wuPreVarReferences);
                               
    SetTime::VarReferences wuPostVarReferences(createWUPostVarRef(sparseSG, "Post")); // R
    model.addCustomUpdate<SetTime>("WUPostSetTime", "Test",
                                   {}, {0.0}, wuPostVarReferences);
                                   
    SetTime::WUVarReferences wuDenseVarReferences(createWUVarRef(denseSG, "g")); // R
    model.addCustomUpdate<SetTime>("WUDenseSetTime", "Test",
                                   {}, {0.0}, wuDenseVarReferences);
                                   
    SetTime::WUVarReferences wuSparseVarReferences(createWUVarRef(sparseSG, "g")); // R
    model.addCustomUpdate<SetTime>("WUSparseSetTime", "Test",
                                   {}, {0.0}, wuSparseVarReferences);
}