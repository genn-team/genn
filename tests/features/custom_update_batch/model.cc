//--------------------------------------------------------------------------
/*! \file custom_update_batch/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class TestNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TestNeuron, 0, 2);

    SET_VARS({{"V","scalar"}, {"U", "scalar", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestNeuron);

class TestWUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(TestWUM, 0, 2, 0, 0);

    SET_VARS({{"V","scalar"}, {"U", "scalar", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(TestWUM);

class SetTimeBatch : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(SetTimeBatch, 0, 1, 1);
    
    SET_UPDATE_CODE(
        "$(V) = ($(batch) * 1000.0) + $(t);\n"
        "$(R) = ($(batch) * 1000.0) + $(t);\n");

    SET_VARS({{"V", "scalar"}});
    SET_VAR_REFS({{"R", "scalar", VarAccessMode::READ_WRITE}})
};
IMPLEMENT_MODEL(SetTimeBatch);

class SetTime : public CustomUpdateModels::Base
{
public:
    DECLARE_CUSTOM_UPDATE_MODEL(SetTime, 0, 1, 1);
    
    SET_UPDATE_CODE(
        "$(R) = $(V) + ($(batch) * 1000.0) + $(t);\n");

    SET_VARS({{"V", "scalar", VarAccess::READ_ONLY}});
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
    model.setName("custom_update_batch");
    model.setBatchSize(5);

    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 50, {}, {});
    auto *ng = model.addNeuronPopulation<TestNeuron>("Neuron", 50, {}, {0.0, 0.0});
    auto *denseSG = model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Dense", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, {0.0, 0.0},
        {}, {});
    auto *sparseSG = model.addSynapsePopulation<TestWUM, PostsynapticModels::DeltaCurr>(
        "Sparse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "Neuron",
        {}, {0.0, 0.0},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.1}));
    
    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    SetTimeBatch::VarReferences neuronDuplicateVarReferences(createVarRef(ng, "V")); // R
    model.addCustomUpdate<SetTimeBatch>("NeuronDuplicateSetTime", "Test",
                                        {}, {0.0}, neuronDuplicateVarReferences);
    
    SetTime::VarReferences neuronSharedVarReferences(createVarRef(ng, "U")); // R
    model.addCustomUpdate<SetTime>("NeuronSharedSetTime", "Test",
                                   {}, {0.0}, neuronSharedVarReferences);
    
    SetTimeBatch::WUVarReferences wumDenseDuplicateVarReferences(createWUVarRef(denseSG, "V")); // R
    model.addCustomUpdate<SetTimeBatch>("WUMDenseDuplicateSetTime", "Test",
                                        {}, {0.0}, wumDenseDuplicateVarReferences);
    
    SetTime::WUVarReferences wumDenseSharedVarReferences(createWUVarRef(denseSG, "U")); // R
    model.addCustomUpdate<SetTime>("WUMDenseSharedSetTime", "Test",
                                   {}, {0.0}, wumDenseSharedVarReferences);
   
    SetTimeBatch::WUVarReferences wumSparseDuplicateVarReferences(createWUVarRef(sparseSG, "V")); // R
    model.addCustomUpdate<SetTimeBatch>("WUMSparseDuplicateSetTime", "Test",
                                        {}, {0.0}, wumSparseDuplicateVarReferences);
    
    SetTime::WUVarReferences wumSparseSharedVarReferences(createWUVarRef(sparseSG, "U")); // R
    model.addCustomUpdate<SetTime>("WUMSparseSharedSetTime", "Test",
                                   {}, {0.0}, wumSparseSharedVarReferences);

}