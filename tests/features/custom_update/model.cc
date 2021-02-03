//--------------------------------------------------------------------------
/*! \file custom_update/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 1);

    SET_VARS({{"V","scalar"}});
};
IMPLEMENT_MODEL(Neuron);

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


    /*WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(0);
                                
    InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement::ParamValues fixedNumTotalParams(1000);
    InitSparseConnectivitySnippet::FixedNumberPostWithReplacement::ParamValues fixedNumPostParams(10);
    InitSparseConnectivitySnippet::FixedNumberPreWithReplacement::ParamValues fixedNumPreParams(10);*/
    
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 100, {}, {});
    auto *ng = model.addNeuronPopulation<Neuron>("Neuron", 100, {}, {0.0});
    
    // Fixed number total connectivity
    /*model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "FixedNumberTotal", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "LIF",
        {}, staticSynapseInit, {}, {},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement>(fixedNumTotalParams));
    
    // Fixed number post connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "FixedNumberPost", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "LIF",
        {}, staticSynapseInit, {}, {},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumPostParams));
    
    // Fixed number pre connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "FixedNumberPre", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "SpikeSource", "LIF",
        {}, staticSynapseInit, {}, {},
        {}, {},
        initConnec*/
        
    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    SetTime::VarReferences neuronVarReferences(createVarRef(ng, "V")); // R
    model.addCustomUpdate<SetTime>("NeuronSetTime", "Test",
                                   {}, {0.0}, neuronVarReferences);

}