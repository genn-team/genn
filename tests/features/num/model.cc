//--------------------------------------------------------------------------
/*! \file num/model.cc

\brief model definition file that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------


#include "modelSpec.h"

//----------------------------------------------------------------------------
// Neuron
//----------------------------------------------------------------------------
class Neuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Neuron, 0, 2);

    SET_VARS({{"num_test", "unsigned int"}, {"num_batch_test", "unsigned int"}});

    SET_SIM_CODE(
        "$(num_test)= $(num);\n"
        "$(num_batch_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(Neuron);

//----------------------------------------------------------------------------
// PSM
//----------------------------------------------------------------------------
class PSM : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(PSM, 0, 2);
    
    SET_VARS({{"num_psm_test", "unsigned int"}, {"num_batch_psm_test", "unsigned int"}});
    
    SET_DECAY_CODE(
        "$(num_psm_test)= $(num);\n"
        "$(num_batch_psm_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(PSM);

//----------------------------------------------------------------------------
// CS
//----------------------------------------------------------------------------
class CS : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CS, 0, 2);
    
    SET_VARS({{"num_cs_test", "unsigned int"}, {"num_batch_cs_test", "unsigned int"}});
    
    SET_INJECTION_CODE(
        "$(num_cs_test)= $(num);\n"
        "$(num_batch_cs_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(CS);

//----------------------------------------------------------------------------
// WUM
//----------------------------------------------------------------------------
class WUM : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WUM, 0, 3, 2, 2);

    SET_VARS({{"num_pre_wum_syn_test", "unsigned int"}, {"num_post_wum_syn_test", "unsigned int"}, {"num_batch_wum_syn_test", "unsigned int"}});
    SET_PRE_VARS({{"num_wum_pre_test", "unsigned int"}, {"num_batch_wum_pre_test", "unsigned int"}});
    SET_POST_VARS({{"num_wum_post_test", "unsigned int"}, {"num_batch_wum_post_test", "unsigned int"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(num_pre_wum_syn_test)= $(num_pre);\n"
        "$(num_post_wum_syn_test)= $(num_post);\n"
        "$(num_batch_wum_syn_test) = $(num_batch);\n");
    SET_PRE_DYNAMICS_CODE(
        "$(num_wum_pre_test)= $(num);\n"
        "$(num_batch_wum_pre_test) = $(num_batch);\n");
    SET_POST_DYNAMICS_CODE(
        "$(num_wum_post_test)= $(num);\n"
        "$(num_batch_wum_post_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(WUM);

//----------------------------------------------------------------------------
// CU
//----------------------------------------------------------------------------
class CU : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(CU, 0, 2, 1);
    
    SET_VARS({{"num_test", "unsigned int"}, {"num_batch_test", "unsigned int"}});
    SET_VAR_REFS({{"ref", "unsigned int", VarAccessMode::READ_ONLY}});
    
    SET_UPDATE_CODE(
        "$(num_test)= $(num);\n"
        "$(num_batch_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(CU);

//----------------------------------------------------------------------------
// CUWUM
//----------------------------------------------------------------------------
class CUWUM : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(CUWUM, 0, 3, 1);
    
    SET_VARS({{"num_pre_test", "unsigned int"}, {"num_post_test", "unsigned int"}, {"num_batch_test", "unsigned int"}});
    SET_VAR_REFS({{"ref", "unsigned int", VarAccessMode::READ_ONLY}});
    
    SET_UPDATE_CODE(
        "$(num_pre_test)= $(num_pre);\n"
        "$(num_post_test)= $(num_post);\n"
        "$(num_batch_test) = $(num_batch);\n");
};
IMPLEMENT_MODEL(CUWUM);

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
    model.setName("num");

    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 2, {}, {});
    auto *post = model.addNeuronPopulation<Neuron>("Post", 4, {}, {0, 0});
    model.addCurrentSource<CS>("CurrSource", "Post", {}, {0, 0});
    auto *syn = model.addSynapsePopulation<WUM, PSM>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, "Pre", "Post",
        {}, {0, 0, 0}, {0, 0}, {0, 0},
        {}, {0, 0});
    
    CU::VarReferences varReferences(createVarRef(post, "num_test")); // ref
    model.addCustomUpdate<CU>("CU", "Test", {}, {0, 0}, varReferences);
    
    CUWUM::WUVarReferences wuVarReferences(createWUVarRef(syn, "num_pre_wum_syn_test")); // R
    model.addCustomUpdate<CUWUM>("CUWM", "Test", {}, {0, 0, 0}, wuVarReferences);
    
    
    model.setPrecision(GENN_FLOAT);
}