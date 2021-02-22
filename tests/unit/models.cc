// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class AlphaCurr : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(AlphaCurr, 1, 1);

    SET_DECAY_CODE(
        "$(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));\n"
        "$(inSyn)*=$(expDecay);\n");

    SET_CURRENT_CONVERTER_CODE("$(x)");

    SET_PARAM_NAMES({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::vector<double> &pars, double dt) { return std::exp(-dt / pars[0]); }},
        {"init", [](const std::vector<double> &pars, double) { return (std::exp(1) / pars[0]); }}});
};
IMPLEMENT_MODEL(AlphaCurr);

class StaticPulseUInt : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseUInt, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");
};
IMPLEMENT_MODEL(StaticPulseUInt);

class Cont : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(Cont, 0, 1, 0, 0);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addToInSyn, $(g) * $(V_pre));\n");
};
IMPLEMENT_MODEL(Cont);

class ContPrePost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(ContPrePost, 0, 1, 1, 1);

    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(preTrace) = ($(preTrace) * exp(-dt / $(tauPlus))) + 1.0;\n");

    SET_POST_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(postTrace) = ($(postTrace) * exp(-dt / $(tauMinus))) + 1.0;\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addToInSyn, $(g) * $(V_pre));\n");
};
IMPLEMENT_MODEL(ContPrePost);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Models, NeuronVarReference)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    const auto *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    auto neuronVoltage = createVarRef(ng, "V");
    ASSERT_EQ(neuronVoltage.getSize(), 10);

    try {
        auto neuronMagic = createVarRef(ng, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Models, NeuronVarReferenceDelay)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    model.addSynapsePopulation<Cont, PostsynapticModels::DeltaCurr>("Syn", SynapseMatrixType::DENSE_INDIVIDUALG,
                                                                    10, "Neurons0", "Neurons1",
                                                                    {}, {0.1}, {}, {});
    
    auto neuronV = createVarRef(pre, "V");
    auto neuronU = createVarRef(pre, "U");

    // Finalize model
    model.finalize();
     
    // Check
    ASSERT_EQ(neuronV.getDelayNeuronGroup(), pre);
    ASSERT_EQ(neuronU.getDelayNeuronGroup(), nullptr);
}
//--------------------------------------------------------------------------
TEST(Models, CurrentSourceVarReference)
{
    ModelSpecInternal model;

    // Add neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one poisson exp current source
    CurrentSourceModels::PoissonExp::ParamValues cs0ParamVals(0.1, 5.0, 10.0);
    CurrentSourceModels::PoissonExp::VarValues cs0VarVals(0.0);
    auto *cs0 = model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS0", "Neurons0",
                                                                        cs0ParamVals, cs0VarVals);
                                                                        
    auto csCurrent = createVarRef(cs0, "current");
    ASSERT_EQ(csCurrent.getSize(), 10);

    try {
        auto csMagic = createVarRef(cs0, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Models, PSMVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});

    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses2", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});

    auto psmX = createPSMVarRef(sg1, "x");
    ASSERT_EQ(psmX.getSize(), 25);

    // Test error if variable doesn't exist
    try {
        auto psmMagic = createPSMVarRef(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if GLOBALG
    try {
        auto psmMagic = createPSMVarRef(sg2, "x");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Models, WUPreVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<ContPrePost, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0}, {0.0}, {0.0},
        {}, {});

    auto *sg2 = model.addSynapsePopulation<ContPrePost, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::DENSE_GLOBALG, 5,
        "Pre", "Post",
        {}, {1.0}, {0.0}, {0.0},
        {}, {});

    auto wuPre = createWUPreVarRef(sg1, "preTrace");
    auto wuPre2 = createWUPreVarRef(sg2, "preTrace");
    
    // Test error if variable doesn't exist
    try {
        auto wuPreMagic = createWUPreVarRef(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Finalize model
    model.finalize();

    ASSERT_EQ(wuPre.getSize(), 10);
    ASSERT_EQ(wuPre.getDelayNeuronGroup(), nullptr);
    ASSERT_EQ(wuPre2.getDelayNeuronGroup(), pre);
}
//--------------------------------------------------------------------------
TEST(Models, WUPostVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<ContPrePost, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0}, {0.0}, {0.0},
        {}, {});

    auto *sg2 = model.addSynapsePopulation<ContPrePost, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0}, {0.0}, {0.0},
        {}, {});

    auto wuPost = createWUPostVarRef(sg1, "postTrace");
    auto wuPost2 = createWUPostVarRef(sg2, "postTrace");
    
    sg2->setBackPropDelaySteps(5);

    // Test error if variable doesn't exist
    try {
        auto wuPostMagic = createWUPostVarRef(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Finalize model
    model.finalize();

    ASSERT_EQ(wuPost.getSize(), 25);
    ASSERT_EQ(wuPost.getDelayNeuronGroup(), nullptr);
    ASSERT_EQ(wuPost2.getDelayNeuronGroup(), post);
}
//--------------------------------------------------------------------------
TEST(Models, WUMVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});

    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("Synapses2", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                       "Pre", "Post",
                                                                                       {}, {1.0},
                                                                                       {5.0}, {0.0});
    auto wuG1 = createWUVarRef(sg1, "g");

    // Test error if variable doesn't exist
    try {
        auto wuMagic = createWUVarRef(sg1, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if GLOBALG
    try {
        auto wuG2 = createWUVarRef(sg2, "x");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Models, WUMTransposeVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sgForward = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {1.0},
        {}, {});

    auto *sgBackwardIndividualG = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Post", "Pre",
        {}, {1.0},
        {}, {});

    auto *sgBackwardGlobalG = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses3", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
        "Post", "Pre",
        {}, {1.0},
        {}, {});
    
    auto *sgBackwardBadShape = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses4", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Pre",
        {}, {1.0},
        {}, {});
    
    auto *sgBackwardSparse = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Synapses5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "Post", "Pre",
        {}, {1.0},
        {}, {});

    auto *sgBackwardBadType = model.addSynapsePopulation<StaticPulseUInt, PostsynapticModels::DeltaCurr>(
        "Synapses6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "Post", "Pre",
        {}, {1.0},
        {}, {});

    auto wuG1 = createWUVarRef(sgForward, "g", sgBackwardIndividualG, "g");

    // Test error if transpose varaible doesn't exist
    try {
        auto wuMagic = createWUVarRef(sgForward, "g", sgBackwardIndividualG, "Magic");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if GLOBALG
    try {
        auto wuG2 = createWUVarRef(sgForward, "g", sgBackwardGlobalG, "g");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if shapes don't match
    try {
        auto wuG2 = createWUVarRef(sgForward, "g", sgBackwardBadShape, "g");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if transpose is sparse
    try {
        auto wuG2 = createWUVarRef(sgForward, "g", sgBackwardSparse, "g");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Test error if transpose is different type
    try {
        auto wuG2 = createWUVarRef(sgForward, "g", sgBackwardBadType, "g");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Models, WUMSlaveVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post1", 25, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post2", 25, paramVals, varVals);

    auto *sgMaster = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "SynapseMaster", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post1",
        {}, {1.0},
        {}, {});
    auto *sgSlave = model.addSlaveSynapsePopulation<PostsynapticModels::DeltaCurr>(
        "SynapseSlave", "SynapseMaster", NO_DELAY,
        "Pre", "Post2",
        {}, {});

    auto wuMaster = createWUVarRef(sgMaster, "g");

    // Test error if referencing slave group
    try {
        auto wuSlave = createWUVarRef(sgSlave, "g");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
