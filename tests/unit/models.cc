// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class AlphaCurr : public PostsynapticModels::Base
{
public:
    DECLARE_SNIPPET(AlphaCurr);

    SET_DECAY_CODE(
        "x = (dt * expDecay * inSyn * init) + (expDecay * x);\n"
        "inSyn *= expDecay;\n");

    SET_CURRENT_CONVERTER_CODE("x");

    SET_PARAMS({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const auto &pars, double dt) { return std::exp(-dt / pars.at("tau").cast<double>()); }},
        {"init", [](const auto &pars, double) { return (std::exp(1) / pars.at("tau").cast<double>()); }}});
};
IMPLEMENT_SNIPPET(AlphaCurr);

class StaticPulseUInt : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseUInt);

    SET_PARAMS({"g"});

    SET_SIM_CODE("addToPost(g);\n");
};
IMPLEMENT_SNIPPET(StaticPulseUInt);

class Cont : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(Cont);

    SET_VARS({{"g", "scalar"}});
    SET_PRE_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "addToPost(g * V);\n");
};
IMPLEMENT_SNIPPET(Cont);

class ContPrePost : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(ContPrePost);

    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    SET_PRE_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = t - sT_pre;\n"
        "preTrace = (preTrace * exp(-dt / tauPlus)) + 1.0;\n");

    SET_POST_SPIKE_CODE(
        "scalar dt = t - sT_post;\n"
        "postTrace = (postTrace * exp(-dt / tauMinus)) + 1.0;\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "addToPost(g * V);\n");
};
IMPLEMENT_SNIPPET(ContPrePost);

class ContPrePostConstantWeight : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(ContPrePostConstantWeight);

    SET_PARAMS({"g"});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    SET_PRE_NEURON_VAR_REFS({{"V", "scalar", VarAccessMode::READ_ONLY}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = t - sT_pre;\n"
        "preTrace = (preTrace * exp(-dt / tauPlus)) + 1.0;\n");

    SET_POST_SPIKE_CODE(
        "scalar dt = t - sT_post;\n"
        "postTrace = (postTrace * exp(-dt / tauMinus)) + 1.0;\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "addToPost(g * V_pre);\n");
};
IMPLEMENT_SNIPPET(ContPrePostConstantWeight);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Models, NeuronVarReference)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

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
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    model.addSynapsePopulation(
        "Syn", SynapseMatrixType::DENSE, 10,
        "Neurons0", "Neurons1",
        initWeightUpdate<Cont>({}, {{"g", 0.1}}, {}, {}, {{"V", createVarRef(pre, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto neuronV = createVarRef(pre, "V");
    auto neuronU = createVarRef(pre, "U");

    // Finalize model
    model.finalise();
     
    // Check
    ASSERT_EQ(neuronV.getDelayNeuronGroup(), pre);
    ASSERT_EQ(neuronU.getDelayNeuronGroup(), nullptr);
}
//--------------------------------------------------------------------------
TEST(Models, CurrentSourceVarReference)
{
    ModelSpecInternal model;

    // Add neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    // Add one poisson exp current source
    ParamValues cs0ParamVals{{"weight", 0.1}, {"tauSyn", 5.0}, {"rate", 10.0}};
    VarValues cs0VarVals{{"current", 0.0}};
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
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<AlphaCurr>({{"tau", 5.0}}, {{"x", 0.0}}));

    auto psmX = createPSMVarRef(sg1, "x");
    ASSERT_EQ(psmX.getSize(), 25);

    // Test error if variable doesn't exist
    try {
        auto psmMagic = createPSMVarRef(sg1, "Magic");
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
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<ContPrePost>({}, {{"g", 1.0}}, {{"preTrace", 0.0}}, {{"postTrace", 0.0}},
                                      {{"V", createVarRef(pre, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::DENSE, 5,
        "Pre", "Post",
        initWeightUpdate<ContPrePostConstantWeight>({{"g", 1.0}}, {}, {{"preTrace", 0.0}}, {{"postTrace", 0.0}},
                                                    {{"V", createVarRef(pre, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

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
    model.finalise();

    ASSERT_EQ(wuPre.getSize(), 10);
    ASSERT_EQ(wuPre.getDelayNeuronGroup(), nullptr);
    ASSERT_EQ(wuPre2.getDelayNeuronGroup(), pre);
}
//--------------------------------------------------------------------------
TEST(Models, WUPostVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<ContPrePost>({}, {{"g", 1.0}}, {{"preTrace", 0.0}}, {{"postTrace", 0.0}},
                                      {{"V", createVarRef(pre, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<ContPrePostConstantWeight>({{"g", 1.0}}, {}, {{"preTrace", 0.0}}, {{"postTrace", 0.0}},
                                                    {{"V", createVarRef(pre, "V")}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

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
    model.finalise();

    ASSERT_EQ(wuPost.getSize(), 25);
    ASSERT_EQ(wuPost.getDelayNeuronGroup(), nullptr);
    ASSERT_EQ(wuPost2.getDelayNeuronGroup(), post);
}
//--------------------------------------------------------------------------
TEST(Models, WUMVarReference)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<AlphaCurr>({{"tau", 5.0}}, {{"x", 0.0}}));

    auto wuG1 = createWUVarRef(sg1, "g");

    // Test error if variable doesn't exist
    try {
        auto wuMagic = createWUVarRef(sg1, "Magic");
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
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    auto *sgForward = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *sgBackwardIndividualG = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::DENSE, NO_DELAY,
        "Post", "Pre",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *sgBackwardGlobalG = model.addSynapsePopulation(
        "Synapses3", SynapseMatrixType::DENSE, NO_DELAY,
        "Post", "Pre",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto *sgBackwardBadShape = model.addSynapsePopulation(
        "Synapses4", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Pre",
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto *sgBackwardSparse = model.addSynapsePopulation(
        "Synapses5", SynapseMatrixType::SPARSE, NO_DELAY,
        "Post", "Pre",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *sgBackwardBadType = model.addSynapsePopulation(
        "Synapses6", SynapseMatrixType::SPARSE, NO_DELAY,
        "Post", "Pre",
        initWeightUpdate<StaticPulseUInt>({{"g", 1.0}}, {}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

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
