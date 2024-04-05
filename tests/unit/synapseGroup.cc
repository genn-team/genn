// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);
    SET_PARAMS({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus").cast<double>()); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus").cast<double>()); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "const scalar dt = t - sT_post; \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g - (Aminus * postTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_POST_SPIKE_SYN_CODE(
        "const scalar dt = t - sT_pre;\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g + (Aplus * preTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("preTrace += 1.0;\n");
    SET_POST_SPIKE_CODE("postTrace += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("preTrace *= tauPlusDecay;\n");
    SET_POST_DYNAMICS_CODE("postTrace *= tauMinusDecay;\n");
};
IMPLEMENT_SNIPPET(STDPAdditive);

class STDPAdditiveSpikeParam : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditiveSpikeParam);
    SET_PARAMS({"tauPlus", "tauMinus", "Aplus", "Aminus",
                "Wmin", "Wmax", "S"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus").cast<double>()); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus").cast<double>()); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "const scalar dt = t - sT_post; \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g - (Aminus * postTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_POST_SPIKE_SYN_CODE(
        "const scalar dt = t - sT_pre;\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g + (Aplus * preTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("preTrace += S;\n");
    SET_POST_SPIKE_CODE("postTrace += S;\n");
    SET_PRE_DYNAMICS_CODE("preTrace *= tauPlusDecay;\n");
    SET_POST_DYNAMICS_CODE("postTrace *= tauMinusDecay;\n");
};
IMPLEMENT_SNIPPET(STDPAdditiveSpikeParam);

class STDPAdditiveDecayParam : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditiveDecayParam);
    SET_PARAMS({"Aplus", "Aminus", "Wmin", "Wmax", "tauPlusDecay", "tauMinusDecay"});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "const scalar dt = t - sT_post; \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g - (Aminus * postTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_POST_SPIKE_SYN_CODE(
        "const scalar dt = t - sT_pre;\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g + (Aplus * preTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("preTrace += 1.0;\n");
    SET_POST_SPIKE_CODE("postTrace += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("preTrace *= tauPlusDecay;\n");
    SET_POST_DYNAMICS_CODE("postTrace *= tauMinusDecay;\n");
};
IMPLEMENT_SNIPPET(STDPAdditiveDecayParam);

class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(Continuous);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("addToPost(g * V_pre);\n");
};
IMPLEMENT_SNIPPET(Continuous);

class ContinuousDenDelay : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(ContinuousDenDelay);

    SET_PARAMS({"g"});

    SET_SYNAPSE_DYNAMICS_CODE("addToPostDelay(g * V_pre, 1);\n");
};
IMPLEMENT_SNIPPET(ContinuousDenDelay);

class GradedDenDelay : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(GradedDenDelay);

    SET_PARAMS({"g"});
    SET_PRE_EVENT_THRESHOLD_CONDITION_CODE("V_pre >= 0.1");
    SET_PRE_EVENT_SYN_CODE("addToPostDelay(g * V_pre, 1);");
};
IMPLEMENT_SNIPPET(GradedDenDelay);

class StaticPulseDendriticDelayConstantWeight : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseDendriticDelayConstantWeight);

    SET_PARAMS({"g", "d"});

    SET_PRE_SPIKE_SYN_CODE("addToPostDelay(g, (uint8_t)d);\n");
};
IMPLEMENT_SNIPPET(StaticPulseDendriticDelayConstantWeight);

class StaticPulseDynamics : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseDynamics);

    SET_VARS({ {"g", "scalar"} });

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
    SET_SYNAPSE_DYNAMICS_CODE("g *= 0.99;\n");
};
IMPLEMENT_SNIPPET(StaticPulseDynamics);

class StaticPulsePostLearn : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulsePostLearn);

    SET_VARS({ {"g", "scalar"} });

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
    SET_POST_SPIKE_SYN_CODE("g *= 0.99;\n");
};
IMPLEMENT_SNIPPET(StaticPulsePostLearn);

class PostRepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PostRepeatVal);

    SET_CODE("value = values[id_post % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(PostRepeatVal);

class PreRepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PreRepeatVal);

    SET_CODE("value = values[id_pre % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(PreRepeatVal);

class Sum : public CustomUpdateModels::Base
{
    DECLARE_SNIPPET(Sum);

    SET_UPDATE_CODE("sum = a + b;\n");

    SET_CUSTOM_UPDATE_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}, 
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_SNIPPET(Sum);

class EmptyNeuron : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(EmptyNeuron);

    SET_THRESHOLD_CONDITION_CODE("false");
};
IMPLEMENT_SNIPPET(EmptyNeuron);

class LIFAdditional : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFAdditional);

   SET_ADDITIONAL_INPUT_VARS({{"Isyn2", "scalar", 0.0}});
    SET_SIM_CODE(
        "if (RefracTime <= 0.0) {\n"
        "  scalar alpha = ((Isyn2 + Ioffset) * Rmembrane) + Vrest;\n"
        "  V = alpha - (ExpTC * (alpha - V));\n"
        "}\n"
        "else {\n"
        "  RefracTime -= dt;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= Vthresh");

    SET_RESET_CODE(
        "V = Vreset;\n"
        "RefracTime = TauRefrac;\n");

    SET_PARAMS({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt) { return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rmembrane", [](const ParamValues &pars, double) { return  pars.at("TauM").cast<double>() / pars.at("C").cast<double>(); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});
};
IMPLEMENT_SNIPPET(LIFAdditional);
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(SynapseGroup, WUVarReferencedByCustomUpdate)
{
    ModelSpecInternal model;

    // Add two neuron group to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumPreVarVals{{"preTrace", 0.0}};
    VarValues wumPostVarVals{{"postTrace", 0.0}};

    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg3 = model.addSynapsePopulation(
        "Synapses3", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    VarValues sumVarValues{{"sum", 0.0}};
    WUVarReferences sumVarReferences2{{"a", createWUVarRef(sg2, "g")}, {"b", createWUVarRef(sg2, "g")}};
    VarReferences sumVarReferences3{{"a", createWUPreVarRef(sg3, "preTrace")}, {"b", createWUPreVarRef(sg3, "preTrace")}};

    model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences2);
    model.addCustomUpdate<Sum>("SumWeight3", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences3);
    model.finalise();

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(sg1)->getCustomUpdateReferences().empty());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(sg2)->getCustomUpdateReferences().empty());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(sg3)->getCustomUpdateReferences().empty());
}
//--------------------------------------------------------------------------
TEST(SynapseGroup, CompareWUDifferentModel)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    VarValues staticPulseVarVals{{"g", 0.1}};
    VarValues staticPulseDendriticVarVals{{"g", 0.1}, {"d", 1}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseDendriticDelay>({}, staticPulseDendriticVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPostsynapticUpdateGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDynamicsGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 2);

}

TEST(SynapseGroup, CompareWUDifferentParams)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues staticPulseAParamVals{{"g", 0.1}};
    ParamValues staticPulseBParamVals{{"g", 0.2}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseAParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseAParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseBParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that global g var is heterogeneous
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isWUParamHeterogeneous("g"));
}

TEST(SynapseGroup, CompareWUDifferentProceduralConnectivity)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParamsA{{"prob", 0.1}};
    ParamValues fixedProbParamsB{{"prob", 0.4}};
    ParamValues staticPulseParamVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(staticPulseParamVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsB));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());
}


TEST(SynapseGroup, CompareWUDifferentToeplitzConnectivity)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 64 * 64, paramVals, varVals);
    auto *post1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post1", 62 * 62, paramVals, varVals);
    auto *post2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post2", 64 * 64, paramVals, varVals);

    ParamValues convParamsA{
        {"conv_kh", 3}, {"conv_kw", 3},
        {"conv_ih", 64}, {"conv_iw", 64}, {"conv_ic", 1},
        {"conv_oh", 62}, {"conv_ow", 62}, {"conv_oc", 1}};

    ParamValues convParamsB{
        {"conv_kh", 3}, {"conv_kw", 3},
        {"conv_ih", 64}, {"conv_iw", 64}, {"conv_ic", 1},
        {"conv_oh", 64}, {"conv_ow", 64}, {"conv_oc", 1}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::TOEPLITZ,
        pre, post1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsA));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::TOEPLITZ,
        pre, post1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsA));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::TOEPLITZ,
        pre, post2,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsB));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_EQ(modelSpecMerged.getMergedNeuronUpdateGroups().size(), 3);
    ASSERT_EQ(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size(), 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_EQ(modelSpecMerged.getMergedSynapseInitGroups().size(), 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that connectivity parameter is heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_kh"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_kw"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ih"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_iw"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ic"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_oh"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ow"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_oc"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitDerivedParamHeterogeneous("conv_bh"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().at(0).isToeplitzConnectivityInitDerivedParamHeterogeneous("conv_bw"));
}

TEST(SynapseGroup, CompareWUDifferentProceduralVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues uniformParamsA{{"min", 0.5}, {"max", 1.0}};
    ParamValues uniformParamsB{{"min", 0.25}, {"max", 0.5}};
    VarValues staticPulseVarValsA{{"g", initVar<InitVarSnippet::Uniform>(uniformParamsA)}};
    VarValues staticPulseVarValsB{{"g", initVar<InitVarSnippet::Uniform>(uniformParamsB)}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::PROCEDURAL,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsB),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());
}

TEST(SynapseGroup, CompareWUDifferentProceduralSnippet)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    VarValues staticPulseVarValsA{{"g", initVar<PostRepeatVal>()}};
    VarValues staticPulseVarValsB{{"g", initVar<PreRepeatVal>()}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::DENSE_PROCEDURALG,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::DENSE_PROCEDURALG,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::DENSE_PROCEDURALG,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarValsB),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());
}

TEST(SynapseGroup, InitCompareWUDifferentVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues varValsA{{"g", 0.0}};
    VarValues varValsB{{"g", 1.0}};
    VarValues preVarVals{{"preTrace", 0.0}};
    VarValues postVarVals{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<STDPAdditive>(params, varValsA, preVarVals, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<STDPAdditive>(params, varValsA, preVarVals, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<STDPAdditive>(params, varValsB, preVarVals, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(ng0), sg1Internal->getWUPrePostInitHashDigest(ng0));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(ng0), sg2Internal->getWUPrePostInitHashDigest(ng0));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(ng1), sg1Internal->getWUPrePostInitHashDigest(ng1));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(ng1), sg2Internal->getWUPrePostInitHashDigest(ng1));

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 1);

    // Check that only synaptic weight initialistion parameters are heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitParamHeterogeneous("prob"));
    ASSERT_FALSE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitDerivedParamHeterogeneous("probLogRecip"));
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().at(0).isVarInitParamHeterogeneous("g", "constant"));
}

TEST(SynapseGroup, InitCompareWUDifferentPreVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues synVarVals{{"g", 0.0}};
    VarValues preVarValsA{{"preTrace", 0.0}};
    VarValues preVarValsB{{"preTrace", 1.0}};
    VarValues postVarVals{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarValsA, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarValsA, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarValsB, postVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(pre), sg1Internal->getWUPrePostInitHashDigest(pre));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(pre), sg2Internal->getWUPrePostInitHashDigest(pre));
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(post), sg1Internal->getWUPrePostInitHashDigest(post));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(post), sg2Internal->getWUPrePostInitHashDigest(post));
}

TEST(SynapseGroup, InitCompareWUDifferentPostVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues synVarVals{{"g", 0.0}};
    VarValues preVarVals{{"preTrace", 0.0}};
    VarValues postVarValsA{{"postTrace", 0.0}};
    VarValues postVarValsB{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarVals, postVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarVals, postVarValsA),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(params, synVarVals, preVarVals, postVarValsB),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(post), sg1Internal->getWUPrePostInitHashDigest(post));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(post), sg2Internal->getWUPrePostInitHashDigest(post));
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(pre), sg1Internal->getWUPrePostInitHashDigest(pre));
    ASSERT_EQ(sg0Internal->getWUPrePostInitHashDigest(pre), sg2Internal->getWUPrePostInitHashDigest(pre));
}

TEST(SynapseGroup, InitCompareWUDifferentHeterogeneousParamVarState)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedNumberPostParamsA{{"num", 4}};
    ParamValues fixedNumberPostParamsB{{"num", 8}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParamsA));
    auto *sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParamsB));
    // Finalize model
    model.finalise();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedPostsynapticUpdateGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDynamicsGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 1);

    // Check that fixed number post connectivity row length parameters are heterogeneous
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitParamHeterogeneous("num"));
}


TEST(SynapseGroup, InitCompareWUSynapseDynamicsPostLearn)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedNumberPostParams{{"num", 8}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto* sg0 = model.addSynapsePopulation(
        "Synapses0", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParams));
    auto* sg1 = model.addSynapsePopulation(
        "Synapses1", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<StaticPulseDynamics>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParams));
    auto* sg2 = model.addSynapsePopulation(
        "Synapses2", SynapseMatrixType::SPARSE,
        ng0, ng1,
        initWeightUpdate<StaticPulsePostLearn>({}, staticPulseVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>(),
        initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParams));
    // Finalize model
    model.finalise();

    SynapseGroupInternal* sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal* sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal* sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());
    ASSERT_NE(sg1Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_NE(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_NE(sg1Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticSpikeUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedPostsynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDynamicsGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 3);
}
TEST(SynapseGroup, InvalidMatrixTypes)
{
    ModelSpecInternal model;

    // Add four neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ngA = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsA", 10, paramVals, varVals);
    auto *ngB = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsB", 20, paramVals, varVals);

    // Check that making a synapse group with procedural connectivity fails if no connectivity initialiser is specified
    try {
        model.addSynapsePopulation(
        "NeuronsA_NeuronsB_1", SynapseMatrixType::PROCEDURAL,
        ngA, ngB,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Check that making a synapse group with procedural connectivity and STDP fails
    try {
        ParamValues fixedProbParams{{"prob", 0.1}};
        ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
        VarValues varVals{{"g", 0.0}};
        VarValues preVarVals{{"preTrace", 0.0}};
        VarValues postVarVals{{"postTrace", 0.0}};

        model.addSynapsePopulation(
            "NeuronsA_NeuronsB_2", SynapseMatrixType::PROCEDURAL,
            ngA, ngB,
            initWeightUpdate<STDPAdditive>(params, varVals, preVarVals, postVarVals),
            initPostsynaptic<PostsynapticModels::DeltaCurr>(),
            initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Check that making a synapse group with procedural connectivity and synapse dynamics fails
    try {
        ParamValues fixedProbParams{{"prob", 0.1}};
        model.addSynapsePopulation(
            "NeuronsA_NeuronsB_3", SynapseMatrixType::PROCEDURAL,
            ngA, ngB,
            initWeightUpdate<Continuous>({}, {{"g", 0.0}}, {}, {}),
            initPostsynaptic<PostsynapticModels::DeltaCurr>(),
            initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Check that making a synapse group with dense connections and procedural weights fails if var initialialisers use random numbers
    try {
        model.addSynapsePopulation(
            "NeuronsA_NeuronsB_4", SynapseMatrixType::DENSE_PROCEDURALG,
            ngA, ngB,
            initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}}),
            initPostsynaptic<PostsynapticModels::DeltaCurr>());
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}

TEST(SynapseGroup, IsDendriticDelayRequired)
{
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};

    ModelSpec model;
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

    ParamValues staticPulseDendriticParamVals{{"g", 0.1}, {"d", 1}};
    ParamValues gradedDenDelayParamVars{{"g", 0.1}};
    ParamValues contDenDelayParamVars{{"g", 0.1}};

    auto *syn = model.addSynapsePopulation(
        "Syn", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<StaticPulseDendriticDelayConstantWeight>(staticPulseDendriticParamVals, {}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *synGraded = model.addSynapsePopulation(
        "SynGraded", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<GradedDenDelay>(gradedDenDelayParamVars, {}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *synContinuous = model.addSynapsePopulation(
        "SynContinuous", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<ContinuousDenDelay>(contDenDelayParamVars, {}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(syn)->isDendriticDelayRequired());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(synGraded)->isDendriticDelayRequired());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(synContinuous)->isDendriticDelayRequired());
}

TEST(SynapseGroup, InvalidName)
{
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    
    ModelSpec model;
    auto *pre = model.addNeuronPopulation<EmptyNeuron>("Pre", 10, {}, {});
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    try {
        model.addSynapsePopulation(
            "Syn-6", SynapseMatrixType::DENSE,
            pre, post,
            initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>({{"g", 1.0}}),
            initPostsynaptic<PostsynapticModels::DeltaCurr>());
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}

TEST(SynapseGroup, CanWUMPreUpdateBeFused)
{
    ModelSpecInternal model;

    // Add pre and post neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumSpikeParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}, {"S", 1.0}};
    ParamValues wumDecayParams{{"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}, {"tauPlusDecay", 0.9}, {"tauMinusDecay", 0.9}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumConstPreVarVals{{"preTrace", 0.0}};
    VarValues wumNonConstPreVarVals{{"preTrace", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}};
    VarValues wumPostVarVals{{"postTrace", 0.0}};
    
    auto *constPre = model.addSynapsePopulation(
        "Pre_Post_ConstPre", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumConstPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *nonConstPre = model.addSynapsePopulation(
        "Pre_Post_NonConstPre", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumNonConstPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto *dynamicWMinMax = model.addSynapsePopulation(
        "Pre_Post_DynamicWMinMax", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumConstPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicWMinMax->setWUParamDynamic("Wmin", true);
    dynamicWMinMax->setWUParamDynamic("Wmax", true);

    auto *dynamicSpike = model.addSynapsePopulation(
        "Pre_Post_DynamicSpike", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditiveSpikeParam>(wumSpikeParams, wumVarVals, wumConstPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicSpike->setWUParamDynamic("S", true);

    auto *dynamicDecay = model.addSynapsePopulation(
        "Pre_Post_DynamicDecay", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditiveDecayParam>(wumDecayParams, wumVarVals, wumConstPreVarVals, wumPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicDecay->setWUParamDynamic("tauPlusDecay", true);
    dynamicDecay->setWUParamDynamic("tauMinusDecay", true);

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(constPre)->canWUMPrePostUpdateBeFused(pre));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(nonConstPre)->canWUMPrePostUpdateBeFused(pre));
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(dynamicWMinMax)->canWUMPrePostUpdateBeFused(pre));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(dynamicSpike)->canWUMPrePostUpdateBeFused(pre));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(dynamicDecay)->canWUMPrePostUpdateBeFused(pre));
}

TEST(SynapseGroup, CanWUMPostUpdateBeFused)
{
    ModelSpecInternal model;

    // Add pre and post neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumSpikeParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}, {"S", 1.0}};
    ParamValues wumDecayParams{{"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}, {"tauPlusDecay", 0.9}, {"tauMinusDecay", 0.9}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumPreVarVals{{"preTrace", 0.0}};
    VarValues wumConstPostVarVals{{"postTrace", 0.0}};
    VarValues wumNonConstPostVarVals{{"postTrace", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}};
    
    auto *constPost = model.addSynapsePopulation(
        "Pre_Post_ConstPost", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumConstPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    auto *nonConstPost = model.addSynapsePopulation(
        "Pre_Post_NonConstPost", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumNonConstPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    
    auto *dynamicWMinMax = model.addSynapsePopulation(
        "Pre_Post_DynamicWMinMax", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditive>(wumParams, wumVarVals, wumPreVarVals, wumConstPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicWMinMax->setWUParamDynamic("Wmin", true);
    dynamicWMinMax->setWUParamDynamic("Wmax", true);

    auto *dynamicSpike = model.addSynapsePopulation(
        "Pre_Post_DynamicSpike", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditiveSpikeParam>(wumSpikeParams, wumVarVals, wumPreVarVals, wumConstPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicSpike->setWUParamDynamic("S", true);

    auto *dynamicDecay = model.addSynapsePopulation(
        "Pre_Post_DynamicDecay", SynapseMatrixType::DENSE,
        pre, post,
        initWeightUpdate<STDPAdditiveDecayParam>(wumDecayParams, wumVarVals, wumPreVarVals, wumConstPostVarVals),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());
    dynamicDecay->setWUParamDynamic("tauPlusDecay", true);
    dynamicDecay->setWUParamDynamic("tauMinusDecay", true);

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(constPost)->canWUMPrePostUpdateBeFused(post));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(nonConstPost)->canWUMPrePostUpdateBeFused(post));
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(dynamicWMinMax)->canWUMPrePostUpdateBeFused(post));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(dynamicSpike)->canWUMPrePostUpdateBeFused(post));
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(dynamicDecay)->canWUMPrePostUpdateBeFused(post));
}

TEST(SynapseGroup, InvalidPSOutputVar)
{
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};

    ModelSpec model;
    auto *pre = model.addNeuronPopulation<EmptyNeuron>("Pre", 10, {}, {});
    auto *post = model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);
    auto *prePost = model.addSynapsePopulation(
        "PrePost", SynapseMatrixType::SPARSE,
        pre, post,
        initWeightUpdate<WeightUpdateModels::StaticPulse>({}, {{"g", 1.0}}),
        initPostsynaptic<PostsynapticModels::DeltaCurr>());

    prePost->setPostTargetVar("Isyn");
    prePost->setPostTargetVar("Isyn2");
    try {
        prePost->setPostTargetVar("NonExistent");
        FAIL();
    }
    catch (const std::runtime_error &) {
    }
}
