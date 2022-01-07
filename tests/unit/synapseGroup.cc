// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus")); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus")); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * $(postTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * $(preTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += 1.0;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauPlusDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauMinusDecay);\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_SNIPPET(STDPAdditive);

class STDPAdditiveEGPWMinMax : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditiveEGPWMinMax);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus")); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus")); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"Wmin", "scalar"}, {"Wmax", "scalar"}});
    
    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * $(postTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * $(preTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += 1.0;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauPlusDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauMinusDecay);\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_SNIPPET(STDPAdditiveEGPWMinMax);

class STDPAdditiveEGPSpike : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditiveEGPSpike);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus")); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus")); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"S", "scalar"}});
    
    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * $(postTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * $(preTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += $(S);\n");
    SET_POST_SPIKE_CODE("$(postTrace) += $(S);\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauPlusDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauMinusDecay);\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_SNIPPET(STDPAdditiveEGPSpike);

class STDPAdditiveEGPDynamics : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditiveEGPDynamics);
    SET_PARAM_NAMES({"Aplus", "Aminus", "Wmin", "Wmax"});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    SET_EXTRA_GLOBAL_PARAMS({{"tauPlusDecay", "scalar"}, {"tauMinusDecay", "scalar"}});
    
    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * $(postTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * $(preTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += 1.0;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauPlusDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauMinusDecay);\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_SNIPPET(STDPAdditiveEGPDynamics);

class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(Continuous);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(g) * $(V_pre));\n");
};
IMPLEMENT_SNIPPET(Continuous);

class PostRepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PostRepeatVal);

    SET_CODE("$(value) = $(values)[$(id_post) % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(PostRepeatVal);

class PreRepeatVal : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(PreRepeatVal);

    SET_CODE("$(value) = $(values)[$(id_re) % 10];");

    SET_EXTRA_GLOBAL_PARAMS({{"values", "scalar*"}});
};
IMPLEMENT_SNIPPET(PreRepeatVal);

class Sum : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum, 2);

    SET_UPDATE_CODE("$(sum) = $(a) + $(b);\n");

    SET_VARS({{"sum", "scalar"}});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}, 
                  {"b", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_SNIPPET(Sum);

class LIFAdditional : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFAdditional);

    SET_ADDITIONAL_INPUT_VARS({{"Isyn2", "scalar", "$(Ioffset)"}});
    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  scalar alpha = ($(Isyn2) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt) { return std::exp(-dt / pars.at("TauM")); }},
        {"Rmembrane", [](const ParamValues &pars, double) { return  pars.at("TauM") / pars.at("C"); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
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
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 25, paramVals, varVals);

    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumPreVarVals{{"preTrace", 0.0}};
    VarValues wumPostVarVals{{"postTrace", 0.0}};

    auto *sg1 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Synapses1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    auto *sg2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Synapses2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    auto *sg3 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Synapses3", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});

    
    VarValues sumVarValues{{"sum", 0.0}};
    Sum::WUVarReferences sumVarReferences2(createWUVarRef(sg2, "g"), createWUVarRef(sg2, "g"));
    Sum::VarReferences sumVarReferences3(createWUPreVarRef(sg3, "preTrace"), createWUPreVarRef(sg3, "preTrace"));

    model.addCustomUpdate<Sum>("SumWeight2", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences2);
    model.addCustomUpdate<Sum>("SumWeight3", "CustomUpdate",
                               {}, sumVarValues, sumVarReferences3);
    model.finalize();

    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(sg1)->areWUVarReferencedByCustomUpdate());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(sg2)->areWUVarReferencedByCustomUpdate());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(sg3)->areWUVarReferencedByCustomUpdate());
}
//--------------------------------------------------------------------------
TEST(SynapseGroup, CompareWUDifferentModel)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    VarValues staticPulseVarVals{{"g", 0.1}};
    VarValues staticPulseDendriticVarVals{{"g", 0.1}, {"d", 1}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {});
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                                                         "Neurons0", "Neurons1",
                                                                                                                         {}, staticPulseDendriticVarVals,
                                                                                                                         {}, {});
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPostsynapticUpdateGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDynamicsGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 2);

}

TEST(SynapseGroup, CompareWUDifferentGlobalG)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    VarValues staticPulseAVarVals{{"g", 0.1}};
    VarValues staticPulseBVarVals{{"g", 0.2}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseAVarVals,
                                                                                                           {}, {});
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseAVarVals,
                                                                                                           {}, {});
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseBVarVals,
                                                                                                           {}, {});
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that global g var is heterogeneous
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isWUGlobalVarHeterogeneous(0));
}

TEST(SynapseGroup, CompareWUDifferentProceduralConnectivity)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParamsA{{"prob", 0.1}};
    ParamValues fixedProbParamsB{{"prob", 0.4}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA));
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA));
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsB));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that connectivity parameter is heterogeneous
    // **NOTE** raw parameter is NOT as only derived parameter is used in code
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isSparseConnectivityInitParamHeterogeneous("prob"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isSparseConnectivityInitDerivedParamHeterogeneous("probLogRecip"));
}


TEST(SynapseGroup, CompareWUDifferentToeplitzConnectivity)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 64 * 64, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post1", 62 * 62, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post2", 64 * 64, paramVals, varVals);

    ParamValues convParamsA{
        {"conv_kh", 3}, {"conv_kw", 3},
        {"conv_ih", 64}, {"conv_iw", 64}, {"conv_ic", 1},
        {"conv_oh", 62}, {"conv_ow", 62}, {"conv_oc", 1}};

    ParamValues convParamsB{
        {"conv_kh", 3}, {"conv_kw", 3},
        {"conv_ih", 64}, {"conv_iw", 64}, {"conv_ic", 1},
        {"conv_oh", 64}, {"conv_ow", 64}, {"conv_oc", 1}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY,
                                                                                                           "Pre", "Post1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsA));
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY,
                                                                                                           "Pre", "Post1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsA));
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::TOEPLITZ_KERNELG, NO_DELAY,
                                                                                                           "Pre", "Post2",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initToeplitzConnectivity<InitToeplitzConnectivitySnippet::Conv2D>(convParamsB));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_EQ(modelSpecMerged.getMergedNeuronUpdateGroups().size(), 3);
    ASSERT_EQ(modelSpecMerged.getMergedPresynapticUpdateGroups().size(), 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that connectivity parameter is heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_kh"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_kw"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ih"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_iw"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ic"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_oh"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_ow"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitParamHeterogeneous("conv_oc"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitDerivedParamHeterogeneous("conv_bh"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isToeplitzConnectivityInitDerivedParamHeterogeneous("conv_bw"));
}

TEST(SynapseGroup, CompareWUDifferentProceduralVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues uniformParamsA{{"min", 0.5}, {"max", 1.0}};
    ParamValues uniformParamsB{{"min", 0.25}, {"max", 0.5}};
    VarValues staticPulseVarValsA{{"g", initVar<InitVarSnippet::Uniform>(uniformParamsA)}};
    VarValues staticPulseVarValsB{{"g", initVar<InitVarSnippet::Uniform>(uniformParamsB)}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsA,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsA,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::PROCEDURAL_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsB,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());

    // Check that only synaptic weight initialistion parameters are heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isSparseConnectivityInitParamHeterogeneous("prob"));
    ASSERT_FALSE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isSparseConnectivityInitDerivedParamHeterogeneous("probLogRecip"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isWUVarInitParamHeterogeneous(0, "min"));
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().at(0).isWUVarInitParamHeterogeneous(0, "max"));
}

TEST(SynapseGroup, CompareWUDifferentProceduralSnippet)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    VarValues staticPulseVarValsA{{"g", initVar<PostRepeatVal>()}};
    VarValues staticPulseVarValsB{{"g", initVar<PreRepeatVal>()}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::DENSE_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsA,
                                                                                                           {}, {});
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::DENSE_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsA,
                                                                                                           {}, {});
    auto *sg2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::DENSE_PROCEDURALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarValsB,
                                                                                                           {}, {});
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal*>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal*>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal*>(sg2);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_NE(sg0Internal->getWUHashDigest(), sg2Internal->getWUHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().empty());
}

TEST(SynapseGroup, InitCompareWUDifferentVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues varValsA{{"g", 0.0}};
    VarValues varValsB{{"g", 1.0}};
    VarValues preVarVals{{"preTrace", 0.0}};
    VarValues postVarVals{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, varValsA, preVarVals, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, varValsA, preVarVals, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, varValsB, preVarVals, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg1Internal->getWUPreInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg2Internal->getWUPreInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg1Internal->getWUPostInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg2Internal->getWUPostInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDenseInitGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 1);

    // Check that only synaptic weight initialistion parameters are heterogeneous
    ASSERT_FALSE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitParamHeterogeneous("prob"));
    ASSERT_FALSE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitDerivedParamHeterogeneous("prob"));
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().at(0).isWUVarInitParamHeterogeneous(0, "constant"));
}

TEST(SynapseGroup, InitCompareWUDifferentPreVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues synVarVals{{"g", 0.0}};
    VarValues preVarValsA{{"preTrace", 0.0}};
    VarValues preVarValsB{{"preTrace", 1.0}};
    VarValues postVarVals{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarValsA, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarValsA, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarValsB, postVarVals,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg1Internal->getWUPreInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg2Internal->getWUPreInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg1Internal->getWUPostInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg2Internal->getWUPostInitHashDigest());
}

TEST(SynapseGroup, InitCompareWUDifferentPostVars)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedProbParams{{"prob", 0.1}};
    ParamValues params{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues synVarVals{{"g", 0.0}};
    VarValues preVarVals{{"preTrace", 0.0}};
    VarValues postVarValsA{{"postTrace", 0.0}};
    VarValues postVarValsB{{"postTrace", 0.0}};
    auto *sg0 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarVals, postVarValsA,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg1 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarVals, postVarValsA,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    auto *sg2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("Synapses2", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                        "Neurons0", "Neurons1",
                                                                                        params, synVarVals, preVarVals, postVarValsB,
                                                                                        {}, {},
                                                                                        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    SynapseGroupInternal *sg2Internal = static_cast<SynapseGroupInternal *>(sg2);
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg1Internal->getWUPostInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPostInitHashDigest(), sg2Internal->getWUPostInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg2Internal->getWUInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg1Internal->getWUPreInitHashDigest());
    ASSERT_EQ(sg0Internal->getWUPreInitHashDigest(), sg2Internal->getWUPreInitHashDigest());
}

TEST(SynapseGroup, InitCompareWUDifferentHeterogeneousParamVarState)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);

    ParamValues fixedNumberPostParamsA{{"rowLength", 4}};
    ParamValues fixedNumberPostParamsB{{"rowLength", 8}};
    VarValues staticPulseVarVals{{"g", 0.1}};
    auto *sg0 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses0", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParamsA));
    auto *sg1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                                           "Neurons0", "Neurons1",
                                                                                                           {}, staticPulseVarVals,
                                                                                                           {}, {},
                                                                                                           initConnectivity<InitSparseConnectivitySnippet::FixedNumberPostWithReplacement>(fixedNumberPostParamsB));
    // Finalize model
    model.finalize();

    SynapseGroupInternal *sg0Internal = static_cast<SynapseGroupInternal *>(sg0);
    SynapseGroupInternal *sg1Internal = static_cast<SynapseGroupInternal *>(sg1);
    ASSERT_EQ(sg0Internal->getWUHashDigest(), sg1Internal->getWUHashDigest());
    ASSERT_EQ(sg0Internal->getWUInitHashDigest(), sg1Internal->getWUInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedPresynapticUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedPostsynapticUpdateGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseDynamicsGroups().empty());
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseSparseInitGroups().size() == 1);

    // Check that fixed number post connectivity row length parameters are heterogeneous
    ASSERT_TRUE(modelSpecMerged.getMergedSynapseConnectivityInitGroups().at(0).isSparseConnectivityInitParamHeterogeneous("rowLength"));
}

TEST(SynapseGroup, InvalidMatrixTypes)
{
    ModelSpecInternal model;

    // Add four neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsA", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsB", 20, paramVals, varVals);

    // Check that making a synapse group with procedural connectivity fails if no connectivity initialiser is specified
    try {
        model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
            "NeuronsA_NeuronsB_1", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
            "NeuronsA", "NeuronsB",
            {}, {{"g", 1.0}},
            {}, {});
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

        model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>("NeuronsA_NeuronsB_2", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
                                                                                "NeuronsA", "NeuronsB",
                                                                                params, varVals, preVarVals, postVarVals,
                                                                                {}, {},
                                                                                initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Check that making a synapse group with procedural connectivity and synapse dynamics fails
    try {
        ParamValues fixedProbParams{{"prob", 0.1}};
        model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>("NeuronsA_NeuronsB_3", SynapseMatrixType::PROCEDURAL_GLOBALG, NO_DELAY,
                                                                              "NeuronsA", "NeuronsB",
                                                                              {}, {{"g", 0.0}}, {}, {},
                                                                              {}, {},
                                                                              initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParams));
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Check that making a synapse group with dense connections and procedural weights fails if var initialialisers use random numbers
    try {
        model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
            "NeuronsA_NeuronsB_4", SynapseMatrixType::DENSE_PROCEDURALG, NO_DELAY,
            "NeuronsA", "NeuronsB",
            {}, {{"g", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}},
            {}, {});
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}

TEST(SynapseGroup, InvalidName)
{
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    
    ModelSpec model;
    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    try {
        model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
            "Syn-6", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
            "Pre", "Post",
            {}, {{"g", 1.0}},
            {}, {});
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
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumEGPWMinMaxParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}};
    ParamValues wumEGPDynamicsParams{{"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumConstPreVarVals{{"preTrace", 0.0}};
    VarValues wumNonConstPreVarVals{{"preTrace", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}};
    VarValues wumPostVarVals{{"postTrace", 0.0}};
    
    auto *constPre = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Pre_Post_ConstPre", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumConstPreVarVals, wumPostVarVals,
        {}, {});

    auto *nonConstPre = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Pre_Post_NonConstPre", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumNonConstPreVarVals, wumPostVarVals,
        {}, {});
    
    auto *egpWMinMax = model.addSynapsePopulation<STDPAdditiveEGPWMinMax, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPWMinMax", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumEGPWMinMaxParams, wumVarVals, wumConstPreVarVals, wumPostVarVals,
        {}, {});
    
    auto *egpSpike = model.addSynapsePopulation<STDPAdditiveEGPSpike, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPSpike", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumConstPreVarVals, wumPostVarVals,
        {}, {});
    
    auto *egpDynamics = model.addSynapsePopulation<STDPAdditiveEGPDynamics, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPDynamics", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumEGPDynamicsParams, wumVarVals, wumConstPreVarVals, wumPostVarVals,
        {}, {});

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(constPre)->canWUMPreUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(nonConstPre)->canWUMPreUpdateBeFused());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(egpWMinMax)->canWUMPreUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(egpSpike)->canWUMPreUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(egpDynamics)->canWUMPreUpdateBeFused());
}

TEST(SynapseGroup, CanWUMPostUpdateBeFused)
{
    ModelSpecInternal model;

    // Add pre and post neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"v", 0.0}, {"u", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumEGPWMinMaxParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}};
    ParamValues wumEGPDynamicsParams{{"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumPreVarVals{{"preTrace", 0.0}};
    VarValues wumConstPostVarVals{{"postTrace", 0.0}};
    VarValues wumNonConstPostVarVals{{"postTrace", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}};
    
    auto *constPost = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Pre_Post_ConstPost", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumConstPostVarVals,
        {}, {});

    auto *nonConstPost = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Pre_Post_NonConstPost", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumNonConstPostVarVals,
        {}, {});
    
    auto *egpWMinMax = model.addSynapsePopulation<STDPAdditiveEGPWMinMax, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPWMinMax", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumEGPWMinMaxParams, wumVarVals, wumPreVarVals, wumConstPostVarVals,
        {}, {});
    
    auto *egpSpike = model.addSynapsePopulation<STDPAdditiveEGPSpike, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPSpike", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumConstPostVarVals,
        {}, {});
    
    auto *egpDynamics = model.addSynapsePopulation<STDPAdditiveEGPDynamics, PostsynapticModels::DeltaCurr>(
        "Pre_Post_EGPDynamics", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumEGPDynamicsParams, wumVarVals, wumPreVarVals, wumConstPostVarVals,
        {}, {});

    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(constPost)->canWUMPostUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(nonConstPost)->canWUMPostUpdateBeFused());
    ASSERT_TRUE(static_cast<SynapseGroupInternal*>(egpWMinMax)->canWUMPostUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(egpSpike)->canWUMPostUpdateBeFused());
    ASSERT_FALSE(static_cast<SynapseGroupInternal*>(egpDynamics)->canWUMPostUpdateBeFused());
}

TEST(SynapseGroup, InvalidPSOutputVar)
{
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};

    ModelSpec model;
    model.addNeuronPopulation<NeuronModels::SpikeSource>("Pre", 10, {}, {});
    model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);
    auto *prePost = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "PrePost", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, {{"g", 1.0}},
        {}, {});

    prePost->setPSTargetVar("Isyn");
    prePost->setPSTargetVar("Isyn2");
    try {
        prePost->setPSTargetVar("NonExistent");
        FAIL();
    }
    catch (const std::runtime_error &) {
    }
}
