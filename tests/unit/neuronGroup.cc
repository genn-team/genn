// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

using namespace GeNN;

namespace
{
class StaticPulseBack : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseBack);

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "$(addToPre, $(g));\n");
};
IMPLEMENT_SNIPPET(StaticPulseBack);

class StaticPulseBackConstantWeight : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseBackConstantWeight);

    SET_PARAM_NAMES({"g"});

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "$(addToPre, $(g));\n");
};
IMPLEMENT_SNIPPET(StaticPulseBackConstantWeight);

class WeightUpdateModelPost : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(WeightUpdateModelPost);

    SET_PARAM_NAMES({"w", "p"});
    SET_POST_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_POST_SPIKE_CODE("$(s) = $(t) * $(p);\n");
};
IMPLEMENT_SNIPPET(WeightUpdateModelPost);

class WeightUpdateModelPre : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(WeightUpdateModelPre);

    SET_PARAM_NAMES({"w", "p"});
    SET_PRE_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_PRE_SPIKE_CODE("$(s) = $(t) * $(p);\n");
};
IMPLEMENT_SNIPPET(WeightUpdateModelPre);

class AlphaCurr : public PostsynapticModels::Base
{
public:
    DECLARE_SNIPPET(AlphaCurr);

    SET_DECAY_CODE(
        "$(x) = (dt * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));\n"
        "$(inSyn)*=$(expDecay);\n");

    SET_CURRENT_CONVERTER_CODE("$(x)");

    SET_PARAM_NAMES({"tau"});

    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const ParamValues &pars, double dt) { return std::exp(-dt / pars.at("tau")); }},
        {"init", [](const ParamValues &pars, double) { return (std::exp(1) / pars.at("tau")); }}});
};
IMPLEMENT_SNIPPET(AlphaCurr);

class LIFAdditional : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFAdditional);

    SET_ADDITIONAL_INPUT_VARS({{"Isyn2", "scalar", 0.0}});
    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  scalar alpha = (($(Isyn2) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= dt;\n"
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
};
IMPLEMENT_SNIPPET(LIFAdditional);

//----------------------------------------------------------------------------
// LIFRandom
//----------------------------------------------------------------------------
class LIFRandom : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFRandom);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset) + $(gennrand_normal)) * $(Rmembrane)) + $(Vrest);\n"
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
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM")); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("TauM") / pars.at("C"); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});
};
IMPLEMENT_SNIPPET(LIFRandom);

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
};
IMPLEMENT_SNIPPET(STDPAdditive);
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(NeuronGroup, InvalidName)
{
    ModelSpec model;
    try {
        model.addNeuronPopulation<NeuronModels::SpikeSource>("Neurons-0", 10, {}, {});
     FAIL();
    }
    catch(const std::runtime_error &) {
    }
}

TEST(NeuronGroup, ConstantVarIzhikevich)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    model.finalise();

    auto ngInternal = static_cast<NeuronGroupInternal*>(ng);
    ASSERT_FALSE(ngInternal->isZeroCopyEnabled());
    ASSERT_FALSE(ngInternal->isSimRNGRequired());
    ASSERT_FALSE(ngInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);
    ASSERT_FALSE(backend.isGlobalHostRNGRequired(model));
}

TEST(NeuronGroup, UninitialisedVarIzhikevich)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", uninitialisedVar()}, {"U", uninitialisedVar()}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    model.finalise();

    auto ngInternal = static_cast<NeuronGroupInternal*>(ng);
    ASSERT_FALSE(ngInternal->isZeroCopyEnabled());
    ASSERT_FALSE(ngInternal->isSimRNGRequired());
    ASSERT_FALSE(ngInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);
    ASSERT_FALSE(backend.isGlobalHostRNGRequired(model));
}

TEST(NeuronGroup, RandVarIzhikevich)
{
    ModelSpecInternal model;

    ParamValues dist{{"min", 0.0}, {"max", 1.0}};
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", initVar<InitVarSnippet::Uniform>(dist)}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    model.finalise();

    auto ngInternal = static_cast<NeuronGroupInternal*>(ng);
    ASSERT_FALSE(ngInternal->isZeroCopyEnabled());
    ASSERT_FALSE(ngInternal->isSimRNGRequired());
    ASSERT_TRUE(ngInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);
    ASSERT_TRUE(backend.isGlobalHostRNGRequired(model));
}

TEST(NeuronGroup, Poisson)
{
    ModelSpecInternal model;

    ParamValues paramVals{{"rate", 20.0}};
    VarValues varVals{{"timeStepToSpike", 0.0}};
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::PoissonNew>("Neurons0", 10, paramVals, varVals);

    model.finalise();

    auto ngInternal = static_cast<NeuronGroupInternal*>(ng);
    ASSERT_FALSE(ngInternal->isZeroCopyEnabled());
    ASSERT_TRUE(ngInternal->isSimRNGRequired());
    ASSERT_FALSE(ngInternal->isInitRNGRequired());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);
    ASSERT_TRUE(backend.isGlobalHostRNGRequired(model));
}

TEST(NeuronGroup, FuseWUMPrePost)
{
    ModelSpecInternal model;
    model.setFusePrePostWeightUpdateModels(true);
    
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    
    ParamValues wumParams{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumParamsPre{{"tauPlus", 17.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumParamsPost{{"tauPlus", 10.0}, {"tauMinus", 17.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    ParamValues wumParamsSyn{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 2.0}};
    VarValues wumVarVals{{"g", 0.0}};
    VarValues wumPreVarVals{{"preTrace", 0.0}};
    VarValues wumPostVarVals{{"postTrace", 0.0}};
    VarValues wumPreVarVals2{{"preTrace", 2.0}};
    VarValues wumPostVarVals2{{"postTrace", 2.0}};
    
    // Add two neuron groups to model
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    // Create baseline synapse group
    auto *syn = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter accessed in presynaptic code
    auto *synPreParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPreParam", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParamsPre, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter accessed in presynaptic code
    auto *synPostParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPostParam", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParamsPost, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter only accessed in synapse code
    auto *synSynParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynSynParam", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParamsSyn, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different presynaptic variable initialiser
    auto *synPreVar2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPreVar2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals2, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different postsynaptic variable initialiser
    auto *synPostVar2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPostVar2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals2,
        {}, {});
    
    // Create synapse group with axonal delay
    auto *synAxonalDelay = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynAxonalDelay", SynapseMatrixType::DENSE, 10,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with backprop delay
    auto *synBackPropDelay = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynBackPropDelay", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    synBackPropDelay->setBackPropDelaySteps(10);
    
    model.finalise();
    
    // Cast synapse groups to internal types
    auto synInternal = static_cast<SynapseGroupInternal*>(syn);
    auto synPreParamInternal = static_cast<SynapseGroupInternal*>(synPreParam);
    auto synPostParamInternal = static_cast<SynapseGroupInternal*>(synPostParam);
    auto synSynParamInternal = static_cast<SynapseGroupInternal*>(synSynParam);
    auto synPreVar2Internal = static_cast<SynapseGroupInternal*>(synPreVar2);
    auto synPostVar2Internal = static_cast<SynapseGroupInternal*>(synPostVar2);
    auto synAxonalDelayInternal = static_cast<SynapseGroupInternal*>(synAxonalDelay);
    auto synBackPropDelayInternal = static_cast<SynapseGroupInternal*>(synBackPropDelay);
    
    // Only postsynaptic update can be merged for synapse groups with different presynaptic parameters 
    ASSERT_NE(&synInternal->getFusedWUPreTarget(), &synPreParamInternal->getFusedWUPreTarget());
    ASSERT_EQ(&synInternal->getFusedWUPostTarget(), &synPreParamInternal->getFusedWUPostTarget());
    
    // Only presynaptic update can be merged for synapse groups with different postsynaptic parameters 
    ASSERT_EQ(&synInternal->getFusedWUPreTarget(), &synPostParamInternal->getFusedWUPreTarget());
    ASSERT_NE(&synInternal->getFusedWUPostTarget(), &synPostParamInternal->getFusedWUPostTarget());
    
    // Both types of update can be merged for synapse groups with parameters changes which don't effect pre or post update
    ASSERT_EQ(&synInternal->getFusedWUPreTarget(), &synSynParamInternal->getFusedWUPreTarget());
    ASSERT_EQ(&synInternal->getFusedWUPostTarget(), &synSynParamInternal->getFusedWUPostTarget());
    
    // Only postsynaptic update can be merged for synapse groups with different presynaptic variable initialisers 
    ASSERT_NE(&synInternal->getFusedWUPreTarget(), &synPreVar2Internal->getFusedWUPreTarget());
    ASSERT_EQ(&synInternal->getFusedWUPostTarget(), &synPreVar2Internal->getFusedWUPostTarget());
    
    // Only presynaptic update can be merged for synapse groups with different postsynaptic variable initialisers 
    ASSERT_EQ(&synInternal->getFusedWUPreTarget(), &synPostVar2Internal->getFusedWUPreTarget());
    ASSERT_NE(&synInternal->getFusedWUPostTarget(), &synPostVar2Internal->getFusedWUPostTarget());
    
    // Only postsynaptic update can be merged for synapse groups with different axonal delays
    ASSERT_NE(&synInternal->getFusedWUPreTarget(), &synAxonalDelayInternal->getFusedWUPreTarget());
    ASSERT_EQ(&synInternal->getFusedWUPostTarget(), &synAxonalDelayInternal->getFusedWUPostTarget());
    
    // Only presynaptic update can be merged for synapse groups with different back propagation delays
    ASSERT_EQ(&synInternal->getFusedWUPreTarget(), &synBackPropDelayInternal->getFusedWUPreTarget());
    ASSERT_NE(&synInternal->getFusedWUPostTarget(), &synBackPropDelayInternal->getFusedWUPostTarget());
}


TEST(NeuronGroup, FusePSM)
{
    ModelSpecInternal model;
    model.setMergePostsynapticModels(true);
    
   
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};
    ParamValues psmParamVals{{"tau", 5.0}};
    ParamValues psmParamVals2{{"tau", 10.0}};
    VarValues wumVarVals{{"g", 0.1}, {"d", 10}};
    
    // Add two neuron groups to model
    model.addNeuronPopulation<LIFAdditional>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);

    // Create baseline synapse group
    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "Syn", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    
    // Create second synapse group
    auto *syn2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "Syn2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    
    // Create synapse group with different value for PSM parameter
    auto *synParam = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynParam", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals2, {});
    
    // Create synapse group with different target variable
    auto *synTarget = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynTarget", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    synTarget->setPSTargetVar("Isyn2");
    
    // Create synapse group with different max dendritic delay
    auto *synDelay = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynDelay", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    synDelay->setMaxDendriticDelayTimesteps(20);
    
    model.finalise();
    
    // Cast synapse groups to internal types
    auto synInternal = static_cast<SynapseGroupInternal*>(syn);
    auto syn2Internal = static_cast<SynapseGroupInternal*>(syn2);
    auto synParamInternal = static_cast<SynapseGroupInternal*>(synParam);
    auto synTargetInternal = static_cast<SynapseGroupInternal*>(synTarget);
    auto synDelayInternal = static_cast<SynapseGroupInternal*>(synDelay);
 
    // Check all groups can be fused
    ASSERT_TRUE(synInternal->canPSBeFused());
    ASSERT_TRUE(syn2Internal->canPSBeFused());
    ASSERT_TRUE(synParamInternal->canPSBeFused());
    ASSERT_TRUE(synTargetInternal->canPSBeFused());
    ASSERT_TRUE(synDelayInternal->canPSBeFused());

    // Check that identically configured PSMs can be merged
    ASSERT_EQ(&synInternal->getFusedPSTarget(), &syn2Internal->getFusedPSTarget());
    
    // Check that PSMs with different parameters cannot be merged
    ASSERT_NE(&synInternal->getFusedPSTarget(), &synParamInternal->getFusedPSTarget());
    
    // Check that PSMs targetting different variables cannot be merged
    ASSERT_NE(&synInternal->getFusedPSTarget(), &synTargetInternal->getFusedPSTarget());
    
    // Check that PSMs from synapse groups with different dendritic delay cannot be merged
    ASSERT_NE(&synInternal->getFusedPSTarget(), &synDelayInternal->getFusedPSTarget());
}

TEST(NeuronGroup, FuseVarPSM)
{
    ModelSpecInternal model;
    model.setMergePostsynapticModels(true);
    
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};
    ParamValues psmParamVals{{"tau", 5.0}};
    VarValues psmVarValsConst1{{"x", 0.0}};
    VarValues psmVarValsConst2{{"x", 1.0}}; 
    VarValues psmVarValsRand{{"x", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 1.0}})}}; 
    VarValues wumVarVals{{"g", 0.1}, {"d", 10}};

    // Add two neuron groups to model
    auto *pre = model.addNeuronPopulation<LIFAdditional>("Pre", 10, paramVals, varVals);
    auto *post = model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);

    // Create baseline synapse group
    auto *syn1 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, AlphaCurr>(
        "Syn1", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, psmVarValsConst1);
    
    // Create second synapse group with same model and constant initialisers
    auto *syn2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, AlphaCurr>(
        "Syn2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, psmVarValsConst1);

   // Create third synapse group with same model and different constant initialisers
    auto *syn3 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, AlphaCurr>(
        "Syn3", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, psmVarValsConst2);
    
     // Create fourth synapse group with same model and random variable initialisers
    auto *syn4 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, AlphaCurr>(
        "Syn4", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, psmVarValsRand);
    
    
    // **TODO** third safe group with different variable initialisers
    model.finalise();
    
    // Cast neuron groups to internal types
    auto preInternal = static_cast<NeuronGroupInternal*>(pre);
    auto postInternal = static_cast<NeuronGroupInternal*>(post);

    // Cast synapse groups to internal types
    auto syn1Internal = static_cast<SynapseGroupInternal*>(syn1);
    auto syn2Internal = static_cast<SynapseGroupInternal*>(syn2);
    auto syn3Internal = static_cast<SynapseGroupInternal*>(syn3);
    auto syn4Internal = static_cast<SynapseGroupInternal*>(syn4);
    
    // Check only groups with 'safe' model can be fused
    ASSERT_TRUE(syn1Internal->canPSBeFused());
    ASSERT_TRUE(syn2Internal->canPSBeFused());
    ASSERT_TRUE(syn3Internal->canPSBeFused());
    ASSERT_FALSE(syn4Internal->canPSBeFused());
    
    // Check that identically configured PSMs can be merged
    ASSERT_EQ(&syn1Internal->getFusedPSTarget(), &syn2Internal->getFusedPSTarget());

    ASSERT_TRUE(preInternal->getFusedPSMInSyn().empty());
    ASSERT_EQ(postInternal->getFusedPSMInSyn().size(), 3);
}
TEST(NeuronGroup, FusePreOutput)
{
    ModelSpecInternal model;
    model.setMergePostsynapticModels(true);
    
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};
    ParamValues psmParamVals{{"tau", 5.0}};
    ParamValues psmParamVals2{{"tau", 10.0}};
    VarValues wumVarVals{{"g", 0.1}};
    
    // Add two neuron groups to model
    model.addNeuronPopulation<LIFAdditional>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);

    // Create baseline synapse group
    auto *syn = model.addSynapsePopulation<StaticPulseBack, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        {}, {});
    
    // Create second synapse group
    auto *syn2 = model.addSynapsePopulation<StaticPulseBack, PostsynapticModels::DeltaCurr>(
        "Syn2", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        {}, {});

    // Create synapse group with different target variable
    auto *synTarget = model.addSynapsePopulation<StaticPulseBack, PostsynapticModels::DeltaCurr>(
        "SynTarget", SynapseMatrixType::DENSE, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        {}, {});
    synTarget->setPreTargetVar("Isyn2");
  
    model.finalise();
    
    // Cast synapse groups to internal types
    auto synInternal = static_cast<SynapseGroupInternal*>(syn);
    auto syn2Internal = static_cast<SynapseGroupInternal*>(syn2);
    auto synTargetInternal = static_cast<SynapseGroupInternal*>(synTarget);
 
    // Check that identically configured PSMs can be merged
    ASSERT_EQ(&synInternal->getFusedPreOutputTarget(), &syn2Internal->getFusedPreOutputTarget());
    
    // Check that PSMs targetting different variables cannot be merged
    ASSERT_NE(&synInternal->getFusedPreOutputTarget(), &synTargetInternal->getFusedPreOutputTarget());
}

TEST(NeuronGroup, CompareNeuronModels)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramValsA{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    ParamValues paramValsB{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
    VarValues varVals1{{"V", initVar<InitVarSnippet::Uniform>({{"min", 0.0}, {"max", 30.0}})}, {"U", 0.0}};
    VarValues varVals2{{"V", initVar<InitVarSnippet::Uniform>({{"min", -10.0}, {"max", 30.0}})}, {"U", 0.0}};
    VarValues varVals3{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramValsA, varVals1);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramValsA, varVals2);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramValsB, varVals3);

    model.finalise();

    // Check that all groups can be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal*>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal*>(ng1);
    NeuronGroupInternal *ng2Internal = static_cast<NeuronGroupInternal*>(ng2);
    ASSERT_EQ(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getHashDigest(), ng2Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());
    ASSERT_NE(ng0Internal->getInitHashDigest(), ng2Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);

    // Check that only 'd' parameter is heterogeneous in neuron update group
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("a"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("b"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("c"));
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("d"));

    // Find which merged neuron init group is the one with the single population i.e. the one with constant initialisers
    const size_t constantInitIndex = (modelSpecMerged.getMergedNeuronInitGroups().at(0).getGroups().size() == 1) ? 0 : 1;
    const auto &constantInitMergedGroup = modelSpecMerged.getMergedNeuronInitGroups().at(constantInitIndex);
    const auto &uniformInitMergedGroup = modelSpecMerged.getMergedNeuronInitGroups().at(1 - constantInitIndex);

    // Check that only 'V' init 'min' parameter is heterogeneous
    ASSERT_FALSE(constantInitMergedGroup.isVarInitParamHeterogeneous("V", "constant"));
    ASSERT_FALSE(constantInitMergedGroup.isVarInitParamHeterogeneous("U", "constant"));
    ASSERT_TRUE(uniformInitMergedGroup.isVarInitParamHeterogeneous("V", "min"));
    ASSERT_FALSE(uniformInitMergedGroup.isVarInitParamHeterogeneous("V", "max"));
    ASSERT_FALSE(uniformInitMergedGroup.isVarInitParamHeterogeneous("U", "constant"));
    ASSERT_FALSE(uniformInitMergedGroup.isVarInitParamHeterogeneous("U", "constant"));
}

TEST(NeuronGroup, CompareHeterogeneousParamVarState)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramValsA{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    ParamValues paramValsB{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 1.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};
    auto *ng0 = model.addNeuronPopulation<LIFAdditional>("Neurons0", 10, paramValsA, varVals);
    auto *ng1 = model.addNeuronPopulation<LIFAdditional>("Neurons1", 10, paramValsB, varVals);

    model.finalise();

    // Check that all groups can be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal*>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal*>(ng1);
    ASSERT_EQ(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 1);

    // Check that only 'Ioffset' parameter is heterogeneous in neuron update group
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("C"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("TauM"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("Vrest"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("Vreset"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("Vthresh"));
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("Ioffset"));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous("TauRefrac"));
}


TEST(NeuronGroup, CompareSimRNG)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"C", 0.25}, {"TauM", 10.0}, {"Vrest", 0.0}, {"Vreset", 0.0}, {"Vthresh", 20.0}, {"Ioffset", 0.0}, {"TauRefrac", 5.0}};
    VarValues varVals{{"V", 0.0}, {"RefracTime", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::LIF>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<LIFRandom>("Neurons1", 10, paramVals, varVals);

    model.finalise();

    // Check that groups cannot be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal*>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal*>(ng1);
    ASSERT_NE(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_NE(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());

    ASSERT_TRUE(!ng0Internal->isSimRNGRequired());
    ASSERT_TRUE(ng1Internal->isSimRNGRequired());
}

TEST(NeuronGroup, CompareCurrentSources)
{
    ModelSpecInternal model;

    // Add four neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add one poisson exp and one DC current source to Neurons0
    ParamValues cs0ParamVals{{"weight", 0.1}, {"tauSyn", 20.0}, {"rate", 20.0}};
    ParamValues cs1ParamVals{{"weight", 0.1}, {"tauSyn", 40.0}, {"rate", 20.0}};
    VarValues cs0VarVals{{"current", 0.0}};
    VarValues cs1VarVals{{"current", 0.0}};
    ParamValues cs2ParamVals{{"amp", 0.4}};
    model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS0", "Neurons0", cs0ParamVals, cs0VarVals);
    model.addCurrentSource<CurrentSourceModels::DC>("CS1", "Neurons0", cs2ParamVals, {});

    // Do the same for Neuron1
    model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS2", "Neurons1", cs0ParamVals, cs0VarVals);
    model.addCurrentSource<CurrentSourceModels::DC>("CS3", "Neurons1", cs2ParamVals, {});

    // Do the same, but with different parameters for Neuron2
    model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS4", "Neurons2", cs1ParamVals, cs1VarVals);
    model.addCurrentSource<CurrentSourceModels::DC>("CS5", "Neurons2", cs2ParamVals, {});

    // Do the same, but in the opposite order for Neuron3
    model.addCurrentSource<CurrentSourceModels::DC>("CS6", "Neurons3", cs2ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS7", "Neurons3", cs0ParamVals, cs0VarVals);

    // Add two DC sources to Neurons4
    model.addCurrentSource<CurrentSourceModels::DC>("CS8", "Neurons4", cs2ParamVals, {});
    model.addCurrentSource<CurrentSourceModels::DC>("CS9", "Neurons4", cs2ParamVals, {});

    // **TODO** heterogeneous params
    model.finalise();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    NeuronGroupInternal *ng2Internal = static_cast<NeuronGroupInternal *>(ng2);
    NeuronGroupInternal *ng3Internal = static_cast<NeuronGroupInternal *>(ng3);
    NeuronGroupInternal *ng4Internal = static_cast<NeuronGroupInternal *>(ng4);
    ASSERT_EQ(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getHashDigest(), ng2Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getHashDigest(), ng3Internal->getHashDigest());
    ASSERT_NE(ng0Internal->getHashDigest(), ng4Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng2Internal->getInitHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng3Internal->getInitHashDigest());
    ASSERT_NE(ng0Internal->getInitHashDigest(), ng4Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check neurons are merged into two groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);

    // Find which merged neuron group is the one with the single population i.e. the two DC current sources
    const size_t dcDCIndex = (modelSpecMerged.getMergedNeuronUpdateGroups().at(0).getGroups().size() == 4) ? 1 : 0;
    const auto &dcDCMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(dcDCIndex);
    const auto &dcPoissonMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(1 - dcDCIndex);
    ASSERT_TRUE(dcDCMergedGroup.getGroups().size() == 1);
    
    // Find which child in the DC + poisson merged group is the poisson current source
    const size_t poissonIndex = (dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(0).getArchetype().getCurrentSourceModel() == CurrentSourceModels::PoissonExp::getInstance()) ? 0 : 1;
    
    // Check that only the ExpDecay and Init derived parameters of the poisson exp current sources are heterogeneous
    ASSERT_FALSE(dcDCMergedGroup.getMergedCurrentSourceGroups().at(0).isParamHeterogeneous("amp"));
    ASSERT_FALSE(dcDCMergedGroup.getMergedCurrentSourceGroups().at(1).isParamHeterogeneous("amp"));
    ASSERT_FALSE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isParamHeterogeneous("weight"));
    ASSERT_TRUE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isParamHeterogeneous("tauSyn"));
    ASSERT_FALSE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isParamHeterogeneous("rate"));
    ASSERT_FALSE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(1 - poissonIndex).isParamHeterogeneous("amp"));
    ASSERT_TRUE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isDerivedParamHeterogeneous("ExpDecay"));
    ASSERT_TRUE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isDerivedParamHeterogeneous("Init"));
    ASSERT_FALSE(dcPoissonMergedGroup.getMergedCurrentSourceGroups().at(poissonIndex).isDerivedParamHeterogeneous("ExpMinusLambda"));
}

TEST(NeuronGroup, ComparePostsynapticModels)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 10, {}, {});
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    ParamValues staticPulseParamVals{{"g", 0.1}};
    ParamValues alphaCurrParamVals{{"tau", 0.5}};
    ParamValues alphaCurrParamVals1{{"tau", 0.75}};
    VarValues alphaCurrVarVals{{"x", 0.0}};
    VarValues alphaCurrVarVals1{{"x", 0.1}};
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG0", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons0",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, AlphaCurr>(
        "SG1", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons0",
        staticPulseParamVals, {},
        alphaCurrParamVals, alphaCurrVarVals);

    // Do the same for Neuron1
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG2", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons1",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, AlphaCurr>(
        "SG3", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons1",
        staticPulseParamVals, {},
        alphaCurrParamVals, alphaCurrVarVals);

    // Do the same, but with different parameters for Neuron2,
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG4", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons2",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, AlphaCurr>(
        "SG5", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons2",
        staticPulseParamVals, {},
        alphaCurrParamVals1, alphaCurrVarVals1);

    // Do the same, but in the opposite order for Neuron3
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, AlphaCurr>(
        "SG6", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons3",
        staticPulseParamVals, {},
        alphaCurrParamVals, alphaCurrVarVals);
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG7", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons3",
        staticPulseParamVals, {},
        {}, {});

    // Add two incoming synapse groups with DeltaCurr postsynaptic models sources to Neurons4
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG8", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons4",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG9", SynapseMatrixType::SPARSE, NO_DELAY,
        "SpikeSource", "Neurons4",
        staticPulseParamVals, {},
        {}, {});

    model.finalise();

    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal *>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    NeuronGroupInternal *ng2Internal = static_cast<NeuronGroupInternal *>(ng2);
    NeuronGroupInternal *ng3Internal = static_cast<NeuronGroupInternal *>(ng3);
    NeuronGroupInternal *ng4Internal = static_cast<NeuronGroupInternal *>(ng4);
    ASSERT_EQ(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getHashDigest(), ng2Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getHashDigest(), ng3Internal->getHashDigest());
    ASSERT_NE(ng0Internal->getHashDigest(), ng4Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng2Internal->getInitHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng3Internal->getInitHashDigest());
    ASSERT_NE(ng0Internal->getInitHashDigest(), ng4Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a delta and an alpha psm
    const auto deltaAlphaMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                          [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto deltaAlphaMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                        [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Find which child in the DC + gaussian merged group is the gaussian current source
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->getMergedInSynPSMGroups().size() == 2);
    ASSERT_TRUE(deltaAlphaMergedInitGroup->getMergedInSynPSMGroups().size() == 2);
    const size_t alphaUpdateIndex = (deltaAlphaMergedUpdateGroup->getMergedInSynPSMGroups().at(0).getArchetype().getPSModel() == AlphaCurr::getInstance()) ? 0 : 1;
    const size_t alphaInitIndex = (deltaAlphaMergedInitGroup->getMergedInSynPSMGroups().at(0).getArchetype().getPSModel() == AlphaCurr::getInstance()) ? 0 : 1;

    // Check that parameter and both derived parameters are heterogeneous
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->getMergedInSynPSMGroups().at(alphaUpdateIndex).isParamHeterogeneous("tau"));
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->getMergedInSynPSMGroups().at(alphaUpdateIndex).isDerivedParamHeterogeneous("expDecay"));
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->getMergedInSynPSMGroups().at(alphaUpdateIndex).isDerivedParamHeterogeneous("init"));
    ASSERT_TRUE(deltaAlphaMergedInitGroup->getMergedInSynPSMGroups().at(alphaInitIndex).isVarInitParamHeterogeneous("x", "constant"));
}


TEST(NeuronGroup, ComparePreOutput)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    auto *ngPre0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPre0", 10, paramVals, varVals);
    auto *ngPre1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPre1", 10, paramVals, varVals);
    auto *ngPre2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPre2", 10, paramVals, varVals);
    auto *ngPre3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPre3", 10, paramVals, varVals);
    
    model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPost0", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPost1", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("NeuronsPost2", 10, paramVals, varVals);

    // Add two outgoing synapse groups to NeuronsPre0
    ParamValues staticPulseParamVals{{"g", 0.1}};
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG0", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre0", "NeuronsPost0",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG1", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre0", "NeuronsPost1",
        staticPulseParamVals, {},
        {}, {});

    // Do the same for NeuronsPre1
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG2", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre1", "NeuronsPost0",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG3", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre1", "NeuronsPost1",
        staticPulseParamVals, {},
        {}, {});
    
    // Add three outgoing groups to NeuronPre2
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG4", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre2", "NeuronsPost0",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG5", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre2", "NeuronsPost1",
        staticPulseParamVals, {},
        {}, {});
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG6", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre2", "NeuronsPost2",
        staticPulseParamVals, {},
        {}, {});

    // Add one outgoing groups to NeuronPre3
    model.addSynapsePopulation<StaticPulseBackConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG7", SynapseMatrixType::SPARSE, NO_DELAY,
        "NeuronsPre3", "NeuronsPost0",
        staticPulseParamVals, {},
        {}, {});

    model.finalise();

    NeuronGroupInternal *ngPre0Internal = static_cast<NeuronGroupInternal *>(ngPre0);
    NeuronGroupInternal *ngPre1Internal = static_cast<NeuronGroupInternal *>(ngPre1);
    NeuronGroupInternal *ngPre2Internal = static_cast<NeuronGroupInternal *>(ngPre2);
    NeuronGroupInternal *ngPre3Internal = static_cast<NeuronGroupInternal *>(ngPre3);
    ASSERT_EQ(ngPre0Internal->getHashDigest(), ngPre1Internal->getHashDigest());
    ASSERT_NE(ngPre0Internal->getHashDigest(), ngPre2Internal->getHashDigest());
    ASSERT_NE(ngPre0Internal->getHashDigest(), ngPre3Internal->getHashDigest());
    ASSERT_EQ(ngPre0Internal->getInitHashDigest(), ngPre1Internal->getInitHashDigest());
    ASSERT_NE(ngPre0Internal->getInitHashDigest(), ngPre2Internal->getInitHashDigest());
    ASSERT_NE(ngPre0Internal->getInitHashDigest(), ngPre3Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check neurons are merged into six groups (one for each output group and one for each number of incoming synapses)
    ASSERT_EQ(modelSpecMerged.getMergedNeuronUpdateGroups().size(), 6);
    ASSERT_EQ(modelSpecMerged.getMergedNeuronInitGroups().size(), 6);
}

TEST(NeuronGroup, CompareWUPreUpdate)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    ParamValues staticPulseParamVals{{"g", 0.1}};
    ParamValues testParams{{"w", 0.0}, {"p", 1.0}};
    ParamValues testParams2{{"w", 0.0}, {"p", 2.0}};
    VarValues testPreVarVals1{{"s", 0.0}};
    VarValues testPreVarVals2{{"s", 2.0}};

    // Connect neuron group 1 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons1", "Neurons0",
                                                                                    testParams, {}, testPreVarVals1, {},
                                                                                    {}, {});

    // Also connect neuron group 2 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons2", "Neurons0",
                                                                                    testParams, {}, testPreVarVals1, {},
                                                                                    {}, {});

    // Also connect neuron group 3 to neuron group 0 with pre weight update model, but different parameters
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons3", "Neurons0",
                                                                                    testParams2, {}, testPreVarVals1, {},
                                                                                    {}, {});

    // Connect neuron group 4 to neuron group 0 with 2*pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, {}, testPreVarVals1, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, {}, testPreVarVals1, {},
                                                                                    {}, {});

    // Connect neuron group 5 to neuron group 0 with pre weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                    "Neurons5", "Neurons0",
                                                                                    testParams, {}, testPreVarVals2, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG6", SynapseMatrixType::SPARSE, NO_DELAY,
        "Neurons5", "Neurons0",
        staticPulseParamVals, {},
        {}, {});
    model.finalise();

    // Check which groups can be merged together
    // **NOTE** NG1 and NG5 can be merged because the additional static pulse synapse population doesn't add any presynaptic update
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    NeuronGroupInternal *ng2Internal = static_cast<NeuronGroupInternal *>(ng2);
    NeuronGroupInternal *ng3Internal = static_cast<NeuronGroupInternal *>(ng3);
    NeuronGroupInternal *ng4Internal = static_cast<NeuronGroupInternal *>(ng4);
    NeuronGroupInternal *ng5Internal = static_cast<NeuronGroupInternal *>(ng5);
    ASSERT_EQ(ng1Internal->getHashDigest(), ng2Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getHashDigest(), ng3Internal->getHashDigest());
    ASSERT_NE(ng1Internal->getHashDigest(), ng4Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getHashDigest(), ng5Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng2Internal->getInitHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng3Internal->getInitHashDigest());
    ASSERT_NE(ng1Internal->getInitHashDigest(), ng4Internal->getInitHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng5Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check neuron init and update is merged into three groups (NG0 with no outsyns, NG1, NG2, NG3 and NG5 with 1 outsyn and NG4with 2 outsyns)
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPreMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                      [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto wumPreMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                    [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPreMergedUpdateGroup->getMergedOutSynWUMPreCodeGroups().at(0).isParamHeterogeneous("p"));
    ASSERT_TRUE(wumPreMergedInitGroup->getMergedOutSynWUMPreVarGroups().at(0).isVarInitParamHeterogeneous("s", "constant"));
}

TEST(NeuronGroup, CompareWUPostUpdate)
{
    ModelSpecInternal model;

    // **NOTE** we make sure merging is on so last test doesn't fail on that basis
    model.setMergePostsynapticModels(true);

    // Add two neuron groups to model
    ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    VarValues varVals{{"V", 0.0}, {"U", 0.0}};
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    ParamValues staticPulseParamVals{{"g", 0.1}};
    ParamValues testParams{{"w", 0.0}, {"p", 1.0}};
    ParamValues testParams2{{"w", 0.0}, {"p", 2.0}};
    VarValues testPostVarVals1{{"s", 0.0}};
    VarValues testPostVarVals2{{"s", 2.0}};

    // Connect neuron group 0 to neuron group 1 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons1",
                                                                                     testParams, {}, {}, testPostVarVals1,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 2 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons2",
                                                                                     testParams, {}, {}, testPostVarVals1,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 3 with post weight update model but different parameters
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons3",
                                                                                     testParams2, {}, {}, testPostVarVals1,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 3 with 2*post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, {}, {}, testPostVarVals1,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, {}, {}, testPostVarVals1,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 4 with post weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE, NO_DELAY,
                                                                                     "Neurons0", "Neurons5",
                                                                                     testParams, {}, {}, testPostVarVals2,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
        "SG6", SynapseMatrixType::SPARSE, NO_DELAY,
        "Neurons0", "Neurons5",
        staticPulseParamVals, {},
        {}, {});
    model.finalise();

    // **NOTE** NG1 and NG5 can be merged because the additional static pulse synapse population doesn't add any presynaptic update
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal *>(ng1);
    NeuronGroupInternal *ng2Internal = static_cast<NeuronGroupInternal *>(ng2);
    NeuronGroupInternal *ng3Internal = static_cast<NeuronGroupInternal *>(ng3);
    NeuronGroupInternal *ng4Internal = static_cast<NeuronGroupInternal *>(ng4);
    NeuronGroupInternal *ng5Internal = static_cast<NeuronGroupInternal *>(ng5);
    ASSERT_EQ(ng1Internal->getHashDigest(), ng2Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getHashDigest(), ng3Internal->getHashDigest());
    ASSERT_NE(ng1Internal->getHashDigest(), ng4Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getHashDigest(), ng5Internal->getHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng2Internal->getInitHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng3Internal->getInitHashDigest());
    ASSERT_NE(ng1Internal->getInitHashDigest(), ng4Internal->getInitHashDigest());
    ASSERT_EQ(ng1Internal->getInitHashDigest(), ng5Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(backend, model);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPostMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                       [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto wumPostMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                     [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPostMergedUpdateGroup->getMergedInSynWUMPostCodeGroups().at(0).isParamHeterogeneous("p"));
    ASSERT_TRUE(wumPostMergedInitGroup->getMergedInSynWUMPostVarGroups().at(0).isVarInitParamHeterogeneous("s", "constant"));
}
