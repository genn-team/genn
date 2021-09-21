// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

namespace
{
class WeightUpdateModelPost : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPost, 1, 1, 0, 1);

    SET_VARS({{"w", "scalar"}});
    SET_PARAM_NAMES({"p"});
    SET_POST_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_POST_SPIKE_CODE("$(s) = $(t) * $(p);\n");
};
IMPLEMENT_MODEL(WeightUpdateModelPost);

class WeightUpdateModelPre : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(WeightUpdateModelPre, 1, 1, 1, 0);

    SET_VARS({{"w", "scalar"}});
    SET_PARAM_NAMES({"p"});
    SET_PRE_VARS({{"s", "scalar"}});

    SET_SIM_CODE("$(w)= $(s);\n");
    SET_PRE_SPIKE_CODE("$(s) = $(t) * $(p);\n");
};
IMPLEMENT_MODEL(WeightUpdateModelPre);

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

class LIFAdditional : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFAdditional, 7, 2);

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
        {"ExpTC", [](const std::vector<double> &pars, double dt) { return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double) { return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(LIFAdditional);

//----------------------------------------------------------------------------
// LIFRandom
//----------------------------------------------------------------------------
class LIFRandom : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFRandom, 7, 2);

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
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(LIFRandom);

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(STDPAdditive, 6, 1, 1, 1);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"tauMinusDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }}});
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
IMPLEMENT_MODEL(STDPAdditive);
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
    ModelSpec model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, UnitialisedVarIzhikevich)
{
    ModelSpec model;

    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(uninitialisedVar(), uninitialisedVar());
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, UnitialisedVarRand)
{
    ModelSpec model;

    InitVarSnippet::Uniform::ParamValues dist(0.0, 1.0);
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, initVar<InitVarSnippet::Uniform>(dist));
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_FALSE(ng->isSimRNGRequired());
    ASSERT_TRUE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, Poisson)
{
    ModelSpec model;

    NeuronModels::PoissonNew::ParamValues paramVals(20.0);
    NeuronModels::PoissonNew::VarValues varVals(0.0);
    NeuronGroup *ng = model.addNeuronPopulation<NeuronModels::PoissonNew>("Neurons0", 10, paramVals, varVals);

    ASSERT_FALSE(ng->isZeroCopyEnabled());
    ASSERT_TRUE(ng->isSimRNGRequired());
    ASSERT_FALSE(ng->isInitRNGRequired());
}

TEST(NeuronGroup, MergeWUMPrePost)
{
    ModelSpecInternal model;
    model.setMergePrePostWeightUpdateModels(true);
    
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    
    STDPAdditive::ParamValues wumParams(10.0, 10.0, 0.01, 0.01, 0.0, 1.0);
    STDPAdditive::ParamValues wumParamsPre(17.0, 10.0, 0.01, 0.01, 0.0, 1.0);
    STDPAdditive::ParamValues wumParamsPost(10.0, 17.0, 0.01, 0.01, 0.0, 1.0);
    STDPAdditive::ParamValues wumParamsSyn(10.0, 10.0, 0.01, 0.01, 0.0, 2.0);
    STDPAdditive::VarValues wumVarVals(0.0);
    STDPAdditive::PreVarValues wumPreVarVals(0.0);
    STDPAdditive::PostVarValues wumPostVarVals(0.0);
    STDPAdditive::PreVarValues wumPreVarVals2(2.0);
    STDPAdditive::PostVarValues wumPostVarVals2(2.0);
    
    // Add two neuron groups to model
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);
    
    // Create baseline synapse group
    auto *syn = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter accessed in presynaptic code
    auto *synPreParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPreParam", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParamsPre, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter accessed in presynaptic code
    auto *synPostParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPostParam", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParamsPost, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different value for parameter only accessed in synapse code
    auto *synSynParam = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynSynParam", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParamsSyn, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different presynaptic variable initialiser
    auto *synPreVar2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPreVar2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals2, wumPostVarVals,
        {}, {});
    
    // Create synapse group with different postsynaptic variable initialiser
    auto *synPostVar2 = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynPostVar2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals2,
        {}, {});
    
    // Create synapse group with axonal delay
    auto *synAxonalDelay = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynAxonalDelay", SynapseMatrixType::DENSE_INDIVIDUALG, 10,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    
    // Create synapse group with backprop delay
    auto *synBackPropDelay = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
        "SynBackPropDelay", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        wumParams, wumVarVals, wumPreVarVals, wumPostVarVals,
        {}, {});
    synBackPropDelay->setBackPropDelaySteps(10);
    
    model.finalize();
    
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
    ASSERT_NE(synInternal->getWUPreVarMergeSuffix(), synPreParamInternal->getWUPreVarMergeSuffix());
    ASSERT_EQ(synInternal->getWUPostVarMergeSuffix(), synPreParamInternal->getWUPostVarMergeSuffix());
    
    // Only presynaptic update can be merged for synapse groups with different postsynaptic parameters 
    ASSERT_EQ(synInternal->getWUPreVarMergeSuffix(), synPostParamInternal->getWUPreVarMergeSuffix());
    ASSERT_NE(synInternal->getWUPostVarMergeSuffix(), synPostParamInternal->getWUPostVarMergeSuffix());
    
    // Both types of update can be merged for synapse groups with parameters changes which don't effect pre or post update
    ASSERT_EQ(synInternal->getWUPreVarMergeSuffix(), synSynParamInternal->getWUPreVarMergeSuffix());
    ASSERT_EQ(synInternal->getWUPostVarMergeSuffix(), synSynParamInternal->getWUPostVarMergeSuffix());
    
    // Only postsynaptic update can be merged for synapse groups with different presynaptic variable initialisers 
    ASSERT_NE(synInternal->getWUPreVarMergeSuffix(), synPreVar2Internal->getWUPreVarMergeSuffix());
    ASSERT_EQ(synInternal->getWUPostVarMergeSuffix(), synPreVar2Internal->getWUPostVarMergeSuffix());
    
    // Only presynaptic update can be merged for synapse groups with different postsynaptic variable initialisers 
    ASSERT_EQ(synInternal->getWUPreVarMergeSuffix(), synPostVar2Internal->getWUPreVarMergeSuffix());
    ASSERT_NE(synInternal->getWUPostVarMergeSuffix(), synPostVar2Internal->getWUPostVarMergeSuffix());
    
    // Only postsynaptic update can be merged for synapse groups with different axonal delays
    ASSERT_NE(synInternal->getWUPreVarMergeSuffix(), synAxonalDelayInternal->getWUPreVarMergeSuffix());
    ASSERT_EQ(synInternal->getWUPostVarMergeSuffix(), synAxonalDelayInternal->getWUPostVarMergeSuffix());
    
    // Only presynaptic update can be merged for synapse groups with different back propagation delays
    ASSERT_EQ(synInternal->getWUPreVarMergeSuffix(), synBackPropDelayInternal->getWUPreVarMergeSuffix());
    ASSERT_NE(synInternal->getWUPostVarMergeSuffix(), synBackPropDelayInternal->getWUPostVarMergeSuffix());
}


TEST(NeuronGroup, MergePSM)
{
    ModelSpecInternal model;
    model.setMergePostsynapticModels(true);
    
    LIFAdditional::ParamValues paramVals(0.25, 10.0, 0.0, 0.0, 20.0, 0.0, 5.0);
    LIFAdditional::VarValues varVals(0.0, 0.0);
    PostsynapticModels::ExpCurr::ParamValues psmParamVals(5.0);
    PostsynapticModels::ExpCurr::ParamValues psmParamVals2(10.0);
    WeightUpdateModels::StaticPulseDendriticDelay::VarValues wumVarVals(0.1, 10);
    
    // Add two neuron groups to model
    model.addNeuronPopulation<LIFAdditional>("Pre", 10, paramVals, varVals);
    model.addNeuronPopulation<LIFAdditional>("Post", 10, paramVals, varVals);

    // Create baseline synapse group
    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "Syn", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    
    // Create second synapse group
    auto *syn2 = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "Syn2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    
    // Create synapse group with different value for PSM parameter
    auto *synParam = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynParam", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals2, {});
    
    // Create synapse group with different target variable
    auto *synTarget = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynTarget", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    synTarget->setPSTargetVar("Isyn2");
    
    // Create synapse group with different max dendritic delay
    auto *synDelay = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::ExpCurr>(
        "SynDelay", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        {}, wumVarVals,
        psmParamVals, {});
    synDelay->setMaxDendriticDelayTimesteps(20);
    
    model.finalize();
    
    // Cast synapse groups to internal types
    auto synInternal = static_cast<SynapseGroupInternal*>(syn);
    auto syn2Internal = static_cast<SynapseGroupInternal*>(syn2);
    auto synParamInternal = static_cast<SynapseGroupInternal*>(synParam);
    auto synTargetInternal = static_cast<SynapseGroupInternal*>(synTarget);
    auto synDelayInternal = static_cast<SynapseGroupInternal*>(synDelay);
 
    // Check that identically configured PSMs can be merged
    ASSERT_EQ(synInternal->getPSVarMergeSuffix(), syn2Internal->getPSVarMergeSuffix());
    
    // Check that PSMs with different parameters cannot be merged
    ASSERT_NE(synInternal->getPSVarMergeSuffix(), synParamInternal->getPSVarMergeSuffix());
    
    // Check that PSMs targetting different variables cannot be merged
    ASSERT_NE(synInternal->getPSVarMergeSuffix(), synTargetInternal->getPSVarMergeSuffix());
    
    // Check that PSMs from synapse groups with different dendritic delay cannot be merged
    ASSERT_NE(synInternal->getPSVarMergeSuffix(), synDelayInternal->getPSVarMergeSuffix());
}

TEST(NeuronGroup, CompareNeuronModels)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramValsA(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::ParamValues paramValsB(0.02, 0.2, -65.0, 4.0);
    NeuronModels::Izhikevich::VarValues varVals1(initVar<InitVarSnippet::Uniform>({0.0, 30.0}), 0.0);
    NeuronModels::Izhikevich::VarValues varVals2(initVar<InitVarSnippet::Uniform>({-10.0, 30.0}), 0.0);
    NeuronModels::Izhikevich::VarValues varVals3(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramValsA, varVals1);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramValsA, varVals2);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramValsB, varVals3);

    model.finalize();

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
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);
    
    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 2);

    // Find which merged neuron init group is the one with the single population i.e. the one with constant initialisers
    const size_t constantInitIndex = (modelSpecMerged.getMergedNeuronInitGroups().at(0).getGroups().size() == 1) ? 0 : 1;
    const auto constantInitMergedGroup = modelSpecMerged.getMergedNeuronInitGroups().at(constantInitIndex);
    const auto uniformInitMergedGroup = modelSpecMerged.getMergedNeuronInitGroups().at(1 - constantInitIndex);

    // Check that only 'd' parameter is heterogeneous in neuron update group
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(0));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(1));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(2));
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(3));

    // Check that only 'V' init 'min' parameter is heterogeneous
    ASSERT_FALSE(constantInitMergedGroup.isVarInitParamHeterogeneous(0, 0));
    ASSERT_FALSE(constantInitMergedGroup.isVarInitParamHeterogeneous(1, 0));
    ASSERT_TRUE(uniformInitMergedGroup.isVarInitParamHeterogeneous(0, 0));
    ASSERT_FALSE(uniformInitMergedGroup.isVarInitParamHeterogeneous(0, 1));
    ASSERT_FALSE(uniformInitMergedGroup.isVarInitParamHeterogeneous(1, 0));
}

TEST(NeuronGroup, CompareHeterogeneousParamVarState)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    LIFAdditional::ParamValues paramValsA(0.25, 10.0, 0.0, 0.0, 20.0, 0.0, 5.0);
    LIFAdditional::ParamValues paramValsB(0.25, 10.0, 0.0, 0.0, 20.0, 1.0, 5.0);
    LIFAdditional::VarValues varVals(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<LIFAdditional>("Neurons0", 10, paramValsA, varVals);
    auto *ng1 = model.addNeuronPopulation<LIFAdditional>("Neurons1", 10, paramValsB, varVals);

    model.finalize();

    // Check that all groups can be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal*>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal*>(ng1);
    ASSERT_EQ(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_EQ(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check all groups are merged
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 1);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 1);

    // Check that only 'Ioffset' parameter is heterogeneous in neuron update group
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(0));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(1));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(2));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(3));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(4));
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(5));
    ASSERT_FALSE(modelSpecMerged.getMergedNeuronUpdateGroups().at(0).isParamHeterogeneous(6));
}


TEST(NeuronGroup, CompareSimRNG)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    LIFAdditional::ParamValues paramVals(0.25, 10.0, 0.0, 0.0, 20.0, 0.0, 5.0);
    LIFAdditional::VarValues varVals(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::LIF>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<LIFRandom>("Neurons1", 10, paramVals, varVals);

    model.finalize();

    // Check that groups cannot be merged
    NeuronGroupInternal *ng0Internal = static_cast<NeuronGroupInternal*>(ng0);
    NeuronGroupInternal *ng1Internal = static_cast<NeuronGroupInternal*>(ng1);
    ASSERT_NE(ng0Internal->getHashDigest(), ng1Internal->getHashDigest());
    ASSERT_NE(ng0Internal->getInitHashDigest(), ng1Internal->getInitHashDigest());

    ASSERT_TRUE(!ng0->isSimRNGRequired());
    ASSERT_TRUE(ng1->isSimRNGRequired());
}

TEST(NeuronGroup, CompareCurrentSources)
{
    ModelSpecInternal model;

    // Add four neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add one gaussian and one DC current source to Neurons0
    CurrentSourceModels::PoissonExp::ParamValues cs0ParamVals(0.1, 20.0, 20.0);
    CurrentSourceModels::PoissonExp::ParamValues cs1ParamVals(0.1, 40.0, 20.0);
    CurrentSourceModels::PoissonExp::VarValues cs0VarVals(0.0);
    CurrentSourceModels::PoissonExp::VarValues cs1VarVals(0.0);
    CurrentSourceModels::DC::ParamValues cs2ParamVals(0.4);
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
    model.finalize();

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
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into two groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 2);

    // Find which merged neuron group is the one with the single population i.e. the two DC current sources
    const size_t dcDCIndex = (modelSpecMerged.getMergedNeuronUpdateGroups().at(0).getGroups().size() == 4) ? 1 : 0;
    const auto dcDCMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(dcDCIndex);
    const auto dcPoissonMergedGroup = modelSpecMerged.getMergedNeuronUpdateGroups().at(1 - dcDCIndex);
    ASSERT_TRUE(dcDCMergedGroup.getGroups().size() == 1);
    
    // Find which child in the DC + poisson merged group is the poisson current source
    const size_t poissonIndex = (dcPoissonMergedGroup.getSortedArchetypeCurrentSources().at(0)->getCurrentSourceModel() == CurrentSourceModels::PoissonExp::getInstance()) ? 0 : 1;
    
    // Check that only the standard deviation parameter of the gaussian current sources is heterogeneous
    // **NOTE** tau is not heterogeneous because it's not references
    ASSERT_FALSE(dcDCMergedGroup.isCurrentSourceParamHeterogeneous(0, 0));
    ASSERT_FALSE(dcDCMergedGroup.isCurrentSourceParamHeterogeneous(1, 0));
    ASSERT_FALSE(dcPoissonMergedGroup.isCurrentSourceParamHeterogeneous(poissonIndex, 0));
    ASSERT_FALSE(dcPoissonMergedGroup.isCurrentSourceParamHeterogeneous(poissonIndex, 1));
    ASSERT_FALSE(dcPoissonMergedGroup.isCurrentSourceParamHeterogeneous(1 - poissonIndex, 0));
    ASSERT_TRUE(dcPoissonMergedGroup.isCurrentSourceDerivedParamHeterogeneous(poissonIndex, 0));
    ASSERT_TRUE(dcPoissonMergedGroup.isCurrentSourceDerivedParamHeterogeneous(poissonIndex, 1));
    ASSERT_FALSE(dcPoissonMergedGroup.isCurrentSourceDerivedParamHeterogeneous(poissonIndex, 2));
}

TEST(NeuronGroup, ComparePostsynapticModels)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::SpikeSource>("SpikeSource", 10, {}, {});
    auto *ng0 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    AlphaCurr::ParamValues alphaCurrParamVals(0.5);
    AlphaCurr::ParamValues alphaCurrParamVals1(0.75);
    AlphaCurr::VarValues alphaCurrVarVals(0.0);
    AlphaCurr::VarValues alphaCurrVarVals1(0.1);
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons0",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                           "SpikeSource", "Neurons0",
                                                                           {}, staticPulseVarVals,
                                                                           alphaCurrParamVals, alphaCurrVarVals);

    // Do the same for Neuron1
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons1",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                           "SpikeSource", "Neurons1",
                                                                           {}, staticPulseVarVals,
                                                                           alphaCurrParamVals, alphaCurrVarVals);

    // Do the same, but with different parameters for Neuron2,
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons2",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                           "SpikeSource", "Neurons2",
                                                                           {}, staticPulseVarVals,
                                                                           alphaCurrParamVals1, alphaCurrVarVals1);

    // Do the same, but in the opposite order for Neuron3
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                           "SpikeSource", "Neurons3",
                                                                           {}, staticPulseVarVals,
                                                                           alphaCurrParamVals, alphaCurrVarVals);
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG7", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons3",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    // Add two incoming synapse groups with DeltaCurr postsynaptic models sources to Neurons4
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG8", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons4",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG9", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, NO_DELAY,
                                                                                               "SpikeSource", "Neurons4",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});

    model.finalize();

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
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a delta and an alpha psm
    const auto deltaAlphaMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                          [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto deltaAlphaMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                        [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Find which child in the DC + gaussian merged group is the gaussian current source
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->getSortedArchetypeMergedInSyns().size() == 2);
    ASSERT_TRUE(deltaAlphaMergedInitGroup->getSortedArchetypeMergedInSyns().size() == 2);
    const size_t alphaUpdateIndex = (deltaAlphaMergedUpdateGroup->getSortedArchetypeMergedInSyns().at(0)->getPSModel() == AlphaCurr::getInstance()) ? 0 : 1;
    const size_t alphaInitIndex = (deltaAlphaMergedInitGroup->getSortedArchetypeMergedInSyns().at(0)->getPSModel() == AlphaCurr::getInstance()) ? 0 : 1;

    // Check that parameter and both derived parameters are heterogeneous
    // **NOTE** tau is NOT heterogeneous because it's unused
    ASSERT_FALSE(deltaAlphaMergedUpdateGroup->isPSMParamHeterogeneous(alphaUpdateIndex, 0));
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->isPSMDerivedParamHeterogeneous(alphaUpdateIndex, 0));
    ASSERT_TRUE(deltaAlphaMergedUpdateGroup->isPSMDerivedParamHeterogeneous(alphaUpdateIndex, 1));
    ASSERT_TRUE(deltaAlphaMergedInitGroup->isPSMVarInitParamHeterogeneous(alphaInitIndex, 0, 0));
}

TEST(NeuronGroup, CompareWUPreUpdate)
{
    ModelSpecInternal model;

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPre::VarValues testVarVals(0.0);
    WeightUpdateModelPre::ParamValues testParams(1.0);
    WeightUpdateModelPre::ParamValues testParams2(2.0);
    WeightUpdateModelPre::PreVarValues testPreVarVals1(0.0);
    WeightUpdateModelPre::PreVarValues testPreVarVals2(2.0);

    // Connect neuron group 1 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons1", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals1, {},
                                                                                    {}, {});

    // Also connect neuron group 2 to neuron group 0 with pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons2", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals1, {},
                                                                                    {}, {});

    // Also connect neuron group 3 to neuron group 0 with pre weight update model, but different parameters
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons3", "Neurons0",
                                                                                    testParams2, testVarVals, testPreVarVals1, {},
                                                                                    {}, {});

    // Connect neuron group 4 to neuron group 0 with 2*pre weight update model
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals1, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons4", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals1, {},
                                                                                    {}, {});

    // Connect neuron group 5 to neuron group 0 with pre weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPre, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                    "Neurons5", "Neurons0",
                                                                                    testParams, testVarVals, testPreVarVals2, {},
                                                                                    {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons5", "Neurons0",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

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
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neuron init and update is merged into three groups (NG0 with no outsyns, NG1, NG2, NG3 and NG5 with 1 outsyn and NG4with 2 outsyns)
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPreMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                      [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto wumPreMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                    [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPreMergedUpdateGroup->isOutSynWUMParamHeterogeneous(0, 0));
    ASSERT_TRUE(wumPreMergedInitGroup->isOutSynWUMVarInitParamHeterogeneous(0, 0, 0));
}

TEST(NeuronGroup, CompareWUPostUpdate)
{
    ModelSpecInternal model;

    // **NOTE** we make sure merging is on so last test doesn't fail on that basis
    model.setMergePostsynapticModels(true);

    // Add two neuron groups to model
    NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
    NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons0", 10, paramVals, varVals);
    auto *ng1 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons1", 10, paramVals, varVals);
    auto *ng2 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons2", 10, paramVals, varVals);
    auto *ng3 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons3", 10, paramVals, varVals);
    auto *ng4 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons4", 10, paramVals, varVals);
    auto *ng5 = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons5", 10, paramVals, varVals);

    // Add incoming synapse groups with Delta and DeltaCurr postsynaptic models to Neurons0
    WeightUpdateModels::StaticPulse::VarValues staticPulseVarVals(0.1);
    WeightUpdateModelPost::VarValues testVarVals(0.0);
    WeightUpdateModelPost::ParamValues testParams(1.0);
    WeightUpdateModelPost::ParamValues testParams2(2.0);
    WeightUpdateModelPost::PostVarValues testPostVarVals1(0.0);
    WeightUpdateModelPost::PostVarValues testPostVarVals2(2.0);

    // Connect neuron group 0 to neuron group 1 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG0", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons1",
                                                                                     testParams, testVarVals, {}, testPostVarVals1,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 2 with post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons2",
                                                                                     testParams, testVarVals, {}, testPostVarVals1,
                                                                                     {}, {});

    // Also connect neuron group 0 to neuron group 3 with post weight update model but different parameters
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG2", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons3",
                                                                                     testParams2, testVarVals, {}, testPostVarVals1,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 3 with 2*post weight update model
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG3", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, testVarVals, {}, testPostVarVals1,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons4",
                                                                                     testParams, testVarVals, {}, testPostVarVals1,
                                                                                     {}, {});

    // Connect neuron group 0 to neuron group 4 with post weight update model and static pulse
    model.addSynapsePopulation<WeightUpdateModelPost, PostsynapticModels::DeltaCurr>("SG5", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                     "Neurons0", "Neurons5",
                                                                                     testParams, testVarVals, {}, testPostVarVals2,
                                                                                     {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("SG6", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                                                                                               "Neurons0", "Neurons5",
                                                                                               {}, staticPulseVarVals,
                                                                                               {}, {});
    model.finalize();

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
    CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

    // Merge model
    CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

    // Check neurons are merged into three groups
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronUpdateGroups().size() == 3);
    ASSERT_TRUE(modelSpecMerged.getMergedNeuronInitGroups().size() == 3);

    // Find which merged neuron group is the one containing 4 groups i.e. the 4 neuron groups with a single weight update model with presynaptic update
    const auto wumPostMergedUpdateGroup = std::find_if(modelSpecMerged.getMergedNeuronUpdateGroups().cbegin(), modelSpecMerged.getMergedNeuronUpdateGroups().cend(),
                                                       [](const CodeGenerator::NeuronUpdateGroupMerged &ng) { return (ng.getGroups().size() == 4); });
    const auto wumPostMergedInitGroup = std::find_if(modelSpecMerged.getMergedNeuronInitGroups().cbegin(), modelSpecMerged.getMergedNeuronInitGroups().cend(),
                                                     [](const CodeGenerator::NeuronInitGroupMerged &ng) { return (ng.getGroups().size() == 4); });

    // Check that parameter is heterogeneous
    ASSERT_TRUE(wumPostMergedUpdateGroup->isInSynWUMParamHeterogeneous(0, 0));
    ASSERT_TRUE(wumPostMergedInitGroup->isInSynWUMVarInitParamHeterogeneous(0, 0, 0));
}
