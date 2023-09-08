// Standard C++ includes
#include <array>
#include <functional>
#include <vector>

// Google test includes
#include "gtest/gtest.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Anonyous namespace
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

    SET_PARAM_NAMES({"tau"});

    SET_NEURON_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const ParamValues &pars, double dt) { return std::exp(-dt / pars.at("tau")); }},
        {"init", [](const ParamValues &pars, double) { return (std::exp(1) / pars.at("tau")); }}});
};
IMPLEMENT_SNIPPET(AlphaCurr);

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);

    SET_PARAM_NAMES({
      "tauPlus",  // 0 - Potentiation time constant (ms)
      "tauMinus", // 1 - Depression time constant (ms)
      "Aplus",    // 2 - Rate of potentiation
      "Aminus",   // 3 - Rate of depression
      "Wmin",     // 4 - Minimum weight
      "Wmax"});   // 5 - Maximum weight

    SET_SYNAPSE_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(preTrace) = ($(preTrace) * exp(-dt / $(tauPlus))) + 1.0;\n");

    SET_POST_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(postTrace) = ($(postTrace) * exp(-dt / $(tauMinus))) + 1.0;\n");

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar timing = $(postTrace) * exp(-dt / $(tauMinus));\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * timing);\n"
        "    $(g) = fmax($(Wmin), newWeight);\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar timing = $(postTrace) * exp(-dt / $(tauPlus));\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * timing);\n"
        "    $(g) = fmin($(Wmax), newWeight);\n"
        "}\n");
};
IMPLEMENT_SNIPPET(STDPAdditive);

class Sum : public CustomUpdateModels::Base
{
    DECLARE_SNIPPET(Sum);

    SET_UPDATE_CODE("sum = a + b;\n");

    SET_CUSTOM_UPDATE_VARS({{"sum", "scalar"}});
    SET_PARAM_NAMES({"b"});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_SNIPPET(Sum);

class OneToOneOff : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(OneToOneOff);

    SET_ROW_BUILD_CODE("addSynapse(id_pre + 1);\n");

    SET_MAX_ROW_LENGTH(1);
    SET_MAX_COL_LENGTH(1);
};
IMPLEMENT_SNIPPET(OneToOneOff);

class RemoveSynapse : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapse);
    
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(id_post == (id_pre + 1)) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapse);

class RemoveSynapsePrePost : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapsePrePost);
    
    SET_SYNAPSE_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preThresh", "scalar"}});
    SET_POST_VARS({{"postThresh", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(g < preThresh || g < postThresh) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapsePrePost);

class RemoveSynapseParam : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_SNIPPET(RemoveSynapseParam);
    
    SET_VAR_REFS({{"g", "scalar"}});
    SET_PARAM_NAMES({"thresh"});
    
    SET_ROW_UPDATE_CODE(
        "for_each_synapse {\n"
        "   if(g < thresh) {\n"
        "       remove_synapse();\n"
        "       break;\n"
        "   }\n"
        "};\n");
};
IMPLEMENT_SNIPPET(RemoveSynapseParam);

template<typename T, typename M, size_t N>
void test(const std::pair<T, bool> (&modelModifiers)[N], M applyModifierFn)
{
    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;

    // Create array of module digests
    boost::uuids::detail::sha1::digest_type moduleHash[N];

    // Loop through
    for(size_t i = 0; i < N; i++) {
        // Set default model properties
        ModelSpecInternal model;
        model.setName("test");
        model.setDT(0.1);
        model.setTiming(false);
        model.setPrecision(Type::Float);
        model.setBatchSize(1);
        model.setSeed(0);

        // Apply modifier
        applyModifierFn(modelModifiers[i].first, model);
        
        // Finalize model
        model.finalise();

        // Create suitable backend to build model
        CodeGenerator::SingleThreadedCPU::Backend backend(preferences);

        // Created merged model
        CodeGenerator::ModelSpecMerged modelMerged(backend, model);

        // Write hash digests of model to array
        moduleHash[i] = modelMerged.getHashDigest(backend);
    }

    // Loop through modified models
    for(size_t i = 1; i < N; i++) {
        ASSERT_EQ(moduleHash[i] == moduleHash[0], modelModifiers[i].second);
    }
}
//--------------------------------------------------------------------------
template<typename S>
void testNeuronVarLocation(S setVarLocationFn)
{
    // Make array of variable locations to build model with and flags determining whether the hashes should match baseline
    const std::pair<VarLocation, bool> modelModifiers[] = {
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::DEVICE,                   false},
        {VarLocation::HOST_DEVICE_ZERO_COPY,    false}};

    test(modelModifiers, 
         [setVarLocationFn](const VarLocation &varLocation, ModelSpecInternal &model)
         {
             // Default neuron parameters
             ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             VarValues varVals{{"V", 0.0}, {"U", 0.0}};
             auto *pop = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                             paramVals, varVals);

             // Call function to apply var location to model
             setVarLocationFn(pop, varLocation);
         });
}
//--------------------------------------------------------------------------
template<typename S>
void testSynapseVarLocation(S setVarLocationFn)
{
    // Make array of variable locations to build model with and flags determining whether the hashes should match baseline
    const std::pair<VarLocation, bool> modelModifiers[] = {
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::DEVICE,                   false},
        {VarLocation::HOST_DEVICE_ZERO_COPY,    false}};

    test(modelModifiers, 
         [setVarLocationFn](const VarLocation &varLocation, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 100, 
                                                                 neuronParamVals, neuronVarVals);

             ParamValues params{{"tauPlus", 20.0}, {"tauMinus", 20.0}, {"Aplus", 0.001}, {"Aminus", -0.001}, {"Wmin", 0.0}, {"Wmax", 1.0}};
             VarValues varValues{{"g", 0.5}};
             VarValues preVarValues{{"preTrace", 0.0}};
             VarValues postVarValues{{"postTrace", 0.0}};
               
             ParamValues psmParams{{"tau", 5.0}};
             VarValues psmVarValues{{"x", 0.0}};

             auto *sg = model.addSynapsePopulation<STDPAdditive, AlphaCurr>(
                 "Synapse", SynapseMatrixType::SPARSE, NO_DELAY,
                 "Pre", "Post",
                 params, varValues, preVarValues, postVarValues,
                 psmParams, psmVarValues,
                 initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({{"prob", 0.1}}));
             setVarLocationFn(sg, varLocation);
         });
}
//--------------------------------------------------------------------------
template<typename S>
void testCustomConnectivityUpdateVarLocation(S setVarLocationFn)
{
    // Make array of variable locations to build model with and flags determining whether the hashes should match baseline
    const std::pair<VarLocation, bool> modelModifiers[] = {
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::DEVICE,                   false},
        {VarLocation::HOST_DEVICE_ZERO_COPY,    false}};

    test(modelModifiers, 
         [setVarLocationFn](const VarLocation &varLocation, ModelSpecInternal &model)
         {
            // Add two neuron group to model
            ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
            VarValues varVals{{"V", 0.0}, {"U", 0.0}};
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

            // Create synapse group with global weights
            model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
                "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
                "Pre", "Post",
                {{"g", 1.0}}, {},
                {}, {});

            auto *ccu = model.addCustomConnectivityUpdate<RemoveSynapsePrePost>(
                "CustomConnectivityUpdate1", "Test2", "Synapses1",
                {}, {{"g", 1.0}}, {{"preThresh", 1.0}}, {{"postThresh", 1.0}},
                {}, {}, {});
             setVarLocationFn(ccu, varLocation);
         });
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareModelChanges)
{
    // Make array of functions to modify default model and flags determining whether the hashes should match baseline
    const std::pair<std::function<void(ModelSpecInternal &)>, bool> modelModifiers[] = {
        {nullptr, true},
        {nullptr, true},
        {[](ModelSpecInternal &model) { model.setName("interesting_name"); }, false},
        {[](ModelSpecInternal &model) { model.setDT(1.0); }, false},
        {[](ModelSpecInternal &model) { model.setTiming(true); }, false},
        {[](ModelSpecInternal &model) { model.setPrecision(Type::Double); }, false},
        {[](ModelSpecInternal &model) { model.setTimePrecision(Type::Double); }, false},
        {[](ModelSpecInternal &model) { model.setSeed(1234); }, false}};
    
    test(modelModifiers, 
         [](std::function<void(ModelSpecInternal &)> modifier, ModelSpecInternal &model)
         {
             if(modifier) {
                 modifier(model);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronPopSizeChanges)
{
    // Make array of population size to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<unsigned int>, bool> modelModifiers[] = {
        {{10, 50},      true},
        {{10, 50},      true},
        {{50, 10},      false},
        {{20, 20},      false},
        {{},            false},
        {{10, 20, 30},  false},
        {{20},          false}};
    
    test(modelModifiers, 
         [](const std::vector<unsigned int> &popSizes, ModelSpecInternal &model)
         {
             // Default neuron parameters
             ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             VarValues varVals{{"V", 0.0}, {"U", 0.0}};

             // Add desired number and size of populations
             for(size_t p = 0; p < popSizes.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons" + std::to_string(p), popSizes.at(p), 
                                                                    paramVals, varVals);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronNameChanges)
{
    // Make array of neuron population names and flags determining whether the hashes should match baseline
    const std::pair<std::string, bool> modelModifiers[] = {
        {"Neurons",     true},
        {"Neurons",     true},
        {"neurons",     false},
        {"nrns",        false},
        {"n_euron_s",   false},
        {"Neurons_1",   false}};
    
    test(modelModifiers, 
         [](const std::string &name, ModelSpecInternal &model)
         {
             // Default neuron parameters
             ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             VarValues varVals{{"V", 0.0}, {"U", 0.0}};

             // Add population with specified name
             model.addNeuronPopulation<NeuronModels::Izhikevich>(name, 100, 
                                                                 paramVals, varVals);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronParamChanges)
{
    // Izhikevcih parameter sets
    const ParamValues paramVals1{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
    const ParamValues paramVals2{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 2.0}};
    const ParamValues paramVals3{{"a", 0.02}, {"b", 0.2}, {"c", -50.0}, {"d", 2.0}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &popParams, ModelSpecInternal &model)
         {
             // Default neuron parameters
             VarValues varVals{{"V", 0.0}, {"U", 0.0}};

             // Add desired number of populations
             for(size_t p = 0; p < popParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons" + std::to_string(p), 100, 
                                                                     popParams[p], varVals);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronVarInitChanges)
{
    // Izhikevcih parameter sets
    const VarValues varInit1{{"V", 0.0}, {"U", 0.0}};
    const VarValues varInit2{{"V", 30.0}, {"U", 0.0}};

    // Make array of population variable initialiesers to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<VarValues>, bool> modelModifiers[] = {
        {{varInit1, varInit2},  true},
        {{varInit1, varInit2},  true},
        {{varInit2, varInit1},  false},
        {{varInit2},            false},
        {{},                    false},
        {{varInit1},            false}};

    test(modelModifiers, 
         [](const std::vector<VarValues> &popVarInit, ModelSpecInternal &model)
         {
             // Default neuron parameters
             ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};

             // Add desired number of populations
             for(size_t p = 0; p < popVarInit.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons" + std::to_string(p), 100, 
                                                                     paramVals, popVarInit[p]);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronVarLocationChanges)
{
    testNeuronVarLocation([](NeuronGroup *pop, VarLocation varLocation) {pop->setVarLocation("V", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronSpikeLocationChanges)
{
    testNeuronVarLocation([](NeuronGroup *pop, VarLocation varLocation) {pop->setSpikeLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronSpikeEventLocationChanges)
{
    testNeuronVarLocation([](NeuronGroup *pop, VarLocation varLocation) {pop->setSpikeEventLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronSpikeTimeLocationChanges)
{
    testNeuronVarLocation([](NeuronGroup *pop, VarLocation varLocation) {pop->setSpikeTimeLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronSpikeEventTimeLocationChanges)
{
    testNeuronVarLocation([](NeuronGroup *pop, VarLocation varLocation) {pop->setSpikeEventTimeLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceNameChanges)
{
    // Make array of current source names and flags determining whether the hashes should match baseline
    const std::pair<std::string, bool> modelModifiers[] = {
        {"CurrentSource",   true},
        {"CurrentSource",   true},
        {"currentsource",   false},
        {"crns",            false},
        {"c_urrent_source", false},
        {"CurrentSource_1", false}};
    
    test(modelModifiers, 
         [](const std::string &name, ModelSpecInternal &model)
         {
             // Add population
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             ParamValues paramVals{{"amp", 3.0}};
             model.addCurrentSource<CurrentSourceModels::DC>(name, "Neurons",
                                                             paramVals, {});
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceParamChanges)
{
    // Current source parameter sets
    const ParamValues paramVal1{{"amp", 1.0}};
    const ParamValues paramVal2{{"amp", 2.0}};
    const ParamValues paramVal3{{"amp", 0.5}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},    true},
        {{paramVal1, paramVal2},    true},
        {{paramVal1, paramVal1},    false},
        {{paramVal2, paramVal3},    false},
        {{},                        false},
        {{paramVal1},               false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &csParams, ModelSpecInternal &model)
         {
             // Add population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of current sources
             for(size_t p = 0; p < csParams.size(); p++) {
                 model.addCurrentSource<CurrentSourceModels::DC>("CS" + std::to_string(p), "Neurons",
                                                                 csParams[p], {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceVarInitParamChanges)
{
    // Current source parameter sets
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &csVarInit, ModelSpecInternal &model)
         {
             // Add population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of current sources
             for(size_t p = 0; p < csVarInit.size(); p++) {
                 ParamValues paramVals{{"weight", 1.0}, {"tauSyn", 20.0}, {"rate", 10.0}};
                 VarValues varVals{{"current", initVar<InitVarSnippet::Uniform>(csVarInit[p])}};
                 model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS" + std::to_string(p), "Neurons",
                                                                         paramVals, varVals);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceVarLocationChanges)
{
    // Make array of variable locations to build model with and flags determining whether the hashes should match baseline
    const std::pair<VarLocation, bool> modelModifiers[] = {
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::DEVICE,                   false},
        {VarLocation::HOST_DEVICE_ZERO_COPY,    false}};

    test(modelModifiers, 
         [](const VarLocation &varLocation, ModelSpecInternal &model)
         {
              // Add population
              VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
              ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
              model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);
              ParamValues paramVals{{"weight", 1.0}, {"tauSyn", 20.0}, {"rate", 10.0}};
              VarValues varVals{{"current", 0.0}};
              auto *cs = model.addCurrentSource<CurrentSourceModels::PoissonExp>("CS", "Neurons",
                                                                                 paramVals, varVals);
              cs->setVarLocation("current", varLocation);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareSynapseNameChanges)
{
    // Make array of synapse group names and flags determining whether the hashes should match baseline
    const std::pair<std::string, bool> modelModifiers[] = {
        {"SynapseGroup",    true},
        {"SynapseGroup",    true},
        {"synapsegroup",    false},
        {"sgrp",            false},
        {"s_ynapse_group", false},
        {"SynapseGroup_1", false}};
    
    test(modelModifiers, 
         [](const std::string &name, ModelSpecInternal &model)
         {
             // Add populations
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 100, 
                                                                 neuronParamVals, neuronVarVals);

             model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                name, SynapseMatrixType::DENSE, NO_DELAY,
                "Pre", "Post",
                {}, {{"g", 1.0}},
                {}, {});
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, ComparePSMParamChanges)
{
    // Postsynaptic model tau parameters
    const ParamValues paramVal1{{"tau", 5.0}};
    const ParamValues paramVal2{{"tau", 1.0}};
    const ParamValues paramVal3{{"tau", 10.0}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal1},  false},
        {{paramVal2, paramVal3},  false},
        {{},                      false},
        {{paramVal1},             false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &psmParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < psmParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, {{"g", 1.0}},
                    psmParams[p], {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, ComparePSMVarInitParamChanges)
{
    // Postsynaptic model var initialisers
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &psmVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < psmVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 ParamValues params{{"tau", 5.0}};
                 VarValues varValues{{"x", initVar<InitVarSnippet::Uniform>(psmVarInitParams[p])}};
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, {{"g", 1.0}},
                    params, varValues);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, ComparePSMVarLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setPSVarLocation("x", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMParamChanges)
{
    // Weight update model parameters
    const ParamValues paramVals1{{"tLrn", 50.0}, {"tChng", 50.0}, {"tDecay", 50000.0}, {"tPunish10", 100000.0}, {"tPunish01", 200.0}, 
                                          {"gMax", 0.015}, {"gMid", 0.0075}, {"gSlope", 33.33}, {"tauShift", 10.0}, {"gSyn0", 0.00006}};
    const ParamValues paramVals2{{"tLrn", 100.0}, {"tChng", 100.0}, {"tDecay", 50000.0}, {"tPunish10", 100000.0}, {"tPunish01", 200.0}, 
                                          {"gMax", 0.015}, {"gMid", 0.0075}, {"gSlope", 33.33}, {"tauShift", 10.0}, {"gSyn0", 0.00006}};
    const ParamValues paramVals3{{"tLrn", 50.0}, {"tChng", 50.0}, {"tDecay", 50000.0}, {"tPunish10", 100000.0}, {"tPunish01", 200.0}, 
                                          {"gMax", 0.1}, {"gMid", 0.0075}, {"gSlope", 33.33}, {"tauShift", 10.0}, {"gSyn0", 0.00006}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &wumParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 VarValues varInit{{"g", 0.0}, {"gRaw", uninitialisedVar()}};
                 model.addSynapsePopulation<WeightUpdateModels::PiecewiseSTDP, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    wumParams[p], varInit,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMGlobalGVarChanges)
{
    // Weight update model variable initialisers
    const ParamValues varVals1{{"g", 1.0}};
    const ParamValues varVals2{{"g", 0.2}};
    const ParamValues varVals3{{"g", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{varVals1, varVals2},  true},
        {{varVals1, varVals2},  true},
        {{varVals1, varVals1},  false},
        {{varVals2, varVals3},  false},
        {{},                        false},
        {{varVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &wumParamVals, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumParamVals.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    wumParamVals[p], {}, 
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMVarInitParamChanges)
{
    // Weight update model var initialisers
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &wumVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 VarValues varValues{{"g", initVar<InitVarSnippet::Uniform>(wumVarInitParams[p])}};
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, varValues,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMVarLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setWUVarLocation("g", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMPreVarInitParamChanges)
{
    // Weight update model presynaptic var initialisers
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &wumPreVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumPreVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 ParamValues params{{"tauPlus", 20.0}, {"tauMinus", 20.0}, {"Aplus", 0.001}, {"Aminus", -0.001}, {"Wmin", 0.0}, {"Wmax", 1.0}};
                 VarValues varValues{{"g", 0.5}};
                 VarValues preVarValues{{"preTrace", initVar<InitVarSnippet::Uniform>(wumPreVarInitParams[p])}};
                 VarValues postVarValues{{"postTrace", 0.0}};
                
                 model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    params, varValues, preVarValues, postVarValues,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMPreVarLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setWUPreVarLocation("preTrace", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMPostVarInitParamChanges)
{
    // Weight update model postsynaptic var initialisers
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &wumPostVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumPostVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 ParamValues params{{"tauPlus", 20.0}, {"tauMinus", 20.0}, {"Aplus", 0.001}, {"Aminus", -0.001}, {"Wmin", 0.0}, {"Wmax", 1.0}};
                 VarValues varValues{{"g", 0.5}};
                 VarValues preVarValues{{"preTrace", 0.0}};
                 VarValues postVarValues{{"postTrace", initVar<InitVarSnippet::Uniform>(wumPostVarInitParams[p])}};
                 
                 model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    params, varValues, preVarValues, postVarValues,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMPostVarLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setWUPostVarLocation("postTrace", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareConnectivityParamChanges)
{
    // Connectivity parameters
    const ParamValues paramVals1{{"prob", 0.1}};
    const ParamValues paramVals2{{"prob", 0.2}};
    const ParamValues paramVals3{{"prob", 0.9}};
    
    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &connectivityParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < connectivityParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
                     "Synapse" + std::to_string(p), SynapseMatrixType::SPARSE, NO_DELAY,
                     "Pre", "Post" + std::to_string(p),
                     {{"g", 1.0}}, {},
                     {}, {},
                     initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(connectivityParams[p]));
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareConnectivityModelChanges)
{
    // Connectivity models
    const auto *model1 = InitSparseConnectivitySnippet::OneToOne::getInstance();
    const auto *model2 = OneToOneOff::getInstance();
    
    // Make array of connectivity models to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<const InitSparseConnectivitySnippet::Base*>, bool> modelModifiers[] = {
        {{model1, model2},  true},
        {{model1, model2},  true},
        {{model2, model1},  false},
        {{model1, model1},  false},
        {{},                false},
        {{model1},          false}};

    test(modelModifiers, 
         [](const std::vector<const InitSparseConnectivitySnippet::Base*> &connectivityModels, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < connectivityModels.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
                     "Synapse" + std::to_string(p), SynapseMatrixType::SPARSE, NO_DELAY,
                     "Pre", "Post" + std::to_string(p),
                     {{"g", 1.0}}, {},
                     {}, {},
                     InitSparseConnectivitySnippet::Init(connectivityModels[p], {}));
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareConnectivityLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setSparseConnectivityLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareInSynLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setInSynVarLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareDendriticDelayLocationChanges)
{
    testSynapseVarLocation([](SynapseGroup *pop, VarLocation varLocation) {pop->setDendriticDelayLocation(varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateNameChanges)
{
    // Make array of current source names and flags determining whether the hashes should match baseline
    const std::pair<std::string, bool> modelModifiers[] = {
        {"CustomUpdate",    true},
        {"CustomUpdate",    true},
        {"customupdate",    false},
        {"cupdaate",        false},
        {"c_ustom_update",  false},
        {"CustomUpdate_1", false}};
    
    test(modelModifiers, 
         [](const std::string &name, ModelSpecInternal &model)
         {
             // Add population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             auto *neurons = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                                 neuronParamVals, neuronVarVals);

             ParamValues paramVals{{"b", 5.0}};
             VarValues vals{{"sum", 0.0}};
             VarReferences varRefs{{"a", createVarRef(neurons, "V")}};
             model.addCustomUpdate<Sum>(name, "Group", paramVals, vals, varRefs);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateParamChanges)
{
    // Custom update model "b" parameters
    const ParamValues paramVal1{{"b", 5.0}};
    const ParamValues paramVal2{{"b", 1.0}};
    const ParamValues paramVal3{{"b", 10.0}};
    
    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal1},  false},
        {{paramVal2, paramVal3},  false},
        {{},                      false},
        {{paramVal1},             false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &customUpdateParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateParams.size(); c++) {
                 VarValues vals{{"sum", 0.0}};
                 VarReferences varRefs{{"a", createVarRef(pre, "V")}};
                 model.addCustomUpdate<Sum>("CU" + std::to_string(c), "Group", customUpdateParams[c], vals, varRefs);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateVarInitParamChanges)
{
    // Custom update model var initialisers
    const ParamValues paramVals1{{"min", 0.0}, {"max", 1.0}};
    const ParamValues paramVals2{{"min", 0.1}, {"max", 0.2}};
    const ParamValues paramVals3{{"min", 0.3}, {"max", 0.9}};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<ParamValues>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<ParamValues> &customUpdateVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateVarInitParams.size(); c++) {
                 ParamValues paramVals{{"b", 1.0}};
                 VarValues vals{{"sum", initVar<InitVarSnippet::Uniform>(customUpdateVarInitParams[c])}};
                 VarReferences varRefs{{"a", createVarRef(pre, "V")}};
                 model.addCustomUpdate<Sum>("CU" + std::to_string(c), "Group", paramVals, vals, varRefs);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateVarLocationChanges)
{
    // Make array of variable locations to build model with and flags determining whether the hashes should match baseline
    const std::pair<VarLocation, bool> modelModifiers[] = {
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::HOST_DEVICE,              true},
        {VarLocation::DEVICE,                   false},
        {VarLocation::HOST_DEVICE_ZERO_COPY,    false}};

    test(modelModifiers, 
         [](const VarLocation &varLocation, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

            ParamValues paramVals{{"b", 1.0}};
            VarValues vals{{"sum", 0.0}};
            VarReferences varRefs{{"a", createVarRef(pre, "V")}};
            auto *cu = model.addCustomUpdate<Sum>("CU", "Group", paramVals, vals, varRefs);
            cu->setVarLocation("sum", varLocation);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateVarRefTargetChanges)
{
    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::string>, bool> modelModifiers[] = {
        {{"V", "U"},    true},
        {{"V", "U"},    true},
        {{"U", "V"},    false},
        {{"V", "V"},    false},
        {{},            false},
        {{"V"},         false}};

    test(modelModifiers, 
         [](const std::vector<std::string> &customUpdateVarRefTargets, ModelSpecInternal &model)
         {
             // Add pre population
             VarValues neuronVarVals{{"V", 0.0}, {"U", 0.0}};
             ParamValues neuronParamVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 4.0}};
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateVarRefTargets.size(); c++) {
                 ParamValues paramVals{{"b", 1.0}};
                 VarValues vals{{"sum", 0.0}};
                 VarReferences varRefs{{"a", createVarRef(pre, customUpdateVarRefTargets[c])}};
                 model.addCustomUpdate<Sum>("CU" + std::to_string(c), "Group", paramVals, vals, varRefs);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomConnectivityUpdateVarLocationChanges)
{
    testCustomConnectivityUpdateVarLocation([](CustomConnectivityUpdate *ccu, VarLocation varLocation) {ccu->setVarLocation("g", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomConnectivityUpdatePreVarLocationChanges)
{
    testCustomConnectivityUpdateVarLocation([](CustomConnectivityUpdate *ccu, VarLocation varLocation) {ccu->setPreVarLocation("preThresh", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomConnectivityUpdatePostVarLocationChanges)
{
    testCustomConnectivityUpdateVarLocation([](CustomConnectivityUpdate *ccu, VarLocation varLocation) {ccu->setPostVarLocation("postThresh", varLocation); });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomConnectivityUpdateParamChanges)
{
    // Make array of parameter tuples to build model with and flags determining whether the hashes should match baseline
    const std::pair<double, bool> modelModifiers[] = {
        {1.0,   true},
        {1.0,   true},
        {0.0,   false}};
    
    test(modelModifiers, 
         [](double thresh, ModelSpecInternal &model)
         {
            // Add two neuron group to model
            ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
            VarValues varVals{{"V", 0.0}, {"U", 0.0}};
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

            // Create synapse group with global weights
            auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
                "Pre", "Post",
                {}, {{"g", 1.0}},
                {}, {});

            WUVarReferences varRefs{{"g", createWUVarRef(syn, "g")}};
            model.addCustomConnectivityUpdate<RemoveSynapseParam>(
                "CustomConnectivityUpdate1", "Test2", "Synapses1",
                {{"thresh", thresh}}, {}, {}, {},
                varRefs, {}, {});
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomConnectivityUpdateVarInitParamChanges)
{
    const std::array<double, 3> initParams1{1.0, 1.0, 1.0};
    const std::array<double, 3> initParams2{2.0, 1.0, 1.0};
    const std::array<double, 3> initParams3{1.0, 2.0, 1.0};
    const std::array<double, 3> initParams4{1.0, 1.0, 1.0};
    
    // Make array of parameter tuples to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::array<double, 3>, bool> modelModifiers[] = {
        {initParams1,   true},
        {initParams1,   true},
        {initParams2,   true},
        {initParams3,   true},
        {initParams4,   true}};
    
    test(modelModifiers, 
         [](const std::array<double, 3> &params, ModelSpecInternal &model)
         {
            // Add two neuron group to model
            ParamValues paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
            VarValues varVals{{"V", 0.0}, {"U", 0.0}};
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

            // Create synapse group with global weights
            model.addSynapsePopulation<WeightUpdateModels::StaticPulseConstantWeight, PostsynapticModels::DeltaCurr>(
                "Synapses1", SynapseMatrixType::SPARSE, NO_DELAY,
                "Pre", "Post",
                {{"g", 1.0}}, {},
                {}, {});

            model.addCustomConnectivityUpdate<RemoveSynapsePrePost>(
                "CustomConnectivityUpdate1", "Test2", "Synapses1",
                {}, {{"g", params[0]}}, {{"preThresh", params[1]}}, {{"postThresh", params[2]}},
                {}, {}, {});
         });
}
