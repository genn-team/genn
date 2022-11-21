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

//--------------------------------------------------------------------------
// Anonyous namespace
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

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(STDPAdditive, 6, 1, 1, 1);

    SET_PARAM_NAMES({
      "tauPlus",  // 0 - Potentiation time constant (ms)
      "tauMinus", // 1 - Depression time constant (ms)
      "Aplus",    // 2 - Rate of potentiation
      "Aminus",   // 3 - Rate of depression
      "Wmin",     // 4 - Minimum weight
      "Wmax"});   // 5 - Maximum weight

    SET_VARS({{"g", "scalar"}});
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

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_MODEL(STDPAdditive);

class StaticPulseDendriticDelayReverse : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseDendriticDelayReverse, 0, 2, 0, 0);

    SET_VARS({{"d", "uint8_t", VarAccess::READ_ONLY}, {"g", "scalar", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSynDelay, $(g), $(d));\n");
};
IMPLEMENT_MODEL(StaticPulseDendriticDelayReverse);


class Sum : public CustomUpdateModels::Base
{
    DECLARE_CUSTOM_UPDATE_MODEL(Sum, 1, 1, 1);

    SET_UPDATE_CODE("$(sum) = $(a) + $(b);\n");

    SET_VARS({{"sum", "scalar"}});
    SET_PARAM_NAMES({"b"});
    SET_VAR_REFS({{"a", "scalar", VarAccessMode::READ_ONLY}});
};
IMPLEMENT_MODEL(Sum);

class OneToOneOff : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(OneToOneOff, 0);

    SET_ROW_BUILD_CODE(
        "$(addSynapse, $(id_pre) + 1);\n"
        "$(endRow);\n");

    SET_MAX_ROW_LENGTH(1);
    SET_MAX_COL_LENGTH(1);
};
IMPLEMENT_MODEL(OneToOneOff);

class RemoveSynapse : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapse, 0, 0, 0, 0, 0, 0, 0);
    
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(id_post) == ($(id_pre) + 1)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapse);

class RemoveSynapsePrePost : public CustomConnectivityUpdateModels::Base
{
public:
    DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL(RemoveSynapsePrePost, 0, 1, 1, 1, 0, 0, 0);
    
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preThresh", "scalar"}});
    SET_POST_VARS({{"postThresh", "scalar"}});
    SET_ROW_UPDATE_CODE(
        "$(for_each_synapse,\n"
        "{\n"
        "   if($(g) < $(preThresh) || $(g) < $(postThresh)) {\n"
        "       $(remove_synapse);\n"
        "       break;\n"
        "   }\n"
        "});\n");
};
IMPLEMENT_MODEL(RemoveSynapsePrePost);


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
        model.setPrecision(GENN_FLOAT);
        model.setTimePrecision(TimePrecision::DEFAULT);
        model.setBatchSize(1);
        model.setSeed(0);

        // Apply modifier
        applyModifierFn(modelModifiers[i].first, model);
        
        // Finalize model
        model.finalize();

        // Create suitable backend to build model
        CodeGenerator::SingleThreadedCPU::Backend backend(model.getPrecision(), preferences);

         // Merge model
        CodeGenerator::ModelSpecMerged modelSpecMerged(model, backend);

        // Write hash digests of model to array
        moduleHash[i] = modelSpecMerged.getHashDigest(backend);
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
             NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 4.0);
             NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 100, 
                                                                 neuronParamVals, neuronVarVals);

             STDPAdditive::ParamValues params(20.0, 20.0, 0.001, -0.001, 0.0, 1.0);
             STDPAdditive::VarValues varValues(0.5);
             STDPAdditive::PreVarValues preVarValues(0.0);
             STDPAdditive::PostVarValues postVarValues(0.0);
               
             AlphaCurr::ParamValues psmParams(5.0);
             AlphaCurr::VarValues psmVarValues(0.0);

             auto *sg = model.addSynapsePopulation<STDPAdditive, AlphaCurr>(
                 "Synapse", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                 "Pre", "Post",
                 params, varValues, preVarValues, postVarValues,
                 psmParams, psmVarValues,
                 initConnectivity<InitSparseConnectivitySnippet::FixedProbability>({0.1}));
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
            NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
            NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

            // Create synapse group with global weights
            model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                "Synapses1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                "Pre", "Post",
                {}, {1.0},
                {}, {});

            auto *ccu = model.addCustomConnectivityUpdate<RemoveSynapsePrePost>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                                {}, {1.0}, {1.0}, {1.0},
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
        {[](ModelSpecInternal &model) { model.setPrecision(GENN_DOUBLE); }, false},
        {[](ModelSpecInternal &model) { model.setTimePrecision(TimePrecision::DOUBLE); }, false},
        {[](ModelSpecInternal &model) { model.setBatchSize(10); }, false},
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
             NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 4.0);
             NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);

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
             NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 4.0);
             NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);

             // Add population with specified name
             model.addNeuronPopulation<NeuronModels::Izhikevich>(name, 100, 
                                                                 paramVals, varVals);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronParamChanges)
{
    // Izhikevcih parameter sets
    const std::array<double, 4> paramVals1{0.02, 0.2, -65.0, 4.0};
    const std::array<double, 4> paramVals2{0.1, 0.2, -65.0, 2.0};
    const std::array<double, 4> paramVals3{0.02, 0.2, -50.0, 2.0};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 4>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 4>> &popParams, ModelSpecInternal &model)
         {
             // Default neuron parameters
             NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);

             // Add desired number of populations
             for(size_t p = 0; p < popParams.size(); p++) {
                 NeuronModels::Izhikevich::ParamValues paramVals(popParams[p][0], popParams[p][1], 
                                                                 popParams[p][2], popParams[p][3]);

                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons" + std::to_string(p), 100, 
                                                                     paramVals, varVals);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronVarInitChanges)
{
    // Izhikevcih parameter sets
    const std::array<double, 2> varInit1{0.0, 0.0};
    const std::array<double, 2> varInit2{30.0, 0.0};

    // Make array of population variable initialiesers to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{varInit1, varInit2},  true},
        {{varInit1, varInit2},  true},
        {{varInit2, varInit1},  false},
        {{varInit2},            false},
        {{},                    false},
        {{varInit1},            false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &popVarInit, ModelSpecInternal &model)
         {
             // Default neuron parameters
             NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 4.0);

             // Add desired number of populations
             for(size_t p = 0; p < popVarInit.size(); p++) {
                 NeuronModels::Izhikevich::VarValues varVals(popVarInit[p][0], popVarInit[p][1]);
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons" + std::to_string(p), 100, 
                                                                     paramVals, varVals);
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             CurrentSourceModels::DC::ParamValues paramVals(3.0);
             model.addCurrentSource<CurrentSourceModels::DC>(name, "Neurons",
                                                             paramVals, {});
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceParamChanges)
{
    // Current source parameter sets
    const double paramVal1 = 1.0;
    const double paramVal2 = 2.0;
    const double paramVal3 = 0.5;

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<double>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},    true},
        {{paramVal1, paramVal2},    true},
        {{paramVal1, paramVal1},    false},
        {{paramVal2, paramVal3},    false},
        {{},                        false},
        {{paramVal1},               false}};

    test(modelModifiers, 
         [](const std::vector<double> &csParams, ModelSpecInternal &model)
         {
             // Add population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of current sources
             for(size_t p = 0; p < csParams.size(); p++) {
                 CurrentSourceModels::DC::ParamValues paramVals(csParams[p]);
                 model.addCurrentSource<CurrentSourceModels::DC>("CS" + std::to_string(p), "Neurons",
                                                                 paramVals, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCurrentSourceVarInitParamChanges)
{
    // Current source parameter sets
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &csVarInit, ModelSpecInternal &model)
         {
             // Add population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of current sources
             for(size_t p = 0; p < csVarInit.size(); p++) {
                 CurrentSourceModels::PoissonExp::ParamValues paramVals(1.0, 20.0, 10.0);
                 CurrentSourceModels::PoissonExp::VarValues varVals(initVar<InitVarSnippet::Uniform>({csVarInit[p][0], csVarInit[p][1]}));
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
              NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
              NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
              model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                 neuronParamVals, neuronVarVals);
              CurrentSourceModels::PoissonExp::ParamValues paramVals(1.0, 20.0, 10.0);
              CurrentSourceModels::PoissonExp::VarValues varVals(0.0);
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 100, 
                                                                 neuronParamVals, neuronVarVals);

             model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                name, SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                "Pre", "Post",
                {}, {1.0},
                {}, {});
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, ComparePSMParamChanges)
{
    // Postsynaptic model tau parameters
    const double paramVal1 = 5.0;
    const double paramVal2 = 1.0;
    const double paramVal3 = 10.0;

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<double>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal1},  false},
        {{paramVal2, paramVal3},  false},
        {{},                      false},
        {{paramVal1},             false}};

    test(modelModifiers, 
         [](const std::vector<double> &psmParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < psmParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 PostsynapticModels::ExpCurr::ParamValues params(psmParams[p]);
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, {1.0},
                    params, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, ComparePSMVarInitParamChanges)
{
    // Postsynaptic model var initialisers
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &psmVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < psmVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 AlphaCurr::ParamValues params(5.0);
                 AlphaCurr::VarValues varValues(initVar<InitVarSnippet::Uniform>({psmVarInitParams[p][0], psmVarInitParams[p][1]}));
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, {1.0},
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
    const std::array<double, 10> paramVals1{50.0, 50.0, 50000.0, 100000.0, 200.0, 0.015, 0.0075, 33.33, 10.0, 0.00006};
    const std::array<double, 10> paramVals2{100.0, 100.0, 50000.0, 100000.0, 200.0, 0.015, 0.0075, 33.33, 10.0, 0.00006};
    const std::array<double, 10> paramVals3{50.0, 50.0, 50000.0, 100000.0, 200.0, 0.1, 0.0075, 33.33, 10.0, 0.00006};
    

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 10>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 10>> &wumParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 WeightUpdateModels::PiecewiseSTDP::ParamValues params(wumParams[p][0], wumParams[p][1], wumParams[p][2], wumParams[p][3], 
                                                                       wumParams[p][4], wumParams[p][5], wumParams[p][6], wumParams[p][7], 
                                                                       wumParams[p][8], wumParams[p][9]);
                 WeightUpdateModels::PiecewiseSTDP::VarValues varInit(0.0, uninitialisedVar());
                 model.addSynapsePopulation<WeightUpdateModels::PiecewiseSTDP, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    params, varInit,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMGlobalGVarChanges)
{
    // Weight update model variable initialisers
    const double varVals1 = 1.0;
    const double varVals2 = 0.2;
    const double varVals3 = 0.9;

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<double>, bool> modelModifiers[] = {
        {{varVals1, varVals2},  true},
        {{varVals1, varVals2},  true},
        {{varVals1, varVals1},  false},
        {{varVals2, varVals3},  false},
        {{},                        false},
        {{varVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<double> &wumVarVals, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumVarVals.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 WeightUpdateModels::StaticPulse::VarValues varValues(wumVarVals[p]);
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                    "Pre", "Post" + std::to_string(p),
                    {}, varValues,
                    {}, {});
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareWUMVarInitParamChanges)
{
    // Weight update model var initialisers
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &wumVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 WeightUpdateModels::StaticPulse::VarValues varValues(initVar<InitVarSnippet::Uniform>({wumVarInitParams[p][0], wumVarInitParams[p][1]}));
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
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
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &wumPreVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumPreVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 STDPAdditive::ParamValues params(20.0, 20.0, 0.001, -0.001, 0.0, 1.0);
                 STDPAdditive::VarValues varValues(0.5);
                 STDPAdditive::PreVarValues preVarValues(initVar<InitVarSnippet::Uniform>({wumPreVarInitParams[p][0], wumPreVarInitParams[p][1]}));
                 STDPAdditive::PostVarValues postVarValues(0.0);
                
                 model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
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
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &wumPostVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < wumPostVarInitParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 STDPAdditive::ParamValues params(20.0, 20.0, 0.001, -0.001, 0.0, 1.0);
                 STDPAdditive::VarValues varValues(0.5);
                 STDPAdditive::PreVarValues preVarValues(0.0);
                 STDPAdditive::PostVarValues postVarValues(initVar<InitVarSnippet::Uniform>({wumPostVarInitParams[p][0], wumPostVarInitParams[p][1]}));
                
                 model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
                    "Synapse" + std::to_string(p), SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
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
    const double paramVals1 = 0.1;
    const double paramVals2 = 0.2;
    const double paramVals3 = 0.9;

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<double>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<double> &connectivityParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < connectivityParams.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 WeightUpdateModels::StaticPulse::VarValues varValues(1.0);
                 InitSparseConnectivitySnippet::FixedProbability::ParamValues connectParams(connectivityParams[p]);
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                     "Synapse" + std::to_string(p), SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                     "Pre", "Post" + std::to_string(p),
                     {}, varValues,
                     {}, {},
                     initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(connectParams));
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                 neuronParamVals, neuronVarVals);

             // Add desired number of post populations
             for(size_t p = 0; p < connectivityModels.size(); p++) {
                 model.addNeuronPopulation<NeuronModels::Izhikevich>("Post" + std::to_string(p), 100, 
                                                                     neuronParamVals, neuronVarVals);

                 WeightUpdateModels::StaticPulse::VarValues varValues(1.0);
                 model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                     "Synapse" + std::to_string(p), SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
                     "Pre", "Post" + std::to_string(p),
                     {}, varValues,
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             auto *neurons = model.addNeuronPopulation<NeuronModels::Izhikevich>("Neurons", 100, 
                                                                                 neuronParamVals, neuronVarVals);

             Sum::ParamValues paramVals(5.0);
             Sum::VarValues vals(0.0);
             Sum::VarReferences varRefs(createVarRef(neurons, "V"));
             model.addCustomUpdate<Sum>(name, "Group", paramVals, vals, varRefs);
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateParamChanges)
{
    // Custom update model "b" parameters
    const double paramVal1 = 5.0;
    const double paramVal2 = 1.0;
    const double paramVal3 = 10.0;

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<double>, bool> modelModifiers[] = {
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal2},  true},
        {{paramVal1, paramVal1},  false},
        {{paramVal2, paramVal3},  false},
        {{},                      false},
        {{paramVal1},             false}};

    test(modelModifiers, 
         [](const std::vector<double> &customUpdateParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateParams.size(); c++) {
                 Sum::ParamValues paramVals(customUpdateParams[c]);
                 Sum::VarValues vals(0.0);
                 Sum::VarReferences varRefs(createVarRef(pre, "V"));
                 model.addCustomUpdate<Sum>("CU" + std::to_string(c), "Group", paramVals, vals, varRefs);
             }
         });
}
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareCustomUpdateVarInitParamChanges)
{
    // Custom update model var initialisers
    const std::array<double, 2> paramVals1{0.0, 1.0};
    const std::array<double, 2> paramVals2{0.1, 0.2};
    const std::array<double, 2> paramVals3{0.3, 0.9};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, bool> modelModifiers[] = {
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals2},  true},
        {{paramVals1, paramVals1},  false},
        {{paramVals2, paramVals3},  false},
        {{},                        false},
        {{paramVals1},              false}};

    test(modelModifiers, 
         [](const std::vector<std::array<double, 2>> &customUpdateVarInitParams, ModelSpecInternal &model)
         {
             // Add pre population
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateVarInitParams.size(); c++) {
                 Sum::ParamValues paramVals(1.0);
                 Sum::VarValues vals(initVar<InitVarSnippet::Uniform>({customUpdateVarInitParams[c][0], customUpdateVarInitParams[c][1]}));
                 Sum::VarReferences varRefs(createVarRef(pre, "V"));
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

            Sum::ParamValues paramVals(1.0);
            Sum::VarValues vals(0.0);
            Sum::VarReferences varRefs(createVarRef(pre, "V"));
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
             NeuronModels::Izhikevich::VarValues neuronVarVals(0.0, 0.0);
             NeuronModels::Izhikevich::ParamValues neuronParamVals(0.02, 0.2, -65.0, 4.0);
             auto *pre = model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 100, 
                                                                             neuronParamVals, neuronVarVals);

             // Add desired number of custom updates
             for(size_t c = 0; c < customUpdateVarRefTargets.size(); c++) {
                 Sum::ParamValues paramVals(1.0);
                 Sum::VarValues vals(0.0);
                 Sum::VarReferences varRefs(createVarRef(pre, customUpdateVarRefTargets[c]));
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
TEST(ModelSpecMerged, CompareCustomConnectivityUpdateWUMModel)
{
    auto addModel1 = [](ModelSpecInternal &model)
                     {
                         model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
                            "Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                            "Pre", "Post",
                            {}, {1.0},
                            {}, {});
                     };
    
    auto addModel2 = [](ModelSpecInternal &model)
                     {
                         model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, PostsynapticModels::DeltaCurr>(
                            "Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                            "Pre", "Post",
                            {}, {1.0, 1.0},
                            {}, {});
                     };
    
    auto addModel3 = [](ModelSpecInternal &model)
                     {
                         model.addSynapsePopulation<StaticPulseDendriticDelayReverse, PostsynapticModels::DeltaCurr>(
                            "Synapses1", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                            "Pre", "Post",
                            {}, {1.0, 1.0},
                            {}, {});
                     };
    
    // Make array of functions to add synapse groups with and flags determining whether the hashes should match baseline
    const std::pair<std::function<void(ModelSpecInternal &)>, bool> modelModifiers[] = {
        {addModel2, true},
        {addModel2, true},
        {addModel3, true},
        {addModel1, false}};

    test(modelModifiers, 
         [](std::function<void(ModelSpecInternal &)> addSynapsePopulationFn, ModelSpecInternal &model)
         {
            // Add two neuron group to model
            NeuronModels::Izhikevich::ParamValues paramVals(0.02, 0.2, -65.0, 8.0);
            NeuronModels::Izhikevich::VarValues varVals(0.0, 0.0);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Pre", 10, paramVals, varVals);
            model.addNeuronPopulation<NeuronModels::Izhikevich>("Post", 10, paramVals, varVals);

            // Create synapse group 
            addSynapsePopulationFn(model);
            
            model.addCustomConnectivityUpdate<RemoveSynapsePrePost>("CustomConnectivityUpdate1", "Test2", "Synapses1",
                                                                    {}, {1.0}, {1.0}, {1.0},
                                                                    {}, {}, {});
         });
}
