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
enum Module
{
    ModuleNeuronUpdate,
    ModuleSynapseUpdate,
    ModuleCustomUpdate,
    ModuleInit,
    ModuleMax,
};

template<typename T, typename M, size_t N>
void test(const std::pair<T, std::array<bool, ModuleMax>> (&modelModifiers)[N], M applyModifierFn)
{
    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;

    // Create array of module digests
    boost::uuids::detail::sha1::digest_type moduleHash[ModuleMax][N];

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

        // Write hash digests of model modules to arrays
        moduleHash[ModuleNeuronUpdate][i] = modelSpecMerged.getNeuronUpdateHashDigest();
        moduleHash[ModuleSynapseUpdate][i] = modelSpecMerged.getSynapseUpdateHashDigest();
        moduleHash[ModuleCustomUpdate][i] = modelSpecMerged.getCustomUpdateHashDigest();
        moduleHash[ModuleInit][i] = modelSpecMerged.getInitHashDigest();
    }

    // Loop through modified models
    for(size_t i = 1; i < N; i++) {
        // Loop through modules and check that the hashes compare in the expected way to the baseline
        for(size_t m = 0; m < ModuleMax; m++) {
            ASSERT_TRUE((moduleHash[m][i] == moduleHash[m][0]) == modelModifiers[i].second[m]);
        }
    }
}
}   // Anonymous namespace
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareModelChanges)
{
    // Make array of functions to modify default model and flags determining whether the hashes should match baseline
    const std::pair<std::function<void(ModelSpecInternal &)>, std::array<bool, ModuleMax>> modelModifiers[] = {
        {nullptr, {}},
        {nullptr, {true, true, true, true}},
        {[](ModelSpecInternal &model) { model.setName("interesting_name"); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setDT(1.0); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setTiming(true); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setPrecision(GENN_DOUBLE); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setTimePrecision(TimePrecision::DOUBLE); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setBatchSize(10); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setSeed(1234); }, {true, true, true, false}}};
    
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
    const std::pair<std::vector<unsigned int>, std::array<bool, ModuleMax>> populationSizes[] = {
        {{10, 50},      {}},
        {{10, 50},      {true, true, true, true}},
        {{50, 10},      {false, true, true, false}},
        {{20, 20},      {false, true, true, false}},
        {{},            {false, true, true, false}},
        {{10, 20, 30},  {false, true, true, false}},
        {{20},          {false, true, true, false}}};
    
    test(populationSizes, 
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
TEST(ModelSpecMerged, CompareNeuronParamChanges)
{
    // Izhikevcih parameter sets
    const std::array<double, 4> paramVals1{0.02, 0.2, -65.0, 4.0};
    const std::array<double, 4> paramVals2{0.1, 0.2, -65.0, 2.0};
    const std::array<double, 4> paramVals3{0.02, 0.2, -50.0, 2.0};

    // Make array of population parameters to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 4>>, std::array<bool, ModuleMax>> populationSizes[] = {
        {{paramVals1, paramVals2},  {}},
        {{paramVals1, paramVals2},  {true, true, true, true}},
        {{paramVals1, paramVals1},  {false, true, true, true}},
        {{paramVals2, paramVals3},  {false, true, true, true}},
        {{},                        {false, true, true, false}},
        {{paramVals1},              {false, true, true, false}}};

    test(populationSizes, 
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

};
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareNeuronVarInitChanges)
{
    // Izhikevcih parameter sets
    const std::array<double, 2> varInit1{0.0, 0.0};
    const std::array<double, 2> varInit2{30.0, 0.0};

    // Make array of population variable initialiesers to build model with and flags determining whether the hashes should match baseline
    const std::pair<std::vector<std::array<double, 2>>, std::array<bool, ModuleMax>> populationSizes[] = {
        {{varInit1, varInit2},  {}},
        {{varInit1, varInit2},  {true, true, true, true}},
        {{varInit2, varInit1},  {true, true, true, false}},
        {{varInit2},            {false, true, true, false}},
        {{},                    {false, true, true, false}},
        {{varInit1},            {false, true, true, false}}};

    test(populationSizes, 
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

};