// Standard C++ includes
#include <array>
#include <functional>

// Google test includes
#include "gtest/gtest.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// (Single-threaded CPU) backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(ModelSpecMerged, CompareModelChanges)
{
    enum Module
    {
        ModuleNeuronUpdate,
        ModuleSynapseUpdate,
        ModuleCustomUpdate,
        ModuleInit,
        ModuleMax,
    };

    // Make array of functions to modify default model and flags determining whether the hashes should match baseline
    const std::pair<std::function<void(ModelSpecInternal &)>, std::array<bool, ModuleMax>> modelModifiers[] = {
        {nullptr, {}},
        {[](ModelSpecInternal &model) { model.setName("interesting_name"); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setDT(1.0); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setTiming(true); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setPrecision(GENN_DOUBLE); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setTimePrecision(TimePrecision::DOUBLE); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setBatchSize(10); }, {false, false, false, false}},
        {[](ModelSpecInternal &model) { model.setSeed(1234); }, {true, true, true, false}}};
    
    // Count resultant backends
    constexpr size_t numModels = sizeof(modelModifiers) / sizeof(modelModifiers[0]);

    // Create a backend
    CodeGenerator::SingleThreadedCPU::Preferences preferences;

    // Create array of models and nodule digests
    ModelSpecInternal models[numModels];
    boost::uuids::detail::sha1::digest_type moduleHash[ModuleMax][numModels];
    
    // Loop through
    for(size_t i = 0; i < numModels; i++) {
        // Set default model properties
        models[i].setName("test");
        models[i].setDT(0.1);
        models[i].setTiming(false);
        models[i].setPrecision(GENN_FLOAT);
        models[i].setTimePrecision(TimePrecision::DEFAULT);
        models[i].setBatchSize(1);
        models[i].setSeed(0);

        // If there's a modifier, apply it to model
        if(modelModifiers[i].first) {
            modelModifiers[i].first(models[i]);
        }
        
        // Finalize model
        models[i].finalize();

        // Create suitable backend to build model
        CodeGenerator::SingleThreadedCPU::Backend backend(models[i].getPrecision(), preferences);

         // Merge model
        CodeGenerator::ModelSpecMerged modelSpecMerged(models[i], backend);

        // Write hash digests of model modules to arrays
        moduleHash[ModuleNeuronUpdate][i] = modelSpecMerged.getNeuronUpdateHashDigest();
        moduleHash[ModuleSynapseUpdate][i] = modelSpecMerged.getSynapseUpdateHashDigest();
        moduleHash[ModuleCustomUpdate][i] = modelSpecMerged.getCustomUpdateHashDigest();
        moduleHash[ModuleInit][i] = modelSpecMerged.getInitHashDigest();
    }

    // Loop through modified models
    for(size_t i = 1; i < numModels; i++) {
        // Loop through modules and check that the hashes compare in the expected way to the baseline
        for(size_t m = 0; m < ModuleMax; m++) {
            ASSERT_TRUE((moduleHash[m][i] == moduleHash[m][0]) == modelModifiers[i].second[m]);
        }
    }
}