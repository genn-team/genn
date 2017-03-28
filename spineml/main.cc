// Standard C++ includes
#include <iostream>
#include <map>
#include <set>
#include <string>

// Standard C includes
#include <cassert>
#include <cstdlib>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "generateALL.h"
#include "global.h"
#include "modelSpec.h"
#include "utils.h"

// SpineMLGenerator includes
#include "spineMLNeuronModel.h"

using namespace SpineMLGenerator;

int main(int argc,
         char *argv[])
{
    if(argc != 2) {
        std::cerr << "Expected XML file passed as argument" << std::endl;
        return EXIT_FAILURE;
    }

#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    deviceProp = new cudaDeviceProp[deviceCount];
    for (int device = 0; device < deviceCount; device++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(device));
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));
    }
#endif // CPU_ONLY

    // Use filesystem library to get parent path of the network XML file
    auto networkPath = filesystem::path(argv[1]);
    auto basePath = networkPath.parent_path();

    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(networkPath.str().c_str());
    if(!result) {
        std::cerr << "Unable to load XML file:" << argv[1] << ", error:" << result.description() << std::endl;
        return EXIT_FAILURE;
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        std::cerr << "XML file:" << argv[1] << " is not a SpineML network - it has no root SpineML node" << std::endl;
        return EXIT_FAILURE;
    }



    typedef std::pair<std::string, std::set<std::string>> ModelParams;
    std::map<ModelParams, SpineMLNeuronModel> neuronModels;

    // Initialize GeNN
    initGeNN();

    // The neuron model
    NNmodel model;

    std::string networkName = networkPath.filename();
    networkName = networkName.substr(0, networkName.find_last_of("."));

    model.setDT(1.0);
    model.setName(networkName);

    // Loop through populations
    for(auto population : spineML.children("Population")) {
        auto neuron = population.child("Neuron");
        if(!neuron) {
            std::cerr << "Warning: 'Population' node has no 'Neuron' node" << std::endl;
            continue;
        }

        const auto *popName = neuron.attribute("name").value();
        const unsigned int popSize = neuron.attribute("size").as_int();
        std::cout << "Population " << popName << " consisting of ";
        std::cout << popSize << " neurons" << std::endl;

        // Build uniquely identifying model parameters starting with its 'url'
        ModelParams modelParams;
        modelParams.first = (basePath / neuron.attribute("url").value()).str();

        // Determine which properties are variable (therefore
        // can't be substituted directly into auto-generated code)
        std::map<std::string, double> fixedParamVals;
        for(auto param : neuron.children("Property")) {
            const auto *paramName = param.attribute("name").value();

            // If parameter has a fixed value, it can be hard-coded into either model or automatically initialized in simulator
            // **TODO** annotation to say you don't want this to be hard-coded
            auto fixedValue = param.child("FixedValue");
            if(fixedValue) {
                fixedParamVals.insert(std::make_pair(paramName, fixedValue.attribute("value").as_double()));
            }
            // Otherwise, in GeNN terms, it should be treated as a variable
            else {
                modelParams.second.insert(paramName);
            }
        }

        // If no existing model is found that matches parameters
        const auto existingModel = neuronModels.find(modelParams);
        if(existingModel == neuronModels.end())
        {
            // Create new model
            std::cout << "\tCreating new neuron model" << std::endl;
            auto newModel = neuronModels.insert(
                std::make_pair(modelParams, SpineMLNeuronModel(modelParams.first, modelParams.second)));

            // Add population to model
            model.addNeuronPopulation(popName, popSize, &newModel.first->second,
                                      SpineMLNeuronModel::ParamValues(fixedParamVals, newModel.first->second),
                                      SpineMLNeuronModel::VarValues(fixedParamVals, newModel.first->second));
        }
        else
        {
            std::cout << "\tUsing existing model" << std::endl;

            // Add population to model
            model.addNeuronPopulation(popName, popSize, &existingModel->second,
                                      SpineMLNeuronModel::ParamValues(fixedParamVals, existingModel->second),
                                      SpineMLNeuronModel::VarValues(fixedParamVals, existingModel->second));
        }

        // **TODO** neuron model needs to be able to specify that some parameters, initialised with a fixed value are actually state variables

    }

    // Finalize model
    model.finalize();

#ifndef CPU_ONLY
    chooseDevice(model, basePath.str());
#endif // CPU_ONLY
    generate_model_runner(model, basePath.str());

    return EXIT_SUCCESS;
}