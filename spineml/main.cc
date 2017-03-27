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
#include "modelSpec.h"

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

    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(argv[1]);
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

    // Use filesystem library to get parent path of the network XML file
    auto basePath = filesystem::path(argv[1]).parent_path();

    typedef std::pair<std::string, std::set<std::string>> ModelParams;
    std::map<ModelParams, SpineMLNeuronModel> neuronModels;

    // The neuron model
    NNmodel model;

    // Loop through populations
    for(auto population : spineML.children("Population")) {
        auto neuron = population.child("Neuron");
        if(!neuron) {
            std::cerr << "Warning: 'Population' node has no 'Neuron' node" << std::endl;
            continue;
        }

        const auto *popName = neuron.attribute("name").value();
        const unsigned int populationSize = neuron.attribute("size").as_int();
        std::cout << "Population " << popName << " consisting of ";
        std::cout << populationSize << " neurons" << std::endl;

        // Build uniquely identifying model parameters starting with its 'url'
        ModelParams modelParams;
        modelParams.first = (basePath / neuron.attribute("url").value()).str();

        std::map<std::string, double> paramVals;
        std::map<std::string, double> varVals;

        // Determine which properties are variable (therefore
        // can't be substituted directly into auto-generated code)
        for(auto param : neuron.children("Property")) {
            const auto *paramName = param.attribute("name").value();

            // If parameter has a fixed value, it can be hard-coded into generated code
            // **TODO** annotation to say you don't want this to be treated as a fixed value
            auto fixedValue = param.child("FixedValue");
            if(fixedValue) {
                paramVals.insert(
                    std::pair<std::string, double>(paramName, fixedValue.attribute("value").as_double()));
            }
            // Otherwise, in GeNN terms, it should be treated as a variable
            else {
                std::cout << "\t" << paramName << " is variable" << std::endl;
                modelParams.second.insert(paramName);

                varVals.insert(
                    std::pair<std::string, double>(paramName, 0.0));
            }
        }

        // If no existing model is found that matches parameters, add one
        const auto existingModel = neuronModels.find(modelParams);
        if(existingModel == neuronModels.end())
        {
            std::cout << "\tCreating new neuron model" << std::endl;
            neuronModels.insert(
                std::pair<ModelParams, SpineMLNeuronModel>(
                    modelParams, SpineMLNeuronModel(neuron, modelParams.first,
                                                    modelParams.second)));
        }
        else
        {
            std::cout << "\tUsing existing model" << std::endl;

            //model.addNeuronPopulation<SpineMLNeuronModel>(popName, popSize,
            //                                              paramVals, varVals);
        }

        // **TODO** neuron model needs to be able to specify that some parameters, initialised with a fixed value are actually state variables

    }

    return EXIT_SUCCESS;
}