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
#include "spineMLPostsynapticModel.h"
#include "spineMLWeightUpdateModel.h"

using namespace SpineMLGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
typedef std::pair<std::string, std::set<std::string>> ModelParams;

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
std::tuple<ModelParams, std::map<std::string, double>> readModelProperties(const filesystem::path &basePath,
                                                                           const pugi::xml_node &node)
{
    // Build uniquely identifying model parameters starting with its 'url'
    ModelParams modelParams;
    modelParams.first = (basePath / node.attribute("url").value()).str();

    // Determine which properties are variable (therefore
    // can't be substituted directly into auto-generated code)
    std::map<std::string, double> fixedParamVals;
    for(auto param : node.children("Property")) {
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

    return std::make_tuple(modelParams, fixedParamVals);
}

template<typename T>
const T &getCreateModel(const ModelParams &params, std::map<ModelParams, T> &models)
{
    // If no existing model is found that matches parameters
    const auto existingModel = models.find(params);
    if(existingModel == models.end())
    {
        // Create new model
        std::cout << "\tCreating new model" << std::endl;
        auto newModel = models.insert(
            std::make_pair(params, T(params.first, params.second)));

        return newModel.first->second;
    }
    else
    {
        return existingModel->second;
    }
}

}

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
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
        throw std::runtime_error("Unable to load XML file:" + networkPath.str() + ", error:" + result.description());
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        throw std::runtime_error("XML file:" + networkPath.str() + " is not a SpineML network - it has no root SpineML node");
    }

    // Neuron, postsyaptic and weight update models required by network
    std::map<ModelParams, SpineMLNeuronModel> neuronModels;
    std::map<ModelParams, SpineMLPostsynapticModel> postsynapticModels;
    std::map<ModelParams, SpineMLWeightUpdateModel> weightUpdateModels;

    // Get the filename of the network and remove extension
    // to get something usable as a network name
    std::string networkName = networkPath.filename();
    networkName = networkName.substr(0, networkName.find_last_of("."));

    // Initialize GeNN
    initGeNN();

    // The neuron model
    NNmodel model;
    model.setDT(1.0);
    model.setName(networkName);

    // Loop through populations once to build neuron populations
    for(auto population : spineML.children("Population")) {
        auto neuron = population.child("Neuron");
        if(!neuron) {
            throw std::runtime_error("'Population' node has no 'Neuron' node");
        }

        // Read basic population properties
        const auto *popName = neuron.attribute("name").value();
        const unsigned int popSize = neuron.attribute("size").as_int();
        std::cout << "Population " << popName << " consisting of ";
        std::cout << popSize << " neurons" << std::endl;

        // Read neuron properties
        ModelParams modelParams;
        std::map<std::string, double> fixedParamVals;
        tie(modelParams, fixedParamVals) = readModelProperties(basePath, neuron);

        // Either get existing neuron model or create new one of no suitable models are available
        const auto &neuronModel = getCreateModel(modelParams, neuronModels);

        // Add population to model
        model.addNeuronPopulation(popName, popSize, &neuronModel,
                                  SpineMLNeuronModel::ParamValues(fixedParamVals, neuronModel),
                                  SpineMLNeuronModel::VarValues(fixedParamVals, neuronModel));
    }

    // Loop through populations AGAIN to build projections
    for(auto population : spineML.children("Population")) {
        // Read source population name from neuron node
        const auto *srcPopName = population.child("Neuron").attribute("name").value();

        // Loop through outgoing projections
        for(auto projection : population.children("Projection")) {
            // Read destination population name from projection
            const auto *trgPopName = projection.attribute("dst_population").value();

            std::cout << "Projection from population:" << srcPopName << "->" << trgPopName << std::endl;

            // Get main synapse node
            auto synapse = projection.child("Synapse");
            if(!synapse) {
                throw std::runtime_error("'Projection' node has no 'Synapse' node");
            }

            // Get post synapse
            auto postSynapse = synapse.child("PostSynapse");
            if(!postSynapse) {
                throw std::runtime_error("'Synapse' node has no 'PostSynapse' node");
            }

            // Read postsynapse properties
            ModelParams postsynapticModelParams;
            std::map<std::string, double> fixedPostsynapticParamVals;
            tie(postsynapticModelParams, fixedPostsynapticParamVals) = readModelProperties(basePath, postSynapse);

            // Either get existing postsynaptic model or create new one of no suitable models are available
            const auto &postsynapticModel = getCreateModel(postsynapticModelParams, postsynapticModels);

            // Get weight update
            auto weightUpdate = synapse.child("WeightUpdate");
            if(!weightUpdate) {
                throw std::runtime_error("'Synapse' node has no 'WeightUpdate' node");
            }

            // Read weight update properties
            ModelParams weightUpdateModelParams;
            std::map<std::string, double> fixedWeightUpdateParamVals;
            tie(weightUpdateModelParams, fixedWeightUpdateParamVals) = readModelProperties(basePath, weightUpdate);

            // Either get existing postsynaptic model or create new one of no suitable models are available
            const auto &weightUpdateModel = getCreateModel(weightUpdateModelParams, weightUpdateModels);

        }
    }

    // Finalize model
    model.finalize();

#ifndef CPU_ONLY
    chooseDevice(model, basePath.str());
#endif // CPU_ONLY
    generate_model_runner(model, basePath.str());

    return EXIT_SUCCESS;
}