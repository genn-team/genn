// Standard C++ includes
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>

// Standard C includes
#include <cassert>
#include <cstdlib>

// POSIX C includes
extern "C"
{
#include <dlfcn.h>
}

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML simulator includes
#include "connectors.h"
#include "modelPropertyFixed.h"
#include "modelPropertyUniformDistribution.h"
#include "timer.h"

using namespace SpineMLSimulator;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define LOAD_SYMBOL(LIBRARY, TYPE, NAME)                            \
    TYPE NAME = (TYPE)dlsym(LIBRARY, #NAME);                        \
    if(NAME == NULL) {                                              \
        throw std::runtime_error("Cannot find " #NAME " function"); \
    }

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// Typedefines
typedef void (*VoidFunction)(void);

typedef std::map<std::string, std::map<std::string, std::unique_ptr<ModelProperty>>> PopulationProperties;

unsigned int getNeuronPopSize(const std::string &popName, const std::map<std::string, unsigned int> &popSizes)
{
    auto pop = popSizes.find(popName);
    if(pop == popSizes.end()) {
        throw std::runtime_error("Cannot find neuron population:" + popName);
    }
    else {
        return pop->second;
    }
}
std::unique_ptr<ModelProperty> createModelProperty(const pugi::xml_node &node, scalar *hostStateVar, scalar *deviceStateVar, unsigned int size)
{
    auto fixedValue = node.child("FixedValue");
    if(fixedValue) {
        return std::unique_ptr<ModelProperty>(new ModelPropertyFixed(fixedValue, hostStateVar, deviceStateVar, size));
    }

    auto uniformDistribution = node.child("UniformDistribution");
    if(uniformDistribution) {
        return std::unique_ptr<ModelProperty>(new ModelPropertyUniformDistribution(uniformDistribution, hostStateVar, deviceStateVar, size));
    }

    throw std::runtime_error("Unsupported property type");
}

void addProperties(const pugi::xml_node &node, void *modelLibrary, const std::string &popName, unsigned int popSize,
                   PopulationProperties &properties)
{
    // Loop through properties in network
    for(auto param : node.children("Property")) {
        std::string paramName = param.attribute("name").value();
        std::cout << "\t" << paramName << std::endl;

        // Determine name of global variable that will contain
        std::string hostStateVarName = paramName + popName;
        scalar **hostStateVar = (scalar**)dlsym(modelLibrary, hostStateVarName.c_str());

        // If it's found
        // **NOTE** it not being found is not an error condition - it just suggests that
        if(hostStateVar != NULL) {
#ifdef CPU_ONLY
            scalar **deviceStateVar = NULL;
            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << std::endl;
#else
            std::string deviceStateVarName = "d_" + hostStateVarName;
            scalar **deviceStateVar = (scalar**)dlsym(modelLibrary, deviceStateVarName.c_str());
            if(deviceStateVar == NULL) {
                throw std::runtime_error("Cannot find device-side state variable for property:" + paramName);
            }

            std::cout << "\t\tState variable found host pointer:" << hostStateVar << ", device pointer:" << *deviceStateVar << std::endl;
#endif
            // Create model property object
            properties[popName].insert(
                std::make_pair(paramName, createModelProperty(param, *hostStateVar, *deviceStateVar, popSize)));


        }
    }
}
unsigned int initializeConnector(const pugi::xml_node &node, void *modelLibrary,
                                 const std::string &synPopName, unsigned int numPre, unsigned int numPost)
{
    // Find allocate function
    Connectors::AllocateFn allocateFn = (Connectors::AllocateFn)dlsym(modelLibrary, ("allocate" + synPopName).c_str());
    if(allocateFn == NULL) {
        throw std::runtime_error("Cannot find allocate function for synapse population:" + synPopName);
    }

    // Find sparse projection
    SparseProjection *sparseProjection = (SparseProjection*)dlsym(modelLibrary, ("C" + synPopName).c_str());

    /*auto oneToOne = node.child("OneToOneConnection");
    if(oneToOne) {
        return std::make_tuple(globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG,
                               readDelaySteps(oneToOne, dt));
    }

    auto allToAll = node.child("AllToAllConnection");
    if(allToAll) {
        return std::make_tuple(globalG ? SynapseMatrixType::DENSE_GLOBALG : SynapseMatrixType::DENSE_INDIVIDUALG,
                               readDelaySteps(allToAll, dt));
    }*/

    auto fixedProbability = node.child("FixedProbabilityConnection");
    if(fixedProbability) {
        if(sparseProjection != NULL) {
            return Connectors::fixedProbabilitySparse(fixedProbability, numPre, numPost,
                                                      *sparseProjection, allocateFn);
        }
    }

    /*auto connectionList = node.child("ConnectionList");
    if(connectionList) {
        // **TODO** there is almost certainly a number of connections above which dense is better
        return std::make_tuple(globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG,
                               readDelaySteps(connectionList, dt));
    }*/

    throw std::runtime_error("No supported connection type found for projection");
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if(argc != 4) {
        std::cerr << "Expected model library and; network and experiment XML files passed as arguments" << std::endl;
        return EXIT_FAILURE;
    }

    void *modelLibrary = NULL;
    try
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Attempt to load model library
        modelLibrary = dlopen(argv[1], RTLD_NOW);

        // If it fails throw
        if(modelLibrary == NULL)
        {
            throw std::runtime_error("Unable to load library - error:" + std::string(dlerror()));
        }

        // Load statically-named symbols from library
        LOAD_SYMBOL(modelLibrary, VoidFunction, initialize);
        LOAD_SYMBOL(modelLibrary, VoidFunction, allocateMem);
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeCPU);
#ifndef CPU_ONLY
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeGPU);
#endif // CPU_ONLY

        // Get the filename of the network and remove extension
        // to get something usable as a network name
        std::string networkName = filesystem::path(argv[2]).filename();
        networkName = networkName.substr(0, networkName.find_last_of("."));

        // Search for network initialization function
        VoidFunction initializeNetwork = (VoidFunction)dlsym(modelLibrary, ("init" + networkName).c_str());
        if(initializeNetwork == NULL) {
            throw std::runtime_error("Cannot find network initialization function 'init" + networkName + "'");
        }

        // Call library function to allocate memory
        {
            Timer t("Allocation:");
            allocateMem();
        }

        // Call library function to initialize
        // **TODO** this is probably not ENTIRELY necessary as we initialize a lot of stuff again
        {
            Timer t("Init:");
            initialize();
        }

        // Load network document
        pugi::xml_document networkDoc;
        auto result = networkDoc.load_file(argv[2]);
        if(!result) {
            throw std::runtime_error("Unable to load XML file:" + std::string(argv[2]) + ", error:" + result.description());
        }

        // Get SpineML root
        auto spineML = networkDoc.child("SpineML");
        if(!spineML) {
            throw std::runtime_error("XML file:" + std::string(argv[2]) + " is not a SpineML network - it has no root SpineML node");
        }

        // Loop through populations once to initialize neuron population properties
        std::map<std::string, unsigned int> neuronPopulationSizes;
        PopulationProperties neuronProperties;
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

            // Add neuron population properties to dictionary
            addProperties(neuron, modelLibrary, popName, popSize, neuronProperties);

            // Add size to dictionary
            neuronPopulationSizes.insert(std::make_pair(popName, popSize));
        }

        // Loop through populations AGAIN to build synapse population properties
        PopulationProperties postsynapticProperties;
        PopulationProperties weightUpdateProperties;
        for(auto population : spineML.children("Population")) {
            // Read source population name from neuron node
            const auto *srcPopName = population.child("Neuron").attribute("name").value();
            unsigned int srcPopSize = getNeuronPopSize(srcPopName, neuronPopulationSizes);

            // Loop through outgoing projections
            for(auto projection : population.children("Projection")) {
                // Read destination population name from projection
                const auto *trgPopName = projection.attribute("dst_population").value();
                unsigned int trgPopSize = getNeuronPopSize(trgPopName, neuronPopulationSizes);

                std::cout << "Projection from population:" << srcPopName << "->" << trgPopName << std::endl;

                // Build name of synapse population from these two
                std::string synPopName = std::string(srcPopName) + "_" + trgPopName;

                // Get main synapse node
                auto synapse = projection.child("Synapse");
                if(!synapse) {
                    throw std::runtime_error("'Projection' node has no 'Synapse' node");
                }

                // Initialize connector (will result in correct calculation for num synapses)
                unsigned int numSynapses = initializeConnector(synapse, modelLibrary,
                                                               synPopName, srcPopSize, trgPopSize);

                // Get post synapse
                auto postSynapse = synapse.child("PostSynapse");
                if(!postSynapse) {
                    throw std::runtime_error("'Synapse' node has no 'PostSynapse' node");
                }

                // Add postsynapse properties to dictionary
                addProperties(postSynapse, modelLibrary, synPopName, trgPopSize, postsynapticProperties);

                // Get weight update
                auto weightUpdate = synapse.child("WeightUpdate");
                if(!weightUpdate) {
                    throw std::runtime_error("'Synapse' node has no 'WeightUpdate' node");
                }

                // Add weight update properties to dictionary
                addProperties(weightUpdate, modelLibrary, synPopName, numSynapses, weightUpdateProperties);
            }
        }

        // Perform final initialization
        initializeNetwork();

    }
    catch(...)
    {
        // Close model library if loaded successfully
        if(modelLibrary)
        {
            dlclose(modelLibrary);
        }

        // Re-raise
        throw;
    }

    return 0;
}