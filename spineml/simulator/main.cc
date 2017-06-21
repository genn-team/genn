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

// SpineMLCommon includes
#include "spineMLUtils.h"

// SpineML simulator includes
#include "connectors.h"
#include "logOutput.h"
#include "logOutputAnalogue.h"
#include "logOutputSpike.h"
#include "modelPropertyFixed.h"
#include "modelPropertyUniformDistribution.h"
#include "timer.h"

using namespace SpineMLCommon;
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

template <typename T>
std::pair<T*, T*> getStateVar(void *modelLibrary, const std::string &hostStateVarName)
{
    // Get host statevar
    T *hostStateVar = (T*)dlsym(modelLibrary, hostStateVarName.c_str());

#ifdef CPU_ONLY
    T *deviceStateVar = NULL;
#else
    std::string deviceStateVarName = "d_" + hostStateVarName;
     T *deviceStateVar = (T*)dlsym(modelLibrary, deviceStateVarName.c_str());
#endif

    return std::make_pair(hostStateVar, deviceStateVar);
}
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
        // Get pointers to state vars in model library
        scalar **hostStateVar;
        scalar **deviceStateVar;
        std::tie(hostStateVar, deviceStateVar) = getStateVar<scalar*>(modelLibrary, paramName + popName);

        // If it's found
        // **NOTE** it not being found is not an error condition - it just suggests that
        if(hostStateVar != NULL) {
            std::cout << "\t" << paramName << std::endl;
#ifdef CPU_ONLY
            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << std::endl;
#else
            if(deviceStateVar == NULL) {
                throw std::runtime_error("Cannot find device-side state variable for property:" + paramName);
            }

            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << ", device pointer:" << *deviceStateVar << std::endl;
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

std::unique_ptr<LogOutput> createLogOutput(const pugi::xml_node &node, void *modelLibrary, double dt,
                                           const filesystem::path &basePath, const PopulationProperties &neuronProperties)
{
    // Get name of target
    std::string target = SpineMLUtils::getSafeName(node.attribute("target").value());

    // If this is a spike recorder
    std::string port = node.attribute("port").value();
    if(port == "spike") {
        // Get pointers to spike counts in model library
        unsigned int **hostSpikeCount;
        unsigned int **deviceSpikeCount;
        std::tie(hostSpikeCount, deviceSpikeCount) = getStateVar<unsigned int*>(modelLibrary, "glbSpkCnt" + target);
#ifdef CPU_ONLY
        if(hostSpikeCount == NULL) {
#else
        if(hostSpikeCount == NULL || deviceSpikeCount == NULL) {
#endif
            throw std::runtime_error("Cannot find spike count variable for population:" + target);
        }

        // Get pointers to spike counts in model library
        unsigned int **hostSpikes;
        unsigned int **deviceSpikes;
        std::tie(hostSpikes, deviceSpikes) = getStateVar<unsigned int*>(modelLibrary, "glbSpk" + target);
#ifdef CPU_ONLY
        if(hostSpikes == NULL) {
#else
        if(hostSpikes == NULL || deviceSpikes == NULL) {
#endif
            throw std::runtime_error("Cannot find spike variable for population:" + target);
        }

         // Get host statevar
        unsigned int *spikeQueuePtr = (unsigned int*)dlsym(modelLibrary, ("spkQuePtr" + target).c_str());
        if(spikeQueuePtr == NULL) {
            throw std::runtime_error("Cannot find spike queue pointer for population:" + target);
        }

        // Create spike logger
        return std::unique_ptr<LogOutput>(new LogOutputSpike(node, dt, basePath, spikeQueuePtr,
                                                             *hostSpikeCount, *deviceSpikeCount,
                                                             *hostSpikes, *deviceSpikes));
    }
    // Otherwise we assume it's a neuron property
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = neuronProperties.find(target);
        if(targetProperties != neuronProperties.end()) {
            // If there is a model property object for this port return an analogue log output to read it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                return std::unique_ptr<LogOutput>(new LogOutputAnalogue(node, dt, basePath,
                                                                        portProperty->second.get()));
            }
        }
    }

    throw std::runtime_error("No supported logger found");
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if(argc != 3) {
        std::cerr << "Expected model library and; experiment XML files passed as arguments" << std::endl;
        return EXIT_FAILURE;
    }

    // **YUCK** hard coded 0.1ms time step as SpineML specifies this in experiment but GeNN in model
    const double dt = 0.1;

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

        // Use filesystem library to get parent path of the network XML file
        auto experimentPath = filesystem::path(argv[2]);
        auto basePath = experimentPath.parent_path();

        // Load experiment document
        pugi::xml_document experimentDoc;
        auto experimentResult = experimentDoc.load_file(argv[2]);
        if(!experimentResult) {
            throw std::runtime_error("Unable to load experiment XML file:" + std::string(argv[2]) + ", error:" + experimentResult.description());
        }

        // Get SpineML root
        auto experimentSpineML = experimentDoc.child("SpineML");
        if(!experimentSpineML) {
            throw std::runtime_error("XML file:" + std::string(argv[2]) + " is not a SpineML experiment - it has no root SpineML node");
        }

        // Get experiment node
        auto experiment = experimentSpineML.child("Experiment");
        if(!experiment) {
             throw std::runtime_error("No 'Experiment' node found");
        }

        // Get model
        auto model = experiment.child("Model");
        if(!model) {
            throw std::runtime_error("No 'Model' node found in experiment");
        }

        // Build path to network from URL in model
        auto networkPath = basePath / model.attribute("network_layer_url").value();
        std::cout << "Experiment using model:" << networkPath << std::endl;

        // Get the filename of the network and remove extension
        // to get something usable as a network name
        std::string networkName = networkPath.filename();
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
        auto networkResult = networkDoc.load_file(networkPath.str().c_str());
        if(!networkResult) {
            throw std::runtime_error("Unable to load network XML file:" + std::string(argv[2]) + ", error:" + networkResult.description());
        }

        // Get SpineML root
        auto networkSpineML = networkDoc.child("LL:SpineML");
        if(!networkSpineML) {
            throw std::runtime_error("XML file:" + std::string(argv[2]) + " is not a low-level SpineML network - it has no root SpineML node");
        }

        // Loop through populations once to initialize neuron population properties
        std::map<std::string, unsigned int> neuronPopulationSizes;
        PopulationProperties neuronProperties;
        for(auto population : networkSpineML.children("LL:Population")) {
            auto neuron = population.child("LL:Neuron");
            if(!neuron) {
                throw std::runtime_error("'Population' node has no 'Neuron' node");
            }

            // Read basic population properties
            auto popName = SpineMLUtils::getSafeName(neuron.attribute("name").value());
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
        for(auto population : networkSpineML.children("LL:Population")) {
            // Read source population name from neuron node
            auto srcPopName = SpineMLUtils::getSafeName(population.child("LL:Neuron").attribute("name").value());
            unsigned int srcPopSize = getNeuronPopSize(srcPopName, neuronPopulationSizes);

            // Loop through outgoing projections
            for(auto projection : population.children("LL:Projection")) {
                // Read destination population name from projection
                auto trgPopName = SpineMLUtils::getSafeName(projection.attribute("dst_population").value());
                unsigned int trgPopSize = getNeuronPopSize(trgPopName, neuronPopulationSizes);

                std::cout << "Projection from population:" << srcPopName << "->" << trgPopName << std::endl;

                // Build name of synapse population from these two
                std::string synPopName = std::string(srcPopName) + "_" + trgPopName;

                // Get main synapse node
                auto synapse = projection.child("LL:Synapse");
                if(!synapse) {
                    throw std::runtime_error("'Projection' node has no 'Synapse' node");
                }

                // Initialize connector (will result in correct calculation for num synapses)
                unsigned int numSynapses = initializeConnector(synapse, modelLibrary,
                                                               synPopName, srcPopSize, trgPopSize);

                // Get post synapse
                auto postSynapse = synapse.child("LL:PostSynapse");
                if(!postSynapse) {
                    throw std::runtime_error("'Synapse' node has no 'PostSynapse' node");
                }

                // Add postsynapse properties to dictionary
                addProperties(postSynapse, modelLibrary, synPopName, trgPopSize, postsynapticProperties);

                // Get weight update
                auto weightUpdate = synapse.child("LL:WeightUpdate");
                if(!weightUpdate) {
                    throw std::runtime_error("'Synapse' node has no 'WeightUpdate' node");
                }

                // Add weight update properties to dictionary
                addProperties(weightUpdate, modelLibrary, synPopName, numSynapses, weightUpdateProperties);
            }
        }

        // Loop through output loggers specified by experiment and create handler
        std::vector<std::unique_ptr<LogOutput>> loggers;
        for(auto logOutput : experiment.children("LogOutput")) {
            loggers.push_back(createLogOutput(logOutput, modelLibrary, dt, basePath, neuronProperties));
        }

        auto simulation = experiment.child("Simulation");
        if(!simulation) {
            throw std::runtime_error("No 'Simulation' node found in experiment");
        }

        // Read duration from simulation and convert to timesteps
        double durationMs = simulation.attribute("duration").as_double() * 1000.0;
        unsigned int numTimeSteps = (unsigned int)std::ceil(durationMs / dt);
        std::cout << "Simulating for " << numTimeSteps << " " << dt << "ms timesteps" << std::endl;

        // Perform final initialization
        initializeNetwork();

        // Loop through time
        for(unsigned int i = 0; i < numTimeSteps; i++)
        {
            // Advance time
#ifdef CPU_ONLY
            stepTimeCPU();
#else
            stepTimeGPU();
#endif

            // Perform any recording required this timestep
            for(auto &logger : loggers)
            {
                logger->record(dt, i);
            }
        }
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