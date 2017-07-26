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

#ifdef _WIN32
#include <windows.h>
#else
// POSIX C includes
extern "C"
{
#include <dlfcn.h>
}
#endif

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineMLCommon includes
#include "spineMLUtils.h"

// SpineML simulator includes
#include "connectors.h"
#include "input.h"
#include "inputValue.h"
#include "logOutput.h"
#include "modelProperty.h"
#include "timer.h"

using namespace SpineMLCommon;
using namespace SpineMLSimulator;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#ifdef _WIN32
#define LIBRARY_HANDLE HMODULE 

#define LOAD_SYMBOL(LIBRARY, TYPE, NAME)                            \
    TYPE NAME = (TYPE)GetProcAddress(LIBRARY, #NAME);               \
    if(NAME == nullptr) {                                           \
        throw std::runtime_error("Cannot find " #NAME " function"); \
    }
#else
#define LIBRARY_HANDLE void*

#define LOAD_SYMBOL(LIBRARY, TYPE, NAME)                            \
    TYPE NAME = (TYPE)dlsym(LIBRARY, #NAME);                        \
    if(NAME == nullptr) {                                           \
        throw std::runtime_error("Cannot find " #NAME " function"); \
    }
#endif
//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// Typedefines
typedef void (*VoidFunction)(void);

typedef std::map<std::string, std::map<std::string, std::unique_ptr<ModelProperty::Base>>> ComponentProperties;

typedef std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>> ComponentEventPorts;

//----------------------------------------------------------------------------
void *getLibrarySymbol(LIBRARY_HANDLE modelLibrary, const char *name) {
#ifdef _WIN32
    return GetProcAddress(modelLibrary, name);
#else
    return dlsym(modelLibrary, name);
#endif
}
//----------------------------------------------------------------------------
template <typename T>
std::pair<T*, T*> getStateVar(LIBRARY_HANDLE modelLibrary, const std::string &hostStateVarName)
{
    // Get host statevar
    T *hostStateVar = (T*)getLibrarySymbol(modelLibrary, hostStateVarName.c_str());

#ifdef CPU_ONLY
    T *deviceStateVar = nullptr;
#else
    std::string deviceStateVarName = "d_" + hostStateVarName;
    T *deviceStateVar = (T*)getLibrarySymbol(modelLibrary, deviceStateVarName.c_str());
#endif

    return std::make_pair(hostStateVar, deviceStateVar);
}
//----------------------------------------------------------------------------
unsigned int getComponentSize(const std::string &componentName, const std::map<std::string, unsigned int> &componentSizes)
{
    auto component = componentSizes.find(componentName);
    if(component == componentSizes.end()) {
        throw std::runtime_error("Cannot find neuron population:" + componentName);
    }
    else {
        return component->second;
    }
}
//----------------------------------------------------------------------------
bool isEventSendPort(const std::string &targetName, const std::string &portName,
                     const std::map<std::string, std::string> &componentURLs,
                     const ComponentEventPorts &componentEventPorts)
{
    auto targetURL = componentURLs.find(targetName);
    if(targetURL != componentURLs.end()) {
        auto urlEventPorts = componentEventPorts.find(targetURL->second);
        if(urlEventPorts != componentEventPorts.end()) {

            const auto &eventSendPorts = urlEventPorts->second.first;
            return (eventSendPorts.find(portName) != eventSendPorts.end());
        }
    }

    return false;
}
//----------------------------------------------------------------------------
bool isEventReceivePort(const std::string &targetName, const std::string &portName,
                     const std::map<std::string, std::string> &componentURLs,
                     const ComponentEventPorts &componentEventPorts)
{
    auto targetURL = componentURLs.find(targetName);
    if(targetURL != componentURLs.end()) {
        auto urlEventPorts = componentEventPorts.find(targetURL->second);
        if(urlEventPorts != componentEventPorts.end()) {
            const auto &eventReceivePorts = urlEventPorts->second.second;
            return (eventReceivePorts.find(portName) != eventReceivePorts.end());
        }
    }

    return false;
}
//----------------------------------------------------------------------------
std::tuple<unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int*> getNeuronPopSpikeVars(LIBRARY_HANDLE modelLibrary, const std::string &popName)
{
    // Get pointers to spike counts in model library
    unsigned int **hostSpikeCount;
    unsigned int **deviceSpikeCount;
    std::tie(hostSpikeCount, deviceSpikeCount) = getStateVar<unsigned int*>(modelLibrary, "glbSpkCnt" + popName);
#ifdef CPU_ONLY
    if(hostSpikeCount == nullptr) {
#else
    if(hostSpikeCount == nullptr || deviceSpikeCount == nullptr) {
#endif
        throw std::runtime_error("Cannot find spike count variable for population:" + popName);
    }

    // Get pointers to spike counts in model library
    unsigned int **hostSpikes;
    unsigned int **deviceSpikes;
    std::tie(hostSpikes, deviceSpikes) = getStateVar<unsigned int*>(modelLibrary, "glbSpk" + popName);
#ifdef CPU_ONLY
    if(hostSpikes == nullptr) {
#else
    if(hostSpikes == nullptr || deviceSpikes == nullptr) {
#endif
        throw std::runtime_error("Cannot find spike variable for population:" + popName);
    }

    // Get spike queue
    unsigned int *spikeQueuePtr = (unsigned int*)getLibrarySymbol(modelLibrary, ("spkQuePtr" + popName).c_str());

    // Return pointers in tutple
#ifdef CPU_ONLY
    return std::make_tuple(*hostSpikeCount, nullptr,
                           *hostSpikes, nullptr,
                           spikeQueuePtr);
#else
    return std::make_tuple(*hostSpikeCount, *deviceSpikeCount,
                           *hostSpikes, *deviceSpikes,
                           spikeQueuePtr);
#endif
}
//----------------------------------------------------------------------------
void addEventPorts(const filesystem::path &basePath, const pugi::xml_node &node,
                   std::map<std::string, std::string> &componentURLs, ComponentEventPorts &componentEventPorts)
{
    // Read component name and URL
    auto name = node.attribute("name").value();
    auto url = node.attribute("url").value();

    // Add mapping between name and URL to map
    componentURLs.insert(std::make_pair(name, url));

    // If this component's ports have already been cached, stop
    if(componentEventPorts.find(url) != componentEventPorts.end()) {
        return;
    }

    auto &urlEventPorts = componentEventPorts[url];

    // If this is a SpikeSource add event receive port called 'spike'
    if(strcmp(url, "SpikeSource") == 0) {
        urlEventPorts.second.insert("spike");
        return;
    }

    // Get absolute URL
    auto absoluteURL = (basePath / url).str();

    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(absoluteURL.c_str());
    if(!result) {
        throw std::runtime_error("Could not open file:" + absoluteURL + ", error:" + result.description());
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        throw std::runtime_error("XML file:" + absoluteURL + " is not a SpineML component - it has no root SpineML node");
    }

    // Get component class
    auto componentClass = spineML.child("ComponentClass");
    if(!componentClass) {
        throw std::runtime_error("XML file:" + absoluteURL + " is not a SpineML component - it's ComponentClass node is missing ");
    }

    // Loop through send ports
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();
        if(nodeType == "EventSendPort") {
            urlEventPorts.first.insert(portName);
        }
    }
    // Loop through receive ports
    std::cout << "\t\tEvent receive ports:" << std::endl;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        if(nodeType == "EventReceivePort") {
            urlEventPorts.second.insert(portName);
        }
    }
}
//----------------------------------------------------------------------------
void addPropertiesAndSizes(const pugi::xml_node &node, LIBRARY_HANDLE modelLibrary, const std::string &geNNPopName, unsigned int popSize,
                           std::map<std::string, unsigned int> &sizes, ComponentProperties &properties)
{
    // Get SpineML name of component
    const char *spineMLName = node.attribute("name").value();

    // Add sizes to map
    if(!sizes.insert(std::make_pair(spineMLName, popSize)).second) {
        throw std::runtime_error("Component name '" + std::string(spineMLName) + "' not unique");
    }

    // Loop through properties in network
    for(auto param : node.children("Property")) {
        std::string paramName = param.attribute("name").value();
        // Get pointers to state vars in model library
        scalar **hostStateVar;
        scalar **deviceStateVar;
        std::tie(hostStateVar, deviceStateVar) = getStateVar<scalar*>(modelLibrary, paramName + geNNPopName);

        // If it's found
        // **NOTE** it not being found is not an error condition - it just suggests that
        if(hostStateVar != nullptr) {
            std::cout << "\t" << paramName << std::endl;
#ifdef CPU_ONLY
            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << std::endl;

            // Create model property object
            properties[spineMLName].insert(
                std::make_pair(paramName, ModelProperty::create(param, *hostStateVar, nullptr, popSize)));
#else
            if(deviceStateVar == nullptr) {
                throw std::runtime_error("Cannot find device-side state variable for property:" + paramName);
            }

            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << ", device pointer:" << *deviceStateVar << std::endl;

            // Create model property object
            properties[spineMLName].insert(
                std::make_pair(paramName, ModelProperty::create(param, *hostStateVar, *deviceStateVar, popSize)));
#endif
        }
    }
}
//----------------------------------------------------------------------------
std::unique_ptr<Input::Base> createInput(const pugi::xml_node &node, LIBRARY_HANDLE modelLibrary, double dt,
                                         const std::map<std::string, unsigned int> &componentSizes,
                                         const ComponentProperties &componentProperties,
                                         const std::map<std::string, std::string> &componentURLs,
                                         const ComponentEventPorts &componentEventPorts)
{

    // Get name of target
    std::string target = node.attribute("target").value();
    std::cout << "Input targetting '" << target << "'" << std::endl;

    // Find size of target population
    auto targetSize = componentSizes.find(target);
    if(targetSize == componentSizes.end()) {
        throw std::runtime_error("Cannot find component '" + target + "'");
    }

    // Create suitable input value
    std::unique_ptr<InputValue::Base> inputValue = InputValue::create(dt, targetSize->second, node);

    // If target is an event receive port
    std::string port = node.attribute("port").value();
    if(isEventReceivePort(target, port, componentURLs, componentEventPorts)) {
        // Get host and device (if applicable) pointers to spike counts, spikes and queue
        unsigned int *hostSpikeCount;
        unsigned int *deviceSpikeCount;
        unsigned int *hostSpikes;
        unsigned int *deviceSpikes;
        unsigned int *spikeQueuePtr;
        std::tie(hostSpikeCount, deviceSpikeCount, hostSpikes, deviceSpikes, spikeQueuePtr) = getNeuronPopSpikeVars(
            modelLibrary, SpineMLUtils::getSafeName(target));

        // If this input has a rate distribution
        auto rateDistribution = node.attribute("rate_based_input");
        if(rateDistribution) {
            if(strcmp(rateDistribution.value(), "regular") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::RegularSpikeRate(dt, node, std::move(inputValue),
                                                spikeQueuePtr,
                                                hostSpikeCount, deviceSpikeCount,
                                                hostSpikes, deviceSpikes));
            }
            else if(strcmp(rateDistribution.value(), "poisson") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::PoissonSpikeRate(dt, node, std::move(inputValue),
                                                spikeQueuePtr,
                                                hostSpikeCount, deviceSpikeCount,
                                                hostSpikes, deviceSpikes));
            }
            else {
                throw std::runtime_error("Unsupport spike rate distribution '" + std::string(rateDistribution.value()) + "'");
            }
        }
        // Otherwise, create an exact spike-time input
        else {
            return std::unique_ptr<Input::Base>(
                new Input::SpikeTime(dt, node, std::move(inputValue),
                                     spikeQueuePtr,
                                     hostSpikeCount, deviceSpikeCount,
                                     hostSpikes, deviceSpikes));
        }
    }
    // Otherwise we assume it's an analogue receive port
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = componentProperties.find(target);
        if(targetProperties != componentProperties.end()) {
            // If there is a model property object for this port return an analogue input to stimulate it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                return std::unique_ptr<Input::Base>(
                    new Input::Analogue(dt, node, std::move(inputValue),
                                        portProperty->second.get()));
            }
        }
    }

    throw std::runtime_error("No supported input found");

}
//----------------------------------------------------------------------------
std::unique_ptr<LogOutput::Base> createLogOutput(const pugi::xml_node &node, LIBRARY_HANDLE modelLibrary, double dt,
                                                 unsigned int numTimeSteps, const filesystem::path &basePath,
                                                 const std::map<std::string, unsigned int> &componentSizes,
                                                 const ComponentProperties &componentProperties,
                                                 const std::map<std::string, std::string> &componentURLs,
                                                 const ComponentEventPorts &componentEventPorts)
{
    // Get name of target
    std::string target = node.attribute("target").value();

    // Find size of target component
    auto targetSize = componentSizes.find(target);
    if(targetSize == componentSizes.end()) {
        throw std::runtime_error("Cannot find component '" + target + "'");
    }

    // If target is an event send port
    std::string port = node.attribute("port").value();
    if(isEventSendPort(target, port, componentURLs, componentEventPorts)) {
        // Get host and device (if applicable) pointers to spike counts, spikes and queue
        unsigned int *hostSpikeCount;
        unsigned int *deviceSpikeCount;
        unsigned int *hostSpikes;
        unsigned int *deviceSpikes;
        unsigned int *spikeQueuePtr;
        std::tie(hostSpikeCount, deviceSpikeCount, hostSpikes, deviceSpikes, spikeQueuePtr) = getNeuronPopSpikeVars(
            modelLibrary, SpineMLUtils::getSafeName(target));

        // Create event logger
        return std::unique_ptr<LogOutput::Base>(new LogOutput::Event(node, dt, numTimeSteps, port, targetSize->second,
                                                                     basePath, spikeQueuePtr,
                                                                     hostSpikeCount, deviceSpikeCount,
                                                                     hostSpikes, deviceSpikes));
    }
    // Otherwise we assume it's an analogue send port
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = componentProperties.find(target);
        if(targetProperties != componentProperties.end()) {
            // If there is a model property object for this port return an analogue log output to read it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                return std::unique_ptr<LogOutput::Base>(new LogOutput::Analogue(node, dt, numTimeSteps, port, targetSize->second,
                                                                                basePath, portProperty->second.get()));
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
    if(argc != 2) {
        std::cerr << "Expected experiment XML file passed as arguments" << std::endl;
        return EXIT_FAILURE;
    }

    // **YUCK** hard coded 0.1ms time step as SpineML specifies this in experiment but GeNN in model
    const double dt = 0.1;

    LIBRARY_HANDLE modelLibrary = nullptr;
    try
    {
        std::mt19937 gen;

        // Use filesystem library to get parent path of the network XML file
        auto experimentPath = filesystem::path(argv[1]);
        auto basePath = experimentPath.parent_path();

        // Load experiment document
        pugi::xml_document experimentDoc;
        auto experimentResult = experimentDoc.load_file(argv[1]);
        if(!experimentResult) {
            throw std::runtime_error("Unable to load experiment XML file:" + std::string(argv[1]) + ", error:" + experimentResult.description());
        }

        // Get SpineML root
        auto experimentSpineML = experimentDoc.child("SpineML");
        if(!experimentSpineML) {
            throw std::runtime_error("XML file:" + std::string(argv[1]) + " is not a SpineML experiment - it has no root SpineML node");
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

        // Attempt to load model library
#ifdef _WIN32
        auto libraryPath = basePath / (networkName + "_CODE") / "runner.dll";
        std::cout << "Experiment using model library:" << libraryPath  << std::endl;
        modelLibrary = LoadLibrary(libraryPath.str().c_str());
#else
        auto libraryPath = basePath / (networkName + "_CODE") / "librunner.so";
        std::cout << "Experiment using model library:" << libraryPath  << std::endl;
        modelLibrary = dlopen(libraryPath.str().c_str(), RTLD_NOW);
#endif
        
        // If it fails throw
        if(modelLibrary == nullptr)
        {
#ifdef _WIN32
            throw std::runtime_error("Unable to load library - error:" + std::to_string(GetLastError()));
#else
            throw std::runtime_error("Unable to load library - error:" + std::string(dlerror()));
#endif
        }

        // Load statically-named symbols from library
        LOAD_SYMBOL(modelLibrary, VoidFunction, initialize);
        LOAD_SYMBOL(modelLibrary, VoidFunction, allocateMem);
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeCPU);
#ifndef CPU_ONLY
        LOAD_SYMBOL(modelLibrary, VoidFunction, stepTimeGPU);
#endif // CPU_ONLY

        // Search for network initialization function
        VoidFunction initializeNetwork = (VoidFunction)getLibrarySymbol(modelLibrary, ("init" + networkName).c_str());
        if(initializeNetwork == nullptr) {
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

        std::map<std::string, unsigned int> componentSizes;
        ComponentProperties componentProperties;
        std::map<std::string, std::string> componentURLs;
        ComponentEventPorts componentEventPorts;

        // Loop through populations once to initialize neuron population properties
        for(auto population : networkSpineML.children("LL:Population")) {
            auto neuron = population.child("LL:Neuron");
            if(!neuron) {
                throw std::runtime_error("'Population' node has no 'Neuron' node");
            }

            // Read basic population properties
            const char *popName = neuron.attribute("name").value();
            const unsigned int popSize = neuron.attribute("size").as_int();
            std::cout << "Population '" << popName << "' consisting of ";
            std::cout << popSize << " neurons" << std::endl;

            // Add neuron population properties to dictionary
            auto geNNPopName = SpineMLUtils::getSafeName(popName);
            addPropertiesAndSizes(neuron, modelLibrary, geNNPopName, popSize,
                                  componentSizes, componentProperties);
            addEventPorts(basePath, neuron, componentURLs, componentEventPorts);
        }

        // Loop through populations AGAIN to build synapse population properties
        for(auto population : networkSpineML.children("LL:Population")) {
            auto neuron = population.child("LL:Neuron");

            // Read source population name from neuron node
            const char *popName = neuron.attribute("name").value();
            const unsigned int popSize = getComponentSize(popName, componentSizes);

            // Loop through low-level inputs
            for(auto input : neuron.children("LL:Input")) {
                auto srcPopName = SpineMLUtils::getSafeName(input.attribute("src").value());
                const unsigned int srcPopSize = getComponentSize(srcPopName, componentSizes);

                std::string srcPort = input.attribute("src_port").value();
                std::string dstPort = input.attribute("dst_port").value();

                std::cout << "Low-level input from population:" << srcPopName << "(" << srcPort << ")->" << popName << "(" << dstPort << ")" << std::endl;

                std::string geNNSynPopName = std::string(srcPopName) + "_" + srcPort + "_" + popName + "_"  + dstPort;

                // Find allocate function and sparse projection
                Connectors::AllocateFn allocateFn = (Connectors::AllocateFn)getLibrarySymbol(modelLibrary, ("allocate" + geNNSynPopName).c_str());
                SparseProjection *sparseProjection = (SparseProjection*)getLibrarySymbol(modelLibrary, ("C" + geNNSynPopName).c_str());

                // Create connector
                Connectors::create(input, srcPopSize, popSize, sparseProjection, allocateFn, basePath);
            }

            // Loop through outgoing projections
            for(auto projection : population.children("LL:Projection")) {
                // Read destination population name from projection
                auto trgPopName = projection.attribute("dst_population").value();
                const unsigned int trgPopSize = getComponentSize(trgPopName, componentSizes);

                std::cout << "Projection from population:" << popName << "->" << trgPopName << std::endl;

                // Get main synapse node
                auto synapse = projection.child("LL:Synapse");
                if(!synapse) {
                    throw std::runtime_error("'Projection' node has no 'Synapse' node");
                }
                
                // Get weight update
                auto weightUpdate = synapse.child("LL:WeightUpdate");
                if(!weightUpdate) {
                    throw std::runtime_error("'Synapse' node has no 'WeightUpdate' node");
                }

                // Get post synapse
                auto postSynapse = synapse.child("LL:PostSynapse");
                if(!postSynapse) {
                    throw std::runtime_error("'Synapse' node has no 'PostSynapse' node");
                }
                
                // Build synapse population name from name of weight update
                // **NOTE** this is an arbitrary choice but these are guaranteed unique
                std::string geNNSynPopName = SpineMLUtils::getSafeName(weightUpdate.attribute("name").value());

                // Find allocate function and sparse projection
                Connectors::AllocateFn allocateFn = (Connectors::AllocateFn)getLibrarySymbol(modelLibrary, ("allocate" + geNNSynPopName).c_str());
                SparseProjection *sparseProjection = (SparseProjection*)getLibrarySymbol(modelLibrary, ("C" + geNNSynPopName).c_str());

                // Create connector
                const unsigned int numSynapses = Connectors::create(synapse, popSize, trgPopSize,
                                                                    sparseProjection, allocateFn, basePath);

                // Add postsynapse properties to dictionary
                addPropertiesAndSizes(postSynapse, modelLibrary, geNNSynPopName, trgPopSize,
                                      componentSizes, componentProperties);
                addEventPorts(basePath, postSynapse, componentURLs, componentEventPorts);

                // Add weight update properties to dictionary
                addPropertiesAndSizes(weightUpdate, modelLibrary, geNNSynPopName, numSynapses,
                                      componentSizes, componentProperties);
                addEventPorts(basePath, weightUpdate, componentURLs, componentEventPorts);
            }
        }


        auto simulation = experiment.child("Simulation");
        if(!simulation) {
            throw std::runtime_error("No 'Simulation' node found in experiment");
        }

        // Read duration from simulation and convert to timesteps
        const double durationMs = simulation.attribute("duration").as_double() * 1000.0;
        const unsigned int numTimeSteps = (unsigned int)std::ceil(durationMs / dt);

        // Loop through output loggers specified by experiment and create handler
        std::vector<std::unique_ptr<LogOutput::Base>> loggers;
        for(auto logOutput : experiment.children("LogOutput")) {
            loggers.push_back(createLogOutput(logOutput, modelLibrary, dt, numTimeSteps, basePath,
                                              componentSizes, componentProperties,
                                              componentURLs, componentEventPorts));
        }

        // Loop through inputs specified by experiment and create handlers
        std::vector<std::unique_ptr<Input::Base>> inputs;
        for(auto input : experiment.select_nodes(SpineMLUtils::xPathNodeHasSuffix("Input").c_str())) {
            inputs.push_back(createInput(input.node(), modelLibrary, dt,
                                         componentSizes, componentProperties,
                                         componentURLs, componentEventPorts));
        }

        // Call library function to perform final initialize
        {
            Timer t("Init network:");
            initializeNetwork();
        }


        std::cout << "Simulating for " << numTimeSteps << " " << dt << "ms timesteps" << std::endl;

        // Loop through time
        for(unsigned int i = 0; i < numTimeSteps; i++)
        {
            // Apply inputs
            for(auto &input : inputs)
            {
                input->apply(dt, i);
            }

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
#ifdef _WIN32
            FreeLibrary(modelLibrary);
#else
            dlclose(modelLibrary);
#endif
        }

        // Re-raise
        throw;
    }

    return 0;
}