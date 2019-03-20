// Standard C++ includes
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>

// Standard C includes
#include <cassert>
#include <cmath>
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
#include "path.h"

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
#else
    #define LIBRARY_HANDLE void*
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
void *getLibrarySymbol(LIBRARY_HANDLE modelLibrary, const char *name, bool allowMissing = false) {
#ifdef _WIN32
    void *symbol = GetProcAddress(modelLibrary, name);
#else
    void *symbol = dlsym(modelLibrary, name);
#endif

    // If this symbol isn't allowed to be missing but it is, raise exception
    if(!allowMissing && symbol == nullptr) {
        throw std::runtime_error("Cannot find symbol '" + std::string(name) + "'");
    }

    return symbol;
}
//----------------------------------------------------------------------------
template <typename T>
std::tuple<T*, ModelProperty::Base::PushFunc, ModelProperty::Base::PullFunc> getStateVar(LIBRARY_HANDLE modelLibrary, const std::string &stateVarName)
{
    // Get host statevar
    T *hostStateVar = (T*)getLibrarySymbol(modelLibrary, stateVarName.c_str(), true);

    // Get push and pull functions
    auto pushFunc = (ModelProperty::Base::PushFunc)getLibrarySymbol(modelLibrary, ("push" + stateVarName + "ToDevice").c_str(), true);
    auto pullFunc = (ModelProperty::Base::PullFunc)getLibrarySymbol(modelLibrary, ("pull" + stateVarName + "FromDevice").c_str(), true);

    // Return in tuple
    return std::make_tuple(hostStateVar, pushFunc, pullFunc);
}
//----------------------------------------------------------------------------
void closeLibrary(LIBRARY_HANDLE modelLibrary)
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
    // Find URL of target component
    auto targetURL = componentURLs.find(targetName);
    if(targetURL != componentURLs.end()) {
        // If then target is a spike source
        if(targetURL->second == "SpikeSource" && portName == "spike") {
            return true;
        }
        // Otherwise
        else {
            // If there is a map of event ports for components with this URL,
            // Return true if our portname is within that map
            auto urlEventPorts = componentEventPorts.find(targetURL->second);
            if(urlEventPorts != componentEventPorts.end()) {
                const auto &eventSendPorts = urlEventPorts->second.first;
                return (eventSendPorts.find(portName) != eventSendPorts.end());
            }
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
std::tuple<unsigned int*, unsigned int*, unsigned int*, VoidFunction, VoidFunction> getNeuronPopSpikeVars(LIBRARY_HANDLE modelLibrary, const std::string &popName)
{
    // Get pointers to spike counts in model library
    unsigned int **hostSpikeCount = (unsigned int **)getLibrarySymbol(modelLibrary, ("glbSpkCnt" + popName).c_str());
    unsigned int **hostSpikes = (unsigned int **)getLibrarySymbol(modelLibrary, ("glbSpk" + popName).c_str());

    // Get push and pull functions
    auto pushFunc = (VoidFunction)getLibrarySymbol(modelLibrary, ("push" + popName + "CurrentSpikesToDevice").c_str());
    auto pullFunc = (VoidFunction)getLibrarySymbol(modelLibrary, ("pull" + popName + "CurrentSpikesFromDevice").c_str());

    // Get spike queue
    // **NOTE** neuron populations without any outgoing synapses with delay won't have one so it can be missing
    unsigned int *spikeQueuePtr = (unsigned int*)getLibrarySymbol(modelLibrary, ("spkQuePtr" + popName).c_str(), true);

    // Return pointers in tutple
    return std::make_tuple(*hostSpikeCount, *hostSpikes, spikeQueuePtr, pushFunc, pullFunc);
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
    for(auto eventSendPort : componentClass.children("EventSendPort")) {
        urlEventPorts.first.insert(eventSendPort.attribute("name").value());
    }
    // Loop through receive ports
    for(auto eventReceivePort : componentClass.children("EventReceivePort")) {
        urlEventPorts.second.insert(eventReceivePort.attribute("name").value());
    }
}
//----------------------------------------------------------------------------
void addPropertiesAndSizes(const filesystem::path &basePath, const pugi::xml_node &node, const pugi::xml_node &modelNode, LIBRARY_HANDLE modelLibrary, const std::string &geNNPopName, unsigned int popSize,
                           std::map<std::string, unsigned int> &sizes, ComponentProperties &properties, const std::vector<unsigned int> *remapIndices = nullptr)
{
    // Get SpineML name of component
    const char *spineMLName = node.attribute("name").value();

    // Add sizes to map
    if(!sizes.insert(std::make_pair(spineMLName, popSize)).second) {
        throw std::runtime_error("Component name '" + std::string(spineMLName) + "' not unique");
    }

    // Use XPath query to determine whether there's a corresponding
    // 'Configuration' containing overriden properties
    pugi::xpath_variable_set configurationTargetVars;
    configurationTargetVars.set("target", spineMLName);
    auto overridenProperties = modelNode.select_node("Configuration[@target=$target]",
                                                     &configurationTargetVars);

    // Get map to hold properties associated with this component
    auto &componentProperties = properties[spineMLName];

    // Loop through properties in network
    for(auto param : node.children("Property")) {
        std::string paramName = param.attribute("name").value();

        // Get pointers to state vars in model library
        scalar **hostStateVar;
        ModelProperty::Base::PushFunc pushFunc;
        ModelProperty::Base::PullFunc pullFunc;
        std::tie(hostStateVar, pushFunc, pullFunc) = getStateVar<scalar*>(modelLibrary, paramName + geNNPopName);

        // If it's found
        // **NOTE** it not being found is not an error condition - it just suggests that it was optimised out by generator
        if(hostStateVar != nullptr) {
            std::cout << "\t" << paramName << std::endl;

            // If any properties are overriden
            pugi::xml_node overridenParam;
            if(overridenProperties) {
                // Use XPath to determine whether THIS property is overriden
                pugi::xpath_variable_set propertyNameVars;
                propertyNameVars.set("name", paramName.c_str());
                if((overridenParam = overridenProperties.node().select_node("UL:Property[@name=$name]",
                                                                            &propertyNameVars).node()))
                {
                    std::cout << "\t\tOverriden in experiment" << std::endl;
                }
            }

            // Skip initialisation for properties whose type means that will have already been
            // initialised by GeNN UNLESS parameter is overriden in experiment
            const bool skipGeNNInitialised = !overridenParam;

            // If property is overriden then the value types will be in 'UL:' namespace otherwise root
            const std::string valueNamespace = overridenParam ? "UL:" : "";

            if(pushFunc == nullptr || pullFunc == nullptr) {
                throw std::runtime_error("Cannot find push and pull functions for property:" + paramName);
            }

            std::cout << "\t\tState variable found host pointer:" << *hostStateVar << ", push function:" << pushFunc << ", pull function:" << pullFunc << std::endl;

            // Create model property object
            componentProperties.insert(
                std::make_pair(paramName, ModelProperty::create(overridenParam ? overridenParam : param,
                                                                *hostStateVar, pushFunc, pullFunc, popSize,
                                                                skipGeNNInitialised, basePath, valueNamespace, remapIndices)));
        }
    }

    auto url = node.attribute("url").value();

    // If this is a SpikeSource add event receive port called 'spike'
    if(strcmp(url, "SpikeSource") == 0) {
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

    // Loop through analogue ports which might also be implemented as GeNN state variables
    for(auto analoguePort : componentClass.select_nodes("*[name() = 'AnalogSendPort' or name() = 'AnalogReducePort' or name() = 'AnalogReceivePort']")) {
        const std::string paramName = analoguePort.node().attribute("name").value();
        const std::string portType = analoguePort.node().name();
        if(componentProperties.find(paramName) == componentProperties.end()) {
            // Get pointers to state vars in model library
            scalar **hostStateVar;
            ModelProperty::Base::PushFunc pushFunc;
            ModelProperty::Base::PullFunc pullFunc;
            std::tie(hostStateVar, pushFunc, pullFunc) = getStateVar<scalar*>(modelLibrary, paramName + geNNPopName);

            // If it's found
            // **NOTE** it not being found is not an error condition - it just suggests that it was optimised out by generator
            if(hostStateVar != nullptr) {
                std::cout << "\t" << paramName << std::endl;

                if(pushFunc == nullptr || pullFunc == nullptr) {
                    throw std::runtime_error("Cannot find push and pull functions for property:" + paramName);
                }

                std::cout << "\t\t" << portType << " found host pointer:" << *hostStateVar << ", push function:" << pushFunc << ", pull function:" << pullFunc << std::endl;

                // Create model property object
                componentProperties.insert(
                    std::make_pair(paramName, std::unique_ptr<ModelProperty::Base>(new ModelProperty::Base(*hostStateVar, pushFunc, pullFunc, popSize))));
            }
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
        unsigned int *hostSpikes;
        unsigned int *spikeQueuePtr;
        VoidFunction pushFunc;
        VoidFunction pullFunc;
        std::tie(hostSpikeCount, hostSpikes, spikeQueuePtr, pushFunc, pullFunc) = getNeuronPopSpikeVars(
            modelLibrary, SpineMLUtils::getSafeName(target));

        // If this input has a rate distribution
        auto rateDistribution = node.attribute("rate_based_input");
        if(rateDistribution) {
            if(strcmp(rateDistribution.value(), "regular") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::RegularSpikeRate(dt, node, std::move(inputValue),
                                                targetSize->second, spikeQueuePtr, hostSpikeCount, hostSpikes,
                                                pushFunc));
            }
            else if(strcmp(rateDistribution.value(), "poisson") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::PoissonSpikeRate(dt, node, std::move(inputValue),
                                                targetSize->second, spikeQueuePtr, hostSpikeCount, hostSpikes,
                                                pushFunc));
            }
            else {
                throw std::runtime_error("Unsupport spike rate distribution '" + std::string(rateDistribution.value()) + "'");
            }
        }
        // Otherwise, create an exact spike-time input
        else {
            return std::unique_ptr<Input::Base>(
                new Input::SpikeTime(dt, node, std::move(inputValue),
                                     targetSize->second, spikeQueuePtr, hostSpikeCount, hostSpikes,
                                     pushFunc));
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
            else {
                throw std::runtime_error("Port '" + port + "' not found on target '" + target + "'");
            }
        }
        else {
            throw std::runtime_error("No properties found for '" + target + "'");
        }
    }
}
//----------------------------------------------------------------------------
std::unique_ptr<LogOutput::Base> createLogOutput(const pugi::xml_node &node, LIBRARY_HANDLE modelLibrary, double dt,
                                                 unsigned int numTimeSteps, const filesystem::path &logPath,
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

     // If this writing to file or network
    const bool shouldLogToFile = (strcmp(node.attribute("host").value(), "") == 0);

    // If target is an event send port
    const std::string port = node.attribute("port").value();
    if(isEventSendPort(target, port, componentURLs, componentEventPorts)) {
        // **TODO** spike-based network logging not supported
        assert(shouldLogToFile);

        // Get host and device (if applicable) pointers to spike counts, spikes and queue
        unsigned int *hostSpikeCount;
        unsigned int *hostSpikes;
        unsigned int *spikeQueuePtr;
        VoidFunction pushFunc;
        VoidFunction pullFunc;
        std::tie(hostSpikeCount, hostSpikes, spikeQueuePtr, pushFunc, pullFunc) = getNeuronPopSpikeVars(
            modelLibrary, SpineMLUtils::getSafeName(target));

        // Create event logger
        return std::unique_ptr<LogOutput::Base>(new LogOutput::Event(node, dt, numTimeSteps, port, targetSize->second,
                                                                     logPath, spikeQueuePtr,
                                                                     hostSpikeCount, hostSpikes, pullFunc));
    }
    // Otherwise we assume it's an analogue send port
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = componentProperties.find(target);
        if(targetProperties != componentProperties.end()) {
            // If there is a model property object for this port return an analogue log output to read it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                if(shouldLogToFile) {
                    return std::unique_ptr<LogOutput::Base>(new LogOutput::AnalogueFile(node, dt, numTimeSteps, port, targetSize->second,
                                                                                        logPath, portProperty->second.get()));
                }
                else {
                    return std::unique_ptr<LogOutput::Base>(new LogOutput::AnalogueNetwork(node, dt, numTimeSteps, port, targetSize->second,
                                                                                           logPath, portProperty->second.get()));
                }
            }
            else {
                throw std::runtime_error("Port '" + port + "' not found on target '" + target + "'");
            }
        }
        else {
            throw std::runtime_error("No properties found for '" + target + "'");
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    LIBRARY_HANDLE modelLibrary = nullptr;
    try
    {
        if(argc < 2) {
            throw std::runtime_error("Expected experiment XML file passed as arguments");
        }

#ifdef _WIN32
        // Startup WinSock 2
        WSADATA wsaData;
        if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }

#endif  // _WIN32

        std::mt19937 gen;

        // Use filesystem library to get parent path of the network XML file
        const auto experimentPath = filesystem::path(argv[1]).make_absolute();
        const auto basePath = experimentPath.parent_path();

        // If 2nd argument is specified use as output path otherwise use SpineCreator-compliant location
        const auto outputPath = (argc > 2) ? filesystem::path(argv[2]).make_absolute() : basePath.parent_path();

        std::cout << "Output path:" << outputPath.str() << std::endl;

        // Load experiment document
        pugi::xml_document experimentDoc;
        auto experimentResult = experimentDoc.load_file(experimentPath.str().c_str());
        if(!experimentResult) {
            throw std::runtime_error("Unable to load experiment XML file:" + experimentPath.str() + ", error:" + experimentResult.description());
        }

        // Get SpineML root
        auto experimentSpineML = experimentDoc.child("SpineML");
        if(!experimentSpineML) {
            throw std::runtime_error("XML file:" + experimentPath.str() + " is not a SpineML experiment - it has no root SpineML node");
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
        auto libraryPath = outputPath / "run" / (networkName + "_CODE") / "runner.dll";
        std::cout << "Experiment using model library:" << libraryPath  << std::endl;
        modelLibrary = LoadLibrary(libraryPath.str().c_str());
#else
        auto libraryPath = outputPath / "run" / (networkName + "_CODE") / "librunner.so";
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
        VoidFunction initialize = (VoidFunction)getLibrarySymbol(modelLibrary, "initialize");
        VoidFunction initializeSparse = (VoidFunction)getLibrarySymbol(modelLibrary, "initializeSparse");
        VoidFunction allocateMem = (VoidFunction)getLibrarySymbol(modelLibrary, "allocateMem");
        VoidFunction stepTime = (VoidFunction)getLibrarySymbol(modelLibrary, "stepTime");

        // Search for internal time counter
        // **NOTE** this is only used for checking timesteps are configured correctly
        float *simulationTime = (float*)getLibrarySymbol(modelLibrary, "t");

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
            addPropertiesAndSizes(basePath, neuron, model, modelLibrary, geNNPopName, popSize,
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
                const char *srcPopName = input.attribute("src").value();
                const unsigned int srcPopSize = getComponentSize(srcPopName, componentSizes);

                std::string srcPort = input.attribute("src_port").value();
                std::string dstPort = input.attribute("dst_port").value();

                std::cout << "Low-level input from population:" << srcPopName << "(" << srcPort << ")->" << popName << "(" << dstPort << ")" << std::endl;

                std::string geNNSynPopName = SpineMLUtils::getSafeName(srcPopName) + "_" + srcPort + "_" + SpineMLUtils::getSafeName(popName) + "_"  + dstPort;

                // Find row lengths, indices and max row length associated with sparse connection
                unsigned int **rowLength = (unsigned int**)getLibrarySymbol(modelLibrary, ("rowLength" + geNNSynPopName).c_str(), true);
                unsigned int **ind = (unsigned int**)getLibrarySymbol(modelLibrary, ("rowLength" + geNNSynPopName).c_str(), true);
                unsigned int *maxRowLength = (unsigned int*)getLibrarySymbol(modelLibrary, ("maxRowLength" + geNNSynPopName).c_str(), true);

                // Create connector
                std::vector<unsigned int> remapIndices;
                Connectors::create(input, srcPopSize, popSize, 
                                   rowLength, ind, maxRowLength,
                                   basePath, remapIndices);
            }

            // Loop through outgoing projections
            for(auto projection : population.children("LL:Projection")) {
                // Read destination population name from projection
                auto trgPopName = projection.attribute("dst_population").value();
                const unsigned int trgPopSize = getComponentSize(trgPopName, componentSizes);

                // Loop through synapse children
                // **NOTE** multiple projections between the same two populations of neurons are implemented in this way
                for(auto synapse : projection.children("LL:Synapse")) {
                    std::cout << "Projection from population:" << popName << "->" << trgPopName << std::endl;

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

                    // Find row lengths, indices and max row length associated with sparse connection
                    unsigned int **rowLength = (unsigned int**)getLibrarySymbol(modelLibrary, ("rowLength" + geNNSynPopName).c_str(), true);
                    unsigned int **ind = (unsigned int**)getLibrarySymbol(modelLibrary, ("rowLength" + geNNSynPopName).c_str(), true);
                    unsigned int *maxRowLength = (unsigned int*)getLibrarySymbol(modelLibrary, ("maxRowLength" + geNNSynPopName).c_str(), true);

                    // Create connector
                    std::vector<unsigned int> remapIndices;
                    const unsigned int numSynapses = Connectors::create(synapse, popSize, trgPopSize,
                                                                        rowLength, ind, maxRowLength,
                                                                        basePath, remapIndices);

                    // Add postsynapse properties to dictionary
                    addPropertiesAndSizes(basePath, postSynapse, model, modelLibrary, geNNSynPopName, trgPopSize,
                                          componentSizes, componentProperties);
                    addEventPorts(basePath, postSynapse, componentURLs, componentEventPorts);

                    // Add weight update properties to dictionary
                    addPropertiesAndSizes(basePath, weightUpdate, model, modelLibrary, geNNSynPopName, numSynapses,
                                          componentSizes, componentProperties, remapIndices.empty() ? nullptr : &remapIndices);
                    addEventPorts(basePath, weightUpdate, componentURLs, componentEventPorts);
                }
            }
        }

        // Call library function to perform final initialize
        {
            Timer t("Initialize sparse:");
            initializeSparse();
        }

        auto simulation = experiment.child("Simulation");
        if(!simulation) {
            throw std::runtime_error("No 'Simulation' node found in experiment");
        }

        auto eulerIntegration = simulation.child("EulerIntegration");
        if(!eulerIntegration) {
            throw std::runtime_error("GeNN only currently supports Euler integration scheme");
        }

        // Read integration timestep
        const double dt = eulerIntegration.attribute("dt").as_double(0.1);
        std::cout << "DT = " << dt << "ms" << std::endl;
        
        // Read duration from simulation and convert to timesteps
        const double durationMs = simulation.attribute("duration").as_double() * 1000.0;
        const unsigned int numTimeSteps = (unsigned int)std::ceil(durationMs / dt);

        // Create directory for logs (if required)
        const auto logPath = outputPath / "log";
        filesystem::create_directory(logPath);

        // Loop through output loggers specified by experiment and create handler
        std::vector<std::unique_ptr<LogOutput::Base>> loggers;
        for(auto logOutput : experiment.children("LogOutput")) {
            loggers.push_back(createLogOutput(logOutput, modelLibrary, dt, numTimeSteps, logPath,
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

        std::cout << "Simulating for " << numTimeSteps << " " << dt << "ms timesteps" << std::endl;

        // Loop through time
        double inputMs = 0.0;
        double simulateMs = 0.0;
        double logMs = 0.0;
        for(unsigned int i = 0; i < numTimeSteps; i++)
        {
            // Apply inputs
            {
                TimerAccumulate t(inputMs);

                for(auto &input : inputs)
                {
                    input->apply(dt, i);
                }
            }

            // Advance time
            {
                TimerAccumulate t(simulateMs);
                stepTime();
            }

            // If this is the first timestep
            if(i == 0) {
                // Calculate difference between the time elapsed according to GeNN's internal counter and our desired tiemstep
                const scalar timestepDifference = fabs(*simulationTime - (scalar)dt);

                // If they differ, give an error
                if(timestepDifference > std::numeric_limits<scalar>::epsilon()) {
                    throw std::runtime_error("Timestep mismatch - model was built with " + std::to_string(*simulationTime) + "ms timestep but we are simulating with " + std::to_string(dt) + "ms timestep");
                }
            }

            // Perform any recording required this timestep
            {
                TimerAccumulate t(logMs);

                for(auto &logger : loggers)
                {
                    logger->record(dt, i);
                }
            }
        }

        std::cout << "Applying input: " << inputMs << "ms, simulating:" << simulateMs << "ms, logging:" << logMs << "ms" << std::endl;
    }
    catch(const std::exception &exception)
    {
        std::cerr << exception.what() << std::endl;
        closeLibrary(modelLibrary);
    }
    catch(...)
    {
        closeLibrary(modelLibrary);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
