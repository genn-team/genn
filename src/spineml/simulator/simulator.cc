#include "simulator.h"

// Standard C++ includes
#include <functional>
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

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// SpineMLCommon includes
#include "spineMLUtils.h"

// SpineML simulator includes
#include "connectors.h"
#include "input.h"
#include "inputValue.h"
#include "logOutput.h"
#include "modelProperty.h"
#include "stateVar.h"
#include "timer.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// SpineMLSimulator::Simulator
//----------------------------------------------------------------------------
namespace SpineMLSimulator
{
Simulator::Simulator()
:   m_ModelLibrary(nullptr), m_StepTime(nullptr), m_SimulationTime(nullptr), m_SimulationTimestep(nullptr),
    m_DT(0.0), m_DurationMs(0.0), m_InputMs(0.0), m_SimulateMs(0.0), m_LogMs(0.0)
{}
//----------------------------------------------------------------------------
Simulator::Simulator(const std::string &experimentXML, const std::string &overrideOutputPath)
: Simulator()
{
    load(experimentXML, overrideOutputPath);
}
//----------------------------------------------------------------------------
Simulator::~Simulator()
{
     // Close model library if loaded successfully
    if(m_ModelLibrary) {
#ifdef _WIN32
        FreeLibrary(m_ModelLibrary);
#else
        dlclose(m_ModelLibrary);
#endif
    }
}
//----------------------------------------------------------------------------
void Simulator::load(const std::string &experimentXML, const std::string &overrideOutputPath)
{
    // Use filesystem library to get parent path of the network XML file
    const auto experimentPath = filesystem::path(experimentXML).make_absolute();
    const auto basePath = experimentPath.parent_path();

    // If 2nd argument is specified use as output path otherwise use SpineCreator-compliant location
    const filesystem::path outputPath = overrideOutputPath.empty() ? basePath.parent_path() : filesystem::path(overrideOutputPath).make_absolute();

    LOGI << "Output path:" << outputPath.str();

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
    LOGI << "Experiment using model:" << networkPath;

    // Get the filename of the network and remove extension
    // to get something usable as a network name
    std::string networkName = networkPath.filename();
    networkName = networkName.substr(0, networkName.find_last_of("."));

    // Attempt to load model library
#ifdef _WIN32
    auto libraryPath = outputPath / "run" / "runner_Release.dll";
    LOGI << "Experiment using model library:" << libraryPath;
    m_ModelLibrary = LoadLibrary(libraryPath.str().c_str());
#else
    auto libraryPath = outputPath / "run" / (networkName + "_CODE") / "librunner.so";
    LOGI << "Experiment using model library:" << libraryPath;
    m_ModelLibrary = dlopen(libraryPath.str().c_str(), RTLD_NOW);
#endif

    // If it fails throw
    if(m_ModelLibrary == nullptr) {
#ifdef _WIN32
        throw std::runtime_error("Unable to load library - error:" + std::to_string(GetLastError()));
#else
        throw std::runtime_error("Unable to load library - error:" + std::string(dlerror()));
#endif
    }

    // Load statically-named symbols from library
    VoidFunction initialize = (VoidFunction)getLibrarySymbol("initialize");
    VoidFunction initializeSparse = (VoidFunction)getLibrarySymbol("initializeSparse");
    VoidFunction allocateMem = (VoidFunction)getLibrarySymbol("allocateMem");
    m_StepTime = (VoidFunction)getLibrarySymbol("stepTime");

    // Search for internal time counter
    // **NOTE** this is only used for checking timesteps are configured correctly
    m_SimulationTime = (float*)getLibrarySymbol("t");
    m_SimulationTimestep = (unsigned long long*)getLibrarySymbol("iT");

    // Call library function to allocate memory
    {
        Timer t("Allocation:");
        allocateMem();
    }

    // Call library function to initialize
    {
        Timer t("Init:");
        initialize();
    }

    // Load network document
    pugi::xml_document networkDoc;
    auto networkResult = networkDoc.load_file(networkPath.str().c_str());
    if(!networkResult) {
        throw std::runtime_error("Unable to load network XML file:" + networkPath.str() + ", error:" + networkResult.description());
    }

    // Get SpineML root
    auto networkSpineML = networkDoc.child("LL:SpineML");
    if(!networkSpineML) {
        throw std::runtime_error("XML file:" + networkPath.str() + " is not a low-level SpineML network - it has no root SpineML node");
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
    m_DT = eulerIntegration.attribute("dt").as_double(0.1);
    LOGI << "DT = " << m_DT << "ms";

    // Read duration from simulation and convert to timesteps
    m_DurationMs = simulation.attribute("duration").as_double() * 1000.0;


    std::map<std::string, unsigned int> componentSizes;
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
        const unsigned int popSize = neuron.attribute("size").as_uint();
        LOGI << "Population '" << popName << "' consisting of " << popSize << " neurons";

        // Add neuron population properties to dictionary
        auto geNNPopName = SpineMLUtils::getSafeName(popName);
        addPropertiesAndSizes(basePath, neuron, model, geNNPopName, popSize,
                              componentSizes);
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

            LOGI << "Low-level input from population:" << srcPopName << "(" << srcPort << ")->" << popName << "(" << dstPort << ")";

            std::string geNNSynPopName = SpineMLUtils::getSafeName(srcPopName) + "_" + srcPort + "_" + SpineMLUtils::getSafeName(popName) + "_"  + dstPort;

            // Find row lengths, indices and max row length associated with sparse connection
            unsigned int **rowLength = (unsigned int**)getLibrarySymbol(("rowLength" + geNNSynPopName).c_str(), true);
            unsigned int **ind = (unsigned int**)getLibrarySymbol(("ind" + geNNSynPopName).c_str(), true);
            uint8_t **delay = (uint8_t**)getLibrarySymbol(("_delay" + geNNSynPopName).c_str(), true);
            const unsigned int *maxRowLength = (const unsigned int*)getLibrarySymbol(("maxRowLength" + geNNSynPopName).c_str(), true);

            // Create connector
            std::vector<unsigned int> remapIndices;
            Connectors::create(input, getDT(), srcPopSize, popSize,
                                rowLength, ind, delay, maxRowLength,
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
                LOGI << "Projection from population:" << popName << "->" << trgPopName;

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
                unsigned int **rowLength = (unsigned int**)getLibrarySymbol(("rowLength" + geNNSynPopName).c_str(), true);
                unsigned int **ind = (unsigned int**)getLibrarySymbol(("ind" + geNNSynPopName).c_str(), true);
                uint8_t **delay = (uint8_t**)getLibrarySymbol(("_delay" + geNNSynPopName).c_str(), true);
                const unsigned int *maxRowLength = (const unsigned int*)getLibrarySymbol(("maxRowLength" + geNNSynPopName).c_str(), true);

                // Create connector
                std::vector<unsigned int> remapIndices;
                const unsigned int synapseVarSize = Connectors::create(synapse, getDT(), popSize, trgPopSize,
                                                                       rowLength, ind, delay, maxRowLength,
                                                                       basePath, remapIndices);

                // Add postsynapse properties to dictionary
                addPropertiesAndSizes(basePath, postSynapse, model, geNNSynPopName, trgPopSize, componentSizes);
                addEventPorts(basePath, postSynapse, componentURLs, componentEventPorts);

                // Add weight update properties to dictionary
                addPropertiesAndSizes(basePath, weightUpdate, model, geNNSynPopName, synapseVarSize, componentSizes,
                                      remapIndices.empty() ? nullptr : &remapIndices);
                addEventPorts(basePath, weightUpdate, componentURLs, componentEventPorts);
            }
        }
    }

    // Call library function to perform final initialize
    {
        Timer t("Initialize sparse:");
        initializeSparse();
    }

    // Create directory for logs (if required)
    const auto logPath = outputPath / "log";
    filesystem::create_directory(logPath);

    // Loop through output loggers specified by experiment and create handler
    for(auto logOutput : experiment.children("LogOutput")) {
        m_Loggers.push_back(createLogOutput(logOutput, logPath, componentSizes, componentURLs, componentEventPorts));
    }

    // Loop through inputs specified by experiment and create handlers
    for(auto input : experiment.select_nodes(SpineMLUtils::xPathNodeHasSuffix("Input").c_str())) {
        m_Inputs.push_back(createInput(input.node(), componentSizes, componentURLs, componentEventPorts));
    }
}
//----------------------------------------------------------------------------
void Simulator::stepTime()
{
    // Get GeNN timestep at start of step
    const unsigned long long i = *m_SimulationTimestep;

    // Apply inputs
    {
        TimerAccumulate t(m_InputMs);

        for(auto &input : m_Inputs) {
            input->apply(getDT(), i);
        }
    }

    // Advance time
    {
        TimerAccumulate t(m_SimulateMs);
        m_StepTime();
    }

    // If this is the first timestep
    if(i == 0) {
        // Calculate difference between the time elapsed according to GeNN's internal counter and our desired tiemstep
        const scalar timestepDifference = fabs(*m_SimulationTime - (scalar)getDT());

        // If they differ, give an error
        if(timestepDifference > std::numeric_limits<scalar>::epsilon()) {
            throw std::runtime_error("Timestep mismatch - model was built with " + std::to_string(*m_SimulationTime) + "ms timestep but we are simulating with " + std::to_string(getDT()) + "ms timestep");
        }
    }

    // Perform any recording required this timestep
    {
        TimerAccumulate t(m_LogMs);

        for(auto &logger : m_Loggers) {
            logger->record(getDT(), i);
        }
    }
}
//----------------------------------------------------------------------------
const LogOutput::AnalogueExternal *Simulator::getExternalLogger(const std::string &name) const
{
    auto logger = m_ExternalLoggers.find(name);
    if(logger != m_ExternalLoggers.cend()) {
        return logger->second;
    }
    else {
        return nullptr;
    }
}
//----------------------------------------------------------------------------
InputValue::External *Simulator::getExternalInput(const std::string &name) const
{
    auto logger = m_ExternalInputs.find(name);
    if(logger != m_ExternalInputs.cend()) {
        return logger->second;
    }
    else {
        return nullptr;
    }
}
//----------------------------------------------------------------------------
double Simulator::getNeuronUpdateTime() const
{
    return *(double*)getLibrarySymbol("neuronUpdateTime");
}
//----------------------------------------------------------------------------
double Simulator::getInitTime() const
{
    return *(double*)getLibrarySymbol("initTime");
}
//----------------------------------------------------------------------------
double Simulator::getPresynapticUpdateTime() const
{
    return *(double*)getLibrarySymbol("presynapticUpdateTime");
}
//----------------------------------------------------------------------------
double Simulator::getPostsynapticUpdateTime() const
{
    return *(double*)getLibrarySymbol("postsynapticUpdateTime");
}
//----------------------------------------------------------------------------
double Simulator::getSynapseDynamicsTime() const
{
    return *(double*)getLibrarySymbol("synapseDynamicsTime");
}
//----------------------------------------------------------------------------
double Simulator::getInitSparseTime() const
{
    return *(double*)getLibrarySymbol("initSparseTime");
}
//----------------------------------------------------------------------------
void *Simulator::getLibrarySymbol(const char *name, bool allowMissing) const
{
#ifdef _WIN32
    void *symbol = GetProcAddress(m_ModelLibrary, name);
#else
    void *symbol = dlsym(m_ModelLibrary, name);
#endif

    // If this symbol isn't allowed to be missing but it is, raise exception
    if(!allowMissing && symbol == nullptr) {
        throw std::runtime_error("Cannot find symbol '" + std::string(name) + "'");
    }

    return symbol;
}
//----------------------------------------------------------------------------
Simulator::NeuronPopSpikeVars Simulator::getNeuronPopSpikeVars(const std::string &popName) const
{
    // Get pointers to spike counts in model library
    unsigned int **hostSpikeCount = (unsigned int **)getLibrarySymbol(("glbSpkCnt" + popName).c_str());
    unsigned int **hostSpikes = (unsigned int **)getLibrarySymbol(("glbSpk" + popName).c_str());

    // Get push and pull functions
    auto pushFunc = (VoidFunction)getLibrarySymbol(("push" + popName + "CurrentSpikesToDevice").c_str());
    auto pullFunc = (VoidFunction)getLibrarySymbol(("pull" + popName + "CurrentSpikesFromDevice").c_str());

    // Get spike queue
    // **NOTE** neuron populations without any outgoing synapses with delay won't have one so it can be missing
    unsigned int *spikeQueuePtr = (unsigned int*)getLibrarySymbol(("spkQuePtr" + popName).c_str(), true);

    // Return pointers in tutple
    return std::make_tuple(*hostSpikeCount, *hostSpikes, spikeQueuePtr, pushFunc, pullFunc);
}
//----------------------------------------------------------------------------
void Simulator::addPropertiesAndSizes(const filesystem::path &basePath, const pugi::xml_node &node,
                                      const pugi::xml_node &modelNode, const std::string &geNNPopName,
                                      unsigned int popSize, std::map<std::string, unsigned int> &sizes,
                                      const std::vector<unsigned int> *remapIndices)
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
    auto &componentProperties = m_ComponentProperties[spineMLName];

    // Bind ths to getLibrarySymbol to get free function we can pass to StateVar constructor
    // **NOTE** using functions rather than just passing around Simulator objects is more to break circular dependencies than anything else
    auto getLibrarySymbolFunc = std::bind(&Simulator::getLibrarySymbol, this,
                                          std::placeholders::_1, std::placeholders::_2);

    // Loop through properties in network
    for(auto param : node.children("Property")) {
        std::string paramName = param.attribute("name").value();

        // Find state var in model library
        StateVar<scalar> stateVar(paramName + geNNPopName, getLibrarySymbolFunc);

        // If it's accessible
        // **NOTE** it not being found is not an error condition - it just suggests that it was optimised out by generator
        if(stateVar.isAccessible()) {
            // If any properties are overriden
            pugi::xml_node overridenParam;
            if(overridenProperties) {
                // Use XPath to determine whether THIS property is overriden
                pugi::xpath_variable_set propertyNameVars;
                propertyNameVars.set("name", paramName.c_str());
                if((overridenParam = overridenProperties.node().select_node("UL:Property[@name=$name]",
                                                                            &propertyNameVars).node()))
                {
                    LOGD << "\t\tOverriden in experiment";
                }
            }

            // Skip initialisation for properties whose type means that will have already been
            // initialised by GeNN UNLESS parameter is overriden in experiment
            const bool skipGeNNInitialised = !overridenParam;

            // If property is overriden then the value types will be in 'UL:' namespace otherwise root
            const std::string valueNamespace = overridenParam ? "UL:" : "";

            // Create model property object
            componentProperties.insert(
                std::make_pair(paramName, ModelProperty::create(overridenParam ? overridenParam : param, stateVar, popSize,
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
            StateVar<scalar> stateVar(paramName + geNNPopName, getLibrarySymbolFunc);

            // If it's found
            // **NOTE** it not being found is not an error condition - it just suggests that it was optimised out by generator
            if(stateVar.isAccessible()) {
                // Create model property object
                componentProperties.insert(
                    std::make_pair(paramName, std::unique_ptr<ModelProperty::Base>(new ModelProperty::Base(stateVar, popSize))));
            }
        }
    }
}
//----------------------------------------------------------------------------
std::unique_ptr<Input::Base> Simulator::createInput(const pugi::xml_node &node,
                                                    const std::map<std::string, unsigned int> &componentSizes,
                                                    const std::map<std::string, std::string> &componentURLs,
                                                    const ComponentEventPorts &componentEventPorts)
{

    // Get name of target
    std::string target = node.attribute("target").value();
    LOGI << "Input targetting '" << target << "'";

    // Find size of target population
    auto targetSize = componentSizes.find(target);
    if(targetSize == componentSizes.end()) {
        throw std::runtime_error("Cannot find component '" + target + "'");
    }

    // Create suitable input value
    std::unique_ptr<InputValue::Base> inputValue = InputValue::create(m_DT, targetSize->second, node,
                                                                      m_ExternalInputs);

    // If target is an event receive port
    std::string port = node.attribute("port").value();
    if(isEventReceivePort(target, port, componentURLs, componentEventPorts)) {
        // Get host and device (if applicable) pointers to spike counts, spikes and queue
        unsigned int *hostSpikeCount;
        unsigned int *hostSpikes;
        unsigned int *spikeQueuePtr;
        VoidFunction pushFunc;
        VoidFunction pullFunc;
        std::tie(hostSpikeCount, hostSpikes, spikeQueuePtr, pushFunc, pullFunc) = getNeuronPopSpikeVars(SpineMLUtils::getSafeName(target));

        // If this input has a rate distribution
        auto rateDistribution = node.attribute("rate_based_input");
        if(rateDistribution) {
            if(strcmp(rateDistribution.value(), "regular") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::RegularSpikeRate(m_DT, node, std::move(inputValue),
                                                targetSize->second, spikeQueuePtr, hostSpikeCount, hostSpikes,
                                                pushFunc));
            }
            else if(strcmp(rateDistribution.value(), "poisson") == 0) {
                return std::unique_ptr<Input::Base>(
                    new Input::PoissonSpikeRate(m_DT, node, std::move(inputValue),
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
                new Input::SpikeTime(m_DT, node, std::move(inputValue),
                                     targetSize->second, spikeQueuePtr, hostSpikeCount, hostSpikes,
                                     pushFunc));
        }
    }
    // Otherwise we assume it's an analogue receive port
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = m_ComponentProperties.find(target);
        if(targetProperties != m_ComponentProperties.end()) {
            // If there is a model property object for this port return an analogue input to stimulate it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                return std::unique_ptr<Input::Base>(
                    new Input::Analogue(m_DT, node, std::move(inputValue),
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
std::unique_ptr<LogOutput::Base> Simulator::createLogOutput(const pugi::xml_node &node,
                                                            const filesystem::path &logPath,
                                                            const std::map<std::string, unsigned int> &componentSizes,
                                                            const std::map<std::string, std::string> &componentURLs,
                                                            const ComponentEventPorts &componentEventPorts)
{
    // Get name of target
    const std::string target = node.attribute("target").value();

    // Find size of target component
    auto targetSize = componentSizes.find(target);
    if(targetSize == componentSizes.end()) {
        throw std::runtime_error("Cannot find component '" + target + "'");
    }

    // If this writing to file or network
    const std::string hostName = node.attribute("host").value();
    const bool shouldLogToFile = hostName.empty();

    // If target is an event send port
    const unsigned long long numTimeSteps = calcNumTimesteps();
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
        std::tie(hostSpikeCount, hostSpikes, spikeQueuePtr, pushFunc, pullFunc) = getNeuronPopSpikeVars(SpineMLUtils::getSafeName(target));

        // Create event logger
        return std::unique_ptr<LogOutput::Base>(new LogOutput::Event(node, getDT(), numTimeSteps, port, targetSize->second,
                                                                     logPath, spikeQueuePtr,
                                                                     hostSpikeCount, hostSpikes, pullFunc));
    }
    // Otherwise we assume it's an analogue send port
    else {
        // If there is a dictionary of properties for target population
        auto targetProperties = m_ComponentProperties.find(target);
        if(targetProperties != m_ComponentProperties.end()) {
            // If there is a model property object for this port return an analogue log output to read it
            auto portProperty = targetProperties->second.find(port);
            if(portProperty != targetProperties->second.end()) {
                if(shouldLogToFile) {
                    return std::unique_ptr<LogOutput::Base>(new LogOutput::AnalogueFile(node, getDT(), numTimeSteps, port, targetSize->second,
                                                                                        logPath, portProperty->second.get()));
                }
                else if(hostName == "0.0.0.0") {
                    // Create logger
                    std::unique_ptr<LogOutput::AnalogueExternal> log(
                        new LogOutput::AnalogueExternal(node, getDT(), port, targetSize->second,
                                                        logPath, portProperty->second.get()));

                    // Add to map of external loggers
                    const std::string name = node.attribute("name").value();
                    if(!m_ExternalLoggers.emplace(name, log.get()).second) {
                        LOGW << "External logger with duplicate name '" << name << "' encountered";
                    }

                    // Return pointer
                    return log;
                }
                else {
                    return std::unique_ptr<LogOutput::Base>(new LogOutput::AnalogueNetwork(node, getDT(), port, targetSize->second,
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
//----------------------------------------------------------------------------
unsigned int Simulator::getComponentSize(const std::string &componentName,
                                         const std::map<std::string, unsigned int> &componentSizes)
{
    auto component = componentSizes.find(componentName);
    if(component == componentSizes.end()) {
        throw std::runtime_error("Cannot find component:" + componentName);
    }
    else {
        return component->second;
    }
}
//----------------------------------------------------------------------------
bool Simulator::isEventSendPort(const std::string &targetName, const std::string &portName,
                                const std::map<std::string, std::string> &componentURLs,
                                const ComponentEventPorts &componentEventPorts) const
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
bool Simulator::isEventReceivePort(const std::string &targetName, const std::string &portName,
                                   const std::map<std::string, std::string> &componentURLs,
                                   const ComponentEventPorts &componentEventPorts) const
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
void Simulator::addEventPorts(const filesystem::path &basePath, const pugi::xml_node &node,
                              std::map<std::string, std::string> &componentURLs,
                              ComponentEventPorts &componentEventPorts)
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
}   // namespace SpineMLSimulator

