// Standard C++ includes
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdlib>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// CLI11 includes
#include "CLI11.hpp"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/generateAll.h"
#include "code_generator/generateMakefile.h"
#include "code_generator/generateMSBuild.h"

// GeNN backend includes
#include "optimiser.h"

// SpineMLCommon includes
#include "connectors.h"
#include "spineMLUtils.h"

// SpineMLGenerator includes
#include "modelParams.h"
#include "neuronModel.h"
#include "passthroughPostsynapticModel.h"
#include "passthroughWeightUpdateModel.h"
#include "postsynapticModel.h"
#include "weightUpdateModel.h"

using namespace SpineMLCommon;
using namespace SpineMLGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
struct SynapseMatrixProps
{
    SynapseMatrixConnectivity connectivityType;
    unsigned int axonalDelay;
    unsigned int maxDendriticDelay;
    unsigned int maxRowLength;
    InitSparseConnectivitySnippet::Init connectivityInitSnippet;
};

// Helper function to either find existing model that provides desired parameters or create new one
template<typename Param, typename Model, typename ...Args>
const Model &getCreateModel(const Param &params, std::map<Param, Model> &models, Args... args)
{
    // If no existing model is found that matches parameters
    const auto existingModel = models.find(params);
    if(existingModel == models.end())
    {
        // Load XML document
        pugi::xml_document doc;
        auto result = doc.load_file(params.getURL().c_str());
        if(!result) {
            throw std::runtime_error("Could not open file:" + params.getURL() + ", error:" + result.description());
        }

        // Get SpineML root
        auto spineML = doc.child("SpineML");
        if(!spineML) {
            throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML component - it has no root SpineML node");
        }

        // Get component class
        auto componentClass = spineML.child("ComponentClass");
        if(!componentClass || strcmp(componentClass.attribute("type").value(), Model::componentClassName) != 0) {
            throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML component - "
                                    "it's ComponentClass node is either missing or of the incorrect type");
        }

        // Create new model
        // **THINK** some sort of move-semantic magic could probably make this a move
        LOGD << "\tCreating new model";
        auto newModel = models.insert(
            std::make_pair(params, Model(params, componentClass, args...)));

        return newModel.first->second;
    }
    else
    {
        return existingModel->second;
    }
}
//----------------------------------------------------------------------------
// Helper function to either find existing model that provides desired parameters or create new one
template<typename Param, typename Model, typename ...Args>
const Model &getCreatePassthroughModel(const Param &params, std::map<Param, Model> &models, Args... args)
{
    // If no existing model is found that matches parameters
    const auto existingModel = models.find(params);
    if(existingModel == models.end())
    {
        // Create new model
        // **THINK** some sort of move-semantic magic could probably make this a move
        LOGI << "\tCreating new model";
        auto newModel = models.insert(
            std::make_pair(params, Model(params, args...)));

        return newModel.first->second;
    }
    else
    {
        return existingModel->second;
    }
}
//----------------------------------------------------------------------------
// Helper function to read the delay value from a SpineML 'Synapse' node
unsigned int readDelaySteps(const pugi::xml_node &node, double dt)
{
    // Get delay node
    auto delay = node.child("Delay");
    if(delay) {
        auto fixedValue = delay.child("FixedValue");
        if(fixedValue) {
            double delay = fixedValue.attribute("value").as_double();
            return (unsigned int)std::round(delay / dt);
        }
        else {
            throw std::runtime_error("GeNN currently only supports projections with a single delay value");
        }
    }
    else
    {
        throw std::runtime_error("Connector has no 'Delay' node");
    }
}
//----------------------------------------------------------------------------
// Helper function to determine the correct type of GeNN projection to use for a SpineML 'Synapse' node
SynapseMatrixProps getSynapticMatrixProps(const filesystem::path &basePath, const pugi::xml_node &node, 
                                          unsigned int numPre, unsigned int numPost, double dt)
{
    auto oneToOne = node.child("OneToOneConnection");
    if(oneToOne) {
        return {Connectors::OneToOne::getMatrixConnectivity(oneToOne, numPre, numPost),
                readDelaySteps(oneToOne, dt), 1, 0,
                Connectors::OneToOne::getConnectivityInit(oneToOne)};
    }

    auto allToAll = node.child("AllToAllConnection");
    if(allToAll) {
        return {Connectors::AllToAll::getMatrixConnectivity(allToAll, numPre, numPost),
                readDelaySteps(allToAll, dt), 1, 0,
                uninitialisedConnectivity()};
    }

    auto fixedProbability = node.child("FixedProbabilityConnection");
    if(fixedProbability) {
        return {Connectors::FixedProbability::getMatrixConnectivity(fixedProbability, numPre, numPost),
                readDelaySteps(fixedProbability, dt), 1, 0,
                Connectors::FixedProbability::getConnectivityInit(fixedProbability)};
    }

    auto connectionList = node.child("ConnectionList");
    if(connectionList) {
        // Read maximum row length and any explicit delay from connector
        unsigned int maxRowLength;
        Connectors::List::DelayType delayType;
        float maxDelay;
        std::tie(maxRowLength, delayType, maxDelay) = Connectors::List::readMaxRowLengthAndDelay(basePath, connectionList,
                                                                                                 numPre, numPost);

        // If connector didn't specify delay, read it from delay child. Otherwise convert max delay to timesteps
        unsigned int axonalDelay = NO_DELAY;
        unsigned int maxDendriticDelay = 1;
        if(delayType == Connectors::List::DelayType::None) {
            axonalDelay = readDelaySteps(connectionList, dt);
        }
        else if(delayType == Connectors::List::DelayType::Homogeneous) {
            axonalDelay = (unsigned int)std::round(maxDelay / dt);
        }
        else {
            maxDendriticDelay = (unsigned int)std::round(maxDelay / dt);
        }
        
        // If explicit delay wasn't specified, read it from delay child. Otherwise convert explicit delay to timesteps
        return {Connectors::List::getMatrixConnectivity(connectionList, numPre, numPost),
                axonalDelay, maxDendriticDelay, maxRowLength, uninitialisedConnectivity()};
    }

    throw std::runtime_error("No supported connection type found for projection");
}
//----------------------------------------------------------------------------
const std::set<std::string> *getNamedSet(const std::map<std::string, std::set<std::string>> &sets, const std::string &name)
{
    // If there is no set with this name return NULL
    auto s = sets.find(name);
    if(s == sets.end()) {
        return nullptr;
    }
    // Otherwise, return a pointer to the set
    else {
        return &s->second;
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        CLI::App app{"SpineML generator for GeNN"};

        std::string experimentFilename;
        std::string outputDirectory;
        bool timing = false;
        unsigned int logLevel = plog::info;

        app.add_option("experiment,-e,--experiment", experimentFilename, "Experiment xml file")->required();
        app.add_option("output,-o,--output", outputDirectory, "Output directory for generated code");
        app.add_flag("-t,--timing", timing, "Generate GeNN timing code, allowing more fine-grained profiling");
        app.add_flag("--log-error{2},--log-warning{3},--log-info{4},--log-debug{5}", logLevel, "Verbosity of logging to show");

        CLI11_PARSE(app, argc, argv);

        // Initialise log channels, appending all to console
        plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
        plog::init((plog::Severity)logLevel, &consoleAppender);

        // Use filesystem library to get parent path of the network XML file
        const auto experimentPath = filesystem::path(experimentFilename).make_absolute();
        const auto basePath = experimentPath.parent_path();

        // If 2nd argument is specified use as output path otherwise use SpineCreator-compliant location
        const auto outputPath = outputDirectory.empty() ? basePath.parent_path() : filesystem::path(outputDirectory).make_absolute();

        LOGI << "Output path:" << outputPath.str();
        LOGI << "Parsing experiment '" << experimentPath.str() << "'";

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

        // Loop through inputs
        std::map<std::string, std::set<std::string>> externalInputs;
        for(auto input : experiment.select_nodes(SpineMLUtils::xPathNodeHasSuffix("Input").c_str())) {
            // Read target and port
            const std::string target = SpineMLUtils::getSafeName(input.node().attribute("target").value());
            const std::string port = input.node().attribute("port").value();

            // Add to map
            LOGD << "\tInput targetting: " << target << ":" << port;
            if(!externalInputs[target].emplace(port).second) {
                throw std::runtime_error("Multiple inputs targetting " + target + ":" + port);
            }
        }

        // Get model
        auto experimentModel = experiment.child("Model");
        if(!experimentModel) {
            throw std::runtime_error("No 'Model' node found in experiment");
        }

        // Build path to network from URL in model
        auto networkPath = basePath / experimentModel.attribute("network_layer_url").value();
        LOGI << "\tExperiment using model:" << networkPath;

        // Loop through configurations (overriden property values)
        std::map<std::string, std::set<std::string>> overridenProperties;
        for(auto config : experimentModel.children("Configuration")) {
            const std::string target = SpineMLUtils::getSafeName(config.attribute("target").value());

            // If this configuration has a property (it probably should)
            auto property = config.child("UL:Property");
            if(property) {
                const std::string propertyName = property.attribute("name").value();

                // Add to map
                LOGD << "\tOverriding property " << target << ":" << propertyName ;
                if(!overridenProperties[target].emplace(propertyName).second) {
                    throw std::runtime_error("Multiple overrides for property " + target + ":" + propertyName);
                }
            }
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
        LOGI << "\tDT = " << dt << "ms";

        // Load XML document
        pugi::xml_document doc;
        auto result = doc.load_file(networkPath.str().c_str());
        if(!result) {
            throw std::runtime_error("Unable to load XML file:" + networkPath.str() + ", error:" + result.description());
        }

        // Get SpineML root
        auto spineML = doc.child("LL:SpineML");
        if(!spineML) {
            throw std::runtime_error("XML file:" + networkPath.str() + " is not a low-level SpineML network - it has no root SpineML node");
        }

        // Neuron, postsyaptic and weight update models required by network
        std::map<ModelParams::Neuron, NeuronModel> neuronModels;
        std::map<ModelParams::Postsynaptic, PostsynapticModel> postsynapticModels;
        std::map<ModelParams::WeightUpdate, WeightUpdateModel> weightUpdateModels;
        std::map<std::string, PassthroughWeightUpdateModel> passthroughWeightUpdateModels;
        std::map<std::string, PassthroughPostsynapticModel> passthroughPostsynapticModels;

        // Get the filename of the network and remove extension
        // to get something usable as a network name
        std::string networkName = networkPath.filename();
        networkName = networkName.substr(0, networkName.find_last_of("."));

        // The neuron model
        ModelSpecInternal model;
        model.setDT(dt);
        model.setName(networkName);
        model.setTiming(timing);

        // Loop through populations once to build neuron populations
        for(auto population : spineML.children("LL:Population")) {
            auto neuron = population.child("LL:Neuron");
            if(!neuron) {
                throw std::runtime_error("'Population' node has no 'Neuron' node");
            }

            // Read basic population properties
            auto popName = SpineMLUtils::getSafeName(neuron.attribute("name").value());
            const unsigned int popSize = neuron.attribute("size").as_uint();
            LOGD << "Population " << popName << " consisting of " << popSize << " neurons";

            // If population is a spike source add GeNN spike source
            // **TODO** is this the only special case?
            if(strcmp(neuron.attribute("url").value(), "SpikeSource") == 0) {
                model.addNeuronPopulation<NeuronModels::SpikeSource>(popName, popSize, {}, {});
            }
            else {
                // Get sets of external input and overriden properties for this population
                const auto *externalInputPorts = getNamedSet(externalInputs, popName);
                const auto *overridenPropertyNames = getNamedSet(overridenProperties, popName);

                // Read neuron properties
                std::map<std::string, Models::VarInit> varInitialisers;
                ModelParams::Neuron modelParams(basePath, neuron, externalInputPorts,
                                                overridenPropertyNames, varInitialisers);

                // Either get existing neuron model or create new one of no suitable models are available
                const auto &neuronModel = getCreateModel(modelParams, neuronModels);

                // Add population to model
                model.addNeuronPopulation(popName, popSize, &neuronModel,
                                          NeuronModel::ParamValues(varInitialisers, neuronModel),
                                          NeuronModel::VarValues(varInitialisers, neuronModel));
            }
        }

        // Loop through populations AGAIN to build projections and low-level inputs
        for(auto population : spineML.children("LL:Population")) {
            auto neuron = population.child("LL:Neuron");

            // Read source population name from neuron node
            auto popName = SpineMLUtils::getSafeName(neuron.attribute("name").value());
            const NeuronGroup *neuronGroup = model.findNeuronGroup(popName);
            const NeuronModel *neuronModel = dynamic_cast<const NeuronModel*>(neuronGroup->getNeuronModel());

            // Loop through low-level inputs
            for(auto input : neuron.children("LL:Input")) {
                auto srcPopName = SpineMLUtils::getSafeName(input.attribute("src").value());
                const NeuronGroup *srcNeuronGroup = model.findNeuronGroup(srcPopName);
                const NeuronModel *srcNeuronModel = dynamic_cast<const NeuronModel*>(srcNeuronGroup->getNeuronModel());

                std::string srcPort = input.attribute("src_port").value();
                std::string dstPort = input.attribute("dst_port").value();

                LOGD << "Low-level input from population:" << srcPopName << "(" << srcPort << ")->" << popName << "(" << dstPort << ")";

                // Determine the GeNN matrix type, number of delay steps, max row length (if required) and connectivity initialiser
                const auto synapseMatrixProps = getSynapticMatrixProps(basePath, input,
                                                                       srcNeuronGroup->getNumNeurons(),
                                                                       neuronGroup->getNumNeurons(),
                                                                       dt);
                
                // Are heterogeneous delays required
                const bool heterogeneousDelay = (synapseMatrixProps.maxDendriticDelay > 1);
                
                // Either get existing passthrough weight update model or create new one of no suitable models are available
                const auto &passthroughWeightUpdateModel = getCreatePassthroughModel(srcPort, passthroughWeightUpdateModels,
                                                                                     srcNeuronModel, heterogeneousDelay);

                // Either get existing passthrough postsynaptic model or create new one of no suitable models are available
                const auto &passthroughPostsynapticModel = getCreatePassthroughModel(dstPort, passthroughPostsynapticModels,
                                                                                     neuronModel);

                
                // Create synapse population
                std::string passthroughSynapsePopName = std::string(srcPopName) + "_" + srcPort + "_" + popName + "_"  + dstPort;
                auto synapsePop = model.addSynapsePopulation(passthroughSynapsePopName, 
                                                             SynapseMatrixWeight::INDIVIDUAL | synapseMatrixProps.connectivityType, 
                                                             synapseMatrixProps.axonalDelay, 
                                                             srcPopName, popName,
                                                             &passthroughWeightUpdateModel, {}, {}, {}, {},
                                                             &passthroughPostsynapticModel, {}, {},
                                                             synapseMatrixProps.connectivityInitSnippet);

                // If matrix uses sparse connectivity and no initialiser is specified
                if(synapseMatrixProps.connectivityType == SynapseMatrixConnectivity::SPARSE
                    && synapseMatrixProps.connectivityInitSnippet.getSnippet()->getRowBuildCode().empty())
                {
                    // Check that max connections has been specified
                    assert(synapseMatrixProps.maxRowLength != 0);
                    synapsePop->setMaxConnections(synapseMatrixProps.maxRowLength);
                }
                
                // Set maximum dendritic delay for synapse population
                assert(synapseMatrixProps.maxDendriticDelay >= 1);
                synapsePop->setMaxDendriticDelayTimesteps(synapseMatrixProps.maxDendriticDelay);
            }

            // Loop through outgoing projections
            for(auto projection : population.children("LL:Projection")) {
                // Read destination population name from projection
                auto trgPopName = SpineMLUtils::getSafeName(projection.attribute("dst_population").value());
                const NeuronGroup *trgNeuronGroup = model.findNeuronGroup(trgPopName);
                const NeuronModel *trgNeuronModel = dynamic_cast<const NeuronModel*>(trgNeuronGroup->getNeuronModel());

                // Loop through synapse children
                // **NOTE** multiple projections between the same two populations of neurons are implemented in this way
                for(auto synapse : projection.children("LL:Synapse")) {
                    LOGD << "Projection from population:" << popName << "->" << trgPopName;

                    // Get weight update
                    auto weightUpdate = synapse.child("LL:WeightUpdate");
                    if(!weightUpdate) {
                        throw std::runtime_error("'Synapse' node has no 'WeightUpdate' node");
                    }

                    // Get name of weight update
                    const std::string weightUpdateName = SpineMLUtils::getSafeName(weightUpdate.attribute("name").value());

                    // Determine the GeNN matrix type, number of delay steps, max row length (if required) and connectivity initialiser
                    const auto synapseMatrixProps = getSynapticMatrixProps(basePath, synapse,
                                                                           neuronGroup->getNumNeurons(),
                                                                           trgNeuronGroup->getNumNeurons(),
                                                                           dt);

                    // Get sets of external input and overriden properties for this weight update
                    const auto *weightUpdateExternalInputPorts = getNamedSet(externalInputs, weightUpdateName);
                    const auto *weightUpdateOverridenPropertyNames = getNamedSet(overridenProperties, weightUpdateName);

                    // Read weight update properties
                    std::map<std::string, Models::VarInit> weightUpdateVarInitialisers;
                    ModelParams::WeightUpdate weightUpdateModelParams(basePath, weightUpdate,
                                                                      popName, trgPopName,
                                                                      weightUpdateExternalInputPorts,
                                                                      weightUpdateOverridenPropertyNames,
                                                                      weightUpdateVarInitialisers,
                                                                      synapseMatrixProps.maxDendriticDelay);

                    // Either get existing postsynaptic model or create new one of no suitable models are available
                    const auto &weightUpdateModel = getCreateModel(weightUpdateModelParams, weightUpdateModels,
                                                                   neuronModel, trgNeuronModel);

                    // Get post synapse
                    auto postSynapse = synapse.child("LL:PostSynapse");
                    if(!postSynapse) {
                        throw std::runtime_error("'Synapse' node has no 'PostSynapse' node");
                    }

                    // Get name of post synapse
                    const std::string postSynapseName = SpineMLUtils::getSafeName(postSynapse.attribute("name").value());

                    // Get sets of external input and overriden properties for this post synapse
                    const auto *postSynapseExternalInputPorts = getNamedSet(externalInputs, postSynapseName);
                    const auto *postSynapseOverridenPropertyNames = getNamedSet(overridenProperties, postSynapseName);

                    // Read postsynapse properties
                    std::map<std::string, Models::VarInit> postsynapticVarInitialisers;
                    ModelParams::Postsynaptic postsynapticModelParams(basePath, postSynapse,
                                                                      trgPopName,
                                                                      postSynapseExternalInputPorts,
                                                                      postSynapseOverridenPropertyNames,
                                                                      postsynapticVarInitialisers);

                    // Either get existing postsynaptic model or create new one of no suitable models are available
                    const auto &postsynapticModel = getCreateModel(postsynapticModelParams, postsynapticModels,
                                                                   trgNeuronModel, &weightUpdateModel);

                    // Add synapse population to model
                    // **NOTE** using weight update name is an arbitrary choice but these are guaranteed unique
                    auto synapsePop = model.addSynapsePopulation(weightUpdateName, 
                                                                 SynapseMatrixWeight::INDIVIDUAL | synapseMatrixProps.connectivityType, 
                                                                 synapseMatrixProps.axonalDelay, 
                                                                 popName, trgPopName,
                                                                 &weightUpdateModel, WeightUpdateModel::ParamValues(weightUpdateVarInitialisers, weightUpdateModel), WeightUpdateModel::VarValues(weightUpdateVarInitialisers, weightUpdateModel), {}, {},
                                                                 &postsynapticModel, PostsynapticModel::ParamValues(postsynapticVarInitialisers, postsynapticModel), PostsynapticModel::VarValues(postsynapticVarInitialisers, postsynapticModel),
                                                                 synapseMatrixProps.connectivityInitSnippet);

                    // If matrix uses sparse connectivity and no initialiser is specified
                    if(synapseMatrixProps.connectivityType == SynapseMatrixConnectivity::SPARSE
                        && synapseMatrixProps.connectivityInitSnippet.getSnippet()->getRowBuildCode().empty())
                    {
                        // Check that max connections has been specified
                        assert(synapseMatrixProps.maxRowLength != 0);
                        synapsePop->setMaxConnections(synapseMatrixProps.maxRowLength);
                    }
                    
                    // Set maximum dendritic delay for synapse population
                    assert(synapseMatrixProps.maxDendriticDelay >= 1);
                    synapsePop->setMaxDendriticDelayTimesteps(synapseMatrixProps.maxDendriticDelay);
                }
            }
        }
    

        // Finalize model
        model.finalize();

        // Write generated code to run directory beneath output path (creating it if necessary)
        auto runPath = (outputPath / "run");
        filesystem::create_directory(runPath);
        runPath = runPath.make_absolute();

        // Create directory for generated code within run path
        const auto codePath = runPath / (model.getName() + "_CODE");
        filesystem::create_directory(codePath);

        // **NOTE** SpineML doesn't support MPI for now so set local host ID to zero
        const int localHostID = 0;
        CodeGenerator::BACKEND_NAMESPACE::Preferences preferences;
        
        // Create backend
        auto backend = CodeGenerator::BACKEND_NAMESPACE::Optimiser::createBackend(model, codePath, localHostID, preferences);
    
        // Generate code
        const auto moduleNames = CodeGenerator::generateAll(model, backend, codePath);

#ifdef _WIN32
        // Create MSBuild project to compile and link all generated modules
        // **NOTE** scope requiredso it gets closed before being built
        {
            std::ofstream makefile((codePath / "runner.vcxproj").str());
            CodeGenerator::generateMSBuild(makefile, backend, "", moduleNames);
        }

        // Generate command to build using msbuild
        const std::string buildCommand = "msbuild /m /p:Configuration=Release  /verbosity:minimal \"" + (codePath / "runner.vcxproj").str() + "\"";
#else
        // Create makefile to compile and link all generated modules
        // **NOTE** scope requiredso it gets closed before being built
        {
            std::ofstream makefile((codePath / "Makefile").str());
            CodeGenerator::generateMakefile(makefile, backend, moduleNames);
        }

        // Generate command to build using make, using as many threads as possible
        const unsigned int numThreads = std::thread::hardware_concurrency();
        LOGD << "Using " << numThreads << " threads to build model";
        const std::string buildCommand = "make -C \"" + codePath.str() + "\" -j " + std::to_string(numThreads);
#endif

        // Execute build command
        const int retval = system(buildCommand.c_str());
        if (retval != 0){
            throw std::runtime_error("Building generated code with call:'" + buildCommand + "' failed with return value:" + std::to_string(retval));
        }
    }
    catch(const std::exception &exception)
    {
        std::cerr << exception.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
