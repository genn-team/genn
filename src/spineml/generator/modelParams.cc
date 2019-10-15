#include "modelParams.h"

// Standard C includes
#include <cmath>

// Standard C++ includes
#include <iostream>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "initVarSnippet.h"
#include "models.h"

// SpineMLCommon includes
#include "spineMLUtils.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// SpineMLGenerator::Base
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Base::Base(const filesystem::path &basePath, const pugi::xml_node &node,
                                          const std::set<std::string> *externalInputPorts,
                                          const std::set<std::string> *overridenPropertyNames,
                                          std::map<std::string, Models::VarInit> &varInitialisers)
{
    m_URL = (basePath / node.attribute("url").value()).str();

    // Determine which properties are variable (therefore
    // can't be substituted directly into auto-generated code)
    for(auto param : node.children("Property")) {
        const auto *paramName = param.attribute("name").value();

        // If parameter has a fixed value
        auto fixedValue = param.child("FixedValue");
        if(fixedValue) {
            // Add initialiser
            varInitialisers.emplace(paramName, Models::VarInit(InitVarSnippet::Constant::getInstance(), {
                fixedValue.attribute("value").as_double() }));

            // Typically fixed-value parameters are candidates for hard-coding into model however,
            // if they are overriden in experiment, they should be implemented as state variables so mark them as such
            if(overridenPropertyNames && overridenPropertyNames->find(paramName) != overridenPropertyNames->cend()) {
                m_VariableParams.insert(paramName);
            }
        }
        // Otherwise
        else {
            // In GeNN terms, it should be treated as a variable so add it to set of potential variable names
            m_VariableParams.insert(paramName);

            // If property is uniformly distributed, add uniform initialiser
            if(pugi::xml_node uniformDistribution = param.child("UniformDistribution")) {
                varInitialisers.emplace(paramName, Models::VarInit(InitVarSnippet::Uniform::getInstance(), {
                    uniformDistribution.attribute("minimum").as_double(), uniformDistribution.attribute("maximum").as_double() }));
            }
            // Otherwise, if property is normally distributed, add normal initialiser
            else if(pugi::xml_node normalDistribution = param.child("NormalDistribution")) {
                varInitialisers.emplace(paramName, Models::VarInit(InitVarSnippet::Normal::getInstance(), {
                    normalDistribution.attribute("mean").as_double(), std::sqrt(normalDistribution.attribute("variance").as_double()) }));
            }
            // Otherwise, if property is exponentially distributed, add poisson initialiser
            // **NOTE** Poisson distribution isn't actually one - it is the exponential
            else if(pugi::xml_node exponentialDistribution = param.child("PoissonDistribution")) {
                varInitialisers.emplace(paramName, Models::VarInit(InitVarSnippet::Exponential::getInstance(), {
                    exponentialDistribution.attribute("mean").as_double() }));
            }
            // Otherwise, as this type of property cannot be initialised by GeNN, mark it as unitialised
            else {
                varInitialisers.emplace(paramName, Models::VarInit(InitVarSnippet::Uninitialised::getInstance(), {}));
            }
        }
    }

    // If this model has any external input ports, add input port mapping
    if(externalInputPorts != nullptr) {
        for(const auto &p : *externalInputPorts) {
            addInputPortMapping(p, PortSource::EXTERNAL, "");
        }
    }
}
//----------------------------------------------------------------------------
const std::pair<SpineMLGenerator::ModelParams::Base::PortSource, std::string> &SpineMLGenerator::ModelParams::Base::getInputPortSrc(const std::string &dstPort) const
{
    auto port = m_InputPortSources.find(dstPort);
    if(port == m_InputPortSources.end()) {
        throw std::runtime_error("Cannot find destination port:" + dstPort);
    }
    else {
        return port->second;
    }
}
//----------------------------------------------------------------------------
const std::pair<SpineMLGenerator::ModelParams::Base::PortSource, std::string> &SpineMLGenerator::ModelParams::Base::getOutputPortTrg(const std::string &srcPort) const
{
    auto port = m_OutputPortTargets.find(srcPort);
    if(port == m_OutputPortTargets.end()) {
        throw std::runtime_error("Cannot find source port:" + srcPort);
    }
    else {
        return port->second;
    }
}
//----------------------------------------------------------------------------
bool SpineMLGenerator::ModelParams::Base::isInputPortExternal(const std::string &dstPort) const
{
    auto port = m_InputPortSources.find(dstPort);

    return (port != m_InputPortSources.end() && port->second.first == PortSource::EXTERNAL);
}
//----------------------------------------------------------------------------
void SpineMLGenerator::ModelParams::Base::addInputPortMapping(const std::string &dstPort, PortSource srcComponent, const std::string &srcPort)
{
    if(!m_InputPortSources.emplace(std::piecewise_construct, std::forward_as_tuple(dstPort), std::forward_as_tuple(srcComponent, srcPort)).second) {
        throw std::runtime_error("Duplicate input port destination:" + dstPort);
    }
}
//----------------------------------------------------------------------------
void SpineMLGenerator::ModelParams::Base::addOutputPortMapping(const std::string &srcPort, PortSource dstComponent, const std::string &dstPort)
{
    if(!m_OutputPortTargets.emplace(std::piecewise_construct, std::forward_as_tuple(srcPort), std::forward_as_tuple(dstComponent, dstPort)).second) {
        throw std::runtime_error("Duplicate output port source:" + srcPort);
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Neuron
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Neuron::Neuron(const filesystem::path &basePath, const pugi::xml_node &node,
                                              const std::set<std::string> *externalInputPorts,
                                              const std::set<std::string> *overridenPropertyNames,
                                              std::map<std::string, Models::VarInit> &varInitialisers)
: Base(basePath, node, externalInputPorts, overridenPropertyNames, varInitialisers)
{
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::WeightUpdate
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::WeightUpdate::WeightUpdate(const filesystem::path &basePath, const pugi::xml_node &node,
                                                          const std::string &srcPopName, const std::string &trgPopName,
                                                          const std::set<std::string> *externalInputPorts,
                                                          const std::set<std::string> *overridenPropertyNames,
                                                          std::map<std::string, Models::VarInit> &varInitialisers,
                                                          unsigned int maxDendriticDelay)
: Base(basePath, node, externalInputPorts, overridenPropertyNames, varInitialisers), m_MaxDendriticDelay(maxDendriticDelay)
{
    // If an input src and destination port are specified add them to input port mapping
    auto inputSrcPort = node.attribute("input_src_port");
    auto inputDstPort = node.attribute("input_dst_port");
    if(inputSrcPort && inputDstPort) {
        addInputPortMapping(inputDstPort.value(), PortSource::PRESYNAPTIC_NEURON, inputSrcPort.value());
    }

    // If a feedback src and destination port are specified add them to input port mapping
    auto feedbackSrcPort = node.attribute("feedback_src_port");
    auto feedbackDstPort = node.attribute("feedback_dst_port");
    if(feedbackSrcPort && feedbackDstPort) {
        addInputPortMapping(feedbackDstPort.value(), PortSource::POSTSYNAPTIC_NEURON, feedbackSrcPort.value());
    }

    // Loop through low-level inputs
    for(auto input : node.children("LL:Input")) {
        // If this input is connected with a one-to-one connector
        auto oneToOneConnector = input.child("OneToOneConnection");
        if(oneToOneConnector) {
            // If the source of this input is our target neuron population, add port mapping
            auto safeInputSrc = SpineMLUtils::getSafeName(input.attribute("src").value());
            if(safeInputSrc == srcPopName) {
                addInputPortMapping(input.attribute("dst_port").value(), PortSource::PRESYNAPTIC_NEURON, input.attribute("src_port").value());
            }
            else if(safeInputSrc == trgPopName) {
                addInputPortMapping(input.attribute("dst_port").value(), PortSource::POSTSYNAPTIC_NEURON, input.attribute("src_port").value());
            }
            else {
                throw std::runtime_error("GeNN weight update models can only receive input from the pre or postsynaptic neuron population");
            }
        }
        else {
            throw std::runtime_error("GeNN weight update models can only receive input through OneToOneConnections");
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Postsynaptic
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Postsynaptic::Postsynaptic(const filesystem::path &basePath, const pugi::xml_node &node,
                                                          const std::string &trgPopName,
                                                          const std::set<std::string> *externalInputPorts,
                                                          const std::set<std::string> *overridenPropertyNames,
                                                          std::map<std::string, Models::VarInit> &varInitialisers)
: Base(basePath, node, externalInputPorts, overridenPropertyNames, varInitialisers)
{
    // If an input src and destination port are specified add them to input port mapping
    auto inputSrcPort = node.attribute("input_src_port");
    auto inputDstPort = node.attribute("input_dst_port");
    if(inputSrcPort && inputDstPort) {
        addInputPortMapping(inputDstPort.value(), PortSource::WEIGHT_UPDATE, inputSrcPort.value());
    }

    // If an output src and destination port are specified add them to input port mapping
    auto outputSrcPort = node.attribute("output_src_port");
    auto outputDstPort = node.attribute("output_dst_port");
    if(outputSrcPort && outputDstPort) {
        addOutputPortMapping(outputSrcPort.value(), PortSource::POSTSYNAPTIC_NEURON, outputDstPort.value());
    }

    // Loop through low-level inputs
    for(auto input : node.children("LL:Input")) {
         // If this input is connected with a one-to-one connector
        auto oneToOneConnector = input.child("OneToOneConnection");
        if(oneToOneConnector) {
            // If the source of this input is our target neuron population, add port mapping
            if(SpineMLUtils::getSafeName(input.attribute("src").value()) == trgPopName) {
                addInputPortMapping(input.attribute("dst_port").value(), PortSource::POSTSYNAPTIC_NEURON, input.attribute("src_port").value());
            }
            else {
                throw std::runtime_error("GeNN postsynaptic models can only receive input from the postsynaptic neuron population");
            }
        }
        else {
            throw std::runtime_error("GeNN postsynaptic models can only receive input through OneToOneConnections");
        }
    }
}
