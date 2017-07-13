#include "modelParams.h"

// Standard C++ includes
#include <iostream>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineMLCommon includes
#include "spineMLUtils.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// SpineMLGenerator::Base
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Base::Base(const filesystem::path &basePath, const pugi::xml_node &node,
                                          std::map<std::string, double> &fixedParamVals)
{
    m_URL = (basePath / node.attribute("url").value()).str();

    // Determine which properties are variable (therefore
    // can't be substituted directly into auto-generated code)
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
            m_VariableParams.insert(paramName);
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
void SpineMLGenerator::ModelParams::Base::addInputPortMapping(const std::string &dstPort, PortSource srcComponent, const std::string &srcPort)
{
    if(!m_InputPortSources.insert(std::make_pair(dstPort, std::make_pair(srcComponent, srcPort))).second) {
        throw std::runtime_error("Duplicate input port destination:" + dstPort);
    }
}
//----------------------------------------------------------------------------
void SpineMLGenerator::ModelParams::Base::addOutputPortMapping(const std::string &srcPort, PortSource dstComponent, const std::string &dstPort)
{
    if(!m_OutputPortTargets.insert(std::make_pair(srcPort, std::make_pair(dstComponent, dstPort))).second) {
        throw std::runtime_error("Duplicate output port source:" + srcPort);
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Neuron
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Neuron::Neuron(const filesystem::path &basePath, const pugi::xml_node &node,
                                              std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
{
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::WeightUpdate
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::WeightUpdate::WeightUpdate(const filesystem::path &basePath, const pugi::xml_node &node,
                                                          const std::string &srcPopName, const std::string &trgPopName,
                                                          std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
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
                                                          std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
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