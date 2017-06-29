#include "modelParams.h"

// Standard C++ includes
#include <iostream>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"


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
void SpineMLGenerator::ModelParams::Base::addPortMapping(const std::string &dstPort, PortSource srcComponent, const std::string &srcPort)
{
    m_PortMappings.insert(std::make_pair(dstPort, std::make_pair(srcComponent, srcPort)));
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
                                                          std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
{
    // If an input src and destination port are specified add them to input port mapping
    auto inputSrcPort = node.attribute("input_src_port");
    auto inputDstPort = node.attribute("input_dst_port");
    if(inputSrcPort && inputDstPort) {
        addPortMapping(inputDstPort.value(), PortSource::PRESYNAPTIC_NEURON, inputSrcPort.value());
    }

    // If a feedback src and destination port are specified add them to input port mapping
    auto feedbackSrcPort = node.attribute("feedback_src_port");
    auto feedbackDstPort = node.attribute("feedback_dst_port");
    if(feedbackSrcPort && feedbackDstPort) {
        addPortMapping(feedbackDstPort.value(), PortSource::POSTSYNAPTIC_NEURON, feedbackSrcPort.value());
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Postsynaptic
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Postsynaptic::Postsynaptic(const filesystem::path &basePath, const pugi::xml_node &node,
                                                          std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
{
    // If an input src and destination port are specified add them to input port mapping
    auto inputSrcPort = node.attribute("input_src_port");
    auto inputDstPort = node.attribute("input_dst_port");
    if(inputSrcPort && inputDstPort) {
        addPortMapping(inputDstPort.value(), PortSource::WEIGHT_UPDATE, inputSrcPort.value());
    }

    // If an output src and destination port are specified add them to input port mapping
    auto outputSrcPort = node.attribute("output_src_port");
    auto outputDstPort = node.attribute("output_dst_port");
    if(outputSrcPort && outputDstPort) {
        addPortMapping(outputDstPort.value(), PortSource::POSTSYNAPTIC_NEURON, outputSrcPort.value());
    }
}