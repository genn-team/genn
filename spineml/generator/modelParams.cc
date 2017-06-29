#include "modelParams.h"

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
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ModelParams::Postsynaptic
//----------------------------------------------------------------------------
SpineMLGenerator::ModelParams::Postsynaptic::Postsynaptic(const filesystem::path &basePath, const pugi::xml_node &node,
                                                          std::map<std::string, double> &fixedParamVals)
: Base(basePath, node, fixedParamVals)
{
}