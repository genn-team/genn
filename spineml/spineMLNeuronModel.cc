#include "spineMLNeuronModel.h"

// Standard C++ includes
#include <algorithm>

// pugixml includes
#include "pugixml/pugixml.hpp"

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLNeuronModel::SpineMLNeuronModel(const pugi::xml_node &neuronNode,
                                                         const std::string &url,
                                                         const std::set<std::string> &variableParams)
{
    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(url.c_str());
    if(!result) {
        throw std::runtime_error("Could not open file:" + url + ", error:" + result.description());
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        throw std::runtime_error("XML file:" + url + " is not a SpineML component - it has no root SpineML node");
    }
}



//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel::ParamValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::SpineMLNeuronModel::ParamValues::getValues() const
{
    // Reserve vector to hold values
    std::vector<double> values;
    values.reserve(m_Values.size());

    // Transform values of value map into vector and return
    std::transform(std::begin(m_Values), std::end(m_Values),
                   std::back_inserter(values),
                   [](const std::pair<std::string, double> &v){ return v.second; });
    return values;
}