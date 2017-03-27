#include "spineMLNeuronModel.h"

// Standard C++ includes
#include <algorithm>

// pugixml includes
#include "pugixml.hpp"

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLNeuronModel::SpineMLNeuronModel(const pugi::xml_node &neuronNode,
                                                         const std::string &url,
                                                         const std::set<std::string> &variableParams)
{

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