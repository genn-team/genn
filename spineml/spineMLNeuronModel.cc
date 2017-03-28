#include "spineMLNeuronModel.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <sstream>

// Standard C includes
#include <cstring>

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

    // Get component class
    auto componentClass = spineML.child("ComponentClass");
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "neuron_body") != 0) {
        throw std::runtime_error("XML file:" + url + " is not a SpineML neuron body component - it's ComponentClass node is either missing or of the incorrect type");
    }

    std::cout << "\t\tModel name:" << componentClass.attribute("name").value() << std::endl;

    auto dynamics = componentClass.child("Dynamics");

    // Build mapping from regime names to IDs
    std::map<std::string, unsigned int> regimeIDs;
    std::transform(dynamics.children("Regime").begin(), dynamics.children("Regime").end(),
                   std::inserter(regimeIDs, regimeIDs.end()),
                   [&regimeIDs](const pugi::xml_node &n)
                   {
                       return std::make_pair(n.attribute("name").value(), regimeIDs.size());
                   });

    std::stringstream simCode;
    std::stringstream thresholdCondition;

    // If this model has a single regime
    if(regimeIDs.size() == 1) {
        auto regime = dynamics.child("Regime");
    }
    // Otherwise there are multiple regimes
    else if(regimeIDs.size() > 1) {
        // **TODO** add regime unsigned int state variable

        // Loop through regimes
        bool firstRegime = true;
        for(auto regime : dynamics.children("Regime")) {
            const auto *regimeName = regime.attribute("name").value();
            std::cout << "\t\t\tRegime name:" << regimeName << ", id:" << regimeIDs[regimeName] << std::endl;

            // Write regime condition test code to sim code
            if(firstRegime) {
                firstRegime = false;
            }
            else {
                simCode << "else ";
            }
            simCode << "if(_regimeID == " << regimeIDs[regimeName] << ") {" << std::endl;

            // Loop through conditions by which neuron might leave regime
            for(auto condition : regime.children("OnCondition")) {
                if(!condition.attribute("target_regime")) {
                    throw std::runtime_error("Regime has a condition with no target");
                }

                // Get triggering code
                auto triggerCode = condition.child("Trigger").child("MathInline");
                if(!triggerCode) {
                    throw std::runtime_error("No trigger condition for transition between regimes");
                }

                // Write trigger condition
                simCode << "\tif(" << triggerCode.text().get() << ") {" << std::endl;

                // Loop through state assignements
                for(auto stateAssign : condition.children("StateAssignment")) {
                    simCode << "\t\t" << stateAssign.attribute("variable").value() << " = " << stateAssign.child_value("MathInline") << ";" << std::endl;
                }

                // Write transition to target regime
                simCode << "\t\t_regimeID = " << regimeIDs[condition.attribute("target_regime").value()] << ";" << std::endl;

                // End of trigger condition
                simCode << "\t}" << std::endl;

                // If this condition emits a spike
                auto spikeEventOut = condition.select_node("EventOut[@port='spike']");
                if(spikeEventOut) {
                    // If there are existing threshold conditions, OR them with this one
                    if(thresholdCondition.tellp() > 0) {
                        thresholdCondition << " || ";
                    }

                    // Write trigger condition AND regime to threshold condition
                    thresholdCondition << "(_regimeID == " << regimeIDs[regimeName] << " && (" << triggerCode.text().get() << "))";
                }
            }

            // Write dynamics
            // **TODO** identify cases where Euler is REALLY stupid
            auto timeDerivative = regime.child("TimeDerivative");
            if(timeDerivative) {
                simCode << "\t" << timeDerivative.attribute("variable").value() << " += DT * (" << timeDerivative.child_value("MathInline") << ");" << std::endl;
            }

            // End of regime
            simCode << "}" << std::endl;
        }
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