#include "spineMLNeuronModel.h"

// Standard C++ includes
#include <iostream>
#include <sstream>

// pugixml includes
#include "pugixml/pugixml.hpp"

// Spine ML generator includes
#include "objectHandlerCondition.h"
#include "objectHandlerTimeDerivative.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// ObjectHandlerNeuronCondition
//----------------------------------------------------------------------------
class ObjectHandlerNeuronCondition : public SpineMLGenerator::ObjectHandlerCondition
{
public:
    ObjectHandlerNeuronCondition(SpineMLGenerator::CodeStream &codeStream)
        : ObjectHandlerCondition(codeStream){}

    //----------------------------------------------------------------------------
    // SpineMLGenerator::ObjectHandlerCondition virtuals
    //----------------------------------------------------------------------------
    void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                  unsigned int targetRegimeID)
    {
        // Superclass
        SpineMLGenerator::ObjectHandlerCondition::onObject(node, currentRegimeID,
                                                           targetRegimeID);

        // If this condition emits a spike
        auto spikeEventOut = node.select_node("EventOut[@port='spike']");
        if(spikeEventOut) {
            // If there are existing threshold conditions, OR them with this one
            if(m_ThresholdCodeStream.tellp() > 0) {
                m_ThresholdCodeStream << " || ";
            }

            // Write trigger condition AND regime to threshold condition
            auto triggerCode = node.child("Trigger").child("MathInline");
            m_ThresholdCodeStream << "(_regimeID == " << currentRegimeID << " && (" << triggerCode.text().get() << "))";
        }
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::string getThresholdCode() const{ return m_ThresholdCodeStream.str(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ostringstream m_ThresholdCodeStream;
};
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLNeuronModel::SpineMLNeuronModel(const std::string &url,
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
        throw std::runtime_error("XML file:" + url + " is not a SpineML 'neuron_body' component - "
                                 "it's ComponentClass node is either missing or of the incorrect type");
    }


    // Create a code stream for generating sim code
    CodeStream simCodeStream;

    // Create lambda function to end regime on all code streams when required
    auto regimeEndFunc =
        [&simCodeStream]
        (bool multipleRegimes, unsigned int currentRegimeID)
        {
            simCodeStream.onRegimeEnd(multipleRegimes, currentRegimeID);
        };

    // Generate model code using specified condition handler
    ObjectHandlerError objectHandlerError;
    ObjectHandlerNeuronCondition objectHandlerCondition(simCodeStream);
    ObjectHandlerTimeDerivative objectHandlerTimeDerivative(simCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass, objectHandlerError,
                                                   objectHandlerCondition, objectHandlerError,
                                                   objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_ThresholdConditionCode = objectHandlerCondition.getThresholdCode();

    // Build the final vectors of parameter names and variables from model and
    // correctly wrap references to them in newly-generated code strings
    tie(m_ParamNames, m_Vars) = processModelVariables(componentClass, variableParams,
                                                      multipleRegimes, {&m_SimCode, &m_ThresholdConditionCode});

    // If there is an analogue reduce port using the addition operator, it's probably a synaptic input current
    auto linearReducePorts = componentClass.select_nodes("AnalogReducePort[@reduce_op='+']");
    if(linearReducePorts.size() == 1) {
        const auto *linearReducePortName = linearReducePorts.first().node().attribute("name").value();
        wrapAndReplaceVariableNames(m_SimCode, linearReducePortName, "Isyn");
        wrapAndReplaceVariableNames(m_ThresholdConditionCode, linearReducePortName, "Isyn");
    }
    // Otherwise, throw an exception
    else if(linearReducePorts.size() > 1) {
        // **TODO** 'Alias' nodes in dynamics may be used to combine these together
        throw std::runtime_error("GeNN doesn't support multiple input currents going into neuron");
    }


    std::cout << "SIM CODE:" << std::endl << m_SimCode << std::endl;
    std::cout << "THRESHOLD CONDITION CODE:" << std::endl << m_ThresholdConditionCode << std::endl;
}