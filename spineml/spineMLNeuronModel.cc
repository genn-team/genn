#include "spineMLNeuronModel.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

// Standard C includes
#include <cstring>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "codeGenUtils.h"
#include "CodeHelper.h"

// Spine ML generator includes
#include "objectHandlerCondition.h"
#include "objectHandlerTimeDerivative.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
void wrapAndReplaceVariableNames(std::string &code, const std::string &variableName,
                                 const std::string &replaceVariableName)
{
    // Build a regex to match variable name with at least one
    // character that can't be in a variable name on either side
    std::regex regex("([^a-zA-Z_])" + variableName + "([^a-zA-Z_])");

    // Insert GeNN $(XXXX) wrapper around variable name
    code = std::regex_replace(code,  regex, "$1$(" + replaceVariableName + ")$2");
}

void wrapVariableNames(std::string &code, const std::string &variableName)
{
    wrapAndReplaceVariableNames(code, variableName, variableName);
}

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
    // Load the component class from file and check it's type
    auto componentClass = loadComponent(url, "neuron_body");

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

    //*******************
    // Generic begin
    //*******************
    // Starting with those the model needs to vary, create a set of genn variables
    std::set<string> gennVariables(variableParams);

    // Add model state variables to this
    auto dynamics = componentClass.child("Dynamics");
    std::transform(dynamics.children("StateVariable").begin(), dynamics.children("StateVariable").end(),
                   std::inserter(gennVariables, gennVariables.end()),
                   [](const pugi::xml_node &n){ return n.attribute("name").value(); });

    // Loop through model parameters
    std::cout << "\t\tParameters:" << std::endl;
    for(auto param : componentClass.children("Parameter")) {
        // If parameter hasn't been declared variable by model, add it to vector of parameter names
        std::string paramName = param.attribute("name").value();
        if(gennVariables.find(paramName) == gennVariables.end()) {
            std::cout << "\t\t\t" << paramName << std::endl;
            m_ParamNames.push_back(paramName);

            // Wrap variable names so GeNN code generator can find them
            wrapVariableNames(m_SimCode, paramName);
            wrapVariableNames(m_ThresholdConditionCode, paramName);
        }
    }

    // Add all GeNN variables
    std::transform(gennVariables.begin(), gennVariables.end(), std::back_inserter(m_Vars),
                   [](const std::string &vname){ return std::make_pair(vname, "scalar"); });

    // If model has multiple regimes, add unsigned int regime ID to values
    if(multipleRegimes) {
        m_Vars.push_back(std::make_pair("_regimeID", "unsigned int"));
    }

    std::cout << "\t\tVariables:" << std::endl;
    for(const auto &v : m_Vars) {
        std::cout << "\t\t\t" << v.first << ":" << v.second << std::endl;

        // Wrap variable names so GeNN code generator can find them
        wrapVariableNames(m_SimCode, v.first);
        wrapVariableNames(m_ThresholdConditionCode, v.first);
    }

    //*******************
    // Generic end
    //*******************

    // If there is an analogue reduce port using the addition operator, it's probably a synaptic input current
    auto linearReducePorts = componentClass.select_nodes("AnalogReducePort[@reduce_op='+']");
    if(linearReducePorts.size() == 1) {
        const auto *linearReducePortName = linearReducePorts.first().node().attribute("name").value();
        wrapAndReplaceVariableNames(m_SimCode, linearReducePortName, "Isyn");
        wrapAndReplaceVariableNames(m_ThresholdConditionCode, linearReducePortName, "Isyn");
    }
    // Otherwise, throw and exception
    else if(linearReducePorts.size() > 1) {
        // **TODO** 'Alias' nodes in dynamics may be used to combine these together
        throw std::runtime_error("GeNN doesn't support multiple input currents going into neuron");
    }


    std::cout << "SIM CODE:" << std::endl << m_SimCode << std::endl;
    std::cout << "THRESHOLD CONDITION CODE:" << std::endl << m_ThresholdConditionCode << std::endl;
}