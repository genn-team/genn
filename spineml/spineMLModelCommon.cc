#include "spineMLModelCommon.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "newModels.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::ParamValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::ParamValues::getValues() const
{
    // Get parameter names from model
    auto modelParamNames = m_Model.getParamNames();

    // Reserve vector of values to match it
    std::vector<double> paramValues;
    paramValues.reserve(modelParamNames.size());

    // Populate this vector with either values from map or 0s
    std::transform(modelParamNames.begin(), modelParamNames.end(),
                   std::back_inserter(paramValues),
                   [this](const std::string &n)
                   {
                       auto value = m_Values.find(n);
                       if(value == m_Values.end()) {
                           return 0.0;
                       }
                       else {
                           return value->second;
                       }
                   });
    return paramValues;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::VarValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::VarValues::getValues() const
{
    // Get variables from model
    auto modelVars = m_Model.getVars();

    // Reserve vector of values to match it
    std::vector<double> varValues;
    varValues.reserve(modelVars.size());

    // Populate this vector with either values from map or 0s
    std::transform(modelVars.begin(), modelVars.end(),
                   std::back_inserter(varValues),
                   [this](const std::pair<std::string, std::string> &n)
                   {
                       auto value = m_Values.find(n.first);
                       if(value == m_Values.end()) {
                           return 0.0;
                       }
                       else {
                           return value->second;
                       }
                   });
    return varValues;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::CodeStream
//----------------------------------------------------------------------------
void SpineMLGenerator::CodeStream::onRegimeEnd(bool multipleRegimes, unsigned int currentRegimeID)
{
    // If any code was written for this regime
    if(m_CurrentRegimeCodeStream.tellp() > 0)
    {
        if(multipleRegimes) {
            if(m_FirstNonEmptyRegime) {
                m_FirstNonEmptyRegime = false;
            }
            else {
                m_CodeStream << "else ";
            }
            m_CodeStream << "if(_regimeID == " << currentRegimeID << ")" << ob(1);
        }

        // Write contents of current region code stream to main code stream
        m_CodeStream << m_CurrentRegimeCodeStream.str();

        // Clear current regime code stream
        std::ostringstream().swap(m_CurrentRegimeCodeStream);

        // End of regime
        if(multipleRegimes) {
            m_CodeStream << cb(1);
        }
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandlerError
//----------------------------------------------------------------------------
void SpineMLGenerator::ObjectHandlerError::onObject(const pugi::xml_node &node, unsigned int, unsigned int)
{
    throw std::runtime_error("GeNN cannot handle " + std::string(node.name()) + " objects in this context");
}

//----------------------------------------------------------------------------
// Helper functions
//----------------------------------------------------------------------------
pugi::xml_node SpineMLGenerator::loadComponent(const std::string &url, const std::string &expectedType)
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
        throw std::runtime_error("XML file:" + url + " is not a SpineML " + expectedType + " component - it's ComponentClass node is either missing or of the incorrect type");
    }

    return componentClass;
}
//----------------------------------------------------------------------------
bool SpineMLGenerator::generateModelCode(const pugi::xml_node &componentClass, ObjectHandler &objectHandlerEvent,
                                         ObjectHandler &objectHandlerCondition, ObjectHandler &objectHandlerImpulse,
                                         ObjectHandler &objectHandlerTimeDerivative,
                                         std::function<void(bool, unsigned int)> regimeEndFunc)
{
    std::cout << "\t\tModel name:" << componentClass.attribute("name").value() << std::endl;

    // Build mapping from regime names to IDs
    auto dynamics = componentClass.child("Dynamics");
    std::map<std::string, unsigned int> regimeIDs;
    std::transform(dynamics.children("Regime").begin(), dynamics.children("Regime").end(),
                   std::inserter(regimeIDs, regimeIDs.end()),
                   [&regimeIDs](const pugi::xml_node &n)
                   {
                       return std::make_pair(n.attribute("name").value(), regimeIDs.size());
                   });
    const bool multipleRegimes = (regimeIDs.size() > 1);

    // Loop through regimes
    std::cout << "\t\tRegimes:" << std::endl;
    for(auto regime : dynamics.children("Regime")) {
        const auto *currentRegimeName = regime.attribute("name").value();
        const unsigned int currentRegimeID = regimeIDs[currentRegimeName];
        std::cout << "\t\t\tRegime name:" << currentRegimeName << ", id:" << currentRegimeID << std::endl;

        // Loop through conditions by which model might leave regime
        for(auto condition : regime.children("OnCondition")) {
            const auto *targetRegimeName = condition.attribute("target_regime").value();
            const unsigned int targetRegimeID = regimeIDs[targetRegimeName];
            objectHandlerCondition.onObject(condition, currentRegimeID, targetRegimeID);
        }

        // Write out time derivatives
        for(auto timeDerivative : regime.children("TimeDerivative")) {
            objectHandlerTimeDerivative.onObject(timeDerivative, currentRegimeID, 0);
        }

        // Call function to notify all code streams of end of regime
        regimeEndFunc(multipleRegimes, currentRegimeID);
    }

    return multipleRegimes;
}