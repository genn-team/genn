#include "modelCommon.h"

// Standard C++ includes
#include <iostream>
#include <list>
#include <regex>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"

// SpineML common includes
#include "spineMLLogging.h"

// SpineML generator includes
#include "objectHandler.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::ParamValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::ParamValues::getInitialisers() const
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
                       auto v = m_VarInitialisers.find(n);
                       if(v == m_VarInitialisers.end()) {
                           return 0.0;
                       }
                       else {
                           // Check that this parameter actually has a constant value
                           assert(dynamic_cast<const InitVarSnippet::Constant*>(v->second.getSnippet()) != nullptr);

                           // Return the first parameter (the value)
                           return v->second.getParams()[0];
                       }
                   });
    return paramValues;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::CodeStream
//----------------------------------------------------------------------------
void SpineMLGenerator::CodeStream::onRegimeEnd(bool multipleRegimes, unsigned int currentRegimeID)
{
    // If any code was written for this regime
    if(m_CurrentRegimeStream.tellp() > 0)
    {
        if(multipleRegimes) {
            if(m_FirstNonEmptyRegime) {
                m_FirstNonEmptyRegime = false;
            }
            else {
                m_CodeStream << "else ";
            }
            m_CodeStream << "if(_regimeID == " << currentRegimeID << ")" << CodeStream::OB(1);
        }

        // Flush contents of current regime to main codestream
        flush();

        // End of regime
        if(multipleRegimes) {
            m_CodeStream << CodeStream::CB(1);
        }
    }
}
//----------------------------------------------------------------------------
void SpineMLGenerator::CodeStream::flush()
{
     // Write contents of current region code stream to main code stream
    m_CodeStream << m_CurrentRegimeStream.str();

    // Clear current regime code stream
    m_CurrentRegimeStream.str("");
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Aliases
//----------------------------------------------------------------------------
SpineMLGenerator::Aliases::Aliases(const pugi::xml_node &componentClass)
{
    LOGD_SPINEML << "\t\tAliases:";

    // Loop through aliases and add to map
    auto dynamics = componentClass.child("Dynamics");
    for(auto alias : dynamics.children("Alias")) {
        const std::string name = alias.attribute("name").value();
        const std::string code = alias.child_value("MathInline");

        LOGD_SPINEML << "\t\t\t" << name;

        m_Aliases.emplace(name, code);
    }


    LOGD_SPINEML << "\t\t\tDependencies:";
    // Loop through aliases
    for(auto alias = m_Aliases.begin(); alias != m_Aliases.end(); alias++) {
        // Loop through other aliases
        for(auto otherAlias = m_Aliases.cbegin(); otherAlias != m_Aliases.cend(); otherAlias++) {
            if(otherAlias != alias) {
                // Build a regex to find alias name with at least one character that
                // can't be in a variable name on either side (or an end/beginning of string)
                // **NOTE** the suffix is non-capturing so two instances of variables separated by a single character are matched e.g. a*a
                const std::regex regex("(^|[^0-9a-zA-Z_])" + otherAlias->first + "(?=$|[^a-zA-Z0-9_])");

                // If 'alias' references 'otherAlias', add to set
                if(std::regex_search(alias->second.code, regex)) {
                    LOGD_SPINEML << "\t\t\t\t" << alias->first << " depends on " << otherAlias->first;
                    alias->second.dependencies.push_back(otherAlias);
                }

            }
        }
    }
}
//----------------------------------------------------------------------------
void SpineMLGenerator::Aliases::genAliases(std::ostream &os, std::initializer_list<std::string> codeStrings,
                                           const std::unordered_set<std::string> &excludeAliases) const
{
    // Output list of required aliases in correct order
    std::list<AliasIter> allRequiredAliases;

    // Set of string hashes used to implement depth-first-search
    std::unordered_set<size_t> discoveredAliases;

    // Stack used to implment depth-first search
    std::vector<AliasIter> aliasStack;

    LOGD_SPINEML << "\t\t\tCode alias requirements:";
    // Loop through aliases
    for(auto alias = m_Aliases.cbegin(); alias != m_Aliases.cend(); alias++) {
        // Build a regex to find alias name with at least one character that
        // can't be in a variable name on either side (or an end/beginning of string)
        // **NOTE** the suffix is non-capturing so two instances of variables separated by a single character are matched e.g. a*a
        const std::regex regex("(^|[^0-9a-zA-Z_])" + alias->first + "(?=$|[^a-zA-Z0-9_])");

        // If this alias isn't in the exclude set and if any code strings references it
        if((excludeAliases.find(alias->first) == excludeAliases.cend()) &&
            std::any_of(codeStrings.begin(), codeStrings.end(),
                        [&regex](const std::string &code){ return std::regex_search(code, regex); }))
        {
            assert(aliasStack.empty());

            // Add starting alias to vector
            aliasStack.push_back(alias);
            LOGD_SPINEML << "\t\t\t\tStart:" << alias->first;

            // While there are aliases on the stack
            std::list<AliasIter> requiredAliases;
            while(!aliasStack.empty()) {
                // Pop alias off top of stack
                auto v = aliasStack.back();
                aliasStack.pop_back();

                // If this alias hasn't already been discovered
                if(discoveredAliases.insert(std::hash<std::string>{}(v->first)).second) {
                    // Add it to the front of the required aliases list
                    requiredAliases.push_front(v);

                    // Push alias's non-excluded dependencies onto the top of the stack
                    for(auto d : v->second.dependencies) {
                        if(excludeAliases.find(d->first) == excludeAliases.cend()) {
                            aliasStack.push_back(d);
                        }
                    }
                }
            }

            // Splice all elements out of the ordered list of dependencies for this alias into main list
            allRequiredAliases.splice(allRequiredAliases.end(), requiredAliases);

        }
    }

    // If ANY aliases are required
    if(!allRequiredAliases.empty()) {
        os << "// Aliases" << std::endl;
        // Use stringstream to generate alias code
        for(const auto &r : allRequiredAliases) {
            os << "const scalar " << r->first << " = " << r->second.code << ";" << std::endl;
        }
        os << std::endl;
    }
}
//----------------------------------------------------------------------------
bool SpineMLGenerator::Aliases::isAlias(const std::string &name) const
{
    return (m_Aliases.find(name) != m_Aliases.end());
}
//----------------------------------------------------------------------------
const std::string &SpineMLGenerator::Aliases::getAliasCode(const std::string &name) const
{
    auto alias = m_Aliases.find(name);
    if(alias == m_Aliases.end()) {
        throw std::runtime_error("Cannot find alias '" + name + "'");
    }
    else {
       return alias->second.code;
    }
}

//----------------------------------------------------------------------------
// Helper functions
//----------------------------------------------------------------------------
std::pair<bool, unsigned int> SpineMLGenerator::generateModelCode(const pugi::xml_node &componentClass,
                                                                  const std::map<std::string, ObjectHandler::Base*> &objectHandlerEvent,
                                                                  ObjectHandler::Base *objectHandlerCondition,
                                                                  const std::map<std::string, ObjectHandler::Base*> &objectHandlerImpulse,
                                                                  ObjectHandler::Base *objectHandlerTimeDerivative,
                                                                  std::function<void(bool, unsigned int)> regimeEndFunc)
{
    LOGD_SPINEML << "\t\tModel name:" << componentClass.attribute("name").value();

    // Build mapping from regime names to IDs
    auto dynamics = componentClass.child("Dynamics");
    std::map<std::string, unsigned int> regimeIDs;
    std::transform(dynamics.children("Regime").begin(), dynamics.children("Regime").end(),
                   std::inserter(regimeIDs, regimeIDs.end()),
                   [&regimeIDs](const pugi::xml_node &n)
                   {
                       return std::make_pair(n.attribute("name").value(), (unsigned int)regimeIDs.size());
                   });
    const bool multipleRegimes = (regimeIDs.size() > 1);

    // Loop through regimes
    LOGD_SPINEML << "\t\tRegimes:";
    for(auto regime : dynamics.children("Regime")) {
        const auto *currentRegimeName = regime.attribute("name").value();
        const unsigned int currentRegimeID = regimeIDs[currentRegimeName];
        LOGD_SPINEML << "\t\t\tRegime name:" << currentRegimeName << ", id:" << currentRegimeID;

        // Loop through internal conditions by which model might leave regime
        for(auto condition : regime.children("OnCondition")) {
            if(objectHandlerCondition) {
                const auto *targetRegimeName = condition.attribute("target_regime").value();
                const unsigned int targetRegimeID = regimeIDs[targetRegimeName];
                objectHandlerCondition->onObject(condition, currentRegimeID, targetRegimeID);
            }
            else {
                throw std::runtime_error("No handler for OnCondition in models of type '"
                                         + std::string(componentClass.attribute("type").value()));
            }
        }

        // Loop through events the model might receive
        for(auto event : regime.children("OnEvent")) {
            // Search for object handler matching source port
            const auto *srcPort = event.attribute("src_port").value();
            auto objectHandler = objectHandlerEvent.find(srcPort);
            if(objectHandler != objectHandlerEvent.end()) {
                const auto *targetRegimeName = event.attribute("target_regime").value();
                const unsigned int targetRegimeID = regimeIDs[targetRegimeName];
                objectHandler->second->onObject(event, currentRegimeID, targetRegimeID);
            }
            else {
                throw std::runtime_error("No handler for events from source port '" + std::string(srcPort)
                                         + "' to model of type '" + componentClass.attribute("type").value());
            }
        }

        // Loop through impulses the model might receive
        for(auto impulse : regime.children("OnImpulse")) {
            // Search for object handler matching source port
            const auto *srcPort = impulse.attribute("src_port").value();
            auto objectHandler = objectHandlerImpulse.find(srcPort);
            if(objectHandler != objectHandlerImpulse.end()) {
                const auto *targetRegimeName = impulse.attribute("target_regime").value();
                const unsigned int targetRegimeID = regimeIDs[targetRegimeName];
                objectHandler->second->onObject(impulse, currentRegimeID, targetRegimeID);
            }
            else {
                throw std::runtime_error("No handler for impulses from source port '" + std::string(srcPort)
                                         + "' to model of type '" + componentClass.attribute("type").value());
            }
        }

        // Write out time derivatives
        for(auto timeDerivative : regime.children("TimeDerivative")) {
            if(objectHandlerTimeDerivative) {
                objectHandlerTimeDerivative->onObject(timeDerivative, currentRegimeID, 0);
            }
            else {
                throw std::runtime_error("No handler for TimeDerivative in models of type '"
                                         + std::string(componentClass.attribute("type").value()));
            }
        }

        // Call function to notify all code streams of end of regime
        regimeEndFunc(multipleRegimes, currentRegimeID);
    }

    // Search for initial regime
    auto initialRegime = regimeIDs.find(dynamics.attribute("initial_regime").value());
    if(initialRegime == regimeIDs.end()) {
        throw std::runtime_error("No initial regime set");
    }

    LOGD_SPINEML << "\t\t\tInitial regime ID:" << initialRegime->second;

    // Return whether this model has multiple regimes and what the ID of the initial regime is
    return std::make_pair(multipleRegimes, initialRegime->second);
}
//----------------------------------------------------------------------------
void SpineMLGenerator::wrapAndReplaceVariableNames(std::string &code, const std::string &variableName,
                                                   const std::string &replaceVariableName)
{
    // Replace variable name with replacement variable name, within GeNN $(XXXX) wrapper
    CodeGenerator::regexVarSubstitute(code, variableName, "$(" + replaceVariableName + ")");
}
//----------------------------------------------------------------------------
void SpineMLGenerator::wrapVariableNames(std::string &code, const std::string &variableName)
{
    wrapAndReplaceVariableNames(code, variableName, variableName);
}
//----------------------------------------------------------------------------
std::tuple<Models::Base::StringVec, Models::Base::VarVec> SpineMLGenerator::findModelVariables(
    const pugi::xml_node &componentClass, const std::set<std::string> &variableParams, bool multipleRegimes)
{
    // Starting with those the model needs to vary, create a set of genn variables
    std::set<std::string> gennVariables(variableParams);

    // Add model state variables to this
    auto dynamics = componentClass.child("Dynamics");
    std::transform(dynamics.children("StateVariable").begin(), dynamics.children("StateVariable").end(),
                   std::inserter(gennVariables, gennVariables.end()),
                   [](const pugi::xml_node &n){ return n.attribute("name").value(); });

    // Loop through model parameters
    Models::Base::StringVec paramNames;
    for(auto param : componentClass.children("Parameter")) {
        // If parameter hasn't been declared variable by model, add it to vector of parameter names
        std::string paramName = param.attribute("name").value();
        if(gennVariables.find(paramName) == gennVariables.end()) {
            paramNames.push_back(paramName);
        }
    }

    // Add all GeNN variables
    Models::Base::VarVec vars;
    std::transform(gennVariables.begin(), gennVariables.end(), std::back_inserter(vars),
                   [](const std::string &vname){ return Models::Base::Var{vname, "scalar"}; });

    // If model has multiple regimes, add unsigned int regime ID to values
    if(multipleRegimes) {
        vars.push_back({"_regimeID", "unsigned int"});
    }

    // Return parameter names and variables
    return std::make_tuple(paramNames, vars);
}
//----------------------------------------------------------------------------
void SpineMLGenerator::substituteModelVariables(const Models::Base::StringVec &paramNames,
                                                const Models::Base::VarVec &vars,
                                                const Models::Base::DerivedParamVec &derivedParams,
                                                const std::vector<std::string*> &codeStrings)
{
    // Loop through model parameters
    LOGD_SPINEML << "\t\tParameters:";
    for(const auto &p : paramNames) {
        LOGD_SPINEML << "\t\t\t" << p;

        // Wrap variable names so GeNN code generator can find them
        for(std::string *c : codeStrings) {
            wrapVariableNames(*c, p);
        }
    }

    LOGD_SPINEML << "\t\tVariables:";
    for(const auto &v : vars) {
        LOGD_SPINEML << "\t\t\t" << v.name << ":" << v.type;

        // Wrap variable names so GeNN code generator can find them
        for(std::string *c : codeStrings) {
            wrapVariableNames(*c, v.name);
        }
    }

    LOGD_SPINEML << "\t\tDerived params:";
    for(const auto &d : derivedParams) {
        LOGD_SPINEML << "\t\t\t" << d.name;

        // Wrap derived param names so GeNN code generator can find them
        for(std::string *c : codeStrings) {
            wrapVariableNames(*c, d.name);
        }
    }

    // Loop throug code strings to perform some standard substitutions
    for(std::string *c : codeStrings) {
        // Wrap time
        wrapVariableNames(*c, "t");

        // Replace standard functions with their GeNN equivalent so GeNN code
        // generator can correcly insert platform-specific versions
        wrapAndReplaceVariableNames(*c, "randomNormal", "gennrand_normal");
        wrapAndReplaceVariableNames(*c, "randomUniform", "gennrand_uniform");

        // **TODO** random.binomial(N,P), random.poisson(L), random.exponential(L)
    }
}
