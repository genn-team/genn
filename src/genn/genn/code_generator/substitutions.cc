#include "code_generator/substitutions.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"

//--------------------------------------------------------------------------
// CodeGenerator::Substitutions
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::addParamValueSubstitution(const std::vector<std::string> &paramNames, const std::unordered_map<std::string, double> &values,
                                                             const std::string &sourceSuffix)
{
    if(paramNames.size() != values.size()) {
        throw std::runtime_error("Number of parameters does not match number of values");
    }

    for(const auto &p : paramNames) {
        addVarSubstitution(p + sourceSuffix,
                           "(" + Utils::writePreciseString(values.at(p)) + ")");
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::addVarSubstitution(const std::string &source, const std::string &destionation, bool allowOverride)
{
    auto res = m_VarSubstitutions.emplace(source, destionation);
    if(!allowOverride && !res.second) {
        throw std::runtime_error("'" + source + "' already has a variable substitution");
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::addFuncSubstitution(const std::string &source, unsigned int numArguments, 
                                                       const std::string &funcTemplate, bool allowOverride)
{
    auto res = m_FuncSubstitutions.emplace(std::piecewise_construct,
                                           std::forward_as_tuple(source),
                                           std::forward_as_tuple(numArguments, funcTemplate));
    if(!allowOverride && !res.second) {
        throw std::runtime_error("'" + source + "' already has a function substitution");
    }
}
//--------------------------------------------------------------------------
bool CodeGenerator::Substitutions::hasVarSubstitution(const std::string &source) const
{
    return (m_VarSubstitutions.find(source) != m_VarSubstitutions.end());
}
//--------------------------------------------------------------------------
const std::string &CodeGenerator::Substitutions::getVarSubstitution(const std::string &source) const
{
    auto var = m_VarSubstitutions.find(source);
    if(var != m_VarSubstitutions.end()) {
        return var->second;
    }
    else if(m_Parent) {
        return m_Parent->getVarSubstitution(source);
    }
    else {
        throw std::runtime_error("Nothing to substitute for '" + source + "'");
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::apply(std::string &code) const
{
    // Apply function and variable substitutions
    // **NOTE** functions may contain variables so evaluate ALL functions first
    applyFuncs(code);
    applyVars(code);
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::applyCheckUnreplaced(std::string &code, const std::string &context) const
{
    apply(code);
    checkUnreplacedVariables(code, context);
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::applyFuncs(std::string &code) const
{
    // Apply function substitutions
    for(const auto &f : m_FuncSubstitutions) {
        functionSubstitute(code, f.first, f.second.first, f.second.second);
    }

    // If we have a parent, apply their function substitutions too
    if(m_Parent) {
        m_Parent->applyFuncs(code);
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::Substitutions::applyVars(std::string &code) const
{
    // Apply variable substitutions
    for(const auto &v : m_VarSubstitutions) {
        LOGD_CODE_GEN << "Substituting '$(" << v.first << ")' for '" << v.second << "'";
        substitute(code, "$(" + v.first + ")", v.second);
    }

    // If we have a parent, apply their variable substitutions too
    if(m_Parent) {
        m_Parent->applyVars(code);
    }
}