#include "code_generator/environment.h"

using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternal
//----------------------------------------------------------------------------
std::string EnvironmentExternal::define(const Transpiler::Token&)
{
    throw std::runtime_error("Cannot declare variable in external environment");
}
//----------------------------------------------------------------------------    
CodeStream &EnvironmentExternal::getContextStream() const
{
    return std::visit(
        Transpiler::Utils::Overload{
            [](std::reference_wrapper<EnvironmentBase> enclosing)->CodeStream& { return enclosing.get().getStream(); },
            [](std::reference_wrapper<CodeStream> os)->CodeStream& { return os.get(); }},
        getContext());
}
//----------------------------------------------------------------------------
std::string EnvironmentExternal::getContextName(const Transpiler::Token &name) const
{
    return std::visit(
        Transpiler::Utils::Overload{
            [&name](std::reference_wrapper<EnvironmentBase> enclosing)->std::string { return enclosing.get().getName(name); },
            [&name](std::reference_wrapper<CodeStream>)->std::string { throw std::runtime_error("Variable '" + name.lexeme + "' undefined"); }},
        getContext());
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentSubstitute
//----------------------------------------------------------------------------
std::string EnvironmentSubstitute::getName(const Transpiler::Token &name)
{
    // If there isn't a substitution for this name, try and get name from context
    auto sub = m_VarSubstitutions.find(name.lexeme);
    if(sub == m_VarSubstitutions.end()) {
        return getContextName(name);
    }
    // Otherwise, return substitution
    else {
        return sub->second;
    }
}
//------------------------------------------------------------------------
void EnvironmentSubstitute::addSubstitution(const std::string &source, const std::string &destination)
{
    if(!m_VarSubstitutions.emplace(source, destination).second) {
        throw std::runtime_error("Redeclaration of substitution '" + source + "'");
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentSubstituteCondInit
//----------------------------------------------------------------------------
EnvironmentSubstituteCondInit::~EnvironmentSubstituteCondInit()
{
    // Loop through substitututions
    for(const auto &v : m_VarSubstitutions) {
        // If variable has been referenced, write out initialiser
        if (std::get<0>(v.second)) {
            getContextStream() << std::get<2>(v.second) << std::endl;
        }
    }
        
    // Write contents to context stream
    getContextStream() << m_ContentsStream.str();
}
//------------------------------------------------------------------------
std::string EnvironmentSubstituteCondInit::getName(const Transpiler::Token &name)
{
    // If variable with this name isn't found, try and get name from context
    auto var = m_VarSubstitutions.find(name.lexeme);
    if(var == m_VarSubstitutions.end()) {
        return getContextName(name);
    }
    // Otherwise
    else {
        // Set flag to indicate that variable has been referenced
        std::get<0>(var->second) = true;
            
        // Add local prefix to variable name
        return std::get<1>(var->second);
    }
}

//------------------------------------------------------------------------
void EnvironmentSubstituteCondInit::addSubstitution(const std::string &source, const std::string &destination,
                                                    const std::string &initialiser)
{
    if(!m_VarSubstitutions.try_emplace(source, false, destination, initialiser).second) {
        throw std::runtime_error("Redeclaration of substitution '" + source + "'");
    }
}