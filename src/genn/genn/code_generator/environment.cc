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