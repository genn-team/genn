#include "code_generator/environment.h"

// Standard C++ includes
#include <algorithm>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennUtils.h"

// Transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentExternalBase
//----------------------------------------------------------------------------
std::string EnvironmentExternalBase::define(const std::string&)
{
    throw std::runtime_error("Cannot declare variable in external environment");
}

//----------------------------------------------------------------------------    
void EnvironmentExternalBase::define(const Token&, const Type::ResolvedType&, ErrorHandlerBase&)
{
    throw std::runtime_error("Cannot declare variable in external environment");
}
//----------------------------------------------------------------------------    
CodeStream &EnvironmentExternalBase::getContextStream() const
{
    return std::visit(
        Utils::Overload{
            [](std::pair<TypeChecker::EnvironmentBase*, PrettyPrinter::EnvironmentBase*> enclosing)->CodeStream& 
            { 
                assert(enclosing.second != nullptr);
                return enclosing.second->getStream(); 
            },
            [](std::reference_wrapper<CodeStream> os)->CodeStream& 
            { 
                return os.get(); 
            }},
        m_Context);
}
//----------------------------------------------------------------------------
std::string EnvironmentExternalBase::getContextName(const std::string &name, std::optional<Type::ResolvedType> type) const
{
    return std::visit(
        Utils::Overload{
            [&name, &type](std::pair<TypeChecker::EnvironmentBase*, PrettyPrinter::EnvironmentBase*> enclosing)->std::string
            { 
                assert(enclosing.second != nullptr);
                return enclosing.second->getName(name, type);
            },
            [&name](std::reference_wrapper<CodeStream>)->std::string 
            { 
                throw std::runtime_error("Identifier '" + name + "' undefined"); 
            }},
        m_Context);
}
//----------------------------------------------------------------------------
std::vector<Type::ResolvedType> EnvironmentExternalBase::getContextTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler)  const
{
    return std::visit(
        Utils::Overload{
            [&errorHandler, &name](std::pair<TypeChecker::EnvironmentBase*, PrettyPrinter::EnvironmentBase*> enclosing)->std::vector<Type::ResolvedType>
            { 
                assert(enclosing.first != nullptr);
                return enclosing.first->getTypes(name, errorHandler); 
            },
            [&errorHandler, &name](std::reference_wrapper<CodeStream>)->std::vector<Type::ResolvedType>
            { 
                errorHandler.error(name, "Undefined identifier");
                throw TypeChecker::TypeCheckError();
            }},
        m_Context);
}

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::EnvironmentLibrary
//---------------------------------------------------------------------------
std::vector<Type::ResolvedType> EnvironmentLibrary::getTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler)
{
    const auto [typeBegin, typeEnd] = m_Library.get().equal_range(name.lexeme);
    if (typeBegin == typeEnd) {
         return getContextTypes(name, errorHandler);
    }
    else {
        std::vector<Type::ResolvedType> types;
        types.reserve(std::distance(typeBegin, typeEnd));
        std::transform(typeBegin, typeEnd, std::back_inserter(types),
                       [](const auto &t) { return t.second.first; });
        return types;
    }
}
//---------------------------------------------------------------------------
std::string EnvironmentLibrary::getName(const std::string &name, std::optional<Type::ResolvedType> type)
{
    const auto [libTypeBegin, libTypeEnd] = m_Library.get().equal_range(name);
    if (libTypeBegin == libTypeEnd) {
        return getContextName(name, type);
    }
    else {
        if (!type) {
            throw std::runtime_error("Ambiguous reference to '" + name + "' but no type provided to disambiguate");
        }
        const auto libType = std::find_if(libTypeBegin, libTypeEnd,
                                          [type](const auto &t){ return t.second.first == type; });
        assert(libType != libTypeEnd);
        return libType->second.second;
    }
}
//---------------------------------------------------------------------------
CodeStream &EnvironmentLibrary::getStream()
{
    return getContextStream();
}