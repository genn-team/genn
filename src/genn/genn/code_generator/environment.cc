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
    // If context includes a code stream
    if(std::get<2>(m_Context)) {
        return *std::get<2>(m_Context);
    }
    // Otherwise
    else {
        // Assert that there is a pretty printing environment
        assert(std::get<1>(m_Context));

        // Return its stream
        return std::get<1>(m_Context)->getStream();
    }
}
//----------------------------------------------------------------------------
std::string EnvironmentExternalBase::getContextName(const std::string &name, std::optional<Type::ResolvedType> type) const
{
    // If context includes a pretty-printing environment, get name from it
    if(std::get<1>(m_Context)) {
        return std::get<1>(m_Context)->getName(name, type);
    }
    // Otherwise, give error
    else {
        throw std::runtime_error("Identifier '" + name + "' undefined"); 
    }

}
//----------------------------------------------------------------------------
std::vector<Type::ResolvedType> EnvironmentExternalBase::getContextTypes(const Transpiler::Token &name, Transpiler::ErrorHandlerBase &errorHandler)  const
{
    // If context includes a type-checking environment, get type from it
    if(std::get<0>(m_Context)) {
        return std::get<0>(m_Context)->getTypes(name, errorHandler); 
    }
    // Otherwise, give error
    else {
        errorHandler.error(name, "Undefined identifier");
        throw TypeChecker::TypeCheckError();
    }
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