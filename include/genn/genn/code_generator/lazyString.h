#pragma once

// Standard C++ includes
#include <string>
#include <variant>
#include <vector>

// Forward declarations
namespace GeNN::CodeGenerator
{
class EnvironmentExternalBase;
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::LazyString
//----------------------------------------------------------------------------
//! Lazily-evaluated string class - constructed from a format string containing $(XX) references to variables in environment
namespace GeNN::CodeGenerator
{
class LazyString
{
public:
    LazyString(const std::string &format, EnvironmentExternalBase &env);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Evaluate lazy string
    std::string str() const;

private:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::variant<std::string, std::pair<std::reference_wrapper<EnvironmentExternalBase>, std::string>> Element;
    typedef std::vector<Element> Payload;

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    Payload m_Payload;
};
}   // namespace GeNN::CodeGenerator