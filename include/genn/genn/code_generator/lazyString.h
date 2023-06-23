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
//! Base class for external environments i.e. those defines OUTSIDE of transpiled code by code generator
namespace GeNN::CodeGenerator
{
class LazyString
{
public:
    typedef std::variant<std::string, std::pair<std::reference_wrapper<EnvironmentExternalBase>, std::string>> Element;
    typedef std::vector<Element> Payload;
    
    LazyString(const std::string &str) : m_Payload{str}
    {}
    LazyString(const char *chr) : m_Payload{chr}
    {}
    LazyString(EnvironmentExternalBase &env, const std::string &name) : m_Payload{std::make_pair(std::ref(env), name)}
    {}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Evaluate lazy string
    std::string str() const;

    
private:
    LazyString(const Payload &payload) : m_Payload(payload){}

    //----------------------------------------------------------------------------
    // Friends
    //----------------------------------------------------------------------------
    friend LazyString operator + (const LazyString& lhs, const LazyString &rhs);

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    Payload m_Payload;
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline LazyString operator + (const LazyString& lhs, const LazyString &rhs)
{
    std::vector<LazyString::Element> payload(lhs.m_Payload);
    payload.insert(payload.end(), rhs.m_Payload.cbegin(), rhs.m_Payload.cend());
    return LazyString(payload);
}

inline LazyString operator + (const char *lhs, const LazyString &rhs)
{
    return LazyString(lhs) + rhs;
}
 
inline LazyString operator + (const LazyString &lhs, const char *rhs)
{
    return lhs + LazyString(rhs);
}  
}   // namespace GeNN::CodeGenerator