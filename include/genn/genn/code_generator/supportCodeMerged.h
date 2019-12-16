#pragma once

// Standard C++ includes
#include <unordered_set>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

//--------------------------------------------------------------------------
// CodeGenerator::SupportCodeMerged
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class SupportCodeMerged
{
public:
    SupportCodeMerged(const std::string &namespacePrefix) : m_NamespacePrefix(namespacePrefix)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Add support code string, returning namespace that should be used to access it
    void addSupportCode(const std::string &code)
    {
        // If there is any support code
        if(!code.empty()) {
            // Try and add code to set, assuming that a new namespace will be required
            // **NOTE** namespace name is NOT hashed or compared
            const size_t numStrings = m_SupportCode.size();
            m_SupportCode.emplace(m_NamespacePrefix + std::to_string(numStrings), code);
        }
    }

    const std::string &getSupportCodeNamespace(const std::string &code) const
    {
        // Create dummy structure only containing code
        // **NOTE** namespace name does not get hashed or compared
        const SupportCode dummySupportCode{"", code};
        const auto s = m_SupportCode.find(dummySupportCode);
        assert(s != m_SupportCode.cend());

        // Return the name of the namespace which should be included to use it
        return s->namespaceName;
    }

    //! Generate support code
    void gen(CodeStream &os, const std::string &ftype) const
    {
        // Loop through support code
        for(const auto &s : m_SupportCode) {
            // Write namespace containing support code with fixed up floating point type
            os << "namespace " << s.namespaceName;
            {
                CodeStream::Scope b(os);
                os << ensureFtype(s.supportCode, ftype) << std::endl;
            }
            os << std::endl;
        }
    }

    size_t getNumSupportCodeString() const{ return m_SupportCode.size(); }

private:
    //------------------------------------------------------------------------
    // SupportCode
    //------------------------------------------------------------------------
    //! Struct containing unique support code string and namespace it is generated in
    struct SupportCode
    {
        SupportCode(const std::string &n, const std::string &s) : namespaceName(n), supportCode(s){}

        const std::string namespaceName;
        const std::string supportCode;

        bool operator == (const SupportCode &other) const
        {
            return (other.supportCode == supportCode);
        }
    };

    //------------------------------------------------------------------------
    // SupportCodeHash
    //------------------------------------------------------------------------
    //! Functor used for hashing support code strings
    struct SupportCodeHash
    {
        size_t operator() (const SupportCode &supportCode) const
        {
            std::hash<std::string> stringHash;
            return stringHash(supportCode.supportCode);
        }
    };

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Set of unique support code strings
    std::unordered_set<SupportCode, SupportCodeHash> m_SupportCode;

    // Prefix
    const std::string m_NamespacePrefix;
};
}   // namespace CodeGenerator
