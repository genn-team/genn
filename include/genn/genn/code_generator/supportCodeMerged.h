#pragma once

// Standard C++ includes
#include <unordered_map>

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
            m_SupportCode.emplace(code, m_NamespacePrefix + std::to_string(numStrings));
        }
    }

    const std::string &getSupportCodeNamespace(const std::string &code) const
    {
        const auto s = m_SupportCode.find(code);
        assert(s != m_SupportCode.cend());

        // Return the name of the namespace which should be included to use it
        return s->second;
    }

    //! Generate support code
    void gen(CodeStream &os, const std::string &ftype) const
    {
        // Loop through support code
        for(const auto &s : m_SupportCode) {
            // Write namespace containing support code with fixed up floating point type
            os << "namespace " << s.second;
            {
                CodeStream::Scope b(os);
                os << ensureFtype(s.first, ftype) << std::endl;
            }
            os << std::endl;
        }
    }

    size_t getNumSupportCodeString() const{ return m_SupportCode.size(); }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Map of support code strings to namespace names
    std::unordered_map<std::string, std::string> m_SupportCode;

    // Prefix
    const std::string m_NamespacePrefix;
};
}   // namespace CodeGenerator
