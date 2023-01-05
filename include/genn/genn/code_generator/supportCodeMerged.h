#pragma once

// Standard C++ includes
#include <unordered_map>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::SupportCodeMerged
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class SupportCodeMerged
{
public:
    SupportCodeMerged(const std::string &namespacePrefix) : m_NamespacePrefix(namespacePrefix)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Add support code string
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

    //! Gets the name of the support code namespace which should be 'used' to provide this support code
    const std::string &getSupportCodeNamespace(const std::string &code) const
    {
        const auto s = m_SupportCode.find(code);
        assert(s != m_SupportCode.cend());

        // Return the name of the namespace which should be included to use it
        return s->second;
    }

    //! Generate support code
    void gen(CodeStream &os, const std::string &ftype, const bool supportsNamespace = true) const
    {
        // Loop through support code
        for(const auto &s : m_SupportCode) {
            if (supportsNamespace) {
                // Write namespace containing support code with fixed up floating point type
                os << "namespace " << s.second;
                {
                    CodeStream::Scope b(os);
                    os << ensureFtype(s.first, ftype) << std::endl;
                }
            }
            else {
                // Regex for function definition - looks for words with succeeding parentheses with or without any data inside the parentheses (arguments) followed by braces on the same or new line
                std::regex r("\\w+(?=\\(.*\\)\\s*\\{)");
                os << ensureFtype(std::regex_replace(s.first, r, s.second + "_$&"), ftype) << std::endl;
            }
            os << std::endl;
        }
    }

    //! Gets the number of support code strings hence namespaces which will be generated
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
}   // namespace GeNN::CodeGenerator
