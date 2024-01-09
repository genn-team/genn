#include "code_generator/lazyString.h"

// Standard C++ includes
#include <sstream>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/environment.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::LazyString
//----------------------------------------------------------------------------
LazyString::LazyString(const std::string &format, EnvironmentExternalBase &env)
{
    // Create regex iterator to iterate over $(XXX) style varibles in format string
    // **NOTE** this doesn't match function argument $(0)
    std::regex regex("\\$\\(([a-zA-Z_][\\w]+)\\)");
    std::sregex_iterator matchesBegin(format.cbegin(), format.cend(), regex);
    std::sregex_iterator matchesEnd;
    
    // If there are no matches, use format directly as payload
    if(matchesBegin == matchesEnd) {
        m_Payload.push_back(format);
    }
    // Otherwise
    else {
        // Loop through matches to build lazy string payload
        for(std::sregex_iterator m = matchesBegin;;) {
            // Copy the non-matched subsequence (m->prefix()) onto payload
            m_Payload.push_back(std::string{m->prefix().first, m->prefix().second});
    
            // Add lazy environment reference for $(XXX) variable to payload
            m_Payload.push_back(std::make_pair(std::ref(env), (*m)[1]));
    
            // If there are no subsequent matches, add the remaining non-matched
            // characters onto payload, construct lazy string and stop
            if(std::next(m) == matchesEnd) {
                m_Payload.push_back(std::string{m->suffix().first, m->suffix().second});
                break;
            }
            // Otherwise go onto next match
            else {
                m++;
            }
        }
    }
}
//----------------------------------------------------------------------------
std::string LazyString::str() const
{
    std::ostringstream stream;
    for(const auto &e : m_Payload) 
    {
        std::visit(
            Utils::Overload{
                [&stream](const std::string &str)
                { 
                    stream << str;
                },
                [&stream](const std::pair<std::reference_wrapper<EnvironmentExternalBase>, std::string> &envRef)
                { 
                    stream << envRef.first.get()[envRef.second];
                }},
            e);
    }
    return stream.str();
}