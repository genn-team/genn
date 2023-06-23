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
                [&stream](const std::pair<std::reference_wrapper<EnvironmentExternalBase>, std::string> &env)
                { 
                    stream << env.first.get()[env.second];
                }},
            e);
    }
    return stream.str();
}
    