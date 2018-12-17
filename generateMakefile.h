#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// Forward declarations
class NNmodel;

namespace CodeGenerator
{
namespace Backends
{
    class Base;
}
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateMakefile(std::ostream &os, const Backends::Base &backend,
                      const std::vector<std::string> moduleNames);
}