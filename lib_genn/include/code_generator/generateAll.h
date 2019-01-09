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

namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
    std::vector<std::string> generateAll(const NNmodel &model, const Backends::Base &backend, const filesystem::path &outputPath);
}