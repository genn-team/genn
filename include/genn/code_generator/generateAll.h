#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// Forward declarations
class ModelSpec;

namespace CodeGenerator
{
class BackendBase;
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
    std::vector<std::string> generateAll(const ModelSpec &model, const BackendBase &backend, const filesystem::path &outputPath);
}