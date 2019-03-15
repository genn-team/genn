#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// Forward declarations
class ModelSpecInternal;

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
    GENN_EXPORT std::vector<std::string> generateAll(const ModelSpecInternal &model, const BackendBase &backend, const filesystem::path &outputPath);
}
