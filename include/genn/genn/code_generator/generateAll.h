#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// Forward declarations
class ModelSpecMerged;

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
    GENN_EXPORT std::vector<std::string> generateAll(const ModelSpecMerged &model, const BackendBase &backend, const filesystem::path &outputPath, bool standaloneModules=false);
}
