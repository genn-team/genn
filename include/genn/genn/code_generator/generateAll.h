#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "backendBase.h"

// Forward declarations
class ModelSpecInternal;

namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
GENN_EXPORT std::pair<std::vector<std::string>, MemAlloc> generateAll(const ModelSpecInternal &model, const BackendBase &backend, 
                                                                      const filesystem::path &sharePath, const filesystem::path &outputPath,
                                                                      bool standaloneModules=false);
}
