#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
namespace CodeGenerator
{
class ModelSpecMerged;
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
GENN_EXPORT void generateRunner(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                const BackendBase &backend, const std::string &suffix = "");
}
