#pragma once

// GeNN includes
#include "gennExport.h"

// Standard C++ includes
#include <string>

// Forward declarations
namespace CodeGenerator
{
class BackendBase;
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
GENN_EXPORT void generateSynapseUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                       const BackendBase &backend, const std::string &suffix = "");
}
