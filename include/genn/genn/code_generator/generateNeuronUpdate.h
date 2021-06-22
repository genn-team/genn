#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "gennExport.h"

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
GENN_EXPORT void generateNeuronUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                      const BackendBase &backend, const std::string &suffix = "");
}
