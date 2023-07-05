#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class ModelSpecMerged;
}

namespace filesystem
{
class path;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
GENN_EXPORT MemAlloc generateRunner(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, 
                                    const BackendBase &backend, const std::string &suffix = "");
}
