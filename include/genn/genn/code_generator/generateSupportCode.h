#pragma once

// Standard C++ includes
#include <string>

// GeNN includes
#include "gennExport.h"

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
GENN_EXPORT void generateSupportCode(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                     const std::string &suffix = "");
}
