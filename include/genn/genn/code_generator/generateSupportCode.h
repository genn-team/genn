#pragma once

// GeNN includes
#include "gennExport.h"

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
GENN_EXPORT void generateSupportCode(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged);
}
