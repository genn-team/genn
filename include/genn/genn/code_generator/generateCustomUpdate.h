#pragma once

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
GENN_EXPORT void generateCustomUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
