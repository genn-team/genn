#pragma once

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
MemAlloc generateRunner(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
