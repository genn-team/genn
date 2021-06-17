#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
namespace CodeGenerator
{
class ModelSpecMerged;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateCustomUpdate(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                          const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
