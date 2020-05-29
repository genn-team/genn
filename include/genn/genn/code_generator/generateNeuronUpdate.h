#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"

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
void generateNeuronUpdate(CodeStream &os, const MergedStructData &mergedStructData, BackendBase::MemorySpaces &memorySpaces,
                          const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
