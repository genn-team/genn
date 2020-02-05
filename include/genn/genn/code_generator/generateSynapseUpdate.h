#pragma once

// GeNN code generator includes
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
void generateSynapseUpdate(CodeStream &os, const MergedStructData &mergedStructData, const ModelSpecMerged &modelMerged,
                           const BackendBase &backend);
}
