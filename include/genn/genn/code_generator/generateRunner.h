#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"

// Forward declarations
namespace CodeGenerator
{
class CodeStream;
class ModelSpecMerged;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
MemAlloc generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                        MergedStructData &mergedStructData, const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
