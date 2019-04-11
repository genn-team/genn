#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
class ModelSpecInternal;

namespace CodeGenerator
{
class CodeStream;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
MemAlloc generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                        const ModelSpecInternal &model, const BackendBase &backend, int localHostID);
}
