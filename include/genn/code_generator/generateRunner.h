#pragma once

// Forward declarations
class ModelSpec;

namespace CodeGenerator
{
class BackendBase;
class CodeStream;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, 
                    const ModelSpec &model, const BackendBase &backend, int localHostID);
}
