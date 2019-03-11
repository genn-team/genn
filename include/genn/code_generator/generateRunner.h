#pragma once

// Forward declarations
class ModelSpecInternal;

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
                    const ModelSpecInternal &model, const BackendBase &backend, int localHostID);
}
