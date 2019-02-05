#pragma once

// Forward declarations
class NNmodel;

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
                    const NNmodel &model, const BackendBase &backend, int localHostID);
}
