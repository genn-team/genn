#pragma once

// Forward declarations
class CodeStream;
class NNmodel;

namespace CodeGenerator
{
class BackendBase;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateRunner(CodeStream &definitions, CodeStream &runner, const NNmodel &model,
                    const BackendBase &backend, int localHostID);
}