#pragma once

// Forward declarations
class CodeStream;
class NNmodel;

namespace CodeGenerator
{
namespace Backends
{
    class Base;
}
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateRunner(CodeStream &definitions, CodeStream &runner, const NNmodel &model,
                    const Backends::Base &backend, int localHostID);
}