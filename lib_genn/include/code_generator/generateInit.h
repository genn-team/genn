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
void generateInit(CodeStream &os, const NNmodel &model, const Backends::Base &backend);
}