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
void generateInit(CodeStream &os, const NNmodel &model, const BackendBase &backend);
}