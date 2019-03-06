#pragma once

// Forward declarations
class NNmodel;

namespace CodeGenerator
{
class CodeStream;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateSupportCode(CodeStream &os, const NNmodel &model);
}
