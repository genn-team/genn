#pragma once

// Forward declarations
class ModelSpec;

namespace CodeGenerator
{
class CodeStream;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateSupportCode(CodeStream &os, const ModelSpec &model);
}
