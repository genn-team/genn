#pragma once

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
void generateSupportCode(CodeStream &os, const ModelSpecInternal &model);
}
