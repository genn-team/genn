#pragma once

// Forward declarations
namespace CodeGenerator
{
class CodeStream;
class ModelSpecMerged;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateSupportCode(CodeStream &os, const ModelSpecMerged &modelMerged);
}
