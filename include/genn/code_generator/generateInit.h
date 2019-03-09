#pragma once

// Forward declarations
class ModelSpec;

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
void generateInit(CodeStream &os, const ModelSpec &model, const BackendBase &backend);
}