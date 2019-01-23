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
void generateSynapseUpdate(CodeStream &os, const NNmodel &model, const BackendBase &backend);
}