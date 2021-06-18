#pragma once

// Forward declarations
namespace CodeGenerator
{
class ModelSpecMerged;
}

namespace filesystem
{
class path;
}


//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateSupportCode(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged);
}
