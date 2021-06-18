#pragma once

// Forward declarations
namespace CodeGenerator
{
class BackendBase;
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
void generateInit(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, const BackendBase &backend);
}
