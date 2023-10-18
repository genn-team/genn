#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "backendBase.h"

// Forward declarations
namespace GeNN::CodeGenerator
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
namespace GeNN::CodeGenerator
{
GENN_EXPORT std::vector<std::string> generateAll(ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                                 const filesystem::path &sharePath, const filesystem::path &outputPath,
                                                 bool forceRebuild = false);

GENN_EXPORT void generateNeuronUpdate(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, const BackendBase &backend,
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateCustomUpdate(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateSynapseUpdate(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                       BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateInit(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                              BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");
}
