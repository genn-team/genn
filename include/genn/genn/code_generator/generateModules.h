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

GENN_EXPORT void generateNeuronUpdate(std::ostream &stream, ModelSpecMerged &modelMerged, const BackendBase &backend,
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateCustomUpdate(std::ostream &stream, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateSynapseUpdate(std::ostream &stream, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                       BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateInit(std::ostream &stream, ModelSpecMerged &modelMerged, const BackendBase &backend, 
                              BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");
}
