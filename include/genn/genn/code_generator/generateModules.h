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
using FileStreamCreator = std::function<std::ostream &(const std::string &, const std::string &)>;

GENN_EXPORT std::vector<std::string> generateAll(ModelSpecMerged &modelMerged, const BackendBase &backend, 
                                                 const filesystem::path &outputPath, bool alwaysRebuild = false, 
                                                 bool neverRebuild = false);

GENN_EXPORT void generateNeuronUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateCustomUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                                      BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateSynapseUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                                       BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");

GENN_EXPORT void generateInit(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                              BackendBase::MemorySpaces &memorySpaces, const std::string &suffix = "");
}
