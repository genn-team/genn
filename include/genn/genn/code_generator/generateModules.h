#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// GeNN code generator includes
#include "backendBase.h"

// Forward declarations
namespace GeNN
{
class ModelSpecInternal;
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
GENN_EXPORT std::pair<std::vector<std::string>, MemAlloc> generateAll(const ModelSpecInternal &model, const BackendBase &backend, 
                                                                      const filesystem::path &sharePath, const filesystem::path &outputPath,
                                                                      bool forceRebuild = false);

GENN_EXPORT void generateNeuronUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                      const BackendBase &backend, const std::string &suffix = "");

GENN_EXPORT void generateCustomUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                      const BackendBase &backend, const std::string &suffix = "");

GENN_EXPORT void generateSynapseUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                       const BackendBase &backend, const std::string &suffix = "");

GENN_EXPORT void generateInit(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                              const BackendBase &backend, const std::string &suffix = "");
}
