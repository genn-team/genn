// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// Forward declarations
namespace GeNN
{
class ModelSpecInternal;

namespace CodeGenerator
{
class BackendBase;
}
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
GENN_EXPORT void generateMSBuild(std::ostream &os, const ModelSpecInternal &model, const BackendBase &backend, 
                                 const std::vector<std::string> &moduleNames);
}
