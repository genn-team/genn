#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "gennExport.h"

// Forward declarations
class ModelSpecInternal;

namespace CodeGenerator
{
class BackendBase;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void GENN_EXPORT generateMakefile(std::ostream &os, const BackendBase &backend,
                                  const std::vector<std::string> &moduleNames);
}
