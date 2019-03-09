// Standard C++ includes
#include <string>
#include <vector>

// Forward declarations
class ModelSpec;

namespace CodeGenerator
{
class BackendBase;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateMSBuild(std::ostream &os, const BackendBase &backend, const std::string &projectGUID,
                     const std::vector<std::string> &moduleNames);
}