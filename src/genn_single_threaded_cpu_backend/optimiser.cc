#include "optimiser.h"

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
namespace Optimiser
{
Backend createBackend(const ModelSpecInternal &, const filesystem::path &, int localHostID,
                      const Preferences &preferences)
{
    return Backend(localHostID, preferences);
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
