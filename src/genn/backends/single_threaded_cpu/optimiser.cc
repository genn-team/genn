#include "optimiser.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
namespace Optimiser
{
Backend createBackend(const ModelSpecInternal &model, const filesystem::path &,
                      int localHostID, const Preferences &preferences)
{
    return Backend(localHostID, model.getPrecision(), preferences);
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
