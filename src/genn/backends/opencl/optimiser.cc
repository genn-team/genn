#include "optimiser.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
namespace Optimiser
{
Backend createBackend(const ModelSpecInternal& model, const filesystem::path& outputPath, int localHostID,
	const Preferences& preferences)
{
	KernelWorkGroupSize workGroupSize = std::array<size_t, KernelMax>();

    return Backend(workGroupSize, preferences, 0, "scalar", 1);
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
