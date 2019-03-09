#pragma once

// Single-threaded CPU backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
namespace Optimiser
{
Backend createBackend(const ModelSpec &, const filesystem::path &, int localHostID,
                      const Backend::Preferences &preferences)
{
    Backend backend(localHostID, preferences);
    return std::move(backend);
}
}
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator