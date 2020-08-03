#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
struct cudaDeviceProp;
class SynapseGroupInternal;

namespace CodeGenerator
{
class ModelSpecMerged;
namespace CUDA
{
class Backend;
struct Preferences;
}
}

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace PresynapticUpdateStrategy
{
class Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const = 0;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const = 0;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps, const Preferences &preferences) const = 0;

    //! How many bytes of shared memory per thread does this strategy require
    virtual size_t getNumSharedMemoryBytesPerThread(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const = 0;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const Backend &backend, size_t idStart) const = 0;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const = 0;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const Backend &backend, size_t idStart) const = 0;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpan
//--------------------------------------------------------------------------
//! Presynaptic parallelism
class PreSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps, const Preferences &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getNumSharedMemoryBytesPerThread(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                           BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
//--------------------------------------------------------------------------
//! Postsynaptic parallelism
class PostSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps, const Preferences &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getNumSharedMemoryBytesPerThread(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                           BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Are input currents emitted by this presynaptic update accumulated into a register?
    bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const;

};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpanBitmask
//--------------------------------------------------------------------------
//! Postsynaptic parallelism
class PostSpanBitmask : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps, const Preferences &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getNumSharedMemoryBytesPerThread(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                           BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::PresynapticUpdateStrategy::PreSpanProcedural
//--------------------------------------------------------------------------
//! Presynaptic parallelism with procedural connectivity
class PreSpanProcedural : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg) const override;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg) const override;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const cudaDeviceProp &deviceProps, const Preferences &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getNumSharedMemoryBytesPerThread(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const Backend &backend, bool trueSpike, size_t idStart,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, 
                           BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler,
                           BackendBase::PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const Backend &backend, size_t idStart) const override;
};
}   // namespace PresynapticUpdateStrategy
}   // namespace CUDA
}   // namespace CodeGenerator
