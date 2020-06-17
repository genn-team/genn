#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
class ModelSpecMerged;
class PresynapticUpdateGroupMerged;

namespace CodeGenerator
{
namespace OpenCL
{
class Backend;
}
}

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::PresynapticUpdateStrategy::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
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
    virtual bool isCompatible(const SynapseGroupInternal &sg) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const = 0;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler) const = 0;
};

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::PresynapticUpdateStrategy::PreSpan
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
    virtual bool isCompatible(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::PresynapticUpdateStrategy::PostSpan
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
    virtual bool isCompatible(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const PresynapticUpdateGroupMerged &sg, const Backend &backend) const override;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::PresynapticUpdateGroupMergedHandler wumThreshHandler, BackendBase::PresynapticUpdateGroupMergedHandler wumSimHandler) const override;
};
}   // namespace PresynapticUpdateStrategy
}   // namespace OpenCL
}   // namespace CodeGenerator
