#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
class ModelSpecInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
namespace CUDA
{
class Backend;
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

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg, const Backend &backend) const = 0;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg, const Backend &backend) const = 0;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const = 0;
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

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg, const Backend &backend) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg, const Backend &backend) const override;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const override;
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

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a register?
    virtual bool shouldAccumulateInRegister(const SynapseGroupInternal &sg, const Backend &backend) const override;

    //! Are input currents emitted by this presynaptic update accumulated into a shared memory array?
    virtual bool shouldAccumulateInSharedMemory(const SynapseGroupInternal &sg, const Backend &backend) const override;

    //! Generate presynaptic update code
    virtual void genCode(CodeStream &os, const ModelSpecInternal &model, const SynapseGroupInternal &sg, const Substitutions &popSubs, const Backend &backend, bool trueSpike,
                         BackendBase::SynapseGroupHandler wumThreshHandler, BackendBase::SynapseGroupHandler wumSimHandler) const override;
};
}   // namespace PresynapticUpdateStrategy
}   // namespace CUDA
}   // namespace CodeGenerator
