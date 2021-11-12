#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
class SynapseGroupInternal;

namespace CodeGenerator
{
class BackendSIMT;
class ModelSpecMerged;
}

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::Base
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace PresynapticUpdateStrategySIMT
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const = 0;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const = 0;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const = 0;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const = 0;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const = 0;
};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PreSpan
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PostSpan
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const override;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Are input currents emitted by this presynaptic update accumulated into a register?
    bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const;

};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanBitmask
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PreSpanProcedural
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const override;
};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanToeplitz
//--------------------------------------------------------------------------
//! Postsynaptic parallelism for Toeplitz connectivity
class PostSpanToeplitz : public Base
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
    virtual bool isCompatible(const SynapseGroupInternal &sg, const PreferencesBase &preferences) const override;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const override;

    virtual void genPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                             const Substitutions &popSubs, const BackendSIMT &backend) const override;

    //! Generate presynaptic update code
    virtual void genUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                           const Substitutions &popSubs, const BackendSIMT &backend, bool trueSpike) const override;

    virtual void genPostamble(CodeStream &os, const ModelSpecMerged &modelMerged, const PresynapticUpdateGroupMerged &sg,
                              const Substitutions &popSubs, const BackendSIMT &backend) const override;
};
}   // namespace PresynapticUpdateStrategySIMT
}   // namespace CodeGenerator
