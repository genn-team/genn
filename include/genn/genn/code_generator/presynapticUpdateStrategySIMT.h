#pragma once

// GeNN code generator includes
#include "code_generator/backendBase.h"

// Forward declarations
namespace GeNN
{
class SynapseGroupInternal;

namespace CodeGenerator
{
class BackendSIMT;
}
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::Base
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
{
class Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const = 0;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const = 0;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const = 0;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const = 0;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const = 0;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const = 0;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const = 0;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpan
//--------------------------------------------------------------------------
//! Presynaptic parallelism
class PreSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpan
//--------------------------------------------------------------------------
//! Postsynaptic parallelism
class PostSpan : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Are input currents emitted by this presynaptic update accumulated into a register?
    bool shouldAccumulateInRegister(const PresynapticUpdateGroupMerged &sg) const;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanVectorised
//--------------------------------------------------------------------------
//! Postsynaptic parallelism
class PostSpanVectorised : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;

};

//--------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanBitmask
//--------------------------------------------------------------------------
//! GeNN::Postsynaptic parallelism
class PostSpanBitmask : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PreSpanProcedural
//--------------------------------------------------------------------------
//! Presynaptic parallelism with procedural connectivity
class PreSpanProcedural : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateStrategySIMT::PostSpanToeplitz
//--------------------------------------------------------------------------
//! Postsynaptic parallelism for Toeplitz connectivity
class PostSpanToeplitz : public Base
{
public:
    //------------------------------------------------------------------------
    // PresynapticUpdateStrategy::Base virtuals
    //------------------------------------------------------------------------
    //! Get the number of threads that presynaptic updates should be parallelised across
    virtual size_t getNumThreads(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                 const Type::TypeContext &typeContext) const final;

    //! Gets the stride used to access synaptic matrix rows, taking into account sparse data structure, padding etc
    virtual size_t getSynapticMatrixRowStride(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                                              const Type::TypeContext &typeContext) const final;

    //! Is this presynaptic update strategy compatible with a given synapse group?
    virtual bool isCompatible(const SynapseGroupInternal &sg, const BackendSIMT &backend,
                              const Type::TypeContext &typeContext) const final;

    //! How many neurons does each thread accumulate the outputs of into shared memory
    virtual size_t getSharedMemoryPerThread(const PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend) const final;

    virtual void genPreamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                             const BackendSIMT &backend) const final;

    //! Generate presynaptic update code
    virtual void genUpdate(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, const BackendSIMT &backend, 
                           unsigned int batchSize, double dt, bool trueSpike) const final;

    virtual void genPostamble(EnvironmentExternalBase &env, PresynapticUpdateGroupMerged &sg, 
                              const BackendSIMT &backend, unsigned int batchSize) const final;
};
}   // namespace GeNN::CodeGenerator::PresynapticUpdateStrategySIMT
