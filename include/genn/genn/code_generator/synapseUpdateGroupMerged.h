#pragma once

// GeNN includes
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT SynapseGroupMergedBase : public GroupMerged<SynapseGroupInternal>
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    std::string getPreSlot(bool delay, unsigned int batchSize) const;
    std::string getPostSlot(bool delay, unsigned int batchSize) const;

    std::string getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const;

    std::string getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
    {
        return getPrePostVarIndex(delay, batchSize, varDims, index, "pre");
    }

    std::string getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
    {
        return getPrePostVarIndex(delay, batchSize, varDims, index, "post");
    }

    std::string getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    
    std::string getPostISynIndex(unsigned int batchSize, const std::string &index) const
    {
        return ((batchSize == 1) ? "" : "$(_post_batch_offset) + ") + index;
    }

    std::string getPreISynIndex(unsigned int batchSize, const std::string &index) const
    {
        return ((batchSize == 1) ? "" : "$(_pre_batch_offset) + ") + index;
    }

    std::string getPostVarHetDelayIndex(unsigned int batchSize, VarAccessDim varDims,
                                        const std::string &index) const;

    std::string getSynVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getKernelVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

protected:
    using GroupMerged::GroupMerged;
    

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    std::string getPrePostVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims,
                                   const std::string &index, const std::string &prefix) const;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PresynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    using SynapseGroupMergedBase::SynapseGroupMergedBase;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                  unsigned int batchSize, double dt);
    void generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                             unsigned int batchSize, double dt);
    void generateProceduralConnectivity(EnvironmentExternalBase &env);
    void generateToeplitzConnectivity(EnvironmentExternalBase &env,
                                      Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                                      Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PostsynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    using SynapseGroupMergedBase::SynapseGroupMergedBase;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSpikeEventUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                  unsigned int batchSize, double dt);
    void generateSpikeUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                             unsigned int batchSize, double dt);
    
    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDynamicsGroupMerged : public SynapseGroupMergedBase
{
public:
    using SynapseGroupMergedBase::SynapseGroupMergedBase;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSynapseUpdate(const BackendBase &backend, EnvironmentExternalBase &env, 
                               unsigned int batchSize, double dt);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDendriticDelayUpdateGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    using GroupMerged::GroupMerged;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSynapseUpdate(EnvironmentExternalBase &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
