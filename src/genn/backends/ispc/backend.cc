#include "backend.h"

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"

#include <vector>

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::CodeGenerator::ISPC;

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Preferences
//--------------------------------------------------------------------------
void Preferences::updateHash(boost::uuids::detail::sha1 &hash) const
{
    PreferencesBase::updateHash(hash);
}

const char *Preferences::getImportSuffix() const
{
    return "_ISPC";
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::State
//--------------------------------------------------------------------------
State::State(const GeNN::Runtime::Runtime &)
{
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Array
//--------------------------------------------------------------------------
Array::Array(const Type::ResolvedType &type, size_t count, 
             VarLocation, bool uninitialized)
:   Runtime::ArrayBase(type, count, VarLocation::HOST, uninitialized)
{
}
    
Array::~Array()
{
}

void Array::allocate(size_t)
{
}

void Array::free()
{
}

void Array::pushToDevice()
{
}

void Array::pullFromDevice()
{
}

void Array::pushSlice1DToDevice(size_t, size_t)
{
}

void Array::pullSlice1DFromDevice(size_t, size_t)
{
}

void Array::memsetDeviceObject(int)
{
}

void Array::serialiseDeviceObject(std::vector<std::byte>&, bool) const
{
}

void Array::serialiseHostObject(std::vector<std::byte>&, bool) const
{
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
Backend::Backend(const Preferences &preferences)
:   BackendBase(preferences)
{
}

void Backend::genNeuronUpdate(CodeStream &, ModelSpecMerged &, BackendBase::MemorySpaces &, HostHandler) const
{
}

void Backend::genSynapseUpdate(CodeStream &, ModelSpecMerged &, BackendBase::MemorySpaces &, HostHandler) const
{
}

void Backend::genCustomUpdate(CodeStream &, ModelSpecMerged &, BackendBase::MemorySpaces &, HostHandler) const
{
}

void Backend::genInit(CodeStream &, ModelSpecMerged &, BackendBase::MemorySpaces &, HostHandler) const
{
}

size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal &) const
{
    return 0;
}

void Backend::genDefinitionsPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genRunnerPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genAllocateMemPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genFreeMemPreamble(CodeStream &, const ModelSpecMerged &) const
{
}
void Backend::genStepTimeFinalisePreamble(CodeStream &, const ModelSpecMerged &) const
{
}

std::unique_ptr<GeNN::Runtime::StateBase> Backend::createState(const Runtime::Runtime &runtime) const
{
    return std::make_unique<State>(runtime);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createArray(const Type::ResolvedType &type, size_t count, 
                                                        VarLocation location, bool uninitialized) const
{
    return std::make_unique<Array>(type, count, location, uninitialized);
}

std::unique_ptr<Runtime::ArrayBase> Backend::createPopulationRNG(size_t) const
{
    return nullptr;
}

void Backend::genLazyVariableDynamicAllocation(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genLazyVariableDynamicPush(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genLazyVariableDynamicPull(CodeStream &, const Type::ResolvedType &, const std::string &, VarLocation, const std::string &) const
{
}

void Backend::genMergedDynamicVariablePush(CodeStream &, const std::string &, size_t, const std::string &, const std::string &, const std::string &) const
{
}

std::string Backend::getMergedGroupFieldHostTypeName(const Type::ResolvedType &) const
{
    return "";
}

void Backend::genPopVariableInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genVariableInit(EnvironmentExternalBase &, const std::string &, const std::string &, HandlerEnv) const
{
}
void Backend::genSparseSynapseVariableRowInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genDenseSynapseVariableRowInit(EnvironmentExternalBase &, HandlerEnv) const
{
}
void Backend::genKernelSynapseVariableInit(EnvironmentExternalBase &, SynapseInitGroupMerged &, HandlerEnv) const
{
}
void Backend::genKernelCustomUpdateVariableInit(EnvironmentExternalBase &, CustomWUUpdateInitGroupMerged &, HandlerEnv) const
{
}

std::string Backend::getAtomicOperation(const std::string &, const std::string &,
                                       const Type::ResolvedType &, AtomicOperation) const
{
    return "";
}

void Backend::genGlobalDeviceRNG(CodeStream &, CodeStream &, CodeStream &, CodeStream &) const
{
}

void Backend::genTimer(CodeStream &, CodeStream &, CodeStream &, CodeStream &, CodeStream &, const std::string &, bool) const
{
}

void Backend::genReturnFreeDeviceMemoryBytes(CodeStream &os) const
{
    os << "return 0;" << std::endl;
}

void Backend::genAssert(CodeStream &os, const std::string &condition) const
{
    os << "assert(" << condition << ");" << std::endl;
}

void Backend::genMakefilePreamble(std::ostream &) const
{
}

void Backend::genMakefileLinkRule(std::ostream &) const
{
}

void Backend::genMakefileCompileRule(std::ostream &) const
{
}

void Backend::genMSBuildConfigProperties(std::ostream &) const
{
}

void Backend::genMSBuildImportProps(std::ostream &) const
{
}

void Backend::genMSBuildItemDefinitions(std::ostream &) const
{
}

void Backend::genMSBuildCompileModule(const std::string &, std::ostream &) const
{
}

void Backend::genMSBuildImportTarget(std::ostream &) const
{
}

bool Backend::isArrayDeviceObjectRequired() const
{
    return false;
}

bool Backend::isArrayHostObjectRequired() const
{
    return false;
}

bool Backend::isGlobalHostRNGRequired(const ModelSpecInternal &) const
{
    return true;
}

bool Backend::isGlobalDeviceRNGRequired(const ModelSpecInternal &) const
{
    return false;
}

bool Backend::isPopulationRNGInitialisedOnDevice() const
{
    return false;
}

bool Backend::isPostsynapticRemapRequired() const
{
    return true;
}

bool Backend::isHostReductionRequired() const
{
    return false;
}

size_t Backend::getDeviceMemoryBytes() const
{
    return 0;
}

BackendBase::MemorySpaces Backend::getMergedGroupMemorySpaces(const ModelSpecMerged &) const
{
    return {};
}

boost::uuids::detail::sha1::digest_type Backend::getHashDigest() const
{
    return {};
}
