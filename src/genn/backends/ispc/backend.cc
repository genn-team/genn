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
    GeNN::updateHash(targetISA, hash);
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
             VarLocation location, bool uninitialized, size_t alignment)
:   Runtime::ArrayBase(type, count, location, uninitialized), m_Alignment(alignment)
{
    if(count > 0) {
        allocate(count);
    }
}
    
Array::~Array()
{
    if(getCount() > 0) {
        free();
    }
}

void Array::allocate(size_t count)
{
    setCount(count);
    const size_t sizeBytes = getSizeBytes();
    
    // Using std::aligned_alloc
    // Size must be a multiple of alignment
    const size_t alignedSizeBytes = padSize(sizeBytes, m_Alignment);
    setHostPointer(reinterpret_cast<std::byte*>(std::aligned_alloc(m_Alignment, alignedSizeBytes)));
    if (!getHostPointer()) {
        throw std::bad_alloc();
    }
}

void Array::free()
{
    // std::free is used to deallocate memory allocated by std::aligned_alloc
    std::free(getHostPointer());
    setHostPointer(nullptr);
    setCount(0);
}

void Array::pushToDevice()
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pullFromDevice()
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pushSlice1DToDevice(size_t, size_t)
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::pullSlice1DFromDevice(size_t, size_t)
{
    // ISPC runs on the CPU (host), so no transfer is needed
}

void Array::memsetDeviceObject(int)
{
    throw std::runtime_error("ISPC arrays have no device object");
}

void Array::serialiseDeviceObject(std::vector<std::byte>&, bool) const
{
    throw std::runtime_error("ISPC arrays have no device object");
}

void Array::serialiseHostObject(std::vector<std::byte>& result, bool) const
{
    const size_t sizeBytes = getSizeBytes();
    
    result.resize(sizeBytes);
    
    if(sizeBytes > 0) {
        std::memcpy(result.data(), getHostPointer(), sizeBytes);
    }
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Backend
//--------------------------------------------------------------------------
Backend::Backend()
{
    setPreferencesBase(std::make_shared<Preferences>());
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
    const auto &prefs = getPreferences<Preferences>();
    
    // Determine alignment based on target ISA
    // AVX-512 requires 64-byte alignment, AVX/AVX2 use 32, SSE uses 16
    size_t alignment = 16;
    if(prefs.targetISA.find("avx512") != std::string::npos) {
        alignment = 64;
    }
    else if(prefs.targetISA.find("avx") != std::string::npos) {
        alignment = 32;
    }
    
    return std::make_unique<Array>(type, count, location, uninitialized, alignment);
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
