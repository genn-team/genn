#pragma once

// Standard C++ includes
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

// Platform includes
#ifdef _WIN32
#include <windows.h>
#else
extern "C"
{
#include <dlfcn.h>
}
#endif

// FFI includes
#include <ffi.h>

// GeNN includes
#include "gennExport.h"
#include "logging.h"
#include "modelSpecInternal.h"
#include "type.h"
#include "varAccess.h"
#include "varLocation.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"

namespace GeNN
{
class CurrentSource;
class NeuronGroup;
class SynapseGroup;
class CustomUpdateBase;
class CustomConnectivityUpdate;
}

// Forward declarations
namespace GeNN::CodeGenerator
{
class BackendBase;
class ModelSpecMerged;
}

namespace filesystem
{
class path;
}


#define IMPLEMENT_GROUP_OVERLOADS(GROUP)                                                                            \
public:                                                                                                             \
    void setDynamicParam(const GROUP &group, const std::string &paramName,                                          \
                         const Type::NumericValue &value)                                                           \
    {                                                                                                               \
        setDynamicParam(m_##GROUP##DynamicParameters.at(&group).at(paramName),                                      \
                        value);                                                                                     \
    }                                                                                                               \
    MergedDynamicFieldDestinations &getMergedParamDestinations(const GROUP &group, const std::string &paramName)    \
    {                                                                                                               \
        return m_##GROUP##DynamicParameters[&group][paramName];                                                     \
    }                                                                                                               \
    ArrayBase *getArray(const GROUP &group, const std::string &varName) const                                       \
    {                                                                                                               \
        return m_##GROUP##Arrays.at(&group).at(varName).get();                                                      \
    }                                                                                                               \
    void allocateArray(const GROUP &group, const std::string &varName, size_t count)                                \
    {                                                                                                               \
        allocateExtraGlobalParam(m_##GROUP##Arrays.at(&group), varName, count);                                     \
    }                                                                                                               \
private:                                                                                                            \
    void createArray(const GROUP *group, const std::string &varName,                                                \
                     const Type::ResolvedType &type, size_t count,                                                  \
                     VarLocation location, bool uninitialized = false)                                              \
    {                                                                                                               \
        createArray(m_##GROUP##Arrays[group],                                                                       \
                    varName, type, count, location, uninitialized);                                                 \
    }

//--------------------------------------------------------------------------
// GeNN::Runtime::ArrayBase
//--------------------------------------------------------------------------
namespace GeNN::Runtime
{
class GENN_EXPORT ArrayBase
{
public:
    virtual ~ArrayBase()
    {
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    //! Allocate array
    virtual void allocate(size_t count) = 0;

    //! Free array
    virtual void free() = 0;

    //! Copy entire array to device
    virtual void pushToDevice() = 0;

    //! Copy entire array from device
    virtual void pullFromDevice() = 0;

    //! Copy a 1D slice of elements to device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pushSlice1DToDevice(size_t offset, size_t count) = 0;

    //! Copy a 1D slice of elements from device 
    /*! \param offset   Offset in elements to start copying from
        \param count    Number of elements to copy*/
    virtual void pullSlice1DFromDevice(size_t offset, size_t count) = 0;

    //! Memset the host pointer
    virtual void memsetDeviceObject(int value) = 0;

    //! Serialise backend-specific device object to bytes
    virtual void serialiseDeviceObject(std::vector<std::byte> &bytes, bool pointerToPointer) const = 0;

    //! Serialise backend-specific host object to bytes
    virtual void serialiseHostObject(std::vector<std::byte> &bytes, bool pointerToPointer) const = 0;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const Type::ResolvedType &getType() const{ return m_Type; }
    size_t getCount() const{ return m_Count; };
    size_t getSizeBytes() const{ return m_Count * m_Type.getValue().size; };

    VarLocation getLocation() const{ return m_Location; }
    bool isUninitialized() const{ return m_Uninitialized; }

    //! Get array host pointer
    std::byte *getHostPointer() const{ return m_HostPointer; }

    template<typename T>
    T *getHostPointer() const{ return reinterpret_cast<T*>(m_HostPointer); }

    //! Memset the host pointer
    void memsetHostPointer(int value);

    //! Serialise host pointer to bytes
    void serialiseHostPointer(std::vector<std::byte> &bytes, bool pointerToPointer) const;

protected:
    ArrayBase(const Type::ResolvedType &type, size_t count,
              VarLocation location, bool uninitialized)
    :   m_Type(type), m_Count(count), m_Location(location), m_Uninitialized(uninitialized),
        m_HostPointer(nullptr)
    {
    }

    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------    
    void setCount(size_t count) { m_Count = count; }
    void setHostPointer(std::byte *hostPointer) { m_HostPointer = hostPointer; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    Type::ResolvedType m_Type;
    size_t m_Count;
    VarLocation m_Location;
    bool m_Uninitialized;

    std::byte *m_HostPointer;
};

//----------------------------------------------------------------------------
// GeNN::Runtime::MergedDynamicFieldDestinations
//----------------------------------------------------------------------------
//! Data structure for tracking fields pointing at a dynamic variable/parameter
class GENN_EXPORT MergedDynamicFieldDestinations
{
public:
    //--------------------------------------------------------------------------
    // GeNN::Runtime::MergedDynamicFieldDestinations::DynamicField
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking fields of merged group structure
    //! with dynamic values i.e. those that can be modified at runtime
    struct DynamicField
    {
        DynamicField(size_t m, const Type::ResolvedType &t, const std::string &f,
                     CodeGenerator::GroupMergedFieldType g)
        :   mergedGroupIndex(m), type(t), fieldName(f), fieldType(g) {}

        size_t mergedGroupIndex;
        Type::ResolvedType type;
        std::string fieldName;
        CodeGenerator::GroupMergedFieldType fieldType;

        //! Less than operator (used for std::set::insert), 
        //! lexicographically compares all three struct members
        bool operator < (const DynamicField &other) const
        {
            return (std::make_tuple(mergedGroupIndex, type, fieldName, fieldType) 
                    < std::make_tuple(other.mergedGroupIndex, other.type, other.fieldName, other.fieldType));
        }
    };
    
    //--------------------------------------------------------------------------
    // GeNN::Runtime::MergedDynamicFieldDestinations::MergedDynamicField
    //--------------------------------------------------------------------------
    //! Immutable structure for tracking where an extra global variable ends up after merging
    struct MergedDynamicField : public DynamicField
    {
        MergedDynamicField(size_t m, size_t i, const Type::ResolvedType &t, 
                           const std::string &f, CodeGenerator::GroupMergedFieldType g)
        :   DynamicField(m, t, f, g), groupIndex(i) {}

        size_t groupIndex;
    };

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const std::unordered_multimap<std::string, MergedDynamicField> &getDestinationFields() const
    { 
        return m_DestinationFields; 
    }

    template<typename G>
    void addDestinationField(size_t mergedGroupIndex, size_t groupIndex, 
                             const Type::ResolvedType &fieldDataType, const std::string &fieldName, 
                             CodeGenerator::GroupMergedFieldType fieldType)
    {
        // Add reference to this group's variable to data structure
        // **NOTE** this works fine with EGP references because the function to
        // get their value will just return the array associated with the referenced EGP
        m_DestinationFields.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(G::name),
            std::forward_as_tuple(mergedGroupIndex, groupIndex, fieldDataType, 
                                  fieldName, fieldType));
    }

private:
    // Members

    //! Multimap of merged group types e.g. "PostsynapticUpdate"
    // to fields within them that point to the dynamic variable/parameter
    std::unordered_multimap<std::string, MergedDynamicField> m_DestinationFields;
};

//--------------------------------------------------------------------------
// GeNN::Runtime::Runtime
//--------------------------------------------------------------------------
class GENN_EXPORT Runtime
{
    //--------------------------------------------------------------------------
    // Type defines
    //--------------------------------------------------------------------------
    using ArrayMap = std::unordered_map<std::string, std::unique_ptr<ArrayBase>>;
    
    template<typename G>
    using GroupArrayMap = std::unordered_map<const G*, ArrayMap>;

    using BatchEventArray = std::vector<std::pair<std::vector<double>, std::vector<unsigned int>>>;

    IMPLEMENT_GROUP_OVERLOADS(NeuronGroup)
    IMPLEMENT_GROUP_OVERLOADS(CurrentSource)
    IMPLEMENT_GROUP_OVERLOADS(SynapseGroup)
    IMPLEMENT_GROUP_OVERLOADS(CustomUpdateBase)
    IMPLEMENT_GROUP_OVERLOADS(CustomConnectivityUpdate)
public:
    Runtime(const filesystem::path &modelPath, const CodeGenerator::ModelSpecMerged &modelMerged, 
            const CodeGenerator::BackendBase &backend);
    Runtime(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    ~Runtime();

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Allocate memory required for model
    void allocate(std::optional<size_t> numRecordingTimesteps = std::nullopt);

    //! Initialise model
    void initialize();

    //! Initialise parts of model which rely on sparse connectivity
    void initializeSparse();

    //! Simulate one timestep
    void stepTime();

    //! Perform named custom update
    void customUpdate(const std::string &name)
    { 
        m_CustomUpdateFunctions.at(name)(getTimestep());
    }

    //! Get current simulation timestep
    uint64_t getTimestep() const{ return m_Timestep; }

    //! Set current simulation timestep
    void setTimestep(uint64_t timestep){ m_Timestep = timestep; }

    //! Get current simulation time
    double getTime() const;

    double getNeuronUpdateTime() const{ return *(double*)getSymbol("neuronUpdateTime"); }
    double getInitTime() const{ return *(double*)getSymbol("initTime"); }
    double getPresynapticUpdateTime() const{ return *(double*)getSymbol("presynapticUpdateTime"); }
    double getPostsynapticUpdateTime() const{ return *(double*)getSymbol("postsynapticUpdateTime"); }
    double getSynapseDynamicsTime() const{ return *(double*)getSymbol("synapseDynamicsTime"); }
    double getInitSparseTime() const{ return *(double*)getSymbol("initSparseTime"); }
    double getCustomUpdateTime(const std::string &name) const{ return *(double*)getSymbol("customUpdate" + name + "Time"); }
    double getCustomUpdateTransposeTime(const std::string &name) const{ return *(double*)getSymbol("customUpdate" + name + "TransposeTime"); }
    
    void pullRecordingBuffersFromDevice() const;

    //! Get delay pointer associated with neuron group
    unsigned int getDelayPointer(const NeuronGroup &group) const
    {
        return m_DelayQueuePointer.at(&group);
    }

    //! Get recorded spikes from neuron group
    BatchEventArray getRecordedSpikes(const NeuronGroup &group) const
    {
        return getRecordedEvents(group, getArray(group, "recordSpk"));
    }

    //! Get recorded spike-like events from neuron group
    BatchEventArray getRecordedSpikeEvents(const NeuronGroup &group) const
    {
        return getRecordedEvents(group, getArray(group, "recordSpkEvent"));
    }

    //! Write recorded spikes to CSV file
    void writeRecordedSpikes(const NeuronGroup &group, const std::string &path) const
    {
        return writeRecordedEvents(group, getArray(group, "recordSpk"), path);
    }

    //! Write recorded spike-like events to CSV file
    void writeRecordedSpikeEvents(const NeuronGroup &group, const std::string &path) const
    {
        return writeRecordedEvents(group, getArray(group, "recordSpkEvent"), path);
    }

private:
    //----------------------------------------------------------------------------
    // Type defines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);
    typedef void (*StepTimeFunction)(unsigned long long, unsigned long long);
    typedef void (*CustomUpdateFunction)(unsigned long long);

    //! Map of arrays to destinations in merged structures
    using MergedDynamicArrayMap = std::map<const ArrayBase*, MergedDynamicFieldDestinations>;
    
    //! Map of groups to names of dynamic parameters and their destinations
    template<typename G>
    using MergedDynamicParameterMap = std::unordered_map<const G*, std::unordered_map<std::string, MergedDynamicFieldDestinations>>;
    
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    const ModelSpecInternal &getModel() const;
    void *getSymbol(const std::string &symbolName, bool allowMissing = false) const;

    void createArray(ArrayMap &groupArrays, const std::string &varName, const Type::ResolvedType &type, 
                     size_t count, VarLocation location, bool uninitialized = false);

    BatchEventArray getRecordedEvents(const NeuronGroup &group, ArrayBase *array) const;

    void writeRecordedEvents(const NeuronGroup &group, ArrayBase *array, const std::string &path) const;

    template<typename G>
    void addMergedArrays(const G &mergedGroup)
    {
        // Loop through fields
        for(const auto &f : mergedGroup.getFields()) {
            // If field is dynamic
            if((std::get<3>(f) & CodeGenerator::GroupMergedFieldType::DYNAMIC)) {
                // Loop through groups within newly-created merged group
                for(size_t groupIndex = 0; groupIndex < mergedGroup.getGroups().size(); groupIndex++) {
                    auto g = mergedGroup.getGroups()[groupIndex];
                    std::visit(
                        Utils::Overload{
                            // If field contains an array
                            [&f, &mergedGroup, groupIndex, this](const ArrayBase *array) 
                            { 
                                // Add reference to this group's variable to data structure
                                // **NOTE** this works fine with EGP references because the function to
                                // get their value will just return the array associated with the referenced EGP
                                m_MergedDynamicArrays[array].addDestinationField<G>(
                                    mergedGroup.getIndex(), groupIndex, std::get<0>(f), 
                                    std::get<1>(f), std::get<3>(f));
                            },
                            // Otherwise, if it cotnains a dynamic parameter
                            [&f, &mergedGroup, groupIndex](std::pair<Type::NumericValue, MergedDynamicFieldDestinations&> value)
                            {
                                value.second.addDestinationField<G>(
                                    mergedGroup.getIndex(), groupIndex, std::get<0>(f), 
                                    std::get<1>(f), std::get<3>(f));
                            },
                            [](const Type::NumericValue&) 
                            {
                                assert(false);
                            }},
                        std::get<2>(f)(*this, g, groupIndex));
                    
                }
            }
        }
    }

    template<typename A, typename G>
    void createEGPArrays(const G *group)
    {
        A adaptor(*group);
        for(const auto &egp : adaptor.getDefs()) {
            const auto resolvedType = egp.type.resolve(getModel().getTypeContext());
            createArray(group, egp.name, resolvedType, 0, adaptor.getLoc(egp.name));
        }
    }

    template<typename A, typename G, typename S>
    void createVarArrays(const G *group, size_t batchSize, bool batched, S getSizeFn)
    {
        A adaptor(*group);
        for(const auto &var : adaptor.getDefs()) {
            const auto &varInit = adaptor.getInitialisers().at(var.name);
            const bool uninitialized = Utils::areTokensEmpty(varInit.getCodeTokens());
            const auto resolvedType = var.type.resolve(getModel().getTypeContext());
            const auto varDims = adaptor.getVarDims(var);

            const size_t numVarCopies = ((varDims & VarAccessDim::BATCH) && batched) ? batchSize : 1;
            const size_t varSize = getSizeFn(var.name, varDims);
            createArray(group, var.name, resolvedType, numVarCopies * varSize,
                        adaptor.getLoc(var.name), uninitialized);

            // Loop through EGPs required to initialize neuron variable and create
            for(const auto &egp : varInit.getSnippet()->getExtraGlobalParams()) {
                const auto resolvedEGPType = egp.type.resolve(getModel().getTypeContext());
                createArray(group, egp.name + var.name, resolvedEGPType, 0, VarLocation::HOST_DEVICE);
            }
        }
    }

    //! Helper to create arrays for neuron state variables
    /*! \tparam A               Adaptor class used to access 
        \tparam G               Type of group variables are associated with
        \param group            Group array is to be associatd with
        \param numNeurons       Number of neurons in group
        \param batchSize        Batch size of model
        \param numDelaySlots    Number of delay slots in group
        \param batched          Should these variables ever be batched*/
    template<typename A, typename G>
    void createNeuronVarArrays(const G *group, size_t numNeurons, size_t batchSize, 
                               size_t numDelaySlots, bool batched)
    {
        A adaptor(*group);
        createVarArrays<A>(
            group, batchSize, batched,
            [&adaptor, group, numDelaySlots, numNeurons]
            (const std::string &varName, VarAccessDim varDims)
            {
                const size_t numVarDelaySlots = adaptor.isVarDelayed(varName) ? numDelaySlots : 1;
                const size_t numElements = ((varDims & VarAccessDim::ELEMENT) ? numNeurons : 1);
                return numVarDelaySlots * numElements;
            });
                  
    }

    /*template<typename G>
    void createDynamicParameterArrays(const G &group, const Snippet::Base::ParamVec &params, 
                                      bool (G::*isDynamic)(const std::string&) const)
    {
    }*/

    void allocateExtraGlobalParam(ArrayMap &groupArrays, const std::string &varName, size_t count);

    //! Set dynamic parameter value in all merged field destinations
    void setDynamicParam(const MergedDynamicFieldDestinations &mergedDestinations, 
                         const Type::NumericValue &value);

    template<typename G>
    void pushUninitialized(GroupArrayMap<G> &groups)
    {
        // Loop through maps of groups to variables
        for(auto &g : groups) {
            // Loop through maps of variable names to arrays
            LOGD_RUNTIME << "\t" << g.first->getName();
            for(auto &a : g.second) {
                // If array is uninitialized, push to device
                if(a.second->isUninitialized()) {
                    LOGD_RUNTIME << "\t\t" << a.first;
                    a.second->pushToDevice();
                }
            }
        }
    }

    template<typename G>
    void pushMergedGroup(const G &g)
    {
        // Loop through groups
        const auto sortedFields = g.getSortedFields(m_Backend.get());

        // Start vector of argument types with unsigned int group index and them append FFI types of each argument
        // **TODO** allow backend to override type
        std::vector<ffi_type*> argumentTypes{&ffi_type_uint};
        argumentTypes.reserve(sortedFields.size() + 1);
        std::transform(sortedFields.cbegin(), sortedFields.cend(), std::back_inserter(argumentTypes),
                        [](const auto &f){ return std::get<0>(f).getFFIType(); });
        
        // Prepare an FFI Call InterFace for calls to push merged
        ffi_cif cif;
        ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, argumentTypes.size(),
                                         &ffi_type_void, argumentTypes.data());
        if (status != FFI_OK) {
            throw std::runtime_error("ffi_prep_cif failed: " + std::to_string(status));
        }

        // Get push function
        void *pushFunction = getSymbol("pushMerged" + G::name + "Group" + std::to_string(g.getIndex()) + "ToDevice");

        // Loop through groups in merged group
        for(unsigned int groupIndex = 0; groupIndex < g.getGroups().size(); groupIndex++) {
            // Create vector of bytes to serialise arguments into
            std::vector<std::byte> argumentStorage;

            // Create vector of arguments
            std::vector<size_t> argumentOffsets{};
            argumentOffsets.reserve(sortedFields.size());

            // Loop through sorted fields
            for(const auto &f : sortedFields) {
                // Push back offset where this argument will start in the storage
                argumentOffsets.push_back(argumentStorage.size());

                std::visit(
                    Utils::Overload{
                        // If field contains array
                        // **TODO** pointer-to-pointer
                        [&argumentStorage, &f, this](const ArrayBase *array)
                        {
                            // If this field should contain host pointer
                            const bool pointerToPointer = std::get<0>(f).isPointerToPointer();
                            if(std::get<3>(f) & CodeGenerator::GroupMergedFieldType::HOST) {
                                array->serialiseHostPointer(argumentStorage, pointerToPointer);
                            }
                            // Otherwise, if it should contain host object
                            else if(std::get<3>(f) & CodeGenerator::GroupMergedFieldType::HOST_OBJECT) {
                                array->serialiseHostObject(argumentStorage, pointerToPointer);
                            }
                            // Otherwise
                            else {
                                // Serialise device object if backend requires it
                                if(m_Backend.get().isArrayDeviceObjectRequired()) {
                                    array->serialiseDeviceObject(argumentStorage, pointerToPointer);
                                }
                                // Otherwise, host pointer
                                else {
                                    array->serialiseHostPointer(argumentStorage, pointerToPointer);
                                }
                            }
                        },
                        // Otherwise, if field contains numeric value
                        [&argumentStorage, &f](const Type::NumericValue &value)
                        { 
                            Type::serialiseNumeric(value, std::get<0>(f), argumentStorage);
                        },
                        [&argumentStorage, &f](std::pair<Type::NumericValue, MergedDynamicFieldDestinations&> value)
                        {
                            Type::serialiseNumeric(value.first, std::get<0>(f), argumentStorage);
                        }},
                    std::get<2>(f)(*this, g.getGroups()[groupIndex], groupIndex));
            }

            // Build vector of argument pointers
            std::vector<void*> argumentPointers{&groupIndex};
            argumentPointers.reserve(1 + sortedFields.size());
            std::transform(argumentOffsets.cbegin(), argumentOffsets.cend(), std::back_inserter(argumentPointers),
                           [&argumentStorage](size_t offset){ return &argumentStorage[offset]; });

            // Call function
            ffi_call(&cif, FFI_FN(pushFunction), nullptr, argumentPointers.data());
        }
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    //! Handle to model library
#ifdef _WIN32
    HMODULE m_Library;
#else
    void *m_Library;
#endif

    //! Current timestep
    uint64_t m_Timestep;

    std::optional<uint64_t> m_NumRecordingTimesteps;

    //! Reference to merged model being run
    std::reference_wrapper<const CodeGenerator::ModelSpecMerged> m_ModelMerged;

    //! Reference to backend used for running model
    std::reference_wrapper<const CodeGenerator::BackendBase> m_Backend;

    //! Delay queue pointers associated with neuron group names
    std::unordered_map<const NeuronGroup*, unsigned int> m_DelayQueuePointer;

    //! Functions to perform custom updates
    std::unordered_map<std::string, CustomUpdateFunction> m_CustomUpdateFunctions;

    //! Map containing mapping of dynamic arrays to their locations within merged groups
    MergedDynamicArrayMap m_MergedDynamicArrays;

    //! Maps of population pointers to named arrays
    GroupArrayMap<CurrentSource> m_CurrentSourceArrays;
    GroupArrayMap<NeuronGroup> m_NeuronGroupArrays;
    GroupArrayMap<SynapseGroup> m_SynapseGroupArrays;
    GroupArrayMap<CustomUpdateBase> m_CustomUpdateBaseArrays;
    GroupArrayMap<CustomConnectivityUpdate> m_CustomConnectivityUpdateArrays;

    //! Maps of dynamic parameters in populations to their locations within merged groups
    MergedDynamicParameterMap<CurrentSource> m_CurrentSourceDynamicParameters;
    MergedDynamicParameterMap<NeuronGroup> m_NeuronGroupDynamicParameters;
    MergedDynamicParameterMap<SynapseGroup> m_SynapseGroupDynamicParameters;
    MergedDynamicParameterMap<CustomUpdateBase> m_CustomUpdateBaseDynamicParameters;
    MergedDynamicParameterMap<CustomConnectivityUpdate> m_CustomConnectivityUpdateDynamicParameters;

    VoidFunction m_AllocateMem;
    VoidFunction m_FreeMem;
    VoidFunction m_Initialize;
    VoidFunction m_InitializeSparse;
    VoidFunction m_InitializeHost;
    StepTimeFunction m_StepTime;
};
}
