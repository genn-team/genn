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
class NeuronGroupMergedBase;
}

namespace filesystem
{
class path;
}


#define IMPLEMENT_GROUP_OVERLOADS(GROUP)                                                                            \
public:                                                                                                             \
    void setDynamicParamValue(const GROUP &group, const std::string &paramName,                                     \
                              const Type::NumericValue &value)                                                      \
    {                                                                                                               \
        setDynamicParamValue(m_##GROUP##DynamicParameters.at(&group).at(paramName),                                 \
                             value);                                                                                \
    }                                                                                                               \
    void allocateArray(const GROUP &group, const std::string &varName, size_t count)                                \
    {                                                                                                               \
        allocateExtraGlobalParam(m_##GROUP##Arrays.at(&group), varName, count);                                     \
    }                                                                                                               \
    MergedDynamicFieldDestinations &getMergedParamDestinations(const GROUP &group, const std::string &paramName)    \
    {                                                                                                               \
        return m_##GROUP##DynamicParameters.at(&group).at(paramName).second;                                        \
    }                                                                                                               \
    ArrayBase *getArray(const GROUP &group, const std::string &varName) const                                       \
    {                                                                                                               \
        return m_##GROUP##Arrays.at(&group).at(varName).get();                                                      \
    }                                                                                                               \
private:                                                                                                            \
    void createArray(const GROUP *group, const std::string &varName, const Type::ResolvedType &type,                \
                     size_t count, VarLocation location, bool uninitialized = false, unsigned int logIndent = 1)    \
    {                                                                                                               \
        createArray(m_##GROUP##Arrays[group],                                                                       \
                    varName, type, count, location, uninitialized, logIndent);                                      \
    }                                                                                                               \
    void createDynamicParamDestinations(const GROUP *group, const std::string &paramName,                           \
                                        const Type::ResolvedType &type, unsigned int logIndent = 1)                 \
    {                                                                                                               \
        createDynamicParamDestinations(m_##GROUP##DynamicParameters[group], paramName, type, logIndent);            \
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

class GENN_EXPORT StateBase
{
public:
    virtual ~StateBase()
    {
    }
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
    void customUpdate(const std::string &name);

    //! Get current simulation timestep
    uint64_t getTimestep() const{ return m_Timestep; }

    //! Set current simulation timestep
    void setTimestep(uint64_t timestep){ m_Timestep = timestep; }

    //! Get current simulation time
    double getTime() const;

    //! Get backend-specific state
    StateBase *getState(){ return m_State.get(); }

    double getNeuronUpdateTime() const{ return *(double*)getSymbol("neuronUpdateTime"); }
    double getInitTime() const{ return *(double*)getSymbol("initTime"); }
    double getPresynapticUpdateTime() const{ return *(double*)getSymbol("presynapticUpdateTime"); }
    double getPostsynapticUpdateTime() const{ return *(double*)getSymbol("postsynapticUpdateTime"); }
    double getSynapseDynamicsTime() const{ return *(double*)getSymbol("synapseDynamicsTime"); }
    double getInitSparseTime() const{ return *(double*)getSymbol("initSparseTime"); }
    double getCustomUpdateTime(const std::string &name) const{ return *(double*)getSymbol("customUpdate" + name + "Time"); }
    double getCustomUpdateTransposeTime(const std::string &name) const{ return *(double*)getSymbol("customUpdate" + name + "TransposeTime"); }
    double getCustomUpdateRemapTime(const std::string &name) const{ return *(double*)getSymbol("customUpdate" + name + "RemapTime"); }
    
    void pullRecordingBuffersFromDevice() const;

    //! Get delay pointer associated with neuron group
    unsigned int getDelayPointer(const NeuronGroup &group) const
    {
        return m_DelayQueuePointer.at(&group);
    }

    //! Get recorded spikes from neuron group
    BatchEventArray getRecordedSpikes(const NeuronGroup &group) const
    {
        return getRecordedEvents(group.getNumNeurons(), 
                                 getArray(group, "recordSpk"));
    }

    //! Get recorded presynaptic spike-like events from synapse group
    BatchEventArray getRecordedPreSpikeEvents(const SynapseGroup &group) const
    {
        const auto &groupInternal = static_cast<const SynapseGroupInternal&>(group);
        return getRecordedEvents(groupInternal.getSrcNeuronGroup()->getNumNeurons(), 
                                 getFusedSrcSpikeEventArray(groupInternal, "RecordSpkEvent"));
    }

    //! Get recorded postsynaptic spike-like events from synapse group
    BatchEventArray getRecordedPostSpikeEvents(const SynapseGroup &group) const
    {
        const auto &groupInternal = static_cast<const SynapseGroupInternal&>(group);
        return getRecordedEvents(groupInternal.getTrgNeuronGroup()->getNumNeurons(),
                                 getFusedTrgSpikeEventArray(groupInternal, "RecordSpkEvent"));
    }

    //! Write recorded spikes to CSV file
    void writeRecordedSpikes(const NeuronGroup &group, const std::string &path) const
    {
        return writeRecordedEvents(group.getNumNeurons(), getArray(group, "recordSpk"), path);
    }

    //! Write recorded presynaptic spike-like events to CSV file
    void writeRecordedPreSpikeEvents(const SynapseGroup &group, const std::string &path) const
    {
        const auto &groupInternal = static_cast<const SynapseGroupInternal&>(group);
        return writeRecordedEvents(groupInternal.getSrcNeuronGroup()->getNumNeurons(), 
                                   getFusedSrcSpikeEventArray(groupInternal, "RecordSpkEvent"),
                                   path);
    }

     //! Write recorded postsynaptic spike-like events to CSV file
    void writeRecordedPostSpikeEvents(const SynapseGroup &group, const std::string &path) const
    {
        const auto &groupInternal = static_cast<const SynapseGroupInternal&>(group);
        return writeRecordedEvents(groupInternal.getTrgNeuronGroup()->getNumNeurons(), 
                                   getFusedTrgSpikeEventArray(groupInternal, "RecordSpkEvent"),
                                   path);
    }

    //! Get array associated with fused event group (either spike or spike-event)
    /*! \param ng   Parent merged neuron group
        \param i    Index of the group within the merged group
        \param sg   Child synapse group of neuron group at index i
        \param name Name of variable array is associated with*/
    ArrayBase *getFusedEventArray(const CodeGenerator::NeuronGroupMergedBase &ng, size_t i, 
                                  const SynapseGroupInternal &sg, const std::string &name) const;


    ArrayBase *getFusedSrcSpikeArray(const SynapseGroupInternal &g, const std::string &name) const;
    
    ArrayBase *getFusedTrgSpikeArray(const SynapseGroupInternal &g, const std::string &name) const;
    
    ArrayBase *getFusedSrcSpikeEventArray(const SynapseGroupInternal &g, const std::string &name) const;

    ArrayBase *getFusedTrgSpikeEventArray(const SynapseGroupInternal &g, const std::string &name) const;
   
    void *getSymbol(const std::string &symbolName, bool allowMissing = false) const;

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
    using MergedDynamicParameterMap = std::unordered_map<const G*, std::unordered_map<std::string, std::pair<Type::ResolvedType, MergedDynamicFieldDestinations>>>;
    
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    const ModelSpecInternal &getModel() const;

    void createArray(ArrayMap &groupArrays, const std::string &varName, const Type::ResolvedType &type, 
                     size_t count, VarLocation location, bool uninitialized = false, unsigned int logIndent = 1);
    void createDynamicParamDestinations(std::unordered_map<std::string, std::pair<Type::ResolvedType, MergedDynamicFieldDestinations>> &destinations, 
                                        const std::string &paramName, const Type::ResolvedType &type, unsigned int logIndent = 1);
    BatchEventArray getRecordedEvents(unsigned int numNeurons, ArrayBase *array) const;

    void writeRecordedEvents(unsigned int numNeurons, ArrayBase *array, const std::string &path) const;

    template<typename G>
    void addMergedArrays(const G &mergedGroup)
    {
        // Loop through fields
        for(const auto &f : mergedGroup.getFields()) {
            // If field is dynamic
            if((f.fieldType & CodeGenerator::GroupMergedFieldType::DYNAMIC)) {
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
                                    mergedGroup.getIndex(), groupIndex, f.type, 
                                    f.name, f.fieldType);
                            },
                            // Otherwise, if it cotnains a dynamic parameter
                            [&f, &mergedGroup, groupIndex](std::pair<Type::NumericValue, MergedDynamicFieldDestinations&> value)
                            {
                                value.second.addDestinationField<G>(
                                    mergedGroup.getIndex(), groupIndex, f.type, 
                                    f.name, f.fieldType);
                            },
                            [](const Type::NumericValue&) 
                            {
                                assert(false);
                            }},
                        f.getValue(*this, g, groupIndex));
                    
                }
            }
        }
    }

    template<typename A, typename G>
    void createEGPArrays(const G *group, unsigned int logIndent = 1)
    {
        A adaptor(*group);
        for(const auto &egp : adaptor.getDefs()) {
            const auto resolvedType = egp.type.resolve(getModel().getTypeContext());
            createArray(group, egp.name, resolvedType, 0, adaptor.getLoc(egp.name), false, logIndent);
        }
    }

    template<typename A, typename G, typename S>
    void createVarArrays(const G *group, size_t batchSize, bool batched, S getSizeFn, unsigned int logIndent = 1)
    {
        A adaptor(*group);
        for(const auto &var : adaptor.getDefs()) {
            const auto &varInit = adaptor.getInitialisers().at(var.name);
            const bool uninitialized = Utils::areTokensEmpty(varInit.getCodeTokens());
            const auto resolvedType = var.storageType.resolve(getModel().getTypeContext());
            const auto varDims = adaptor.getVarDims(var);

            const size_t numVarCopies = ((varDims & VarAccessDim::BATCH) && batched) ? batchSize : 1;
            const size_t varSize = getSizeFn(var.name, varDims);
            createArray(group, var.name, resolvedType, numVarCopies * varSize,
                        adaptor.getLoc(var.name), uninitialized, logIndent);

            // Loop through EGPs required to initialize neuron variable and create
            for(const auto &egp : varInit.getSnippet()->getExtraGlobalParams()) {
                const auto resolvedEGPType = egp.type.resolve(getModel().getTypeContext());
                createArray(group, egp.name + var.name, resolvedEGPType, 0, VarLocation::HOST_DEVICE,
                            false, logIndent);
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
                               bool batched, unsigned int logIndent = 1)
    {
        A adaptor(*group);
        createVarArrays<A>(
            group, batchSize, batched,
            [&adaptor, numNeurons]
            (const std::string &varName, VarAccessDim varDims)
            {
                const size_t numVarDelaySlots = adaptor.getNumVarDelaySlots(varName).value_or(1);
                const size_t numElements = ((varDims & VarAccessDim::ELEMENT) ? numNeurons : 1);
                return numVarDelaySlots * numElements;
            },
            logIndent);
                  
    }

    template<typename G>
    void createDynamicParamDestinations(const G &group, const Snippet::Base::ParamVec &params, 
                                      bool (G::*isDynamic)(const std::string&) const, unsigned int logIndent = 1)
    {
        const auto &typeContext = getModel().getTypeContext();
        for(const auto &p : params) {
            if(std::invoke(isDynamic, group, p.name)) {
                createDynamicParamDestinations(&group, p.name, p.type.resolve(typeContext), logIndent);
            }
        }
    }

    //! Set dynamic parameter value in all merged field destinations
    void setDynamicParamValue(const std::pair<Type::ResolvedType, MergedDynamicFieldDestinations> &mergedDestinations, 
                              const Type::NumericValue &value);

    void allocateExtraGlobalParam(ArrayMap &groupArrays, const std::string &varName, size_t count);

    
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
        if(g.getFields().empty()) {
            LOGD_RUNTIME << "Skipping empty merged group '" << G::name << "' index: " << g.getIndex();
            return;
        }
        LOGD_RUNTIME << "Pushing merged group '" << G::name << "' index: " << g.getIndex();

        // Loop through groups
        const auto sortedFields = g.getSortedFields(m_Backend.get());

        // Start vector of argument types with unsigned int group index and them append FFI types of each argument
        // **TODO** allow backend to override type
        std::vector<ffi_type*> argumentTypes{&ffi_type_uint};
        argumentTypes.reserve(sortedFields.size() + 1);
        std::transform(sortedFields.cbegin(), sortedFields.cend(), std::back_inserter(argumentTypes),
                       [](const auto &f){ return f.type.getFFIType(); });
        
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
            LOGD_RUNTIME << "\tGroup: " << groupIndex;
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
                            LOGD_RUNTIME << "\t\t" << f.name << " = array (" << array << ")";
                            const bool pointerToPointer = f.type.isPointerToPointer();
                            if(f.fieldType & CodeGenerator::GroupMergedFieldType::HOST) {
                                array->serialiseHostPointer(argumentStorage, pointerToPointer);
                            }
                            // Otherwise, if it should contain host object
                            else if(f.fieldType & CodeGenerator::GroupMergedFieldType::HOST_OBJECT) {
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
                            LOGD_RUNTIME << "\t\t" << f.name << " = numeric (" << Type::writeNumeric(value, f.type) << ")";
                            Type::serialiseNumeric(value, f.type, argumentStorage);
                        },
                        [&argumentStorage, &f](std::pair<Type::NumericValue, MergedDynamicFieldDestinations&> value)
                        {
                            LOGD_RUNTIME << "\t\t" << f.name << " = dynamic numeric (" << Type::writeNumeric(value.first, f.type) << ")";
                            Type::serialiseNumeric(value.first, f.type, argumentStorage);
                        }},
                    f.getValue(*this, g.getGroups()[groupIndex], groupIndex));
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

    //! Backend-specific state object
    std::unique_ptr<StateBase> m_State;

    //! Reference to merged model being run
    std::reference_wrapper<const CodeGenerator::ModelSpecMerged> m_ModelMerged;

    //! Reference to backend used for running model
    std::reference_wrapper<const CodeGenerator::BackendBase> m_Backend;

    //! Delay queue pointers associated with neuron group names
    std::unordered_map<const NeuronGroup*, unsigned int> m_DelayQueuePointer;

    //! Functions to perform custom updates
    std::unordered_map<std::string, CustomUpdateFunction> m_CustomUpdateFunctions;

    //! Arrays containing column length arrays which should be zeroed before updating connectivity
    std::unordered_map<std::string, std::vector<ArrayBase*>> m_CustomUpdateColLengthArrays;

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
