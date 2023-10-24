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
#include "variableMode.h"

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
class ArrayBase;
class BackendBase;
class ModelSpecMerged;
}

namespace filesystem
{
class path;
}

//--------------------------------------------------------------------------
// GeNN::Runtime::Runtime
//--------------------------------------------------------------------------
namespace GeNN::Runtime
{
class GENN_EXPORT Runtime
{
    using ArrayMap = std::unordered_map<std::string, std::unique_ptr<CodeGenerator::ArrayBase>>;
    
    template<typename G>
    using GroupArrayMap = std::unordered_map<const G*, ArrayMap>;

public:
    Runtime(const filesystem::path &modelPath, const CodeGenerator::ModelSpecMerged &modelMerged, 
            const CodeGenerator::BackendBase &backend);
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

    //! Get current simulation timestep
    uint64_t getTimestep() const{ return m_Timestep; }

    //! Get current simulation time
    double getTime() const;

    double getNeuronUpdateTime() const{ return *(double*)getSymbol("neuronUpdateTime"); }
    double getInitTime() const{ return *(double*)getSymbol("initTime"); }
    double getPresynapticUpdateTime() const{ return *(double*)getSymbol("presynapticUpdateTime"); }
    double getPostsynapticUpdateTime() const{ return *(double*)getSymbol("postsynapticUpdateTime"); }
    double getSynapseDynamicsTime() const{ return *(double*)getSymbol("synapseDynamicsTime"); }
    double getInitSparseTime() const{ return *(double*)getSymbol("initSparseTime"); }
    double getCustomUpdateTime(const std::string &name)const{ return *(double*)getSymbol("customUpdate" + name + "Time"); }
    double getCustomUpdateTransposeTime(const std::string &name)const{ return *(double*)getSymbol("customUpdate" + name + "TransposeTime"); }
    
    void pullRecordingBuffersFromDevice() const;


    //! Get array associated with current source variable
    CodeGenerator::ArrayBase *getArray(const CurrentSource &group, const std::string &varName) const
    {
        return m_CurrentSourceArrays.at(&group).at(varName).get();   
    }

    //! Get array associated with neuron group variable
    CodeGenerator::ArrayBase *getArray(const NeuronGroup &group, const std::string &varName) const
    {
        return m_NeuronGroupArrays.at(&group).at(varName).get();   
    }

    //! Get array associated with synapse group variable
    CodeGenerator::ArrayBase *getArray(const SynapseGroup &group, const std::string &varName) const
    {
        return m_SynapseGroupArrays.at(&group).at(varName).get();   
    }

    //! Get array associated with custom update variable
    CodeGenerator::ArrayBase *getArray(const CustomUpdateBase &group, const std::string &varName) const
    {
        return m_CustomUpdateArrays.at(&group).at(varName).get();   
    }

    //! Get array associated with custom connectivity update variable
    CodeGenerator::ArrayBase *getArray(const CustomConnectivityUpdate &group, const std::string &varName) const
    {
        return m_CustomConnectivityUpdateArrays.at(&group).at(varName).get();   
    }

   std::pair<std::vector<double>, std::vector<unsigned int>> getRecordedSpikes(const NeuronGroup &group) const
   {
       return getRecordedEvents(group, getArray(group, "recordSpk"));
   }

   std::pair<std::vector<double>, std::vector<unsigned int>> getRecordedSpikeEvents(const NeuronGroup &group) const
   {
       return getRecordedEvents(group, getArray(group, "recordSpkEvent"));
   }

   void writeRecordedSpikes(const NeuronGroup &group, const std::string &path) const
   {
       return writeRecordedEvents(group, getArray(group, "recordSpk"), path);
   }

   void writeRecordedSpikeEvents(const NeuronGroup &group, const std::string &path) const
   {
       return writeRecordedEvents(group, getArray(group, "recordSpkEvent"), path);
   }

private:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);
    typedef void (*StepTimeFunction)(unsigned long long, unsigned long long);

    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    const ModelSpecInternal &getModel() const;
    void *getSymbol(const std::string &symbolName, bool allowMissing = false) const;

    void createArray(ArrayMap &groupArrays, const std::string &varName, const Type::ResolvedType &type, 
                     size_t count, VarLocation location, bool uninitialized = false);

    void createArray(const CurrentSource *currentSource, const std::string &varName, 
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location, bool uninitialized = false)
    {
        createArray(m_CurrentSourceArrays[currentSource], 
                    varName, type, count, location, uninitialized);
    }

    void createArray(const NeuronGroup *neuronGroup, const std::string &varName, 
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location, bool uninitialized = false)
    {
        createArray(m_NeuronGroupArrays[neuronGroup], 
                    varName, type, count, location, uninitialized);
    }

    void createArray(const SynapseGroup *synapseGroup, const std::string &varName, 
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location, bool uninitialized = false)
    {
        createArray(m_SynapseGroupArrays[synapseGroup], 
                    varName, type, count, location, uninitialized);
    }

    void createArray(const CustomUpdateBase *customUpdate, const std::string &varName, 
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location, bool uninitialized = false)
    {
        createArray(m_CustomUpdateArrays[customUpdate], 
                    varName, type, count, location, uninitialized);
    }


    void createArray(const CustomConnectivityUpdate *customConnectivityUpdate, const std::string &varName, 
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location, bool uninitialized = false)
    {
        createArray(m_CustomConnectivityUpdateArrays[customConnectivityUpdate],
                    varName, type, count, location, uninitialized);
    }

    std::pair<std::vector<double>, std::vector<unsigned int>> getRecordedEvents(const NeuronGroup &group, 
                                                                                CodeGenerator::ArrayBase *array) const;

    void writeRecordedEvents(const NeuronGroup &group, CodeGenerator::ArrayBase *array, const std::string &path) const;

    template<typename A, typename G>
    void createEGPArrays(const G *group)
    {
        A adaptor(*group);
        for(const auto &egp : adaptor.getDefs()) {
            const auto resolvedType = egp.type.resolve(getModel().getTypeContext());
            createArray(group, egp.name, resolvedType, 0, adaptor.getLoc(egp.name));
        }
    }

    //! Helper to create arrays for neuron state variables
    /*! \tparam A               Adaptor class used to access 
        \tparam G               Type of group variables are associated with
        \param group            Group array is to be associatd with
        \param numNeuron        Number of neurons in group
        \param batchSize        Batch size of model
        \param numDelaySlots    Number of delay slots in group
        \param batched          Should these variables ever be batched*/
    template<typename A, typename G>
    void createNeuronVarArrays(const G *group, size_t numNeurons, size_t batchSize, 
                               size_t numDelaySlots, bool batched)
    {
        A adaptor(*group);
        for(const auto &var : adaptor.getDefs()) {
            const auto &varInit = adaptor.getInitialisers().at(var.name);
            const bool uninitialized = Utils::areTokensEmpty(varInit.getCodeTokens());
            const auto resolvedType = var.type.resolve(getModel().getTypeContext());
            const auto varDims = adaptor.getVarDims(var);

            const size_t numVarCopies = ((varDims & VarAccessDim::BATCH) && batched) ? batchSize : 1;
            const size_t numVarElements = (varDims & VarAccessDim::NEURON) ? numNeurons : 1;
            const size_t numVarDelaySlots = adaptor.isVarDelayed(var.name) ? numDelaySlots : 1;
            createArray(group, var.name, resolvedType, numVarCopies * numVarElements * numVarDelaySlots,
                        adaptor.getLoc(var.name), uninitialized);

            // Loop through EGPs required to initialize neuron variable and create
            for(const auto &egp : varInit.getSnippet()->getExtraGlobalParams()) {
                const auto resolvedEGPType = egp.type.resolve(getModel().getTypeContext());
                createArray(group, egp.name + var.name, resolvedEGPType, 0, VarLocation::HOST_DEVICE);
            }
        }
    }

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
    void pushMergedGroup(const G &g) const
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
                        [&argumentStorage, &f, this](const CodeGenerator::ArrayBase *array)
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
    std::unordered_map<std::string, unsigned int> m_DelayQueuePointer;

    //! Maps of population pointers to named arrays
    GroupArrayMap<CurrentSource> m_CurrentSourceArrays;
    GroupArrayMap<NeuronGroup> m_NeuronGroupArrays;
    GroupArrayMap<SynapseGroup> m_SynapseGroupArrays;
    GroupArrayMap<CustomUpdateBase> m_CustomUpdateArrays;
    GroupArrayMap<CustomConnectivityUpdate> m_CustomConnectivityUpdateArrays;

    VoidFunction m_AllocateMem;
    VoidFunction m_FreeMem;
    VoidFunction m_Initialize;
    VoidFunction m_InitializeSparse;
    VoidFunction m_InitializeHost;
    StepTimeFunction m_StepTime;
};
}
