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

// GeNN includes
#include "gennExport.h"
#include "type.h"
#include "varAccess.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class ArrayBase;
class BackendBase;
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
    void initialise();

    //! Initialise parts of model which rely on sparse connectivity
    void initialiseSparse();

    //! Simulate one timestep
    void stepTime();

    void allocateExtraGlobalParam(const std::string &groupName, const std::string &varName,
                                  size_t count);
                                  
    //! Get current simulation timestep
    uint64_t getTimestep() const{ return m_Timestep; }

    //! Get current simulation time
    double getTime() const{ return m_Timestep * m_ModelMerged.get().getModel().getDT(); }

private:
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    //! Helper to allocate array in m_Arrays data structure
    /*! \param groupName    name of group owning array e.g. a NeuronGroup
        \param varName      name of variable this array is representing
        \param type         data type of array
        \param count        number of elements in array
        \param location     location of array e.g. device-only
        \param memAlloc     MemAlloc object for tracking memory usage*/
    void createArray(const std::string &groupName, const std::string &varName,
                     const Type::ResolvedType &type, size_t count, 
                     VarLocation location);
    
    //! Helper to allocate array in m_Arrays data structure
    /*! \tparam A           Adaptor class used to access 
        \tparam G           Type of group variables are associated with
        \param memAlloc     MemAlloc object for tracking memory usage*/
    template<typename A, typename G>
    void allocateNeuronVars(const G &group, size_t numNeurons, size_t batchSize, 
                            size_t delaySlots, bool batched)
    {
        A adaptor(group);
        for(const auto &var : adaptor.getDefs()) {
            const auto resolvedType = var.type.resolve(m_ModelMerged.get().getModel().getTypeContext());
            const auto varDims = adaptor.getVarDims(var);

            const size_t numVarCopies = ((varDims & VarAccessDim::BATCH) && batched) ? batchSize : 1;
            const size_t numVarElements = (varDims & VarAccessDim::NEURON) ? numNeurons : 1;
            const size_t numDelaySlots = adaptor.isVarDelayed(var.name) ? numDelaySlots : 1;
            createArray(adaptor.getNameSuffix(), var.name, resolvedType, numVarCopies * numVarElements * numDelaySlots,
                        adaptor.getLoc(var.name));
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

    //! Map of population names to named arrays
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<CodeGenerator::ArrayBase>>> m_Arrays;
};
}
