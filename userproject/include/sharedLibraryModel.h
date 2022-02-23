#pragma once

// Standard C++ includes
#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <bitset>

// Platform includes
#ifdef _WIN32
#include <windows.h>
#else
extern "C"
{
#include <dlfcn.h>
}
#endif

// GeNN userproject includes
#include "spikeRecorder.h"

//----------------------------------------------------------------------------
// SharedLibraryModel
//----------------------------------------------------------------------------
// Interface for spike recorders
template<typename scalar = float>
class SharedLibraryModel
{
public:
    SharedLibraryModel()
    :   m_Library(nullptr), m_AllocateMem(nullptr), m_AllocateRecordingBuffers(nullptr),
        m_FreeMem(nullptr), m_Initialize(nullptr), m_InitializeSparse(nullptr), 
        m_StepTime(nullptr), m_PullRecordingBuffersFromDevice(nullptr),
        m_NCCLGenerateUniqueID(nullptr), m_NCCLGetUniqueID(nullptr), 
        m_NCCLInitCommunicator(nullptr), m_NCCLUniqueIDBytes(nullptr)
    {
    }

    SharedLibraryModel(const std::string &pathToModel, const std::string &modelName)
    {
        if(!open(pathToModel, modelName)) {
            throw std::runtime_error("Unable to open library");
        }
    }

    virtual ~SharedLibraryModel()
    {
        // Close model library if loaded successfully
        if(m_Library) {
            freeMem();
#ifdef _WIN32
            FreeLibrary(m_Library);
#else
            dlclose(m_Library);
#endif
        }
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool open(const std::string &pathToModel, const std::string &modelName)
    {
#ifdef _WIN32
        const std::string runnerName = "runner_" + modelName;
#ifdef _DEBUG
        const std::string libraryName = pathToModel + "\\" + runnerName + "_Debug.dll";
#else
        const std::string libraryName = pathToModel + "\\" + runnerName + "_Release.dll";
#endif
        m_Library = LoadLibrary(libraryName.c_str());
#else
        const std::string libraryName = pathToModel + modelName + "_CODE/librunner.so";
        m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif

        // If it fails throw
        if(m_Library != nullptr) {
            m_AllocateMem = (VoidFunction)getSymbol("allocateMem");
            m_AllocateRecordingBuffers = (EGPFreeFunction)getSymbol("allocateRecordingBuffers", true);
            m_FreeMem = (VoidFunction)getSymbol("freeMem");
            m_GetFreeDeviceMemBytes = (GetFreeMemFunction)getSymbol("getFreeDeviceMemBytes");

            m_Initialize = (VoidFunction)getSymbol("initialize");
            m_InitializeSparse = (VoidFunction)getSymbol("initializeSparse");

            m_StepTime = (VoidFunction)getSymbol("stepTime");
            m_GetPopulation = (GetPopulationFunction)getSymbol("getPopulation", true);
            if(m_GetPopulation == nullptr) {
                std::cerr << "GeNN model not built without runtime population lookup support - please rebuild" << std::endl;
                return false;
                
            }
            m_PullRecordingBuffersFromDevice = (VoidFunction)getSymbol("pullRecordingBuffersFromDevice", true);
            
            m_T = (scalar*)getSymbol("t");
            m_Timestep = (unsigned long long*)getSymbol("iT");
            
            m_NCCLGenerateUniqueID = (VoidFunction)getSymbol("ncclGenerateUniqueID", true);
            m_NCCLGetUniqueID = (UCharPtrFunction)getSymbol("ncclGetUniqueID", true);
            m_NCCLInitCommunicator = (NCCLInitCommunicatorFunction)getSymbol("ncclInitCommunicator", true);
            m_NCCLUniqueIDBytes = (unsigned int*)getSymbol("ncclUniqueIDBytes", true);
            return true;
        }
        else {
#ifdef _WIN32
            std::cerr << "Unable to load library - error:" << std::to_string(GetLastError()) << std::endl;;
#else
            std::cerr << "Unable to load library - error:" << dlerror() << std::endl;
#endif
            return false;
        }
    }

    void allocateExtraGlobalParam(const std::string &popName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check allocate exists
        const auto funcs = getEGPFunctions(popName, egpName);
        if(std::get<1>(funcs) == nullptr) {
            throw std::runtime_error("You cannot allocate EGP '" + egpName + "' in population '" + popName + "'");
        }

        // Call allocate
        std::get<1>(funcs)(std::get<0>(funcs), count);
    }
    
    void allocateExtraGlobalParam(const std::string &popName, const std::string &varName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check allocate exists
        const auto funcs = getEGPFunctions(popName, egpName + varName);
        if(std::get<1>(funcs) == nullptr) {
            throw std::runtime_error("You cannot allocate EGP '" + egpName + "' for initializing '" + varName + "'in population '" + popName + "'");
        }

        // Call allocate
        std::get<1>(funcs)(std::get<0>(funcs),count);
    }

    void freeExtraGlobalParam(const std::string &popName, const std::string &egpName)
    {
        // Get EGP functions and check free exists
        const auto funcs = getEGPFunctions(popName, egpName);
        if(std::get<2>(funcs) == nullptr) {
            throw std::runtime_error("You cannot free EGP '" + egpName + "' in population '" + popName + "'");
        }

        // Call free
        std::get<2>(funcs)(std::get<0>(funcs));
    }
    
    void freeExtraGlobalParam(const std::string &popName, const std::string &varName, const std::string &egpName)
    {
        // Get EGP functions and check free exists
        const auto funcs = getEGPFunctions(popName, egpName + varName);
        if(std::get<2>(funcs) == nullptr) {
            throw std::runtime_error("You cannot free EGP '" + egpName + "' for initializing '" + varName + "'in population '" + popName + "'");
        }

        // Call free
        std::get<2>(funcs)(std::get<0>(funcs));
    }

    void pullStateFromDevice(const std::string &popName)
    {
        // Get push and pull state functions and check pull exists
        const auto funcs = getVarFunctions(popName, "", "State");
        if(std::get<2>(funcs) == nullptr) {
            throw std::runtime_error("You cannot pull state from population '" + popName + "'");
        }

        // Call pull
        std::get<2>(funcs)(std::get<0>(funcs));
    }

    
    void pullConnectivityFromDevice(const std::string &popName)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getVarFunctions(popName, "", "Connectivity");
        if(std::get<2>(funcs) == nullptr) {
            throw std::runtime_error("You cannot pull connectivity from population '" + popName + "'");
        }

        // Call pull
        std::get<2>(funcs)(std::get<0>(funcs));
    }

    void pullVarFromDevice(const std::string &popName, const std::string &varName)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getVarFunctions(popName, varName, "");
        if(std::get<2>(funcs) == nullptr) {
            throw std::runtime_error("You cannot pull var '" + varName + "' from population '" + popName + "'");
        }

        // Call pull
        std::get<2>(funcs)(std::get<0>(funcs));
    }

    void pullExtraGlobalParam(const std::string &popName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check pull exists
        const auto funcs = getEGPFunctions(popName, egpName);
        if(std::get<4>(funcs) == nullptr) {
            throw std::runtime_error("You cannot pull EGP '" + egpName + "' from population '" + popName + "'");
        }

        // Call pull
        std::get<4>(funcs)(std::get<0>(funcs), count);
    }
    
    void pullExtraGlobalParam(const std::string &popName, const std::string &varName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check pull exists
        const auto funcs = getEGPFunctions(popName, egpName + varName);
        if(std::get<4>(funcs) == nullptr) {
            throw std::runtime_error("You cannot pull EGP '" + egpName + "' for initializing '" + varName + "'in population '" + popName + "'");
        }

        // Call pull
        std::get<4>(funcs)(std::get<0>(funcs), count);
    }

    void pushStateToDevice(const std::string &popName, bool uninitialisedOnly = false)
    {
        // Get push and pull state functions and check pull exists
        const auto funcs = getVarFunctions(popName, "", "State");
        if(std::get<1>(funcs) == nullptr) {
            throw std::runtime_error("You cannot push state to population '" + popName + "'");
        }

        // Call push
        std::get<1>(funcs)(std::get<0>(funcs), uninitialisedOnly);
    }


    void pushConnectivityToDevice(const std::string &popName, bool uninitialisedOnly = false)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getVarFunctions(popName, "", "Connectivity");
        if(std::get<1>(funcs) == nullptr) {
            throw std::runtime_error("You cannot push connectivity to population '" + popName + "'");
        }

        // Call push
        std::get<1>(funcs)(std::get<0>(funcs), uninitialisedOnly);
    }

    void pushVarToDevice(const std::string &popName, const std::string &varName, bool uninitialisedOnly = false)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getVarFunctions(popName, varName, "");
        if(std::get<1>(funcs) == nullptr) {
            throw std::runtime_error("You cannot push var '" + varName + "' to population '" + popName + "'");
        }

        // Call push
        std::get<1>(funcs)(std::get<0>(funcs), uninitialisedOnly);
    }

    void pushExtraGlobalParam(const std::string &popName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check push exists
        const auto funcs = getEGPFunctions(popName, egpName);
        if(std::get<3>(funcs) == nullptr) {
            throw std::runtime_error("You cannot push EGP '" + egpName + "' to population '" + popName + "'");
        }

        // Call push
        std::get<3>(funcs)(std::get<0>(funcs), count);
    }
    
    void pushExtraGlobalParam(const std::string &popName, const std::string &varName, const std::string &egpName, unsigned int count)
    {
        // Get EGP functions and check push exists
        const auto funcs = getEGPFunctions(egpName + varName + popName);
        if(std::get<3>(funcs) == nullptr) {
            throw std::runtime_error("You cannot push EGP '" + egpName + "' for initializing '" + varName + "'in population '" + popName + "'");
        }

        // Call push
        std::get<3>(funcs)(std::get<0>(funcs), count);
    }

    template<typename T>
    T *getVar(const std::string &popName, const std::string &varName)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getVarFunctions(popName, varName, "");
        if(std::get<3>(funcs) == nullptr) {
            throw std::runtime_error("You cannot get var '" + varName + "' from population '" + popName + "'");
        }

        // Call get
        return static_cast<T*>(std::get<3>(funcs)(std::get<0>(funcs)));
    }

    template<typename T>
    T *getEGP(const std::string &popName, const std::string &varName)
    {
        // Get push and pull connectivity functions and check pull exists
        const auto funcs = getEGPFunctions(popName, varName);
        if(std::get<5>(funcs) == nullptr) {
            throw std::runtime_error("You cannot get EGP '" + varName + "' from population '" + popName + "'");
        }

        // Call get
        return static_cast<T*>(std::get<5>(funcs)(std::get<0>(funcs)));
    }

    // Gets a scalar from the shared library
    template<typename T>
    T *getScalar(const std::string &varName)
    {
        return (static_cast<T*>(getSymbol(varName)));
    }

    void allocateMem()
    {
        m_AllocateMem();
    }

    void allocateRecordingBuffers(unsigned int timesteps)
    {
        if(m_AllocateRecordingBuffers == nullptr) {
            throw std::runtime_error("Cannot allocate recording buffers - model may not have recording enabled");
        }
        m_AllocateRecordingBuffers(timesteps);
    }

    void freeMem()
    {
        m_FreeMem();
    }

    size_t getFreeDeviceMemBytes()
    {
        return m_GetFreeDeviceMemBytes();
    }
    
    void ncclGenerateUniqueID()
    {
        if(m_NCCLGenerateUniqueID == nullptr) {
            throw std::runtime_error("Cannot generate NCCL unique ID - model may not have been built with NCCL support");
        }
        m_NCCLGenerateUniqueID();
    }
    
    unsigned char *ncclGetUniqueID()
    {
        if(m_NCCLGetUniqueID == nullptr) {
            throw std::runtime_error("Cannot get NCCL unique ID - model may not have been built with NCCL support");
        }
        return m_NCCLGetUniqueID();
    }
    
    unsigned int ncclGetUniqueIDBytes() const
    {
        if(m_NCCLUniqueIDBytes == nullptr) {
            throw std::runtime_error("Cannot get NCCL unique ID bytes - model may not have been built with NCCL support");
        }
        
        return *m_NCCLUniqueIDBytes;
    }

    void ncclInitCommunicator(int rank, int numRanks)
    {
        if(m_NCCLInitCommunicator == nullptr) {
            throw std::runtime_error("Cannot initialise NCCL communicator - model may not have been built with NCCL support");
        }
        m_NCCLInitCommunicator(rank, numRanks);
    }

    void initialize()
    {
        m_Initialize();
    }

    void initializeSparse()
    {
        m_InitializeSparse();
    }

    void stepTime()
    {
        m_StepTime();
    }
    
    void customUpdate(const std::string &name)
    {
        auto c = m_CustomUpdates.find(name);
        if(c != m_CustomUpdates.cend()) {
            c->second();
        }
        else {
            auto customUpdateFn = (VoidFunction)getSymbol("update" + name);
            m_CustomUpdates.emplace(name, customUpdateFn);
            customUpdateFn();
        }
    }
    
    void pullRecordingBuffersFromDevice()
    {
        if(m_PullRecordingBuffersFromDevice == nullptr) {
            throw std::runtime_error("Cannot pull recording buffers from device - model may not have recording enabled");
        }
        m_PullRecordingBuffersFromDevice();
    }

    scalar getTime() const
    {
        return *m_T;
    }

    unsigned long long getTimestep() const
    {
        return *m_Timestep;
    }

    void setTime(scalar t)
    {
        *m_T = t;
    }

    void setTimestep(unsigned long long iT)
    {
        *m_Timestep = iT;
    }

    double getNeuronUpdateTime() const{ return *(double*)getSymbol("neuronUpdateTime"); }
    double getInitTime() const{ return *(double*)getSymbol("initTime"); }
    double getPresynapticUpdateTime() const{ return *(double*)getSymbol("presynapticUpdateTime"); }
    double getPostsynapticUpdateTime() const{ return *(double*)getSymbol("postsynapticUpdateTime"); }
    double getSynapseDynamicsTime() const{ return *(double*)getSymbol("synapseDynamicsTime"); }
    double getInitSparseTime() const{ return *(double*)getSymbol("initSparseTime"); }
    double getCustomUpdateTime(const std::string &name)const{ return *(double*)getSymbol("customUpdate" + name + "Time"); }
    double getCustomUpdateTransposeTime(const std::string &name)const{ return *(double*)getSymbol("customUpdate" + name + "TransposeTime"); }
    
    void *getSymbol(const std::string &symbolName, bool allowMissing = false, void *defaultSymbol = nullptr) const
    {
#ifdef _WIN32
        void *symbol = GetProcAddress(m_Library, symbolName.c_str());
#else
        void *symbol = dlsym(m_Library, symbolName.c_str());
#endif

        // If this symbol's missing
        if(symbol == nullptr) {
            // If this isn't allowed, throw error
            if(!allowMissing) {
                throw std::runtime_error("Cannot find symbol '" + symbolName + "'");
            }
            // Otherwise, return default
            else {
                return defaultSymbol;
            }
        }
        // Otherwise, return symbolPopulationFuncs
        else {
            return symbol;
        }
    }

private:
    //----------------------------------------------------------------------------
    // Structs
    //----------------------------------------------------------------------------
    //! Structure containing the merged location of a population
    /* NOTE: It is very important that this struct is kept synchronised with the one generated by GeNN */
    struct Population 
    {
        unsigned int mergedGroupIndex;
        unsigned int groupIndex;
        const char *groupType;
    };

    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);
    typedef unsigned char* (*UCharPtrFunction)(void);
    typedef void (*PushFunction)(unsigned int, bool);
    typedef void (*PullFunction)(unsigned int);
    typedef void* (*GetFunction)(unsigned int);
    typedef void (*EGPFunction)(unsigned int, unsigned int);
    typedef void (*EGPFreeFunction)(unsigned int);
    typedef size_t (*GetFreeMemFunction)(void);
    typedef void (*NCCLInitCommunicatorFunction)(int, int);
    typedef Population (*GetPopulationFunction)(const char*);
    
    typedef std::tuple<unsigned int, PushFunction, PullFunction, GetFunction> PushPullFunc;
    typedef std::tuple<unsigned int, EGPFunction, EGPFreeFunction, EGPFunction, EGPFunction, GetFunction> EGPFunc;

    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    PushPullFunc getVarFunctions(const std::string &popName, const std::string &prefix, const std::string &suffix)
    {
        // If description is found, return associated push and pull functions
        const std::string description = prefix + popName + suffix;
        const auto popVar = m_PopulationVars.find(description);
        if(popVar != m_PopulationVars.end()) {
            return popVar->second;
        }
        else {
            // Get population struct for this population
            const auto pop = m_GetPopulation(popName.c_str());
            
            // Get symbols for push and pull functions
            const std::string mergedGroupStem = prefix + pop.groupType + "Group" + std::to_string(pop.mergedGroupIndex) + suffix;
            auto pushFunc = (PushFunction)getSymbol("push" + mergedGroupStem + "ToDevice", true);
            auto pullFunc = (PullFunction)getSymbol("pull" + mergedGroupStem + "FromDevice", true);
            auto getFunc = (GetFunction)getSymbol("get" + mergedGroupStem, true);

            // Add to map
            auto newPopVar = m_PopulationVars.emplace(std::piecewise_construct,
                                                      std::forward_as_tuple(description),
                                                      std::forward_as_tuple(pop.groupIndex, pushFunc, pullFunc, getFunc));

            // Return newly added push and pull functions
            return newPopVar.first->second;
        }
    }

    EGPFunc getEGPFunctions(const std::string &popName, const std::string &prefix)
    {
        // If description is found, return associated EGP functions
        const std::string description = prefix + popName;
        const auto popEGP = m_PopulationEPGs.find(description);
        if(popEGP != m_PopulationEPGs.end()) {
            return popEGP->second;
        }
        else {
            // Get population struct for this population
            const auto pop = m_GetPopulation(popName.c_str());
            
            // Get symbols for push and pull functions
            const std::string mergedGroupStem = prefix + pop.groupType + "Group" + std::to_string(pop.mergedGroupIndex);
            auto allocateFunc = (EGPFunction)getSymbol("allocate" + mergedGroupStem, true);
            auto freeFunc = (EGPFreeFunction)getSymbol("free" + mergedGroupStem, true);
            auto pushFunc = (EGPFunction)getSymbol("push" + mergedGroupStem + "ToDevice", true);
            auto pullFunc = (EGPFunction)getSymbol("pull" + mergedGroupStem + "FromDevice", true);
            auto getFunc = (GetFunction)getSymbol("get" + mergedGroupStem, true);

            // Add to map
            auto newPopEGP = m_PopulationEPGs.emplace(std::piecewise_construct,
                                                      std::forward_as_tuple(description),
                                                      std::forward_as_tuple(pop.groupIndex, allocateFunc, freeFunc, 
                                                                            pushFunc, pullFunc, getFunc));

            // Return newly functions
            return newPopEGP.first->second;
        }
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
#ifdef _WIN32
    HMODULE m_Library;
#else
    void *m_Library;
#endif

    VoidFunction m_AllocateMem;
    EGPFreeFunction m_AllocateRecordingBuffers;
    VoidFunction m_FreeMem;
    GetFreeMemFunction m_GetFreeDeviceMemBytes;
    VoidFunction m_Initialize;
    VoidFunction m_InitializeSparse;
    VoidFunction m_StepTime;
    GetPopulationFunction m_GetPopulation;

    VoidFunction m_PullRecordingBuffersFromDevice;

    VoidFunction m_NCCLGenerateUniqueID;
    UCharPtrFunction m_NCCLGetUniqueID;
    NCCLInitCommunicatorFunction m_NCCLInitCommunicator;
    const unsigned int *m_NCCLUniqueIDBytes;
    
    std::unordered_map<std::string, PushPullFunc> m_PopulationVars;
    std::unordered_map<std::string, EGPFunc> m_PopulationEPGs;
    std::unordered_map<std::string, VoidFunction> m_CustomUpdates;
    scalar *m_T;
    unsigned long long *m_Timestep;
};
