
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

#ifdef _WIN32
#include <windows.h>
#else
// POSIX C includes
extern "C"
{
#include <dlfcn.h>
}
#endif

//----------------------------------------------------------------------------
// SharedLibraryModel
//----------------------------------------------------------------------------
template<typename scalar = float>
class SharedLibraryModel
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef void (*VoidFunction)(void);

    typedef std::array< VoidFunction, 5 > VoidIOFuncs;
    typedef std::tuple< VoidIOFuncs, VoidIOFuncs > PopulationIO;
    typedef std::unordered_map< std::string, PopulationIO > PopIOMap;


    SharedLibraryModel()
    :   m_Library(nullptr), m_AllocateMem(nullptr), m_Initialize(nullptr), m_InitializeSparse(nullptr), m_StepTime(nullptr)
    {
    }

    SharedLibraryModel(const std::string &pathToModel, const std::string &modelName)
    {
        if(!open(pathToModel, modelName)) {
            throw std::runtime_error("Unable to open library");
        }
    }

    ~SharedLibraryModel()
    {
        // Close model library if loaded successfully
        if(m_Library)
        {
            exitGeNN();
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
        const std::string libraryName = pathToModel + modelName + "_CODE\\runner.dll";
        m_Library = LoadLibrary(libraryName.c_str());
#else
        const std::string libraryName = pathToModel + modelName + "_CODE/librunner.so";
        m_Library = dlopen(libraryName.c_str(), RTLD_NOW);
#endif

        // If it fails throw
        if(m_Library != nullptr) {
            m_AllocateMem = (VoidFunction)getSymbol("allocateMem");
            m_Initialize = (VoidFunction)getSymbol("initialize");
            m_InitializeSparse = (VoidFunction)getSymbol("initializeSparse");

            m_StepTime = (VoidFunction)getSymbol("stepTime", true);

            m_ExitGeNN = (VoidFunction)getSymbol("exitGeNN");

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

    // Retrive symbols to pull/push from/to the device from shared model
    // and store them in a map for fast lookup
    void initIO( const std::string &popName,
                 const std::bitset<5> &availableDTypes ) // the bits are indexed backwards
    {
#ifndef CPU_ONLY
      std::array< std::string, 5 > dataTypes = {
        "State",
        "Spikes",
        "SpikeEvents",
        "CurrentSpikes",
        "CurrentSpikeEvents" 
      };
      VoidIOFuncs pushers;
      VoidIOFuncs pullers;
      for ( size_t i = 0; i < dataTypes.size(); ++i )
      {
        if (availableDTypes.test(i)){
          pullers[i] = (VoidFunction)getSymbol( "pull" + popName + dataTypes[i] + "FromDevice" );
          pushers[i] = (VoidFunction)getSymbol( "push" + popName + dataTypes[i] + "ToDevice" );
        }
        else
        {
          // SynapseGroups and CurrentSources only have states. Throw if attempted to pull/push anything but state
          pullers[i] = []{ throw std::runtime_error("You cannot pull from this population"); };
          pushers[i] = []{ throw std::runtime_error("You cannot push to this population"); };
        }
      }

      m_PopulationsIO.emplace( std::piecewise_construct,
                             std::forward_as_tuple( popName ),
                             std::forward_as_tuple( pullers, pushers ) );
#endif
    }

    void initNeuronPopIO( const std::string &popName )
    {
      // the bits are indexed backwards
      initIO( popName, std::bitset<5>("11111") );
    }
    
    void initSynapsePopIO( const std::string &popName )
    {
      // the bits are indexed backwards
      initIO( popName, std::bitset<5>("00001") );
    }

    void initCurrentSourceIO( const std::string &csName )
    {
      // the bits are indexed backwards
      initIO( csName, std::bitset<5>("00001") );
    }
    
    void pullStateFromDevice( const std::string &popName )
    {
        auto tmpPop = m_PopulationsIO.at( popName );
        std::get<0>( std::get<0>( tmpPop ) )();
    }
    
    void pullSpikesFromDevice( const std::string &popName )
    {
        auto tmpPop = m_PopulationsIO.at( popName );
        std::get<1>( std::get<0>( tmpPop ) )();
    }
    
    void pullCurrentSpikesFromDevice( const std::string &popName )
    {
        auto tmpPop = m_PopulationsIO.at( popName );
        std::get<3>( std::get<0>( tmpPop ) )();
    }

    void pushStateToDevice( const std::string &popName )
    {
        auto tmpPop = m_PopulationsIO.at( popName );
        std::get<0>( std::get<1>( tmpPop ) )();
    }
    
    void pushSpikesToDevice( const std::string &popName )
    {
        auto tmpPop = m_PopulationsIO.at( popName );
        std::get<1>( std::get<1>( tmpPop ) )();
    }

    // Assign symbol from shared model to the provided pointer.
    // The symbol is supposed to be an array
    // When used with numpy, wrapper automatically provides varPtr and n1
    template <typename T>
    void assignExternalPointerArray( const std::string &varName,
                                     const int varSize,
                                     T** varPtr, int* n1 )
    {
      *varPtr = *( static_cast<T**>( getSymbol( varName ) ) );
      *n1 = varSize;
    }
    
    // Assign symbol from shared model to the provided pointer.
    // The symbol is supposed to be a single value
    // When used with numpy, wrapper automatically provides varPtr and n1
    template <typename T>
    void assignExternalPointerSingle( const std::string &varName,
                                      T** varPtr, int* n1 )
    {
      *varPtr = static_cast<T*>( getSymbol( varName ) );
      *n1 = 1;
    }

    void *getSymbol(const std::string &symbolName, bool allowMissing = false)
    {
#ifdef _WIN32
        void *symbol = GetProcAddress(m_Library, symbolName.c_str());
#else
        void *symbol = dlsym(m_Library, symbolName.c_str());
#endif

        // If this symbol isn't allowed to be missing but it is, raise exception
        if(!allowMissing && symbol == nullptr) {
            throw std::runtime_error("Cannot find symbol '" + symbolName + "'");
        }

        return symbol;
    }


    void allocateMem()
    {
        m_AllocateMem();
    }

   /* void allocateExtraGlobalParam( const std::string &popName,
                                   const std::string &paramName,
                                   const int size )
    {
        auto egp = static_cast<void**>(getSymbol( paramName + popName ));
#ifndef CPU_ONLY
        CHECK_CUDA_ERRORS( cudaHostAlloc( egp, size * sizeof( scalar ), cudaHostAllocPortable ) );
#else
        *egp = malloc( size * sizeof( scalar ) );
#endif
    }

    void freeExtraGlobalParam( const std::string &popName,
                               const std::string &paramName )
    {
        auto egp = static_cast<void**>( getSymbol( paramName + popName ));
#ifndef CPU_ONLY
        CHECK_CUDA_ERRORS( cudaFreeHost( *egp ) );
#else
        free(*egp);
#endif
    }*/

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

    void exitGeNN()
    {
        m_ExitGeNN();
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
#ifdef _WIN32
    HMODULE m_Library;
#else
    void *m_Library;
#endif

    VoidFunction m_AllocateMem;
    VoidFunction m_Initialize;
    VoidFunction m_InitializeSparse;
    VoidFunction m_StepTime;
    VoidFunction m_ExitGeNN;
    
    PopIOMap m_PopulationsIO;
};
