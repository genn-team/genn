
#pragma once

// Standard C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "sparseProjection.h"
#include "modelSpec.h"

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


    typedef std::unordered_map< std::string, scalar** > VarMap;
    typedef std::unordered_map< std::string, std::tuple< int, VoidFunction, VarMap > > PopMap;
    
    typedef std::array< VoidFunction, 5 > VoidIOFuncs;
    typedef std::tuple< int, VoidIOFuncs, VoidIOFuncs > PopulationIO;
    typedef std::unordered_map< std::string, PopulationIO > PopIOMap;


    SharedLibraryModel() : m_Library(nullptr), m_AllocateMem(nullptr),
        m_Initialize(nullptr), m_InitializeModel(nullptr),
        m_StepTimeGPU(nullptr), m_StepTimeCPU(nullptr)
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
            m_InitializeModel = (VoidFunction)getSymbol("init" + modelName);

            m_StepTimeCPU = (VoidFunction)getSymbol("stepTimeCPU", true);
            m_StepTimeGPU = (VoidFunction)getSymbol("stepTimeGPU", true);

            m_T = (scalar*)getSymbol("t");
            m_Timestep = (unsigned long long*)getSymbol("iT");
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
   
    void initNeuronPopIO( const std::string &popName, int popSize )
    {
      std::array< std::string, 5 > data = {
        "State",
        "Spikes",
        "SpikeEvents",
        "CurrentSpikes",
        "CurrentSpikeEvents" 
      };
      VoidIOFuncs pushers;
      VoidIOFuncs pullers;
      for ( int i = 0; i < data.size(); ++i )
      {
        pullers[i] = (VoidFunction)getSymbol( "pull" + popName + data[i] + "FromDevice" );
        pushers[i] = (VoidFunction)getSymbol( "push" + popName + data[i] + "ToDevice" );
      }

      populationsIO.emplace( std::piecewise_construct,
                             std::forward_as_tuple( popName ),
                             std::forward_as_tuple( popSize, pullers, pushers ) );
    }
    
    void initSynapsePopIO( const std::string &popName, int popSize )
    {
      std::array< std::string, 1 > data = {
        "State",
      };
      VoidIOFuncs pushers;
      VoidIOFuncs pullers;
      for ( int i = 0; i < data.size(); ++i )
      {
        pullers[i] = (VoidFunction)getSymbol( "pull" + popName + data[i] + "FromDevice" );
        pushers[i] = (VoidFunction)getSymbol( "push" + popName + data[i] + "ToDevice" );
      }

      populationsIO.emplace( std::piecewise_construct,
                             std::forward_as_tuple( popName ),
                             std::forward_as_tuple( popSize, pullers, pushers ) );
    }
    
    void pullPopulationStateFromDevice( const std::string &popName )
    {
        auto tmpPop = populationsIO.at( popName );
        std::get<0>( std::get<1>( tmpPop ) )();
    }
    
    void pullPopulationSpikesFromDevice( const std::string &popName )
    {
        auto tmpPop = populationsIO.at( popName );
        std::get<1>( std::get<1>( tmpPop ) )();
    }
    
    void pushPopulationStateToDevice( const std::string &popName )
    {
        auto tmpPop = populationsIO.at( popName );
        std::get<0>( std::get<2>( tmpPop ) )();
    }
    
    void pushPopulationSpikesToDevice( const std::string &popName )
    {
        auto tmpPop = populationsIO.at( popName );
        std::get<1>( std::get<2>( tmpPop ) )();
    }

    void assignExternalPointerToVar( const std::string &popName,
                                     const int popSize,
                                     const std::string &varName,
                                     scalar** varPtr, int* n1 )
    {
      *varPtr = *( (scalar**)getSymbol( varName + popName ));
      *n1 = popSize;
    }

    template <typename T>
    void assignExternalPointer( const std::string varName,
                                     const int varSize,
                                     T** varPtr, int* n1 )
    {
      *varPtr = *( static_cast<T**>( getSymbol( varName ) ) );
      *n1 = varSize;
    }
    
    void assignExternalPointerToSpikes( const std::string &popName,
                                        int popSize, bool spkCnt,
                                        unsigned int** spkPtr, int* n1 )
    {
      *spkPtr = *( (unsigned int**) getSymbol( ( spkCnt ? "glbSpkCnt" : "glbSpk" ) + popName ) );
      *n1 = popSize;
    }

    void assignExternalPointerToT( scalar** tPtr, int* n1 )
    {
      *tPtr = (scalar*)getSymbol( "t" );
      *n1 = 1;
    }
    
    void assignExternalPointerToTimestep( unsigned long long** timestepPtr, int* n1 )
    {
      *timestepPtr = (unsigned long long*)getSymbol( "iT" );
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

    void allocateSparsePop( const std::string &popName, unsigned int nConn )
    {
      typedef void(*UIntFct)(unsigned int);
      ((UIntFct) getSymbol( "allocate" + popName ))( nConn );
    }

    void allocateExtraGlobalParam( const std::string &popName,
                                   const std::string &paramName,
                                   const int size )
    {
        auto egp = static_cast<void**>(getSymbol( paramName + popName ));
        NNmodel::allocateExtraGlobalParam( egp, size * sizeof( scalar ) );
    }

    void freeExtraGlobalParam( const std::string &popName,
                               const std::string &paramName )
    {
        auto egp = *(static_cast<void**>( getSymbol( paramName + popName )));
        cudaFreeHost( egp );
    }

    void initialize()
    {
        m_Initialize();
    }

    void initializeSparsePop( const std::string &popName,
                              unsigned int* _ind, int nConn,
                              unsigned int* _indInG, int nPre,
                              scalar* _g, int nG )
    {
        auto C = (SparseProjection*) getSymbol( "C" + popName );
        auto g = (scalar**) getSymbol( "g" + popName );
        C->connN = nConn;
        for ( int i = 0; i < nConn; ++i )
        {
            C->ind[i] = _ind[i];
        }
        for ( int i = 0; i < nPre; ++i )
        {
            C->indInG[i] = _indInG[i];
        }
        for ( int i = 0; i < nG; ++i )
        {
            (*g)[i] = _g[i];
        }
    }

    void initializeModel()
    {
        m_InitializeModel();
    }

    void stepTimeGPU()
    {
        m_StepTimeGPU();
    }

    void stepTimeCPU()
    {
        m_StepTimeCPU();
    }

    scalar getT() const
    {
        return *m_T;
    }

    unsigned long long getTimestep() const
    {
        return *m_Timestep;
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
    VoidFunction m_InitializeModel;
    VoidFunction m_StepTimeGPU;
    VoidFunction m_StepTimeCPU;
    
    PopIOMap populationsIO;

    int max_pop_size = 0;
    int total_state_vars = 0;
    int totalPopVars = 0;

    scalar *m_T;
    unsigned long long *m_Timestep;
};

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef SharedLibraryModel<float> SharedLibraryModelFloat;
typedef SharedLibraryModel<double> SharedLibraryModelDouble;
