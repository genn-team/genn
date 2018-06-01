
#pragma once

// Standard C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

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


    SharedLibraryModel() : m_Library(nullptr), m_AllocateMem(nullptr),
        m_Initialize(nullptr), m_InitializeSparse(nullptr),
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
            m_InitializeSparse = (VoidFunction)getSymbol("init" + modelName);

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
    
    
    // bool runGPUInPlace( scalar* out, int n1, int n2, int n3, int nSteps )
    // {
    //   if ( n1 < nSteps || n2 < total_state_vars || n3 < max_pop_size ) return false;
    //   scalar* cur = out;
    //   for ( int i = 0; i < nSteps; ++i )
    //   {
    //     stepTimeGPU();
    //
    //     for ( auto pop : populations )
    //     {
    //       auto popData = pop.second;
    //       auto popSize = std::get<0>(popData);
    //       auto popVarPullers = std::get<1>(popData);
    //       auto popStateVars = std::get<2>(popData);
    //       for ( auto funcIter = popVarPullers.begin(); funcIter != popVarPullers.end(); ++funcIter )
    //         (*funcIter)();
    //
    //       for ( auto varIter = popStateVars.begin(); varIter != popStateVars.end();  ++varIter )
    //       {
    //         for ( int j = 0; j < popSize; ++j, ++cur )
    //           *cur = (**varIter)[j];
    //         cur += max_pop_size - popSize;
    //       }
    //     }
    //   }
    //   return true;
    // }

    
    bool runGPUInPlace( scalar* out, int n1, int n2, int nSteps )
    {
      if ( n1 < nSteps || n2 < totalPopVars ) return false;
      scalar* cur = out;
      for ( int i = 0; i < nSteps; ++i )
      {
        stepTimeGPU();
        
        for ( auto pop : populations )
        {
          auto popData = pop.second;
          auto popSize = std::get<0>(popData);
          auto popVarPullers = std::get<1>(popData);
          auto popStateVars = std::get<2>(popData);
          for ( auto funcIter = popVarPullers.begin(); funcIter != popVarPullers.end(); ++funcIter )
            (*funcIter)();
        
          for ( auto varIter = popStateVars.begin(); varIter != popStateVars.end();  ++varIter )
          {
            for ( int j = 0; j < popSize; ++j, ++cur )
              *cur = (**varIter)[j];
            cur += max_pop_size - popSize;
          }
        }
      }
      return true;
    }
    

    void addVars( const std::vector< std::pair< std::string, std::string > > nameTypes, const std::string &popName, int popSize )
    {
        std::vector< VoidFunction > varPullers;
        std::vector< scalar** > stateVars;
        varPullers.emplace_back( (VoidFunction)getSymbol( "pull" + popName + "StateFromDevice" ) );
        
        for ( auto nameType : nameTypes )
        {
          stateVars.emplace_back( (scalar**)getSymbol( nameType.first + popName ) );
        }
        populations.emplace( std::piecewise_construct, std::forward_as_tuple( popName ), std::forward_as_tuple( popSize, varPullers, stateVars ) );
        total_state_vars += stateVars.size();
        max_pop_size = max_pop_size < popSize ? popSize : max_pop_size;
        totalPopVars = stateVars.size() * popSize;
    }


    void pushVarToDevice( const std::string &popName, const std::string &varName, scalar* initData, int n1 )
    {
        scalar** tmpVar = (scalar**) getSymbol( varName + popName );
        VoidFunction tmpPusher = (VoidFunction) getSymbol( "push" + popName + "StateToDevice" );
        scalar* curVar = *tmpVar;
        for ( int i = 0; i < n1; ++i, ++initData, ++curVar )
            *curVar = *initData;
        tmpPusher();
    }


    void pushSpikesToDevice( const std::string &popName, int spikeCount, unsigned int* initData, int n1 )
    {
        unsigned int** tmpSpikeCount = (unsigned int**) getSymbol( "glbSpkCnt" + popName );
        unsigned int** tmpSpike = (unsigned int**) getSymbol( "glbSpk" + popName );
        VoidFunction tmpPusher = (VoidFunction) getSymbol( "push" + popName + "SpikesToDevice" );
        **tmpSpikeCount = spikeCount;
        unsigned int* curSpike = *tmpSpike;
        for ( int i = 0; i < n1; ++i, ++initData, ++curSpike )
            *curSpike = *initData;
        tmpPusher();
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

    void initialize()
    {
        m_Initialize();
    }

    void initializeSparse()
    {
        m_InitializeSparse();
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
    VoidFunction m_InitializeSparse;
    VoidFunction m_StepTimeGPU;
    VoidFunction m_StepTimeCPU;

    std::unordered_map< std::string, std::tuple<int, std::vector< VoidFunction >, std::vector< scalar** > > > populations;

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
