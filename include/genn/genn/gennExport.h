#pragma once

// If we're building on Windows and SWIG isn't defined (amusingly, __declspec breaks SWIG's parser)
#if defined(_WIN32) && !defined(SWIG)
    #ifdef BUILDING_GENN_DLL
        #define GENN_EXPORT __declspec(dllexport)
    #elif defined(LINKING_GENN_DLL)
        #define GENN_EXPORT __declspec(dllimport)
    #else
        #define GENN_EXPORT
    #endif
#else
    #define GENN_EXPORT
#endif