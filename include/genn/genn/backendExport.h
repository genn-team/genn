#pragma once

// If we're building on Windows and SWIG isn't defined (amusingly, __declspec breaks SWIG's parser)
#if defined(_WIN32) && !defined(SWIG)
    #ifdef BUILDING_BACKEND_DLL
        #define BACKEND_EXPORT __declspec(dllexport)
    #elif defined(LINKING_BACKEND_DLL)
        #define BACKEND_EXPORT __declspec(dllimport)
    #else
        #define BACKEND_EXPORT
    #endif
#else
    #define BACKEND_EXPORT
#endif