#pragma once

// Standard C++ includes
#include <iostream>

// GeNN includes
#include "logging.h"

#define CHECK_HIP_ERRORS(call)                                                                                         \
{                                                                                                                       \
    hipError_t error = call;                                                                                           \
    if (error != hipSuccess) {                                                                                         \
        LOGE_BACKEND << __FILE__ << ": " <<  __LINE__ << ": hip runtime error " << error << ": " << hipGetErrorString(error); \
        exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                   \
}

#define CHECK_RCCL_ERRORS(call)                                                                                         \
{                                                                                                                       \
    rcclResult_t error = call;                                                                                          \
    if (error != rcclSuccess) {                                                                                         \
        LOGE_BACKEND << __FILE__ << ": " << __LINE__ << ": rccl error " << error << ": " << rcclGetErrorString(error);  \
        exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                   \
}
