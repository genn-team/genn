#pragma once

// Standard C++ includes
#include <iostream>

// GeNN includes
#include "logging.h"

#if CUDA_VERSION >= 6050
#define CHECK_CU_ERRORS(call)                                                                       \
{                                                                                                   \
    CUresult error = call;                                                                          \
    if (error != CUDA_SUCCESS) {                                                                    \
        const char *errStr;                                                                         \
        cuGetErrorName(error, &errStr);                                                             \
        LOGE_BACKEND << __FILE__ << ": " <<  __LINE__ << ": cuda driver error " << error << ": " << errStr; \
        exit(EXIT_FAILURE);                                                                         \
    }                                                                                               \
}
#else
#define CHECK_CU_ERRORS(call) call
#endif

#define CHECK_CUDA_ERRORS(call)                                                                                         \
{                                                                                                                       \
    cudaError_t error = call;                                                                                           \
    if (error != cudaSuccess) {                                                                                         \
        LOGE_BACKEND << __FILE__ << ": " <<  __LINE__ << ": cuda runtime error " << error << ": " << cudaGetErrorString(error); \
        exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                   \
}

#define CHECK_NCCL_ERRORS(call)                                                                                         \
{                                                                                                                       \
    ncclResult_t error = call;                                                                                          \
    if (error != ncclSuccess) {                                                                                         \
        LOGE_BACKEND << __FILE__ << ": " << __LINE__ << ": nccl error " << error << ": " << ncclGetErrorString(error);  \
        exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                   \
}
