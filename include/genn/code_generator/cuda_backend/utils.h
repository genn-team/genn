#pragma once

// Standard C++ includes
#include <iostream>

#if CUDA_VERSION >= 6050
#define CHECK_CU_ERRORS(call)                                        \
  {                                                                \
    CUresult error = call;                                        \
    if (error != CUDA_SUCCESS)                                        \
      {                                                                \
        const char *errStr;                                        \
        cuGetErrorName(error, &errStr);                                \
        std::cerr << __FILE__ << ": " <<  __LINE__;                        \
        std::cerr << ": cuda driver error " << error << ": ";        \
        std::cerr << errStr << std::endl;                                        \
        exit(EXIT_FAILURE);                                        \
      }                                                                \
  }
#else
#define CHECK_CU_ERRORS(call) call
#endif

#define CHECK_CUDA_ERRORS(call)                                        \
  {                                                                \
    cudaError_t error = call;                                        \
    if (error != cudaSuccess)                                        \
      {                                                                \
        std::cerr << __FILE__ << ": " <<  __LINE__;                        \
        std::cerr << ": cuda runtime error " << error << ": ";        \
        std::cerr << cudaGetErrorString(error) << std::endl;                \
        exit(EXIT_FAILURE);                                        \
      }                                                                \
  }

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Utils
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace Utils
{
inline size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

inline size_t padSize(size_t size, size_t blockSize)
{
    return ceilDivide(size, blockSize) * blockSize;
}
}   // namespace Utils
}   // namespace CUDA
}   // namespace CodeGenerator