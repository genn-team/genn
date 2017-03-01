/*--------------------------------------------------------------------------
  Author/Modifier: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
   
  --------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file utils.h

  \brief This file contains standard utility functions provide within the NVIDIA CUDA software development toolkit (SDK). The remainder of the file contains a function that defines the standard neuron models.
*/
//--------------------------------------------------------------------------

#ifndef _UTILS_H_
#define _UTILS_H_ //!< macro for avoiding multiple inclusion during compilation

#include <iostream>
#include <string>
#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace std;


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Macros for catching errors returned by the CUDA driver and runtime APIs.
 */
//--------------------------------------------------------------------------

#if CUDA_VERSION >= 6050
#define CHECK_CU_ERRORS(call)                                        \
  {                                                                \
    CUresult error = call;                                        \
    if (error != CUDA_SUCCESS)                                        \
      {                                                                \
        const char *errStr;                                        \
        cuGetErrorName(error, &errStr);                                \
        cerr << __FILE__ << ": " <<  __LINE__;                        \
        cerr << ": cuda driver error " << error << ": ";        \
        cerr << errStr << endl;                                        \
        exit(EXIT_FAILURE);                                        \
      }                                                                \
  }
#else
#define CHECK_CU_ERRORS(call) call
#endif

// comment below and uncomment here when using CUDA that does not support cugetErrorName
//#define CHECK_CU_ERRORS(call) call
#define CHECK_CUDA_ERRORS(call)                                        \
  {                                                                \
    cudaError_t error = call;                                        \
    if (error != cudaSuccess)                                        \
      {                                                                \
        cerr << __FILE__ << ": " <<  __LINE__;                        \
        cerr << ": cuda runtime error " << error << ": ";        \
        cerr << cudaGetErrorString(error) << endl;                \
        exit(EXIT_FAILURE);                                        \
      }                                                                \
  }
#endif


//--------------------------------------------------------------------------
/*! \brief Bit tool macros
 */
//--------------------------------------------------------------------------

#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x

#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1

#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

//--------------------------------------------------------------------------
/*! \brief Miscellaneous macros
 */
//--------------------------------------------------------------------------

#define USE(expr) do { (void)(expr); } while (0) //!< Silence 'unused parameter' warnings

#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Function for getting the capabilities of a CUDA device via the driver API.
 */
//--------------------------------------------------------------------------

CUresult cudaFuncGetAttributesDriver(cudaFuncAttributes *attr, CUfunction kern);
#endif


//--------------------------------------------------------------------------
/*! \brief Function called upon the detection of an error. Outputs an error message and then exits.
 */
//--------------------------------------------------------------------------

void gennError(const string &error);


//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(const string &type);


//--------------------------------------------------------------------------
/*! \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &os);


#endif  // _UTILS_H_
