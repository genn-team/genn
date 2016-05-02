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

#include <cstdlib> // for exit() and EXIT_FAIL / EXIT_SUCCESS
#include <iostream>
#include <map>
#include <memory>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

using namespace std;


//--------------------------------------------------------------------------
/*! \brief Macros for catching errors returned by the CUDA driver and runtime APIs.
 */
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
#if CUDA_VERSION >= 6050
#define CHECK_CU_ERRORS(call)					\
  {								\
    CUresult error = call;					\
    if (error != CUDA_SUCCESS)					\
      {								\
	const char *errStr;					\
	cuGetErrorName(error, &errStr);				\
	fprintf(stderr, "%s: %i: cuda driver error %i: %s\n",	\
		__FILE__, __LINE__, (int)error, errStr);	\
	exit(EXIT_FAILURE);					\
      }								\
  }
#else
#define CHECK_CU_ERRORS(call) call
#endif

// comment below and uncomment here when using CUDA that does not support cugetErrorName
//#define CHECK_CU_ERRORS(call) call
#define CHECK_CUDA_ERRORS(call)						\
  {									\
    cudaError_t error = call;						\
    if (error != cudaSuccess)						\
      {									\
	fprintf(stderr, "%s: %i: cuda error %i: %s\n",			\
		__FILE__, __LINE__, (int)error, cudaGetErrorString(error)); \
	exit(EXIT_FAILURE);						\
      }									\
  }
#endif


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

void gennError(string error);


//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(string type);


//--------------------------------------------------------------------------
/*! \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &os);


#endif  // _UTILS_H_
