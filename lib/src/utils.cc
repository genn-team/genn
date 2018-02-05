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

#ifndef UTILS_CC
#define UTILS_CC

#include "utils.h"

// C++ standard includes
#include <fstream>

// C standard includes
#include <cstdint>

// GeNN includes
#include "codeStream.h"

#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Function for getting the capabilities of a CUDA device via the driver API.
 */
//--------------------------------------------------------------------------

CUresult cudaFuncGetAttributesDriver(cudaFuncAttributes *attr, CUfunction kern) {
    int tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: " << tmp << endl;
#endif
    attr->maxThreadsPerBlock= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: " << tmp << endl;
#endif
    attr->sharedSizeBytes= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: " << tmp << endl;
#endif
    attr->constSizeBytes= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: " << tmp << endl;
#endif
    attr->localSizeBytes= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_NUM_REGS, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_NUM_REGS: " << tmp << endl;
#endif
    attr->numRegs= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_PTX_VERSION, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_PTX_VERSION: " << tmp << endl;
#endif
    attr->ptxVersion= tmp;
    CHECK_CU_ERRORS(cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kern));
#ifdef BLOCKSZ_DEBUG
    cerr << "BLOCKSZ_DEBUG: CU_FUNC_ATTRIBUTE_BINARY_VERSION: " << tmp << endl;
#endif
    attr->binaryVersion= tmp;
    return CUDA_SUCCESS;
}
#endif

//--------------------------------------------------------------------------
/*! \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(CodeStream &os)
{
    string s;
    ifstream is("../src/header.src");
    getline(is, s);
    while (is.good()) {
        os << s << endl;
        getline(is, s);
    }
    os << endl;
}


//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(const string &type)
{
  unsigned int size = 0;
  if (type.find("*") != string::npos) size= sizeof(char *); // it's a pointer ... any pointer should have the same size
  if (type == "char") size = sizeof(char);
  //  if (type == "char16_t") size = sizeof(char16_t);
  //  if (type == "char32_t") size = sizeof(char32_t);
  if (type == "wchar_t") size = sizeof(wchar_t);
  if (type == "signed char") size = sizeof(signed char);
  if (type == "short") size = sizeof(short);
  if (type == "signed short") size = sizeof(signed short);
  if (type == "short int") size = sizeof(short int);
  if (type == "signed short int") size = sizeof(signed short int);
  if (type == "int") size = sizeof(int);
  if (type == "signed int") size = sizeof(signed int);
  if (type == "long") size = sizeof(long);
  if (type == "signed long") size = sizeof(signed long);
  if (type == "long int") size = sizeof(long int);
  if (type == "signed long int") size = sizeof(signed long int);
  if (type == "long long") size = sizeof(long long);
  if (type == "signed long long") size = sizeof(signed long long);
  if (type == "long long int") size = sizeof(long long int);
  if (type == "signed long long int") size = sizeof(signed long long int);
  if (type == "unsigned char") size = sizeof(unsigned char);
  if (type == "unsigned short") size = sizeof(unsigned short);
  if (type == "unsigned short int") size = sizeof(unsigned short int);
  if (type == "unsigned") size = sizeof(unsigned);
  if (type == "unsigned int") size = sizeof(unsigned int);
  if (type == "unsigned long") size = sizeof(unsigned long);
  if (type == "unsigned long int") size = sizeof(unsigned long int);
  if (type == "unsigned long long") size = sizeof(unsigned long long);
  if (type == "unsigned long long int") size = sizeof(unsigned long long int);
  if (type == "float") size = sizeof(float);
  if (type == "double") size = sizeof(double);
  if (type == "long double") size = sizeof(long double);
  if (type == "bool") size = sizeof(bool);
  if (type == "intmax_t") size= sizeof(intmax_t);
  if (type == "uintmax_t") size= sizeof(uintmax_t);
  if (type == "int8_t") size= sizeof(int8_t);
  if (type == "uint8_t") size= sizeof(uint8_t);
  if (type == "int16_t") size= sizeof(int16_t);
  if (type == "uint16_t") size= sizeof(uint16_t);
  if (type == "int32_t") size= sizeof(int32_t);
  if (type == "uint32_t") size= sizeof(uint32_t);
  if (type == "int64_t") size= sizeof(int64_t);
  if (type == "uint64_t") size= sizeof(uint64_t);
  if (type == "int_least8_t") size= sizeof(int_least8_t);
  if (type == "uint_least8_t") size= sizeof(uint_least8_t);
  if (type == "int_least16_t") size= sizeof(int_least16_t);
  if (type == "uint_least16_t") size= sizeof(uint_least16_t);
  if (type == "int_least32_t") size= sizeof(int_least32_t);
  if (type == "uint_least32_t") size= sizeof(uint_least32_t);
  if (type == "int_least64_t") size= sizeof(int_least64_t);
  if (type == "uint_least64_t") size= sizeof(uint_least64_t);
  if (type == "int_fast8_t") size= sizeof(int_fast8_t);
  if (type == "uint_fast8_t") size= sizeof(uint_fast8_t);
  if (type == "int_fast16_t") size= sizeof(int_fast16_t);
  if (type == "uint_fast16_t") size= sizeof(uint_fast16_t);
  if (type == "int_fast32_t") size= sizeof(int_fast32_t);
  if (type == "uint_fast32_t") size= sizeof(uint_fast32_t);
  if (type == "int_fast64_t") size= sizeof(int_fast64_t);
  if (type == "uint_fast64_t") size= sizeof(uint_fast64_t);
  return size;
}

#endif  // UTILS_CC
