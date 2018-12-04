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

size_t theSize(const string &type)
{
    if (type.find("*") != string::npos) return sizeof(char *); // it's a pointer ... any pointer should have the same size
    else if (type == "char") return sizeof(char);
    //  else if (type == "char16_t") size = sizeof(char16_t);
    //  else if (type == "char32_t") size = sizeof(char32_t);
    else if (type == "wchar_t") return sizeof(wchar_t);
    else if (type == "signed char") return sizeof(signed char);
    else if (type == "short") return sizeof(short);
    else if (type == "signed short") return sizeof(signed short);
    else if (type == "short int") return sizeof(short int);
    else if (type == "signed short int") return sizeof(signed short int);
    else if (type == "int") return sizeof(int);
    else if (type == "signed int") return sizeof(signed int);
    else if (type == "long") return sizeof(long);
    else if (type == "signed long") return sizeof(signed long);
    else if (type == "long int") return sizeof(long int);
    else if (type == "signed long int") return sizeof(signed long int);
    else if (type == "long long") return sizeof(long long);
    else if (type == "signed long long") return sizeof(signed long long);
    else if (type == "long long int") return sizeof(long long int);
    else if (type == "signed long long int") return sizeof(signed long long int);
    else if (type == "unsigned char") return sizeof(unsigned char);
    else if (type == "unsigned short") return sizeof(unsigned short);
    else if (type == "unsigned short int") return sizeof(unsigned short int);
    else if (type == "unsigned") return sizeof(unsigned);
    else if (type == "unsigned int") return sizeof(unsigned int);
    else if (type == "unsigned long") return sizeof(unsigned long);
    else if (type == "unsigned long int") return sizeof(unsigned long int);
    else if (type == "unsigned long long") return sizeof(unsigned long long);
    else if (type == "unsigned long long int") return sizeof(unsigned long long int);
    else if (type == "float") return sizeof(float);
    else if (type == "double") return sizeof(double);
    else if (type == "long double") return sizeof(long double);
    else if (type == "bool") return sizeof(bool);
    else if (type == "intmax_t") return sizeof(intmax_t);
    else if (type == "uintmax_t") return sizeof(uintmax_t);
    else if (type == "int8_t") return sizeof(int8_t);
    else if (type == "uint8_t") return sizeof(uint8_t);
    else if (type == "int16_t") return sizeof(int16_t);
    else if (type == "uint16_t") return sizeof(uint16_t);
    else if (type == "int32_t") return sizeof(int32_t);
    else if (type == "uint32_t") return sizeof(uint32_t);
    else if (type == "int64_t") return sizeof(int64_t);
    else if (type == "uint64_t") return sizeof(uint64_t);
    else if (type == "int_least8_t") return sizeof(int_least8_t);
    else if (type == "uint_least8_t") return sizeof(uint_least8_t);
    else if (type == "int_least16_t") return sizeof(int_least16_t);
    else if (type == "uint_least16_t") return sizeof(uint_least16_t);
    else if (type == "int_least32_t") return sizeof(int_least32_t);
    else if (type == "uint_least32_t") return sizeof(uint_least32_t);
    else if (type == "int_least64_t") return sizeof(int_least64_t);
    else if (type == "uint_least64_t") return sizeof(uint_least64_t);
    else if (type == "int_fast8_t") return sizeof(int_fast8_t);
    else if (type == "uint_fast8_t") return sizeof(uint_fast8_t);
    else if (type == "int_fast16_t") return sizeof(int_fast16_t);
    else if (type == "uint_fast16_t") return sizeof(uint_fast16_t);
    else if (type == "int_fast32_t") return sizeof(int_fast32_t);
    else if (type == "uint_fast32_t") return sizeof(uint_fast32_t);
    else if (type == "int_fast64_t") return sizeof(int_fast64_t);
    else if (type == "uint_fast64_t") return sizeof(uint_fast64_t);
    else if (type == "curandState") return 44;
    else if (type == "curandStatePhilox4_32_10_t") return 64;
    else if (type == "scalar") return sizeof(float);
    else return 0;
}

#endif  // UTILS_CC
