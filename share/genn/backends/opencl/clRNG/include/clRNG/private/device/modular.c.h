
 /*
 ***********************************************************************
 Copyright (c) 2015 Advanced Micro Devices, Inc. 
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without 
 modification, are permitted provided that the following conditions 
 are met:
 
 1. Redistributions of source code must retain the above copyright 
 notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright 
 notice, this list of conditions and the following disclaimer in the 
 documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 ***********************************************************************
 */

/*! @file Modular arithmetic and linear algebra
 *
 *  This file provides the code common to the host and device.
 *
 *  The preprocessor symbol `MODULAR_NUMBER_TYPE` must be defined as the type
 *  of number (cl_uint, cl_ulong, etc.) on which the modular functions operate.
 *
 *  To use the fixed size variant, the preprocessor constant
 *  `MODULAR_FIXED_SIZE` must be set to the size (number of rows or of columns)
 *  of the matrix.
 *
 *  @note If the project is migrated to C++, this could be rewritten much more
 *  clearly using templates.
 */

#pragma once
#ifndef PRIVATE_MODULAR_CH

#ifndef MODULAR_NUMBER_TYPE
#error "MODULAR_NUMBER_TYPE must be defined"
#endif

#ifdef MODULAR_FIXED_SIZE
#  define N MODULAR_FIXED_SIZE
#  define MATRIX_ELEM(mat, i, j) (mat[i][j])
#else
#  define MATRIX_ELEM(mat, i, j) (mat[i * N + j])
#endif // MODULAR_FIXED_SIZE

//! Compute (a*s + c) % m
#if 1
#define modMult(a, s, c, m) ((MODULAR_NUMBER_TYPE)(((cl_ulong) a * s + c) % m))
#else
static MODULAR_NUMBER_TYPE modMult(MODULAR_NUMBER_TYPE a, MODULAR_NUMBER_TYPE s, MODULAR_NUMBER_TYPE c, MODULAR_NUMBER_TYPE m)
{
    MODULAR_NUMBER_TYPE v;
    v = (MODULAR_NUMBER_TYPE) (((cl_ulong) a * s + c) % m);
    return v;
}
#endif


//! @brief Matrix-vector modular multiplication
//  @details Also works if v = s.
//  @return v = A*s % m
#ifdef MODULAR_FIXED_SIZE
static void modMatVec (MODULAR_NUMBER_TYPE A[N][N], MODULAR_NUMBER_TYPE s[N], MODULAR_NUMBER_TYPE v[N], MODULAR_NUMBER_TYPE m)
#else
void modMatVec (size_t N, MODULAR_NUMBER_TYPE* A, MODULAR_NUMBER_TYPE* s, MODULAR_NUMBER_TYPE* v, MODULAR_NUMBER_TYPE m)
#endif
{
    MODULAR_NUMBER_TYPE x[MODULAR_FIXED_SIZE];     // Necessary if v = s
    for (size_t i = 0; i < N; ++i) {
        x[i] = 0;
        for (size_t j = 0; j < N; j++)
            x[i] = modMult(MATRIX_ELEM(A,i,j), s[j], x[i], m);
    }
    for (size_t i = 0; i < N; ++i)
        v[i] = x[i];
}

#undef MATRIX_ELEM
#undef N

#endif
