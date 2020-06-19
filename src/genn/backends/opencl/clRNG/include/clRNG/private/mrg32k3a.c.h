
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

/*! @file Mrg32k3a.c.h
*  @brief Code for the Mrg32k3a generator common to the host and device
*/
#pragma once
#ifndef PRIVATE_MRG32K3A_CH
#define PRIVATE_MRG32K3A_CH

#define Mrg32k3a_M1 4294967087            
#define Mrg32k3a_M2 4294944443             

#define Mrg32k3a_NORM_cl_double 2.328306549295727688e-10
#define Mrg32k3a_NORM_cl_float  2.3283064e-10

#if defined(CLRNG_ENABLE_SUBSTREAMS) || !defined(__CLRNG_DEVICE_API)

// clrngMrg32k3a_A1p76 and clrngMrg32k3a_A2p76 jump 2^76 steps forward
#if defined(__CLRNG_DEVICE_API)
__constant
#else
static
#endif
cl_ulong clrngMrg32k3a_A1p76[3][3] = {
	{ 82758667, 1871391091, 4127413238 },
	{ 3672831523, 69195019, 1871391091 },
	{ 3672091415, 3528743235, 69195019 }
};

#if defined(__CLRNG_DEVICE_API)
__constant
#else
static
#endif
cl_ulong clrngMrg32k3a_A2p76[3][3] = {
	{ 1511326704, 3759209742, 1610795712 },
	{ 4292754251, 1511326704, 3889917532 },
	{ 3859662829, 4292754251, 3708466080 }
};

#endif


clrngStatus clrngMrg32k3aCopyOverStreams(size_t count, clrngMrg32k3aStream* destStreams, const clrngMrg32k3aStream* srcStreams)
{
	//Check params
	if (!destStreams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): destStreams cannot be NULL", __func__);
	if (!srcStreams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): srcStreams cannot be NULL", __func__);

	for (size_t i = 0; i < count; i++)
		destStreams[i] = srcStreams[i];

	return CLRNG_SUCCESS;
}

/*! @brief Advance the rng one step and returns z such that 1 <= z <= Mrg32k3a_M1
*/
static cl_ulong clrngMrg32k3aNextState(clrngMrg32k3aStreamState* currentState)
{

	cl_ulong* g1 = currentState->g1;
	cl_ulong* g2 = currentState->g2;

	cl_long p0, p1;

	/* component 1 */
	p0 = 1403580 * g1[1] - 810728 * g1[0];
	p0 %= Mrg32k3a_M1;
	if (p0 < 0)
		p0 += Mrg32k3a_M1;
	g1[0] = g1[1];
	g1[1] = g1[2];
	g1[2] = p0;

	/* component 2 */
	p1 = 527612 * g2[2] - 1370589 * g2[0];
	p1 %= Mrg32k3a_M2;
	if (p1 < 0)
		p1 += Mrg32k3a_M2;
	g2[0] = g2[1];
	g2[1] = g2[2];
	g2[2] = p1;

	/* combinations */
	if (p0 > p1)
		return (p0 - p1);
	else return (p0 - p1 + Mrg32k3a_M1);
}


// The following would be much cleaner with C++ templates instead of macros.

// We use an underscore on the r.h.s. to avoid potential recursion with certain
// preprocessors.
#define IMPLEMENT_GENERATE_FOR_TYPE(fptype) \
	\
	fptype clrngMrg32k3aRandomU01_##fptype(clrngMrg32k3aStream* stream) { \
	    return clrngMrg32k3aNextState(&stream->current) * Mrg32k3a_NORM_##fptype; \
	} \
	\
	cl_int clrngMrg32k3aRandomInteger_##fptype(clrngMrg32k3aStream* stream, cl_int i, cl_int j) { \
	    return i + (cl_int)((j - i + 1) * clrngMrg32k3aRandomU01_##fptype(stream)); \
	} \
	\
	clrngStatus clrngMrg32k3aRandomU01Array_##fptype(clrngMrg32k3aStream* stream, size_t count, fptype* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t i = 0; i < count; i++)  \
			buffer[i] = clrngMrg32k3aRandomU01_##fptype(stream); \
		return CLRNG_SUCCESS; \
	} \
	\
	clrngStatus clrngMrg32k3aRandomIntegerArray_##fptype(clrngMrg32k3aStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t k = 0; k < count; k++) \
			buffer[k] = clrngMrg32k3aRandomInteger_##fptype(stream, i, j); \
		return CLRNG_SUCCESS; \
	}

// On the host, implement everything.
// On the device, implement only what is required to avoid cluttering memory.
#if defined(CLRNG_SINGLE_PRECISION)  || !defined(__CLRNG_DEVICE_API)
IMPLEMENT_GENERATE_FOR_TYPE(cl_float)
#endif
#if !defined(CLRNG_SINGLE_PRECISION) || !defined(__CLRNG_DEVICE_API)
IMPLEMENT_GENERATE_FOR_TYPE(cl_double)
#endif

// Clean up macros, especially to avoid polluting device code.
#undef IMPLEMENT_GENERATE_FOR_TYPE



clrngStatus clrngMrg32k3aRewindStreams(size_t count, clrngMrg32k3aStream* streams)
{
	//Check params
	if (!streams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);
	//Reset current state to the stream initial state
	for (size_t j = 0; j < count; j++) {
#ifdef __CLRNG_DEVICE_API
#ifdef CLRNG_ENABLE_SUBSTREAMS
		streams[j].current = streams[j].substream = *streams[j].initial;
#else
		streams[j].current = *streams[j].initial;
#endif
#else
		streams[j].current = streams[j].substream = streams[j].initial;
#endif
	}

	return CLRNG_SUCCESS;
}

#if defined(CLRNG_ENABLE_SUBSTREAMS) || !defined(__CLRNG_DEVICE_API)
clrngStatus clrngMrg32k3aRewindSubstreams(size_t count, clrngMrg32k3aStream* streams)
{
	//Check params
	if (!streams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);
	//Reset current state to the subStream initial state
	for (size_t j = 0; j < count; j++) {
		streams[j].current = streams[j].substream;
	}

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aForwardToNextSubstreams(size_t count, clrngMrg32k3aStream* streams)
{
	//Check params
	if (!streams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);

	for (size_t k = 0; k < count; k++) {
		modMatVec(clrngMrg32k3a_A1p76, streams[k].substream.g1, streams[k].substream.g1, Mrg32k3a_M1);
		modMatVec(clrngMrg32k3a_A2p76, streams[k].substream.g2, streams[k].substream.g2, Mrg32k3a_M2);
		streams[k].current = streams[k].substream;
	}

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aMakeOverSubstreams(clrngMrg32k3aStream* stream, size_t count, clrngMrg32k3aStream* substreams)
{
	for (size_t i = 0; i < count; i++) {
		clrngStatus err;
		// snapshot current stream into substreams[i]
		err = clrngMrg32k3aCopyOverStreams(1, &substreams[i], stream);
		if (err != CLRNG_SUCCESS)
		    return err;
		// advance to next substream
		err = clrngMrg32k3aForwardToNextSubstreams(1, stream);
		if (err != CLRNG_SUCCESS)
		    return err;
	}
	return CLRNG_SUCCESS;
}
#endif

#endif // PRIVATE_Mrg32k3a_CH
