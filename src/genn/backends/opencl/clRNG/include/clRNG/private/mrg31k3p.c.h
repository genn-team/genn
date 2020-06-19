
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

/*! @file mrg31k3p.c.h
 *  @brief Code for the MRG31k3p generator common to the host and device
 */

#pragma once
#ifndef PRIVATE_MRG31K3P_CH
#define PRIVATE_MRG31K3P_CH

#define mrg31k3p_M1 2147483647             /* 2^31 - 1 */
#define mrg31k3p_M2 2147462579             /* 2^31 - 21069 */

#define mrg31k3p_MASK12 511                /* 2^9 - 1 */
#define mrg31k3p_MASK13 16777215           /* 2^24 - 1 */
#define mrg31k3p_MASK2 65535               /* 2^16 - 1 */
#define mrg31k3p_MULT2 21069

#define mrg31k3p_NORM_cl_double 4.656612873077392578125e-10  /* 1/2^31 */
#define mrg31k3p_NORM_cl_float  4.6566126e-10



#if defined(CLRNG_ENABLE_SUBSTREAMS) || !defined(__CLRNG_DEVICE_API)

// clrngMrg31k3p_A1p72 and clrngMrg31k3p_A2p72 jump 2^72 steps forward
#if defined(__CLRNG_DEVICE_API)
__constant
#else
static
#endif
cl_uint clrngMrg31k3p_A1p72[3][3] = { 
    {1516919229,  758510237, 499121365},
    {1884998244, 1516919229, 335398200},
    {601897748,  1884998244, 358115744}
};

#if defined(__CLRNG_DEVICE_API)
__constant
#else
static
#endif
cl_uint clrngMrg31k3p_A2p72[3][3] = { 
    {1228857673, 1496414766,  954677935},
    {1133297478, 1407477216, 1496414766},
    {2002613992, 1639496704, 1407477216}
};

#endif


clrngStatus clrngMrg31k3pCopyOverStreams(size_t count, clrngMrg31k3pStream* destStreams, const clrngMrg31k3pStream* srcStreams)
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

/*! @brief Advance the rng one step and returns z such that 1 <= z <= mrg31k3p_M1
 */
static cl_uint clrngMrg31k3pNextState(clrngMrg31k3pStreamState* currentState)
{
	
	cl_uint* g1 = currentState->g1;
	cl_uint* g2 = currentState->g2;
	cl_uint y1, y2;

	// first component
	y1 = ((g1[1] & mrg31k3p_MASK12) << 22) + (g1[1] >> 9)
		+ ((g1[2] & mrg31k3p_MASK13) << 7) + (g1[2] >> 24);

	if (y1 >= mrg31k3p_M1)
		y1 -= mrg31k3p_M1;

	y1 += g1[2];
	if (y1 >= mrg31k3p_M1)
		y1 -= mrg31k3p_M1;

	g1[2] = g1[1];
	g1[1] = g1[0];
	g1[0] = y1;

	// second component
	y1 = ((g2[0] & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2[0] >> 16));
	if (y1 >= mrg31k3p_M2)
		y1 -= mrg31k3p_M2;
	y2 = ((g2[2] & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2[2] >> 16));
	if (y2 >= mrg31k3p_M2)
		y2 -= mrg31k3p_M2;
	y2 += g2[2];
	if (y2 >= mrg31k3p_M2)
		y2 -= mrg31k3p_M2;
	y2 += y1;
	if (y2 >= mrg31k3p_M2)
		y2 -= mrg31k3p_M2;

	g2[2] = g2[1];
	g2[1] = g2[0];
	g2[0] = y2;

	if (g1[0] <= g2[0])
		return (g1[0] - g2[0] + mrg31k3p_M1);
	else
		return (g1[0] - g2[0]);
}

// The following would be much cleaner with C++ templates instead of macros.

// We use an underscore on the r.h.s. to avoid potential recursion with certain
// preprocessors.
#define IMPLEMENT_GENERATE_FOR_TYPE(fptype) \
	\
	fptype clrngMrg31k3pRandomU01_##fptype(clrngMrg31k3pStream* stream) { \
	    return clrngMrg31k3pNextState(&stream->current) * mrg31k3p_NORM_##fptype; \
	} \
	\
	cl_int clrngMrg31k3pRandomInteger_##fptype(clrngMrg31k3pStream* stream, cl_int i, cl_int j) { \
	    return i + (cl_int)((j - i + 1) * clrngMrg31k3pRandomU01_##fptype(stream)); \
	} \
	\
	clrngStatus clrngMrg31k3pRandomU01Array_##fptype(clrngMrg31k3pStream* stream, size_t count, fptype* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t i = 0; i < count; i++)  \
			buffer[i] = clrngMrg31k3pRandomU01_##fptype(stream); \
		return CLRNG_SUCCESS; \
	} \
	\
	clrngStatus clrngMrg31k3pRandomIntegerArray_##fptype(clrngMrg31k3pStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t k = 0; k < count; k++) \
			buffer[k] = clrngMrg31k3pRandomInteger_##fptype(stream, i, j); \
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



clrngStatus clrngMrg31k3pRewindStreams(size_t count, clrngMrg31k3pStream* streams)
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
clrngStatus clrngMrg31k3pRewindSubstreams(size_t count, clrngMrg31k3pStream* streams)
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

clrngStatus clrngMrg31k3pForwardToNextSubstreams(size_t count, clrngMrg31k3pStream* streams)
{
	//Check params
	if (!streams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);
	
	for (size_t k = 0; k < count; k++) {
		modMatVec (clrngMrg31k3p_A1p72, streams[k].substream.g1, streams[k].substream.g1, mrg31k3p_M1);
		modMatVec (clrngMrg31k3p_A2p72, streams[k].substream.g2, streams[k].substream.g2, mrg31k3p_M2);
		streams[k].current = streams[k].substream;
	}

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg31k3pMakeOverSubstreams(clrngMrg31k3pStream* stream, size_t count, clrngMrg31k3pStream* substreams)
{
	for (size_t i = 0; i < count; i++) {
		clrngStatus err;
		// snapshot current stream into substreams[i]
		err = clrngMrg31k3pCopyOverStreams(1, &substreams[i], stream);
		if (err != CLRNG_SUCCESS)
		    return err;
		// advance to next substream
		err = clrngMrg31k3pForwardToNextSubstreams(1, stream);
		if (err != CLRNG_SUCCESS)
		    return err;
	}
	return CLRNG_SUCCESS;
}
#endif // substreams

#endif // PRIVATE_MRG31K3P_CH
