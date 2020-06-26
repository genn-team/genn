
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

/*! @file Lfsr113.c.h
*  @brief Code for the Lfsr113 generator common to the host and device
*/

#pragma once
#ifndef PRIVATE_LFSR113_CH
#define PRIVATE_LFSR113_CH

#define Lfsr113_NORM_cl_double 1.0 / 0x100000001L   // 2^32 + 1    // 2.328306436538696e-10
#define Lfsr113_NORM_cl_float  2.3283063e-10


clrngStatus clrngLfsr113CopyOverStreams(size_t count, clrngLfsr113Stream* destStreams, const clrngLfsr113Stream* srcStreams)
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

/*! @brief Advance the rng one step and returns z such that 1 <= z <= lfsr113_M1
*/
static cl_ulong clrngLfsr113NextState(clrngLfsr113StreamState *currentState) {

	cl_ulong b;

	b = (((currentState->g[0] << 6) ^ currentState->g[0]) >> 13);
	currentState->g[0] = (((currentState->g[0] & 4294967294U) << 18) ^ b);

	b = (((currentState->g[1] << 2) ^ currentState->g[1]) >> 27);
	currentState->g[1] = (((currentState->g[1] & 4294967288U) << 2) ^ b);

	b = (((currentState->g[2] << 13) ^ currentState->g[2]) >> 21);
	currentState->g[2] = (((currentState->g[2] & 4294967280U) << 7) ^ b);

	b = (((currentState->g[3] << 3) ^ currentState->g[3]) >> 12);
	currentState->g[3] = (((currentState->g[3] & 4294967168U) << 13) ^ b);

	return (currentState->g[0] ^ currentState->g[1] ^ currentState->g[2] ^ currentState->g[3]);

}

// The following would be much cleaner with C++ templates instead of macros.

// We use an underscore on the r.h.s. to avoid potential recursion with certain
// preprocessors.
#define IMPLEMENT_GENERATE_FOR_TYPE(fptype) \
	\
	fptype clrngLfsr113RandomU01_##fptype(clrngLfsr113Stream* stream) { \
	    return clrngLfsr113NextState(&stream->current) * Lfsr113_NORM_##fptype; \
	} \
	\
	cl_int clrngLfsr113RandomInteger_##fptype(clrngLfsr113Stream* stream, cl_int i, cl_int j) { \
	    return i + (cl_int)((j - i + 1) * clrngLfsr113RandomU01_##fptype(stream)); \
	} \
	\
	clrngStatus clrngLfsr113RandomU01Array_##fptype(clrngLfsr113Stream* stream, size_t count, fptype* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t i = 0; i < count; i++)  \
			buffer[i] = clrngLfsr113RandomU01_##fptype(stream); \
		return CLRNG_SUCCESS; \
	} \
	\
	clrngStatus clrngLfsr113RandomIntegerArray_##fptype(clrngLfsr113Stream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer) { \
		if (!stream) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__); \
		if (!buffer) \
			return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__); \
		for (size_t k = 0; k < count; k++) \
			buffer[k] = clrngLfsr113RandomInteger_##fptype(stream, i, j); \
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



clrngStatus clrngLfsr113RewindStreams(size_t count, clrngLfsr113Stream* streams)
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
clrngStatus clrngLfsr113RewindSubstreams(size_t count, clrngLfsr113Stream* streams)
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
void lfsr113ResetNextSubStream(clrngLfsr113Stream* stream){

	/* The following operations make the jump ahead with
	2 ^ 55 iterations for every component of the generator.
	The internal state after the jump, however, is slightly different
	from 2 ^ 55 iterations since it ignores the state in
	which are found the first bits of each components,
	since they are ignored in the recurrence.The state becomes
	identical to what one would with normal iterations
	after a call nextValue().*/

	int z, b;

	cl_uint* subStreamState = stream->substream.g;

	//Calculate the first component
	z = subStreamState[0] & (cl_uint)-2;
	b = (z << 6) ^ z;

	z = (z) ^ (z << 3) ^ (z << 4) ^ (z << 6) ^ (z << 7) ^
		(z << 8) ^ (z << 10) ^ (z << 11) ^ (z << 13) ^ (z << 14) ^
		(z << 16) ^ (z << 17) ^ (z << 18) ^ (z << 22) ^
		(z << 24) ^ (z << 25) ^ (z << 26) ^ (z << 28) ^ (z << 30);

	z ^= ((b >> 1) & 0x7FFFFFFF) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 5) & 0x07FFFFFF) ^
		((b >> 6) & 0x03FFFFFF) ^
		((b >> 7) & 0x01FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 13) & 0x0007FFFF) ^
		((b >> 14) & 0x0003FFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 17) & 0x00007FFF) ^
		((b >> 18) & 0x00003FFF) ^
		((b >> 20) & 0x00000FFF) ^
		((b >> 21) & 0x000007FF) ^
		((b >> 23) & 0x000001FF) ^
		((b >> 24) & 0x000000FF) ^
		((b >> 25) & 0x0000007F) ^
		((b >> 26) & 0x0000003F) ^
		((b >> 27) & 0x0000001F) ^
		((b >> 30) & 0x00000003);
	subStreamState[0] = z;

	//Calculate the second component
	z = subStreamState[1] & (cl_uint)-8;
	b = z ^ (z << 1);
	b ^= (b << 2);
	b ^= (b << 4);
	b ^= (b << 8);

	b <<= 8;
	b ^= (z << 22) ^ (z << 25) ^ (z << 27);
	if ((z & 0x80000000) != 0) b ^= 0xABFFF000;
	if ((z & 0x40000000) != 0) b ^= 0x55FFF800;

	z = b ^ ((z >> 7) & 0x01FFFFFF) ^
		((z >> 20) & 0x00000FFF) ^
		((z >> 21) & 0x000007FF);

	subStreamState[1] = z;

	//Calculate the third component
	z = subStreamState[2] & (cl_uint)-16;
	b = (z << 13) ^ z;
	z = ((b >> 3) & 0x1FFFFFFF) ^
		((b >> 17) & 0x00007FFF) ^
		(z << 10) ^ (z << 11) ^ (z << 25);
	subStreamState[2] = z;

	//Calculate the forth component
	z = subStreamState[3] & (cl_uint)-128;
	b = (z << 3) ^ z;
	z = (z << 14) ^ (z << 16) ^ (z << 20) ^
		((b >> 5) & 0x07FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 11) & 0x001FFFFF);
	subStreamState[3] = z;

	clrngLfsr113RewindSubstreams(1, stream);
}
clrngStatus clrngLfsr113ForwardToNextSubstreams(size_t count, clrngLfsr113Stream* streams)
{
	//Check params
	if (!streams)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);

	for (size_t k = 0; k < count; k++) {

		lfsr113ResetNextSubStream(&streams[k]);
	}

	return CLRNG_SUCCESS;
}

clrngStatus clrngLfsr113MakeOverSubstreams(clrngLfsr113Stream* stream, size_t count, clrngLfsr113Stream* substreams)
{
	for (size_t i = 0; i < count; i++) {
		clrngStatus err;
		// snapshot current stream into substreams[i]
		err = clrngLfsr113CopyOverStreams(1, &substreams[i], stream);
		if (err != CLRNG_SUCCESS)
		    return err;
		// advance to next substream
		err = clrngLfsr113ForwardToNextSubstreams(1, stream);
		if (err != CLRNG_SUCCESS)
		    return err;
	}
	return CLRNG_SUCCESS;
}

#endif

#endif // PRIVATE_Lfsr113_CH
