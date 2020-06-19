
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

/*  @file Mrg32k3a.h
*  @brief Specific interface for the Mrg32k3a generator
*  @see clRNG_template.h
*/

#pragma once
#ifndef MRG32K3A_H
#define MRG32K3A_H

#include <clRNG/clRNG.h>
#include <stdio.h>


/*  @brief State type of a Mrg32k3a stream
*
*  The state is a seed consisting of six unsigned 32-bit integers.
*
*  @see clrngStreamState
*/
typedef struct {
	/*! @brief Seed for the first MRG component
	*/
	cl_ulong g1[3];
	/*! @brief Seed for the second MRG component
	*/
	cl_ulong g2[3];
} clrngMrg32k3aStreamState;


struct clrngMrg32k3aStream_ {
	union {
		struct {
			clrngMrg32k3aStreamState states[3];
		};
		struct {
			clrngMrg32k3aStreamState current;
			clrngMrg32k3aStreamState initial;
			clrngMrg32k3aStreamState substream;
		};
	};
};

/*! @copybrief clrngStream
*  @see clrngStream
*/
typedef struct clrngMrg32k3aStream_ clrngMrg32k3aStream;

struct clrngMrg32k3aStreamCreator_;
/*! @copybrief clrngStreamCreator
*  @see clrngStreamCreator
*/
typedef struct clrngMrg32k3aStreamCreator_ clrngMrg32k3aStreamCreator;


#ifdef __cplusplus
extern "C" {
#endif

	/*! @copybrief clrngCopyStreamCreator()
	*  @see clrngCopyStreamCreator()
	*/
	CLRNGAPI clrngMrg32k3aStreamCreator* clrngMrg32k3aCopyStreamCreator(const clrngMrg32k3aStreamCreator* creator, clrngStatus* err);

	/*! @copybrief clrngDestroyStreamCreator()
	*  @see clrngDestroyStreamCreator()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aDestroyStreamCreator(clrngMrg32k3aStreamCreator* creator);

	/*! @copybrief clrngRewindStreamCreator()
	 *  @see clrngRewindStreamCreator()
	 */
	CLRNGAPI clrngStatus clrngMrg32k3aRewindStreamCreator(clrngMrg32k3aStreamCreator* creator);

	/*! @copybrief clrngSetBaseCreatorState()
	*  @see clrngSetBaseCreatorState()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aSetBaseCreatorState(clrngMrg32k3aStreamCreator* creator, const clrngMrg32k3aStreamState* baseState);

	/*! @copybrief clrngChangeStreamsSpacing()
	*  @see clrngChangeStreamsSpacing()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aChangeStreamsSpacing(clrngMrg32k3aStreamCreator* creator, cl_int e, cl_int c);

	/*! @copybrief clrngAllocStreams()
	*  @see clrngAllocStreams()
	*/
	CLRNGAPI clrngMrg32k3aStream* clrngMrg32k3aAllocStreams(size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngDestroyStreams()
	*  @see clrngDestroyStreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aDestroyStreams(clrngMrg32k3aStream* streams);

	/*! @copybrief clrngCreateOverStreams()
	*  @see clrngCreateOverStreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aCreateOverStreams(clrngMrg32k3aStreamCreator* creator, size_t count, clrngMrg32k3aStream* streams);

	/*! @copybrief clrngCreateStreams()
	*  @see clrngCreateStreams()
	*/
	CLRNGAPI clrngMrg32k3aStream* clrngMrg32k3aCreateStreams(clrngMrg32k3aStreamCreator* creator, size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngCopyOverStreams()
	*  @see clrngCopyOverStreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aCopyOverStreams(size_t count, clrngMrg32k3aStream* destStreams, const clrngMrg32k3aStream* srcStreams);

	/*! @copybrief clrngCopyStreams()
	*  @see clrngCopyStreams()
	*/
	CLRNGAPI clrngMrg32k3aStream* clrngMrg32k3aCopyStreams(size_t count, const clrngMrg32k3aStream* streams, clrngStatus* err);

#define clrngMrg32k3aRandomU01          _CLRNG_TAG_FPTYPE(clrngMrg32k3aRandomU01)
#define clrngMrg32k3aRandomInteger      _CLRNG_TAG_FPTYPE(clrngMrg32k3aRandomInteger)
#define clrngMrg32k3aRandomU01Array     _CLRNG_TAG_FPTYPE(clrngMrg32k3aRandomU01Array)
#define clrngMrg32k3aRandomIntegerArray _CLRNG_TAG_FPTYPE(clrngMrg32k3aRandomIntegerArray)

	/*! @copybrief clrngRandomU01()
	*  @see clrngRandomU01()
	*/
	CLRNGAPI _CLRNG_FPTYPE clrngMrg32k3aRandomU01(clrngMrg32k3aStream* stream);
	CLRNGAPI cl_float  clrngMrg32k3aRandomU01_cl_float (clrngMrg32k3aStream* stream);
	CLRNGAPI cl_double clrngMrg32k3aRandomU01_cl_double(clrngMrg32k3aStream* stream);

	/*! @copybrief clrngRandomInteger()
	*  @see clrngRandomInteger()
	*/
	CLRNGAPI cl_int clrngMrg32k3aRandomInteger(clrngMrg32k3aStream* stream, cl_int i, cl_int j);
	CLRNGAPI cl_int clrngMrg32k3aRandomInteger_cl_float (clrngMrg32k3aStream* stream, cl_int i, cl_int j);
	CLRNGAPI cl_int clrngMrg32k3aRandomInteger_cl_double(clrngMrg32k3aStream* stream, cl_int i, cl_int j);

	/*! @copybrief clrngRandomU01Array()
	*  @see clrngRandomU01Array()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aRandomU01Array(clrngMrg32k3aStream* stream, size_t count, _CLRNG_FPTYPE* buffer);
	CLRNGAPI clrngStatus clrngMrg32k3aRandomU01Array_cl_float (clrngMrg32k3aStream* stream, size_t count, cl_float * buffer);
	CLRNGAPI clrngStatus clrngMrg32k3aRandomU01Array_cl_double(clrngMrg32k3aStream* stream, size_t count, cl_double* buffer);

	/*! @copybrief clrngRandomIntegerArray()
	*  @see clrngRandomIntegerArray()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aRandomIntegerArray(clrngMrg32k3aStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
	CLRNGAPI clrngStatus clrngMrg32k3aRandomIntegerArray_cl_float (clrngMrg32k3aStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
	CLRNGAPI clrngStatus clrngMrg32k3aRandomIntegerArray_cl_double(clrngMrg32k3aStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);

	/*! @copybrief clrngRewindStreams()
	*  @see clrngRewindStreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aRewindStreams(size_t count, clrngMrg32k3aStream* streams);

	/*! @copybrief clrngRewindSubstreams()
	*  @see clrngRewindSubstreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aRewindSubstreams(size_t count, clrngMrg32k3aStream* streams);

	/*! @copybrief clrngForwardToNextSubstreams()
	*  @see clrngForwardToNextSubstreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aForwardToNextSubstreams(size_t count, clrngMrg32k3aStream* streams);

	/*! @copybrief clrngMakeSubstreams()
	 *  @see clrngMakeSubstreams()
	 */
	CLRNGAPI clrngMrg32k3aStream* clrngMrg32k3aMakeSubstreams(clrngMrg32k3aStream* stream, size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngMakeOverSubstreams()
	 *  @see clrngMakeOverSubstreams()
	 */
	CLRNGAPI clrngStatus clrngMrg32k3aMakeOverSubstreams(clrngMrg32k3aStream* stream, size_t count, clrngMrg32k3aStream* substreams);

	/*! @copybrief clrngAdvanceStreams()
	*  @see clrngAdvanceStreams()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aAdvanceStreams(size_t count, clrngMrg32k3aStream* streams, cl_int e, cl_int c);

	/*! @copybrief clrngDeviceRandomU01Array()
	*  @see clrngDeviceRandomU01Array()
	*/
#ifdef CLRNG_SINGLE_PRECISION
#define clrngMrg32k3aDeviceRandomU01Array(...) clrngMrg32k3aDeviceRandomU01Array_(__VA_ARGS__, CL_TRUE)
#else
#define clrngMrg32k3aDeviceRandomU01Array(...) clrngMrg32k3aDeviceRandomU01Array_(__VA_ARGS__, CL_FALSE)
#endif

	/** \internal
	 *  @brief Helper function for clrngMrg32k3aDeviceRandomU01Array()
	 */
	CLRNGAPI clrngStatus clrngMrg32k3aDeviceRandomU01Array_(size_t streamCount, cl_mem streams,
		size_t numberCount, cl_mem outBuffer, cl_uint numQueuesAndEvents,
		cl_command_queue* commQueues, cl_uint numWaitEvents,
		const cl_event* waitEvents, cl_event* outEvents, cl_bool singlePrecision);
/** \endinternal
 */

	/*! @copybrief clrngWriteStreamInfo()
	*  @see clrngWriteStreamInfo()
	*/
	CLRNGAPI clrngStatus clrngMrg32k3aWriteStreamInfo(const clrngMrg32k3aStream* stream, FILE *file);


#if 0
	CLRNGAPI clrngMrg32k3aStream* clrngMrg32k3aGetStreamByIndex(clrngMrg32k3aStream* stream, cl_uint index);
#endif


#ifdef __cplusplus
}
#endif



#endif
