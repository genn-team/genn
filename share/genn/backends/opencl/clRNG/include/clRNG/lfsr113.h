
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

/*  @file Lfsr113.h
*  @brief Specific interface for the Lfsr113 generator
*  @see clRNG_template.h
*/

#pragma once
#ifndef LFSR113_H
#define LFSR113_H

#include <clRNG/clRNG.h>
#include <stdio.h>


/*  @brief State type of a Lfsr113 stream
*
*  The state is a seed consisting of six unsigned 32-bit integers.
*
*  @see clrngStreamState
*/
typedef struct {
	/*! @brief Seed for the first LFSR component
	*/
	cl_uint g[4];
} clrngLfsr113StreamState;


struct clrngLfsr113Stream_ {
	union {
		struct {
			clrngLfsr113StreamState states[3];
		};
		struct {
			clrngLfsr113StreamState current;
			clrngLfsr113StreamState initial;
			clrngLfsr113StreamState substream;
		};
	};
};

/*! @copybrief clrngStream
*  @see clrngStream
*/
typedef struct clrngLfsr113Stream_ clrngLfsr113Stream;

struct clrngLfsr113StreamCreator_;
/*! @copybrief clrngStreamCreator
*  @see clrngStreamCreator
*/
typedef struct clrngLfsr113StreamCreator_ clrngLfsr113StreamCreator;


#ifdef __cplusplus
extern "C" {
#endif

	/*! @copybrief clrngCopyStreamCreator()
	*  @see clrngCopyStreamCreator()
	*/
	CLRNGAPI clrngLfsr113StreamCreator* clrngLfsr113CopyStreamCreator(const clrngLfsr113StreamCreator* creator, clrngStatus* err);

	/*! @copybrief clrngDestroyStreamCreator()
	*  @see clrngDestroyStreamCreator()
	*/
	CLRNGAPI clrngStatus clrngLfsr113DestroyStreamCreator(clrngLfsr113StreamCreator* creator);

	/*! @copybrief clrngRewindStreamCreator()
	 *  @see clrngRewindStreamCreator()
	 */
	CLRNGAPI clrngStatus clrngLfsr113RewindStreamCreator(clrngLfsr113StreamCreator* creator);

	/*! @copybrief clrngSetBaseCreatorState()
	*  @see clrngSetBaseCreatorState()
	*/
	CLRNGAPI clrngStatus clrngLfsr113SetBaseCreatorState(clrngLfsr113StreamCreator* creator, const clrngLfsr113StreamState* baseState);

	/*! @copybrief clrngChangeStreamsSpacing()
	*  @see clrngChangeStreamsSpacing()
	*/
	CLRNGAPI clrngStatus clrngLfsr113ChangeStreamsSpacing(clrngLfsr113StreamCreator* creator, cl_int e, cl_int c);

	/*! @copybrief clrngAllocStreams()
	*  @see clrngAllocStreams()
	*/
	CLRNGAPI clrngLfsr113Stream* clrngLfsr113AllocStreams(size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngDestroyStreams()
	*  @see clrngDestroyStreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113DestroyStreams(clrngLfsr113Stream* streams);

	/*! @copybrief clrngCreateOverStreams()
	*  @see clrngCreateOverStreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113CreateOverStreams(clrngLfsr113StreamCreator* creator, size_t count, clrngLfsr113Stream* streams);

	/*! @copybrief clrngCreateStreams()
	*  @see clrngCreateStreams()
	*/
	CLRNGAPI clrngLfsr113Stream* clrngLfsr113CreateStreams(clrngLfsr113StreamCreator* creator, size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngCopyOverStreams()
	*  @see clrngCopyOverStreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113CopyOverStreams(size_t count, clrngLfsr113Stream* destStreams, const clrngLfsr113Stream* srcStreams);

	/*! @copybrief clrngCopyStreams()
	*  @see clrngCopyStreams()
	*/
	CLRNGAPI clrngLfsr113Stream* clrngLfsr113CopyStreams(size_t count, const clrngLfsr113Stream* streams, clrngStatus* err);

#define clrngLfsr113RandomU01          _CLRNG_TAG_FPTYPE(clrngLfsr113RandomU01)
#define clrngLfsr113RandomInteger      _CLRNG_TAG_FPTYPE(clrngLfsr113RandomInteger)
#define clrngLfsr113RandomU01Array     _CLRNG_TAG_FPTYPE(clrngLfsr113RandomU01Array)
#define clrngLfsr113RandomIntegerArray _CLRNG_TAG_FPTYPE(clrngLfsr113RandomIntegerArray)

	/*! @copybrief clrngRandomU01()
	*  @see clrngRandomU01()
	*/
	CLRNGAPI _CLRNG_FPTYPE clrngLfsr113RandomU01(clrngLfsr113Stream* stream);
	CLRNGAPI cl_float  clrngLfsr113RandomU01_cl_float (clrngLfsr113Stream* stream);
	CLRNGAPI cl_double clrngLfsr113RandomU01_cl_double(clrngLfsr113Stream* stream);

	/*! @copybrief clrngRandomInteger()
	*  @see clrngRandomInteger()
	*/
	CLRNGAPI cl_int clrngLfsr113RandomInteger(clrngLfsr113Stream* stream, cl_int i, cl_int j);
	CLRNGAPI cl_int clrngLfsr113RandomInteger_cl_float (clrngLfsr113Stream* stream, cl_int i, cl_int j);
	CLRNGAPI cl_int clrngLfsr113RandomInteger_cl_double(clrngLfsr113Stream* stream, cl_int i, cl_int j);

	/*! @copybrief clrngRandomU01Array()
	*  @see clrngRandomU01Array()
	*/
	CLRNGAPI clrngStatus clrngLfsr113RandomU01Array(clrngLfsr113Stream* stream, size_t count, _CLRNG_FPTYPE* buffer);
	CLRNGAPI clrngStatus clrngLfsr113RandomU01Array_cl_float (clrngLfsr113Stream* stream, size_t count, cl_float * buffer);
	CLRNGAPI clrngStatus clrngLfsr113RandomU01Array_cl_double(clrngLfsr113Stream* stream, size_t count, cl_double* buffer);

	/*! @copybrief clrngRandomIntegerArray()
	*  @see clrngRandomIntegerArray()
	*/
	CLRNGAPI clrngStatus clrngLfsr113RandomIntegerArray(clrngLfsr113Stream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
	CLRNGAPI clrngStatus clrngLfsr113RandomIntegerArray_cl_float (clrngLfsr113Stream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
	CLRNGAPI clrngStatus clrngLfsr113RandomIntegerArray_cl_double(clrngLfsr113Stream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);

	/*! @copybrief clrngRewindStreams()
	*  @see clrngRewindStreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113RewindStreams(size_t count, clrngLfsr113Stream* streams);

	/*! @copybrief clrngRewindSubstreams()
	*  @see clrngRewindSubstreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113RewindSubstreams(size_t count, clrngLfsr113Stream* streams);

	/*! @copybrief clrngForwardToNextSubstreams()
	*  @see clrngForwardToNextSubstreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113ForwardToNextSubstreams(size_t count, clrngLfsr113Stream* streams);

	/*! @copybrief clrngMakeSubstreams()
	 *  @see clrngMakeSubstreams()
	 */
	CLRNGAPI clrngLfsr113Stream* clrngLfsr113MakeSubstreams(clrngLfsr113Stream* stream, size_t count, size_t* bufSize, clrngStatus* err);

	/*! @copybrief clrngMakeOverSubstreams()
	 *  @see clrngMakeOverSubstreams()
	 */
	CLRNGAPI clrngStatus clrngLfsr113MakeOverSubstreams(clrngLfsr113Stream* stream, size_t count, clrngLfsr113Stream* substreams);

	/*! @copybrief clrngAdvanceStreams()
	*  @see clrngAdvanceStreams()
	*/
	CLRNGAPI clrngStatus clrngLfsr113AdvanceStreams(size_t count, clrngLfsr113Stream* streams, cl_int e, cl_int c);

	/*! @copybrief clrngDeviceRandomU01Array()
	*  @see clrngDeviceRandomU01Array()
	*/
#ifdef CLRNG_SINGLE_PRECISION
#define clrngLfsr113DeviceRandomU01Array(...) clrngLfsr113DeviceRandomU01Array_(__VA_ARGS__, CL_TRUE)
#else
#define clrngLfsr113DeviceRandomU01Array(...) clrngLfsr113DeviceRandomU01Array_(__VA_ARGS__, CL_FALSE)
#endif

	/** \internal
	 *  @brief Helper function for clrngLfsr113DeviceRandomU01Array()
	 */
	CLRNGAPI clrngStatus clrngLfsr113DeviceRandomU01Array_(size_t streamCount, cl_mem streams,
		size_t numberCount, cl_mem outBuffer, cl_uint numQueuesAndEvents,
		cl_command_queue* commQueues, cl_uint numWaitEvents,
		const cl_event* waitEvents, cl_event* outEvents, cl_bool singlePrecision);
/** \endinternal
 */

	/*! @copybrief clrngWriteStreamInfo()
	*  @see clrngWriteStreamInfo()
	*/
	CLRNGAPI clrngStatus clrngLfsr113WriteStreamInfo(const clrngLfsr113Stream* stream, FILE *file);


#if 0
	CLRNGAPI clrngLfsr113Stream* clrngLfsr113GetStreamByIndex(clrngLfsr113Stream* stream, cl_uint index);
#endif


#ifdef __cplusplus
}
#endif



#endif
