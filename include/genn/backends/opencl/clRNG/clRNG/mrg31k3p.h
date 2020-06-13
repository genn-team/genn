
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

/*  @file mrg31k3p.h
 *  @brief Specific interface for the MRG31k3p generator
 *  @see clRNG_template.h
 */

#pragma once
#ifndef MRG31K3P_H
#define MRG31K3P_H

#include <clRNG/clRNG.h>
#include <stdio.h>


/*  @brief State type of a MRG31k3p stream
 *
 *  The state is a seed consisting of six unsigned 32-bit integers.
 *
 *  @see clrngStreamState
 */
typedef struct {
    /*! @brief Seed for the first MRG component
     */
    cl_uint g1[3];
    /*! @brief Seed for the second MRG component
     */
    cl_uint g2[3];
} clrngMrg31k3pStreamState;


struct clrngMrg31k3pStream_ {
	union {
		struct {
			clrngMrg31k3pStreamState states[3];
		};
		struct {
			clrngMrg31k3pStreamState current;
			clrngMrg31k3pStreamState initial;
			clrngMrg31k3pStreamState substream;
		};
	};
};

/*! @copybrief clrngStream
 *  @see clrngStream
 */
typedef struct clrngMrg31k3pStream_ clrngMrg31k3pStream;

struct clrngMrg31k3pStreamCreator_;
/*! @copybrief clrngStreamCreator
 *  @see clrngStreamCreator
 */
typedef struct clrngMrg31k3pStreamCreator_ clrngMrg31k3pStreamCreator;


#ifdef __cplusplus
extern "C" {
#endif

/*! @copybrief clrngCopyStreamCreator()
 *  @see clrngCopyStreamCreator()
 */
CLRNGAPI clrngMrg31k3pStreamCreator* clrngMrg31k3pCopyStreamCreator(const clrngMrg31k3pStreamCreator* creator, clrngStatus* err);

/*! @copybrief clrngDestroyStreamCreator()
 *  @see clrngDestroyStreamCreator()
 */
CLRNGAPI clrngStatus clrngMrg31k3pDestroyStreamCreator(clrngMrg31k3pStreamCreator* creator);

/*! @copybrief clrngRewindStreamCreator()
 *  @see clrngRewindStreamCreator()
 */
CLRNGAPI clrngStatus clrngMrg31k3pRewindStreamCreator(clrngMrg31k3pStreamCreator* creator);

/*! @copybrief clrngSetBaseCreatorState()
 *  @see clrngSetBaseCreatorState()
 */
CLRNGAPI clrngStatus clrngMrg31k3pSetBaseCreatorState(clrngMrg31k3pStreamCreator* creator, const clrngMrg31k3pStreamState* baseState);

/*! @copybrief clrngChangeStreamsSpacing()
 *  @see clrngChangeStreamsSpacing()
 */
CLRNGAPI clrngStatus clrngMrg31k3pChangeStreamsSpacing(clrngMrg31k3pStreamCreator* creator, cl_int e, cl_int c);

/*! @copybrief clrngAllocStreams()
 *  @see clrngAllocStreams()
 */
CLRNGAPI clrngMrg31k3pStream* clrngMrg31k3pAllocStreams(size_t count, size_t* bufSize, clrngStatus* err);

/*! @copybrief clrngDestroyStreams()
 *  @see clrngDestroyStreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pDestroyStreams(clrngMrg31k3pStream* streams);

/*! @copybrief clrngCreateOverStreams()
 *  @see clrngCreateOverStreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pCreateOverStreams(clrngMrg31k3pStreamCreator* creator, size_t count, clrngMrg31k3pStream* streams);

/*! @copybrief clrngCreateStreams()
 *  @see clrngCreateStreams()
 */
CLRNGAPI clrngMrg31k3pStream* clrngMrg31k3pCreateStreams(clrngMrg31k3pStreamCreator* creator, size_t count, size_t* bufSize, clrngStatus* err);

/*! @copybrief clrngCopyOverStreams()
 *  @see clrngCopyOverStreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pCopyOverStreams(size_t count, clrngMrg31k3pStream* destStreams, const clrngMrg31k3pStream* srcStreams);

/*! @copybrief clrngCopyStreams()
 *  @see clrngCopyStreams()
 */
CLRNGAPI clrngMrg31k3pStream* clrngMrg31k3pCopyStreams(size_t count, const clrngMrg31k3pStream* streams, clrngStatus* err);

#define clrngMrg31k3pRandomU01          _CLRNG_TAG_FPTYPE(clrngMrg31k3pRandomU01)
#define clrngMrg31k3pRandomInteger      _CLRNG_TAG_FPTYPE(clrngMrg31k3pRandomInteger)
#define clrngMrg31k3pRandomU01Array     _CLRNG_TAG_FPTYPE(clrngMrg31k3pRandomU01Array)
#define clrngMrg31k3pRandomIntegerArray _CLRNG_TAG_FPTYPE(clrngMrg31k3pRandomIntegerArray)

/*! @copybrief clrngRandomU01()
 *  @see clrngRandomU01()
 */
CLRNGAPI _CLRNG_FPTYPE clrngMrg31k3pRandomU01(clrngMrg31k3pStream* stream);
CLRNGAPI cl_float  clrngMrg31k3pRandomU01_cl_float (clrngMrg31k3pStream* stream);
CLRNGAPI cl_double clrngMrg31k3pRandomU01_cl_double(clrngMrg31k3pStream* stream);

/*! @copybrief clrngRandomInteger()
 *  @see clrngRandomInteger()
 */
CLRNGAPI cl_int clrngMrg31k3pRandomInteger(clrngMrg31k3pStream* stream, cl_int i, cl_int j);
CLRNGAPI cl_int clrngMrg31k3pRandomInteger_cl_float (clrngMrg31k3pStream* stream, cl_int i, cl_int j);
CLRNGAPI cl_int clrngMrg31k3pRandomInteger_cl_double(clrngMrg31k3pStream* stream, cl_int i, cl_int j);

/*! @copybrief clrngRandomU01Array()
 *  @see clrngRandomU01Array()
 */
CLRNGAPI clrngStatus clrngMrg31k3pRandomU01Array(clrngMrg31k3pStream* stream, size_t count, _CLRNG_FPTYPE* buffer);
CLRNGAPI clrngStatus clrngMrg31k3pRandomU01Array_cl_float (clrngMrg31k3pStream* stream, size_t count, cl_float * buffer);
CLRNGAPI clrngStatus clrngMrg31k3pRandomU01Array_cl_double(clrngMrg31k3pStream* stream, size_t count, cl_double* buffer);

/*! @copybrief clrngRandomIntegerArray()
 *  @see clrngRandomIntegerArray()
 */
CLRNGAPI clrngStatus clrngMrg31k3pRandomIntegerArray(clrngMrg31k3pStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
CLRNGAPI clrngStatus clrngMrg31k3pRandomIntegerArray_cl_float (clrngMrg31k3pStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);
CLRNGAPI clrngStatus clrngMrg31k3pRandomIntegerArray_cl_double(clrngMrg31k3pStream* stream, cl_int i, cl_int j, size_t count, cl_int* buffer);

/*! @copybrief clrngRewindStreams()
 *  @see clrngRewindStreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pRewindStreams(size_t count, clrngMrg31k3pStream* streams);

/*! @copybrief clrngRewindSubstreams()
 *  @see clrngRewindSubstreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pRewindSubstreams(size_t count, clrngMrg31k3pStream* streams);

/*! @copybrief clrngForwardToNextSubstreams()
 *  @see clrngForwardToNextSubstreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pForwardToNextSubstreams(size_t count, clrngMrg31k3pStream* streams);

/*! @copybrief clrngMakeSubstreams()
 *  @see clrngMakeSubstreams()
 */
CLRNGAPI clrngMrg31k3pStream* clrngMrg31k3pMakeSubstreams(clrngMrg31k3pStream* stream, size_t count, size_t* bufSize, clrngStatus* err);

/*! @copybrief clrngMakeOverSubstreams()
 *  @see clrngMakeOverSubstreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pMakeOverSubstreams(clrngMrg31k3pStream* stream, size_t count, clrngMrg31k3pStream* substreams);

/*! @copybrief clrngAdvanceStreams()
 *  @see clrngAdvanceStreams()
 */
CLRNGAPI clrngStatus clrngMrg31k3pAdvanceStreams(size_t count, clrngMrg31k3pStream* streams, cl_int e, cl_int c);

/*! @copybrief clrngDeviceRandomU01Array()
 *  @see clrngDeviceRandomU01Array()
 */
#ifdef CLRNG_SINGLE_PRECISION
#define clrngMrg31k3pDeviceRandomU01Array(...) clrngMrg31k3pDeviceRandomU01Array_(__VA_ARGS__, CL_TRUE)
#else
#define clrngMrg31k3pDeviceRandomU01Array(...) clrngMrg31k3pDeviceRandomU01Array_(__VA_ARGS__, CL_FALSE)
#endif

/** \internal
 *  @brief Helper function for clrngMrg31k3pDeviceRandomU01Array()
 */
CLRNGAPI clrngStatus clrngMrg31k3pDeviceRandomU01Array_(size_t streamCount, cl_mem streams,
	size_t numberCount, cl_mem outBuffer, cl_uint numQueuesAndEvents,
	cl_command_queue* commQueues, cl_uint numWaitEvents,
	const cl_event* waitEvents, cl_event* outEvents, cl_bool singlePrecision);
/** \endinternal
 */

/*! @copybrief clrngWriteStreamInfo()
 *  @see clrngWriteStreamInfo()
 */
CLRNGAPI clrngStatus clrngMrg31k3pWriteStreamInfo(const clrngMrg31k3pStream* stream, FILE *file);


#if 0
CLRNGAPI clrngMrg31k3pStream* clrngMrg31k3pGetStreamByIndex(clrngMrg31k3pStream* stream, cl_uint index);
#endif


#ifdef __cplusplus
}
#endif



#endif
