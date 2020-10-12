
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

#include <clRNG/lfsr113.h>

#include "private.h"
#include <stdlib.h>

#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif



struct clrngLfsr113StreamCreator_ {
	clrngLfsr113StreamState initialState;
	clrngLfsr113StreamState nextState;
};

// code that is common to host and device
#include "clRNG/private/lfsr113.c.h"


/*! @brief Default initial seed of the first stream
*/
#define BASE_CREATOR_STATE { 987654321, 987654321, 987654321, 987654321 }


/*! @brief Default stream creator (defaults to \f$2^{134}\f$ steps forward)
*
*  Contains the default seed and the transition matrices to jump \f$\nu\f$ steps forward;
*  adjacent streams are spaced nu steps apart.
*  The default is \f$nu = 2^{134}\f$.
*  The default seed is \f$(12345,12345,12345,12345,12345,12345)\f$.
*/
static clrngLfsr113StreamCreator defaultStreamCreator = {
	{ BASE_CREATOR_STATE },
	{ BASE_CREATOR_STATE }
};

/*! @brief Check the validity of a seed for Lfsr113
*/
static clrngStatus validateSeed(const clrngLfsr113StreamState* seed)
{
	// Check that the seeds have valid values
	if (seed->g[0] < 2)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g[%u] must be greater than 1", 0);

	if (seed->g[1] < 8)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g[%u] must be greater than 7", 1);

	if (seed->g[2] < 16)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g[%u] must be greater than 15", 2);

	if (seed->g[3] < 128)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g[%u] must be greater than 127", 3);

	return CLRNG_SUCCESS;
}

clrngLfsr113StreamCreator* clrngLfsr113CopyStreamCreator(const clrngLfsr113StreamCreator* creator, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;

	// allocate creator
	clrngLfsr113StreamCreator* newCreator = (clrngLfsr113StreamCreator*)malloc(sizeof(clrngLfsr113StreamCreator));

	if (newCreator == NULL)
		// allocation failed
		err_ = clrngSetErrorString(CLRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for stream creator", __func__);
	else {
		if (creator == NULL)
			creator = &defaultStreamCreator;
		// initialize creator
		*newCreator = *creator;
	}

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return newCreator;
}

clrngStatus clrngLfsr113DestroyStreamCreator(clrngLfsr113StreamCreator* creator)
{
	if (creator != NULL)
		free(creator);
	return CLRNG_SUCCESS;
}

clrngStatus clrngLfsr113RewindStreamCreator(clrngLfsr113StreamCreator* creator)
{
	if (creator == NULL)
		creator = &defaultStreamCreator;
	creator->nextState = creator->initialState;
	return CLRNG_SUCCESS;
}

clrngStatus clrngLfsr113SetBaseCreatorState(clrngLfsr113StreamCreator* creator, const clrngLfsr113StreamState* baseState)
{
	//Check params
	if (creator == NULL)
		return clrngSetErrorString(CLRNG_INVALID_STREAM_CREATOR, "%s(): modifying the default stream creator is forbidden", __func__);
	if (baseState == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): baseState cannot be NULL", __func__);

	clrngStatus err = validateSeed(baseState);

	if (err == CLRNG_SUCCESS) {
		// initialize new creator
		creator->initialState = creator->nextState = *baseState;
	}

	return err;
}

clrngStatus clrngLfsr113ChangeStreamsSpacing(clrngLfsr113StreamCreator* creator, cl_int e, cl_int c)
{
	return clrngSetErrorString(CLRNG_FUNCTION_NOT_IMPLEMENTED, "%s(): Not Implemented", __func__);
}

clrngLfsr113Stream* clrngLfsr113AllocStreams(size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;
	size_t bufSize_ = count * sizeof(clrngLfsr113Stream);

	// allocate streams
	clrngLfsr113Stream* buf = (clrngLfsr113Stream*)malloc(bufSize_);

	if (buf == NULL) {
		// allocation failed
		err_ = clrngSetErrorString(CLRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for streams", __func__);
		bufSize_ = 0;
	}

	// set buffer size if needed
	if (bufSize != NULL)
		*bufSize = bufSize_;

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return buf;
}

clrngStatus clrngLfsr113DestroyStreams(clrngLfsr113Stream* streams)
{
	if (streams != NULL)
		free(streams);
	return CLRNG_SUCCESS;
}
void lfsr113AdvanceState(clrngLfsr113StreamState* currentState)
{
	int z, b;
	cl_uint* nextSeed = currentState->g;

	//Calculate the new value for nextSeed[0]
	z = nextSeed[0] & (cl_uint)(-2);
	b = (z << 6) ^ z;
	z = (z) ^ (z << 2) ^ (z << 3) ^ (z << 10) ^ (z << 13) ^
		(z << 16) ^ (z << 19) ^ (z << 22) ^ (z << 25) ^
		(z << 27) ^ (z << 28) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 4) & 0x0FFFFFFF) ^
		((b >> 6) & 0x03FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 12) & 0x000FFFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 18) & 0x00003FFF) ^
		((b >> 21) & 0x000007FF);
	nextSeed[0] = z;

	//Calculate the new value for nextSeed[1]
	z = nextSeed[1] & (cl_uint)(-8);
	b = (z << 2) ^ z;
	z = ((b >> 13) & 0x0007FFFF) ^ (z << 16);
	nextSeed[1] = z;

	//Calculate the new value for nextSeed[2]
	z = nextSeed[2] & (cl_uint)(-16);
	b = (z << 13) ^ z;
	z = (z << 2) ^ (z << 4) ^ (z << 10) ^ (z << 12) ^ (z << 13) ^
		(z << 17) ^ (z << 25) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 11) & 0x001FFFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 16) & 0x0000FFFF) ^
		((b >> 24) & 0x000000FF);
	nextSeed[2] = z;

	//Calculate the new value for nextSeed[3]
	z = nextSeed[3] & (cl_uint)(-128);
	b = (z << 3) ^ z;
	z = (z << 9) ^ (z << 10) ^ (z << 11) ^ (z << 14) ^ (z << 16) ^
		(z << 18) ^ (z << 23) ^ (z << 24) ^
		((b >> 1) & 0x7FFFFFFF) ^
		((b >> 2) & 0x3FFFFFFF) ^
		((b >> 7) & 0x01FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 11) & 0x001FFFFF) ^
		((b >> 14) & 0x0003FFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 16) & 0x0000FFFF) ^
		((b >> 23) & 0x000001FF) ^
		((b >> 24) & 0x000000FF);

	nextSeed[3] = z;
}
static clrngStatus Lfsr113CreateStream(clrngLfsr113StreamCreator* creator, clrngLfsr113Stream* buffer)
{
	//Check params
	if (buffer == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);

	// use default creator if not given
	if (creator == NULL)
		creator = &defaultStreamCreator;

	// initialize stream
	buffer->current = buffer->initial = buffer->substream = creator->nextState;

	//Advance next state in stream creator
	lfsr113AdvanceState(&creator->nextState);

	return CLRNG_SUCCESS;
}

clrngStatus clrngLfsr113CreateOverStreams(clrngLfsr113StreamCreator* creator, size_t count, clrngLfsr113Stream* streams)
{
	// iterate over all individual stream buffers
	for (size_t i = 0; i < count; i++) {

		clrngStatus err = Lfsr113CreateStream(creator, &streams[i]);

		// abort on error
		if (err != CLRNG_SUCCESS)
			return err;
	}

	return CLRNG_SUCCESS;
}

clrngLfsr113Stream* clrngLfsr113CreateStreams(clrngLfsr113StreamCreator* creator, size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_;
	size_t bufSize_;
	clrngLfsr113Stream* streams = clrngLfsr113AllocStreams(count, &bufSize_, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngLfsr113CreateOverStreams(creator, count, streams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return streams;
}

clrngLfsr113Stream* clrngLfsr113CopyStreams(size_t count, const clrngLfsr113Stream* streams, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;
	clrngLfsr113Stream* dest = NULL;

	//Check params
	if (streams == NULL)
		err_ = clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);

	if (err_ == CLRNG_SUCCESS)
		dest = clrngLfsr113AllocStreams(count, NULL, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngLfsr113CopyOverStreams(count, dest, streams);

	if (err != NULL)
		*err = err_;

	return dest;
}

clrngLfsr113Stream* clrngLfsr113MakeSubstreams(clrngLfsr113Stream* stream, size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_;
	size_t bufSize_;
	clrngLfsr113Stream* substreams = clrngLfsr113AllocStreams(count, &bufSize_, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngLfsr113MakeOverSubstreams(stream, count, substreams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return substreams;
}

clrngStatus clrngLfsr113AdvanceStreams(size_t count, clrngLfsr113Stream* streams, cl_int e, cl_int c)
{
	return clrngSetErrorString(CLRNG_FUNCTION_NOT_IMPLEMENTED, "%s(): Not Implemented", __func__);
}

clrngStatus clrngLfsr113WriteStreamInfo(const clrngLfsr113Stream* stream, FILE *file)
{
	//Check params
	if (stream == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);
	if (file == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): file cannot be NULL", __func__);

	// The Initial state of the Stream
	fprintf(file, "\n   initial = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%u, ", stream->initial.g[i]);

	fprintf(file, "%u }\n", stream->initial.g[3]);

	//The Current state of the Stream
	fprintf(file, "\n   Current = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%u, ", stream->current.g[i]);

	fprintf(file, "%u }\n", stream->current.g[3]);

	return CLRNG_SUCCESS;
}

clrngStatus clrngLfsr113DeviceRandomU01Array_(size_t streamCount, cl_mem streams,
	size_t numberCount, cl_mem outBuffer, cl_uint numQueuesAndEvents,
	cl_command_queue* commQueues, cl_uint numWaitEvents,
	const cl_event* waitEvents, cl_event* outEvents, cl_bool singlePrecision)
{
	//Check params
	if (streamCount < 1)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streamCount cannot be less than 1", __func__);
	if (streams == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream_array cannot be NULL", __func__);
	if (numberCount < 1)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): numberCount cannot be less than 1", __func__);
	if (outBuffer == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);
	if (commQueues == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): commQueues cannot be NULL", __func__);
	if (numberCount % streamCount != 0)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): numberCount must be a multiple of streamCount", __func__);
	if (numQueuesAndEvents != 1)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): numQueuesAndEvents can only have the value '1'", __func__);

	//***************************************************************************************
	//Get the context
	cl_int err;

	cl_context ctx;
	err = clGetCommandQueueInfo(commQueues[0], CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot retrieve context", __func__);

	//Get the Device
	cl_device_id dev;
	err = clGetCommandQueueInfo(commQueues[0], CL_QUEUE_DEVICE, sizeof(cl_device_id), &dev, NULL);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot retrieve the device", __func__);

	//create the program
	const char *sources[4] = {
	        singlePrecision ? "#define CLRNG_SINGLE_PRECISION\n" : "",
		"#include <clRNG/lfsr113.clh>\n"
		"__kernel void fillBufferU01(__global clrngLfsr113HostStream* streams, uint numberCount, __global ",
		singlePrecision ? "float" : "double",
		"* numbers) {\n"
		"	int gid = get_global_id(0);\n"
		"       int gsize = get_global_size(0);\n"
		"	//Copy a stream from global stream array to local stream struct\n"
		"	clrngLfsr113Stream local_stream;\n"
		"	clrngLfsr113CopyOverStreamsFromGlobal(1, &local_stream, &streams[gid]);\n"
		"	// wavefront-friendly ordering\n"
		"	for (int i = 0; i < numberCount; i++)\n"
		"		numbers[i * gsize + gid] = clrngLfsr113RandomU01(&local_stream);\n"
		"}\n"
	};
	cl_program program = clCreateProgramWithSource(ctx, 4, sources, NULL, &err);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot create program", __func__);

	// construct compiler options
	const char* includes = clrngGetLibraryDeviceIncludes(&err);
	if (err != CLRNG_SUCCESS)
		return (clrngStatus)err;

	err = clBuildProgram(program, 0, NULL, includes, NULL, NULL);
	if (err < 0) {
		// Find size of log and print to std output
		char *program_log;
		size_t log_size;
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		program_log = (char *)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
		printf("clBuildProgram fails:\n%s\n", program_log);
		free(program_log);
		exit(1);
	}

	// Create the kernel
	cl_kernel kernel = clCreateKernel(program, "fillBufferU01", &err);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot create kernel", __func__);

	//***************************************************************************************
	//Random numbers generated by each work-item
	cl_uint number_count_per_stream = numberCount / streamCount;

	//Work Group Size (local_size)
	size_t local_size;
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot read CL_DEVICE_MAX_WORK_GROUP_SIZE", __func__);

	if (local_size > streamCount)
		local_size = streamCount;

	// Set kernel arguments for kernel and enqueue that kernel.
	err = clSetKernelArg(kernel, 0, sizeof(streams), &streams);
	err |= clSetKernelArg(kernel, 1, sizeof(number_count_per_stream), &number_count_per_stream);
	err |= clSetKernelArg(kernel, 2, sizeof(outBuffer), &outBuffer);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot create kernel arguments", __func__);

	// Enqueue kernel
	err = clEnqueueNDRangeKernel(commQueues[0], kernel, 1, NULL, &streamCount, &local_size, numWaitEvents, waitEvents, outEvents);
	if (err != CLRNG_SUCCESS)
		return clrngSetErrorString(err, "%s(): cannot enqueue kernel", __func__);

	clReleaseKernel(kernel);
	clReleaseProgram(program);

	return(clrngStatus)EXIT_SUCCESS;
}

#if 0
clrngLfsr113Stream* Lfsr113GetStreamByIndex(clrngLfsr113Stream* stream, cl_uint index)
{

	return &stream[index];

}
#endif
