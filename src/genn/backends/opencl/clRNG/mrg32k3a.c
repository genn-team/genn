
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

#include <clRNG/mrg32k3a.h>

#include "private.h"
#include <stdlib.h>

#if defined ( WIN32 )
#define __func__ __FUNCTION__
#endif

struct clrngMrg32k3aStreamCreator_ {
	clrngMrg32k3aStreamState initialState;
	clrngMrg32k3aStreamState nextState;
	/*! @brief Jump matrices for advancing the initial seed of streams
	*/
	cl_ulong nuA1[3][3];
	cl_ulong nuA2[3][3];
};

#define MODULAR_NUMBER_TYPE cl_ulong
#define MODULAR_FIXED_SIZE 3
#include "./modularHost.c.h"

// code that is common to host and device
#include "clRNG/private/mrg32k3a.c.h"



/*! @brief Matrices to advance to the next state
*/
static cl_ulong Mrg32k3a_A1p0[3][3] = {
	{ 0, 1, 0 },
	{ 0, 0, 1 },
	{ 4294156359, 1403580, 0 }
};

static cl_ulong Mrg32k3a_A2p0[3][3] = {
	{ 0, 1, 0 },
	{ 0, 0, 1 },
	{ 4293573854, 0, 527612 }
};


/*! @brief Inverse of Mrg32k3a_A1p0 mod Mrg32k3a_M1
*
*  Matrices to go back to the previous state.
*/
static cl_ulong invA1[3][3] = {
	{ 184888585, 0, 1945170933 },
	{ 1, 0, 0 },
	{ 0, 1, 0 }
};

// inverse of Mrg32k3a_A2p0 mod Mrg32k3a_M1
static cl_ulong invA2[3][3] = {
	{ 0, 360363334, 4225571728 },
	{ 1, 0, 0 },
	{ 0, 1, 0 }
};


/*! @brief Default initial seed of the first stream
*/
#define BASE_CREATOR_STATE { { 12345, 12345, 12345 }, { 12345, 12345, 12345 } }
/*! @brief Jump matrices for \f$2^{127}\f$ steps forward
*/
#define BASE_CREATOR_JUMP_MATRIX_1 { \
        {2427906178, 3580155704, 949770784}, \
        { 226153695, 1230515664, 3580155704}, \
        {1988835001, 986791581, 1230515664} }
#define BASE_CREATOR_JUMP_MATRIX_2 { \
        { 1464411153, 277697599, 1610723613}, \
        {32183930, 1464411153, 1022607788}, \
        {2824425944, 32183930, 2093834863} }

/*! @brief Default stream creator (defaults to \f$2^{134}\f$ steps forward)
*
*  Contains the default seed and the transition matrices to jump \f$\nu\f$ steps forward;
*  adjacent streams are spaced nu steps apart.
*  The default is \f$nu = 2^{134}\f$.
*  The default seed is \f$(12345,12345,12345,12345,12345,12345)\f$.
*/
static clrngMrg32k3aStreamCreator defaultStreamCreator = {
	BASE_CREATOR_STATE,
	BASE_CREATOR_STATE,
	BASE_CREATOR_JUMP_MATRIX_1,
	BASE_CREATOR_JUMP_MATRIX_2
};

/*! @brief Check the validity of a seed for Mrg32k3a
*/
static clrngStatus validateSeed(const clrngMrg32k3aStreamState* seed)
{
	// Check that the seeds have valid values
	for (size_t i = 0; i < 3; ++i)
		if (seed->g1[i] >= Mrg32k3a_M1)
			return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g1[%u] >= Mrg32k3a_M1", i);

	for (size_t i = 0; i < 3; ++i)
		if (seed->g2[i] >= Mrg32k3a_M2)
			return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g2[%u] >= Mrg32k3a_M2", i);

	if (seed->g1[0] == 0 && seed->g1[1] == 0 && seed->g1[2] == 0)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g1 = (0,0,0)");

	if (seed->g2[0] == 0 && seed->g2[1] == 0 && seed->g2[2] == 0)
		return clrngSetErrorString(CLRNG_INVALID_SEED, "seed.g2 = (0,0,0)");

	return CLRNG_SUCCESS;
}

clrngMrg32k3aStreamCreator* clrngMrg32k3aCopyStreamCreator(const clrngMrg32k3aStreamCreator* creator, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;

	// allocate creator
	clrngMrg32k3aStreamCreator* newCreator = (clrngMrg32k3aStreamCreator*)malloc(sizeof(clrngMrg32k3aStreamCreator));

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

clrngStatus clrngMrg32k3aDestroyStreamCreator(clrngMrg32k3aStreamCreator* creator)
{
	if (creator != NULL)
		free(creator);
	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aRewindStreamCreator(clrngMrg32k3aStreamCreator* creator)
{
	if (creator == NULL)
		creator = &defaultStreamCreator;
	creator->nextState = creator->initialState;
	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aSetBaseCreatorState(clrngMrg32k3aStreamCreator* creator, const clrngMrg32k3aStreamState* baseState)
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

clrngStatus clrngMrg32k3aChangeStreamsSpacing(clrngMrg32k3aStreamCreator* creator, cl_int e, cl_int c)
{
	//Check params
	if (creator == NULL)
		return clrngSetErrorString(CLRNG_INVALID_STREAM_CREATOR, "%s(): modifying the default stream creator is forbidden", __func__);
	if (e < 0)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): e must be >= 0", __func__);

	cl_ulong B[3][3];

	if (c >= 0)
		modMatPow(Mrg32k3a_A1p0, creator->nuA1, Mrg32k3a_M1, c);
	else
		modMatPow(invA1, creator->nuA1, Mrg32k3a_M1, -c);
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A1p0, B, Mrg32k3a_M1, e);
		modMatMat(B, creator->nuA1, creator->nuA1, Mrg32k3a_M1);
	}

	if (c >= 0)
		modMatPow(Mrg32k3a_A2p0, creator->nuA2, Mrg32k3a_M2, c);
	else
		modMatPow(invA2, creator->nuA2, Mrg32k3a_M2, -c);
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A2p0, B, Mrg32k3a_M2, e);
		modMatMat(B, creator->nuA2, creator->nuA2, Mrg32k3a_M2);
	}

	return CLRNG_SUCCESS;
}

clrngMrg32k3aStream* clrngMrg32k3aAllocStreams(size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;
	size_t bufSize_ = count * sizeof(clrngMrg32k3aStream);

	// allocate streams
	clrngMrg32k3aStream* buf = (clrngMrg32k3aStream*)malloc(bufSize_);

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

clrngStatus clrngMrg32k3aDestroyStreams(clrngMrg32k3aStream* streams)
{
	if (streams != NULL)
		free(streams);
	return CLRNG_SUCCESS;
}

static clrngStatus Mrg32k3aCreateStream(clrngMrg32k3aStreamCreator* creator, clrngMrg32k3aStream* buffer)
{
	//Check params
	if (buffer == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);

	// use default creator if not given
	if (creator == NULL)
		creator = &defaultStreamCreator;

	// initialize stream
	buffer->current = buffer->initial = buffer->substream = creator->nextState;

	// advance next state in stream creator
	modMatVec(creator->nuA1, creator->nextState.g1, creator->nextState.g1, Mrg32k3a_M1);
	modMatVec(creator->nuA2, creator->nextState.g2, creator->nextState.g2, Mrg32k3a_M2);

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aCreateOverStreams(clrngMrg32k3aStreamCreator* creator, size_t count, clrngMrg32k3aStream* streams)
{
	// iterate over all individual stream buffers
	for (size_t i = 0; i < count; i++) {

		clrngStatus err = Mrg32k3aCreateStream(creator, &streams[i]);

		// abort on error
		if (err != CLRNG_SUCCESS)
			return err;
	}

	return CLRNG_SUCCESS;
}

clrngMrg32k3aStream* clrngMrg32k3aCreateStreams(clrngMrg32k3aStreamCreator* creator, size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_;
	size_t bufSize_;
	clrngMrg32k3aStream* streams = clrngMrg32k3aAllocStreams(count, &bufSize_, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngMrg32k3aCreateOverStreams(creator, count, streams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return streams;
}

clrngMrg32k3aStream* clrngMrg32k3aCopyStreams(size_t count, const clrngMrg32k3aStream* streams, clrngStatus* err)
{
	clrngStatus err_ = CLRNG_SUCCESS;
	clrngMrg32k3aStream* dest = NULL;

	//Check params
	if (streams == NULL)
		err_ = clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);

	if (err_ == CLRNG_SUCCESS)
		dest = clrngMrg32k3aAllocStreams(count, NULL, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngMrg32k3aCopyOverStreams(count, dest, streams);

	if (err != NULL)
		*err = err_;

	return dest;
}

clrngMrg32k3aStream* clrngMrg32k3aMakeSubstreams(clrngMrg32k3aStream* stream, size_t count, size_t* bufSize, clrngStatus* err)
{
	clrngStatus err_;
	size_t bufSize_;
	clrngMrg32k3aStream* substreams = clrngMrg32k3aAllocStreams(count, &bufSize_, &err_);

	if (err_ == CLRNG_SUCCESS)
		err_ = clrngMrg32k3aMakeOverSubstreams(stream, count, substreams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return substreams;
}

clrngStatus clrngMrg32k3aAdvanceStreams(size_t count, clrngMrg32k3aStream* streams, cl_int e, cl_int c)
{
	//Check params
	if (streams == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);

	//Advance Stream
	cl_ulong B1[3][3], C1[3][3], B2[3][3], C2[3][3];

	// if e == 0, do not add 2^0; just behave as in docs
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A1p0, B1, Mrg32k3a_M1, e);
		modMatPowLog2(Mrg32k3a_A2p0, B2, Mrg32k3a_M2, e);
	}
	else if (e < 0) {
		modMatPowLog2(invA1, B1, Mrg32k3a_M1, -e);
		modMatPowLog2(invA2, B2, Mrg32k3a_M2, -e);
	}

	if (c >= 0) {
		modMatPow(Mrg32k3a_A1p0, C1, Mrg32k3a_M1, c);
		modMatPow(Mrg32k3a_A2p0, C2, Mrg32k3a_M2, c);
	}
	else {
		modMatPow(invA1, C1, Mrg32k3a_M1, -c);
		modMatPow(invA2, C2, Mrg32k3a_M2, -c);
	}

	if (e) {
		modMatMat(B1, C1, C1, Mrg32k3a_M1);
		modMatMat(B2, C2, C2, Mrg32k3a_M2);
	}

	for (size_t i = 0; i < count; i++) {
		modMatVec(C1, streams[i].current.g1, streams[i].current.g1, Mrg32k3a_M1);
		modMatVec(C2, streams[i].current.g2, streams[i].current.g2, Mrg32k3a_M2);
	}

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aWriteStreamInfo(const clrngMrg32k3aStream* stream, FILE *file)
{
	//Check params
	if (stream == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);
	if (file == NULL)
		return clrngSetErrorString(CLRNG_INVALID_VALUE, "%s(): file cannot be NULL", __func__);

	// The Initial state of the Stream
	fprintf(file, "\n   initial = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%lu, ", stream->initial.g1[i]);

	for (size_t i = 0; i < 2; i++)
		fprintf(file, "%lu, ", stream->initial.g2[i]);

	fprintf(file, "%lu }\n", stream->initial.g2[2]);
	//The Current state of the Stream
	fprintf(file, "\n   Current = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%lu, ", stream->current.g1[i]);

	for (size_t i = 0; i < 2; i++)
		fprintf(file, "%lu, ", stream->current.g2[i]);

	fprintf(file, "%lu }\n", stream->current.g2[2]);

	return CLRNG_SUCCESS;
}

clrngStatus clrngMrg32k3aDeviceRandomU01Array_(size_t streamCount, cl_mem streams,
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
		"#include <clRNG/mrg32k3a.clh>\n"
		"__kernel void fillBufferU01(__global clrngMrg32k3aHostStream* streams, uint numberCount, __global ",
		singlePrecision ? "float" : "double",
		"* numbers) {\n"
		"	int gid = get_global_id(0);\n"
		"       int gsize = get_global_size(0);\n"
		"	//Copy a stream from global stream array to local stream struct\n"
		"	clrngMrg32k3aStream local_stream;\n"
		"	clrngMrg32k3aCopyOverStreamsFromGlobal(1, &local_stream, &streams[gid]);\n"
		"	// wavefront-friendly ordering\n"
		"	for (int i = 0; i < numberCount; i++)\n"
		"		numbers[i * gsize + gid] = clrngMrg32k3aRandomU01(&local_stream);\n"
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
clrngMrg32k3aStream* Mrg32k3aGetStreamByIndex(clrngMrg32k3aStream* stream, cl_uint index)
{

	return &stream[index];

}
#endif
