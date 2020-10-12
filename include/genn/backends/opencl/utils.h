#pragma once

// Standard C++ includes
#include <iostream>

// OpenCL includes
#include "../../../../share/genn/backends/opencl/cl2.hpp"

// PLOG includes
#include <plog/Log.h>

// Check Compile-time errors (driver independent)
#define CHECK_OPENCL_ERRORS(call)                                                                                       \
{                                                                                                                       \
    cl_int error = call;                                                                                                \
    if (error != CL_SUCCESS) {                                                                                          \
        LOGE_BACKEND << __FILE__ << ": " <<  __LINE__ << ": opencl error " << error << ": " << clGetErrorString(error); \
        exit(EXIT_FAILURE);                                                                                             \
    }                                                                                                                   \
}

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Utils
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
namespace Utils
{
// OpenCL error string
const char *clGetErrorString(cl_int error) 
{
    #define GEN_CL_ERROR_CASE(ERR) case ERR: return #ERR
    switch(error) {
        // run-time and JIT compiler errors
        GEN_CL_ERROR_CASE(CL_SUCCESS);
        GEN_CL_ERROR_CASE(CL_DEVICE_NOT_FOUND);
        GEN_CL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
        GEN_CL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
        GEN_CL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        GEN_CL_ERROR_CASE(CL_OUT_OF_RESOURCES);
        GEN_CL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
        GEN_CL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        GEN_CL_ERROR_CASE(CL_MEM_COPY_OVERLAP);
        GEN_CL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
        GEN_CL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        GEN_CL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
        GEN_CL_ERROR_CASE(CL_MAP_FAILURE);
        GEN_CL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        GEN_CL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        GEN_CL_ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE);
        GEN_CL_ERROR_CASE(CL_LINKER_NOT_AVAILABLE);
        GEN_CL_ERROR_CASE(CL_LINK_PROGRAM_FAILURE);
        GEN_CL_ERROR_CASE(CL_DEVICE_PARTITION_FAILED);
        GEN_CL_ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);

        // compile-time errors
        GEN_CL_ERROR_CASE(CL_INVALID_VALUE);
        GEN_CL_ERROR_CASE(CL_INVALID_DEVICE_TYPE);
        GEN_CL_ERROR_CASE(CL_INVALID_PLATFORM);
        GEN_CL_ERROR_CASE(CL_INVALID_DEVICE);
        GEN_CL_ERROR_CASE(CL_INVALID_CONTEXT);
        GEN_CL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
        GEN_CL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
        GEN_CL_ERROR_CASE(CL_INVALID_HOST_PTR);
        GEN_CL_ERROR_CASE(CL_INVALID_MEM_OBJECT);
        GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_SAMPLER);
        GEN_CL_ERROR_CASE(CL_INVALID_BINARY);
        GEN_CL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
        GEN_CL_ERROR_CASE(CL_INVALID_PROGRAM);
        GEN_CL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_NAME);
        GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
        GEN_CL_ERROR_CASE(CL_INVALID_KERNEL);
        GEN_CL_ERROR_CASE(CL_INVALID_ARG_INDEX);
        GEN_CL_ERROR_CASE(CL_INVALID_ARG_VALUE);
        GEN_CL_ERROR_CASE(CL_INVALID_ARG_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_KERNEL_ARGS);
        GEN_CL_ERROR_CASE(CL_INVALID_WORK_DIMENSION);
        GEN_CL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
        GEN_CL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
        GEN_CL_ERROR_CASE(CL_INVALID_EVENT);
        GEN_CL_ERROR_CASE(CL_INVALID_OPERATION);
        GEN_CL_ERROR_CASE(CL_INVALID_GL_OBJECT);
        GEN_CL_ERROR_CASE(CL_INVALID_BUFFER_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_MIP_LEVEL);
        GEN_CL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
        GEN_CL_ERROR_CASE(CL_INVALID_PROPERTY);
        GEN_CL_ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
        GEN_CL_ERROR_CASE(CL_INVALID_COMPILER_OPTIONS);
        GEN_CL_ERROR_CASE(CL_INVALID_LINKER_OPTIONS);
        GEN_CL_ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);

        default:    return "Unknown OpenCL error";
    }   
#undef GEN_CL_ERROR_CASE
}
}   // namespace Utils
}   // namespace OpenCL
}   // namespace CodeGenerator