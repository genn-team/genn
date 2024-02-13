// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "modelSpecInternal.h"

// CUDA backend includes
#include "optimiser.h"

// Doc strings
#include "cudaBackendDocStrings.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator::CUDA;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DOC_CUDA(...) DOC(CodeGenerator, CUDA, __VA_ARGS__)
#define WRAP_ENUM(ENUM, VAL) .value(#VAL, ENUM::VAL, DOC_CUDA(ENUM, VAL))
#define WRAP_MEMBER(NAME, CLASS, VAL) .def_readwrite(NAME, &CLASS::VAL, DOC_CUDA(CLASS, VAL))

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
Backend createBackend(const ModelSpecInternal &model, const std::string &outputPath, 
                      plog::Severity backendLevel, const Preferences &preferences)
{
    auto *consoleAppender = new plog::ConsoleAppender<plog::TxtFormatter>;
    return Optimiser::createBackend(model, filesystem::path(outputPath), backendLevel, consoleAppender, preferences);
}
}

//----------------------------------------------------------------------------
// cuda_backend
//----------------------------------------------------------------------------
PYBIND11_MODULE(cuda_backend, m) 
{
    pybind11::module_::import("pygenn._genn");
    pybind11::module_::import("pygenn.runtime");

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    pybind11::enum_<DeviceSelect>(m, "DeviceSelect", DOC_CUDA(DeviceSelect))
        WRAP_ENUM(DeviceSelect, OPTIMAL)
        WRAP_ENUM(DeviceSelect, MOST_MEMORY)
        WRAP_ENUM(DeviceSelect, MANUAL);

    pybind11::enum_<BlockSizeSelect>(m, "BlockSizeSelect", DOC_CUDA(BlockSizeSelect))
        WRAP_ENUM(BlockSizeSelect, OCCUPANCY)
        WRAP_ENUM(BlockSizeSelect, MANUAL);

    //------------------------------------------------------------------------
    // cuda_backend.Preferences
    //------------------------------------------------------------------------
    pybind11::class_<Preferences, CodeGenerator::PreferencesBase>(m, "Preferences", DOC_CUDA(Preferences))
        .def(pybind11::init<>())
        
        WRAP_MEMBER("show_ptx_info", Preferences, showPtxInfo)
        WRAP_MEMBER("generate_line_info", Preferences, generateLineInfo)
        WRAP_MEMBER("enable_nccl_reductions", Preferences, enableNCCLReductions)
        WRAP_MEMBER("device_select_method", Preferences, deviceSelectMethod)
        WRAP_MEMBER("manual_device_id", Preferences, manualDeviceID)
        WRAP_MEMBER("block_size_select_method", Preferences, blockSizeSelectMethod)
        // **TODO** some weirdness with "opaque types" means this doesn't work
        WRAP_MEMBER("manual_block_sizes", Preferences, manualBlockSizes)
        WRAP_MEMBER("constant_cache_overhead", Preferences, constantCacheOverhead);

    //------------------------------------------------------------------------
    // cuda_backend.State
    //------------------------------------------------------------------------
    pybind11::class_<State, Runtime::StateBase>(m, "_Runtime")
        .def("nccl_generate_unique_id", &State::ncclGenerateUniqueID)
        .def("nccl_init_communicator", &State::ncclInitCommunicator)

        .def_property_readonly("nccl_unique_id",
            [](State &a)
            {
               return pybind11::memoryview::from_memory(a.ncclGetUniqueID(),
                                                        a.ncclGetUniqueIDSize());
            });

    //------------------------------------------------------------------------
    // cuda_backend.Backend
    //------------------------------------------------------------------------
    pybind11::class_<Backend, CodeGenerator::BackendBase>(m, "_Backend");
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("_create_backend", &createBackend, pybind11::return_value_policy::move);
}
