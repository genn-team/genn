// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "modelSpecInternal.h"

// CUDA backend includes
#include "optimiser.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator::CUDA;

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
    pybind11::enum_<DeviceSelect>(m, "DeviceSelect")
        .value("OPTIMAL", DeviceSelect::OPTIMAL)
        .value("MOST_MEMORY", DeviceSelect::MOST_MEMORY)
        .value("MANUAL", DeviceSelect::MANUAL);

    pybind11::enum_<BlockSizeSelect>(m, "BlockSizeSelect")
        .value("OCCUPANCY", BlockSizeSelect::OCCUPANCY)
        .value("MANUAL", BlockSizeSelect::MANUAL);

    //------------------------------------------------------------------------
    // cuda_backend.Preferences
    //------------------------------------------------------------------------
    pybind11::class_<Preferences, CodeGenerator::PreferencesBase>(m, "Preferences")
        .def(pybind11::init<>())

        .def_readwrite("show_ptx_info", &Preferences::showPtxInfo)
        .def_readwrite("generate_line_info", &Preferences::generateLineInfo)
        .def_readwrite("enable_nccl_reductions", &Preferences::enableNCCLReductions)
        .def_readwrite("device_select_method", &Preferences::deviceSelectMethod)
        .def_readwrite("manual_device_id", &Preferences::manualDeviceID)
        .def_readwrite("block_size_select_method", &Preferences::blockSizeSelectMethod)
        // **TODO** some weirdness with "opaque types" means this doesn't work
        .def_readwrite("manual_block_sizes", &Preferences::manualBlockSizes)
        .def_readwrite("constant_cache_overhead", &Preferences::constantCacheOverhead);

    //------------------------------------------------------------------------
    // cuda_backend.State
    //------------------------------------------------------------------------
    pybind11::class_<State, Runtime::StateBase>(m, "Runtime")
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
    pybind11::class_<Backend, CodeGenerator::BackendBase>(m, "Backend");
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("create_backend", &createBackend, pybind11::return_value_policy::move);
}
