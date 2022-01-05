// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "modelSpecInternal.h"

// CUDA backend includes
#include "optimiser.h"


using namespace CodeGenerator::CUDA;

// Anonymous namespace
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
    pybind11::module_::import("genn_wrapper.genn");
    
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
        .def_readwrite("manual_block_size", &Preferences::manualBlockSizes)
        .def_readwrite("constant_cache_overhead", &Preferences::constantCacheOverhead);
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("create_backend", &createBackend, pybind11::return_value_policy::move);
}
