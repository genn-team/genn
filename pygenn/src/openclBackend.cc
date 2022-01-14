// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "modelSpecInternal.h"

// CUDA backend includes
#include "optimiser.h"


using namespace CodeGenerator::OpenCL;

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
// opencl_backend
//----------------------------------------------------------------------------
PYBIND11_MODULE(opencl_backend, m)
{
    pybind11::module_::import("pygenn.genn");
    
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    pybind11::enum_<PlatformSelect>(m, "PlatformSelect")
        .value("MANUAL", PlatformSelect::MANUAL);

    pybind11::enum_<DeviceSelect>(m, "DeviceSelect")
        .value("MOST_MEMORY", DeviceSelect::MOST_MEMORY)
        .value("MANUAL", DeviceSelect::MANUAL);

    pybind11::enum_<WorkGroupSizeSelect>(m, "WorkGroupSizeSelect")
        .value("MANUAL", WorkGroupSizeSelect::MANUAL);

    //------------------------------------------------------------------------
    // opencl_backend.Preferences
    //------------------------------------------------------------------------
    pybind11::class_<Preferences, CodeGenerator::PreferencesBase>(m, "Preferences")
        .def(pybind11::init<>())
        
        .def_readwrite("platform_select_method", &Preferences::platformSelectMethod)
        .def_readwrite("manual_platform_id", &Preferences::manualPlatformID)
        .def_readwrite("device_select_method", &Preferences::deviceSelectMethod)
        .def_readwrite("manual_device_id", &Preferences::manualDeviceID)
        .def_readwrite("work_group_size_select_method", &Preferences::workGroupSizeSelectMethod)
        // **TODO** some weirdness with "opaque types" means this doesn't work
        .def_readwrite("manual_work_group_sizes", &Preferences::manualWorkGroupSizes);
    
    //------------------------------------------------------------------------
    // opencl_backend.Backend
    //------------------------------------------------------------------------
    pybind11::class_<Backend, CodeGenerator::BackendBase>(m, "Backend");
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("create_backend", &createBackend, pybind11::return_value_policy::move);
}
