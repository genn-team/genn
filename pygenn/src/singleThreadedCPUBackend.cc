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
using namespace GeNN::CodeGenerator::SingleThreadedCPU;

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
// single_threaded_cpu_backend
//----------------------------------------------------------------------------
PYBIND11_MODULE(single_threaded_cpu_backend, m)
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // single_threaded_cpu_backend.Preferences
    //------------------------------------------------------------------------
    pybind11::class_<Preferences, CodeGenerator::PreferencesBase>(m, "Preferences")
        .def(pybind11::init<>());

    //------------------------------------------------------------------------
    // single_threaded_cpu_backend.Backend
    //------------------------------------------------------------------------
    pybind11::class_<Backend, CodeGenerator::BackendBase>(m, "Backend");
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("create_backend", &createBackend, pybind11::return_value_policy::move);
}
