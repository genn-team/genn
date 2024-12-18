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
#include "hipBackendDocStrings.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator::HIP;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DOC_HIP(...) DOC(CodeGenerator, HIP, __VA_ARGS__)
#define WRAP_ENUM(ENUM, VAL) .value(#VAL, ENUM::VAL, DOC_HIP(ENUM, VAL))
#define WRAP_ATTR(NAME, CLASS, ATTR) .def_readwrite(NAME, &CLASS::ATTR, DOC_HIP(CLASS, ATTR))

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
// hip_backend
//----------------------------------------------------------------------------
PYBIND11_MODULE(hip_backend, m) 
{
    pybind11::module_::import("pygenn._genn");
    pybind11::module_::import("pygenn._runtime");

    //------------------------------------------------------------------------
    // hip_backend.Preferences
    //------------------------------------------------------------------------
    pybind11::class_<Preferences, CodeGenerator::PreferencesBase>(m, "Preferences", DOC_CUDA(Preferences))
        .def(pybind11::init<>())
        
        //WRAP_ATTR("show_ptx_info", Preferences, showPtxInfo)
        //WRAP_ATTR("generate_line_info", Preferences, generateLineInfo)
        //WRAP_ATTR("enable_nccl_reductions", Preferences, enableNCCLReductions)
        //WRAP_ATTR("device_select_method", Preferences, deviceSelectMethod)
        WRAP_ATTR("manual_device_id", Preferences, manualDeviceID)
        //WRAP_ATTR("block_size_select_method", Preferences, blockSizeSelectMethod)
        // **TODO** some weirdness with "opaque types" means this doesn't work
        WRAP_ATTR("manual_block_sizes", Preferences, manualBlockSizes)
        WRAP_ATTR("constant_cache_overhead", Preferences, constantCacheOverhead);

    //------------------------------------------------------------------------
    // hip_backend.State
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
    // hip_backend.Backend
    //------------------------------------------------------------------------
    pybind11::class_<Backend, CodeGenerator::BackendBase>(m, "_Backend");
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("_create_backend", &createBackend, pybind11::return_value_policy::move);
}
