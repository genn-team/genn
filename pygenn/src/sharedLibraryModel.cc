// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// GeNN includes
#include "../../userproject/include/sharedLibraryModel.h"

using namespace pybind11::literals;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
template<typename T>
void declareSharedLibraryModel(pybind11::module &m, const std::string &typeString) 
{
    const std::string className = "SharedLibraryModel" + typeString;
    pybind11::class_<SharedLibraryModel<T>>(m, className.c_str())
        .def(pybind11::init<>())
        .def(pybind11::init<const std::string&, const std::string&>())

        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("neuron_update_time", &SharedLibraryModel<T>::getNeuronUpdateTime)
        .def_property_readonly("init_time", &SharedLibraryModel<T>::getInitTime)
        .def_property_readonly("init_sparse_time", &SharedLibraryModel<T>::getInitSparseTime)
        .def_property_readonly("presynaptic_update_time", &SharedLibraryModel<T>::getPresynapticUpdateTime)
        .def_property_readonly("postsynaptic_update_time", &SharedLibraryModel<T>::getPostsynapticUpdateTime)
        .def_property_readonly("synapse_dynamics_time", &SharedLibraryModel<T>::getSynapseDynamicsTime)
        .def_property_readonly("free_device_mem_bytes", &SharedLibraryModel<T>::getFreeDeviceMemBytes)
        .def_property("time", &SharedLibraryModel<T>::getTime, &SharedLibraryModel<T>::setTime)
        .def_property("timestep", &SharedLibraryModel<T>::getTimestep, &SharedLibraryModel<T>::setTimestep)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("open", &SharedLibraryModel<T>::open)
        .def("allocate_mem", &SharedLibraryModel<T>::allocateMem)
        .def("allocate_recording_buffers", &SharedLibraryModel<T>::allocateRecordingBuffers)
        .def("free_mem", &SharedLibraryModel<T>::freeMem)
        .def("nccl_init_communicator", &SharedLibraryModel<T>::ncclInitCommunicator)
        .def("initialize", &SharedLibraryModel<T>::initialize)
        .def("initialize_sparse", &SharedLibraryModel<T>::initializeSparse)
        .def("step_time", &SharedLibraryModel<T>::stepTime)
        .def("custom_update", &SharedLibraryModel<T>::customUpdate)
        .def("pull_recording_buffers_from_device", &SharedLibraryModel<T>::pullRecordingBuffersFromDevice)
        .def("allocate_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::allocateExtraGlobalParam))
        .def("free_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&>(&SharedLibraryModel<T>::freeExtraGlobalParam))
        .def("pull_state_from_device", &SharedLibraryModel<T>::pullStateFromDevice)
        .def("pull_connectivity_from_device", &SharedLibraryModel<T>::pullConnectivityFromDevice)
        .def("pull_var_from_device", &SharedLibraryModel<T>::pullVarFromDevice)
        .def("pull_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::pullExtraGlobalParam))
        .def("push_state_to_device", &SharedLibraryModel<T>::pushStateToDevice,
            "pop_name"_a, "uninitialised_only"_a = false)
        .def("push_connectivity_to_device", &SharedLibraryModel<T>::pushConnectivityToDevice,
            "pop_name"_a, "uninitialised_only"_a = false)
        .def("push_var_to_device", &SharedLibraryModel<T>::pushVarToDevice,
            "pop_name"_a, "var_name"_a, "uninitialised_only"_a = false)
        .def("push_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::pushExtraGlobalParam))
        .def("get_custom_update_time", &SharedLibraryModel<T>::getCustomUpdateTime)
        .def("get_custom_update_transpose_time", &SharedLibraryModel<T>::getCustomUpdateTransposeTime)
        .def("get_nccl_unique_id", 
            [](SharedLibraryModel<T> &s) 
            { 
               return pybind11::memoryview::from_memory(s.ncclGetUniqueID(), 
                                                        s.ncclGetUniqueIDBytes(), true);
            })
        .def("get_scalar",
             [](SharedLibraryModel<T> &s, const std::string &varName, size_t bytes)
             {
                 return pybind11::memoryview::from_memory(s.getSymbol(varName), bytes);
             })
        .def("get_var",
             [](SharedLibraryModel<T> &s, const std::string &popName, const std::string &varName, size_t bytes)
             {
                 return pybind11::memoryview::from_memory(s.template getVar<void>(popName, varName), bytes);
             })
        .def("get_egp",
             [](SharedLibraryModel<T> &s, const std::string &popName, const std::string &varName, size_t bytes)
             {
                 return pybind11::memoryview::from_memory(s.template getEGP<void>(popName, varName), bytes);
             });
}
}

//----------------------------------------------------------------------------
// shared_library_model
//----------------------------------------------------------------------------
PYBIND11_MODULE(shared_library_model, m) 
{
    //------------------------------------------------------------------------
    // shared_library_model.SharedLibraryModelFloat
    //------------------------------------------------------------------------
    declareSharedLibraryModel<float>(m, "Float");

    //------------------------------------------------------------------------
    // shared_library_model.SharedLibaryModelDouble
    //------------------------------------------------------------------------
    declareSharedLibraryModel<double>(m, "Double");
}
