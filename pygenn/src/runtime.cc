// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// GeNN includes
#include "runtime/runtime.h"

using namespace GeNN::Runtime;
using namespace pybind11::literals;

//----------------------------------------------------------------------------
// runtime
//----------------------------------------------------------------------------
PYBIND11_MODULE(runtime, m) 
{
    //------------------------------------------------------------------------
    // runtime.ArrayBase
    //------------------------------------------------------------------------
    pybind11::class_<ArrayBase>(m, "ArrayBase")    
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("host_view", 
            [](ArrayBase &a) 
            { 
               return pybind11::memoryview::from_memory(a.getHostPointer(), 
                                                        a.getSizeBytes());
            })
        
        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("push_to_device", &ArrayBase::pushToDevice)
        .def("pull_from_device", &ArrayBase::pullFromDevice);
        
    //------------------------------------------------------------------------
    // runtime.Runtime
    //------------------------------------------------------------------------
    pybind11::class_<Runtime>(m, "Runtime")
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("neuron_update_time", &Runtime::getNeuronUpdateTime)
        .def_property_readonly("init_time", &Runtime::getInitTime)
        .def_property_readonly("init_sparse_time", &Runtime::getInitSparseTime)
        .def_property_readonly("presynaptic_update_time", &Runtime::getPresynapticUpdateTime)
        .def_property_readonly("postsynaptic_update_time", &Runtime::getPostsynapticUpdateTime)
        .def_property_readonly("synapse_dynamics_time", &Runtime::getSynapseDynamicsTime)
        //.def_property_readonly("free_device_mem_bytes", &Runtime::getFreeDeviceMemBytes)
        .def_property_readonly("time", &Runtime::getTime)
        .def_property("timestep", &Runtime::getTimestep, &Runtime::setTimestep)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        //.def("open", &SharedLibraryModel<T>::open)
        .def("allocate", &Runtime::allocate)
        //.def("nccl_init_communicator", &SharedLibraryModel<T>::ncclInitCommunicator)
        .def("initialize", &Runtime::initialize)
        .def("initialize_sparse", &Runtime::initializeSparse)
        .def("step_time", &Runtime::stepTime)
        //.def("custom_update", &Runtime::customUpdate)
        .def("pull_recording_buffers_from_device", &Runtime::pullRecordingBuffersFromDevice)
        //.def("allocate_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::allocateExtraGlobalParam))
        //.def("free_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&>(&SharedLibraryModel<T>::freeExtraGlobalParam))
         /*.def("pull_state_from_device", &SharedLibraryModel<T>::pullStateFromDevice)
        .def("pull_connectivity_from_device", &SharedLibraryModel<T>::pullConnectivityFromDevice)
        .def("pull_var_from_device", &SharedLibraryModel<T>::pullVarFromDevice)
        .def("pull_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::pullExtraGlobalParam))
        .def("push_state_to_device", &SharedLibraryModel<T>::pushStateToDevice,
            "pop_name"_a, "uninitialised_only"_a = false)
        .def("push_connectivity_to_device", &SharedLibraryModel<T>::pushConnectivityToDevice,
            "pop_name"_a, "uninitialised_only"_a = false)
        .def("push_var_to_device", &SharedLibraryModel<T>::pushVarToDevice,
            "pop_name"_a, "var_name"_a, "uninitialised_only"_a = false)
        .def("push_extra_global_param", pybind11::overload_cast<const std::string&, const std::string&, unsigned int>(&SharedLibraryModel<T>::pushExtraGlobalParam))*/
        .def("get_custom_update_time", &Runtime::getCustomUpdateTime)
        .def("get_custom_update_transpose_time", &Runtime::getCustomUpdateTransposeTime);
        /*.def("get_nccl_unique_id", 
            [](SharedLibraryModel<T> &s) 
            { 
               return pybind11::memoryview::from_memory(s.ncclGetUniqueID(), 
                                                        s.ncclGetUniqueIDBytes(), true);
            })*/

}