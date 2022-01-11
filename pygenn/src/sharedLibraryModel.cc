// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// GeNN includes
#include "../../userproject/include/sharedLibraryModel.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
//#define DECLARE_GET_VAR(TYPE) declareGetVar<T, TYPE>(slm, #TYPE)
//#define DECLARE_GET_STDINT_VAR(TYPE) declareGetVar<T, TYPE##_t>(slm, #TYPE)

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
/*template<typename T, typename V>
void declareGetVar(pybind11::class_<SharedLibraryModel<T>> &c, const std::string &typeName)
{
    // **NOTE** based on https://github.com/pybind/pybind11/issues/323 and 
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    // we use the class as a 'base' which I think defines lifetime
    const std::string getArrayName = "get_array_" + typeName;
    const std::string getScalarName = "get_scalar_" + typeName;
    c.def(getArrayName.c_str(), 
          [&c](SharedLibraryModel<T> &s, const std::string &varName, size_t count) 
          { 
             return pybind11::array_t<V>(count, s.getArray<V>(varName), c);
          });
    c.def(getScalarName.c_str(), 
          [&c](SharedLibraryModel<T> &s, const std::string &varName) 
          { 
             return pybind11::array_t<V>(1, s.getScalar<V>(varName), c);
          });
}*/
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
        .def("push_state_to_device", &SharedLibraryModel<T>::pushStateToDevice)
        .def("push_connectivity_to_device", &SharedLibraryModel<T>::pushConnectivityToDevice)
        .def("push_var_to_device", &SharedLibraryModel<T>::pushVarToDevice)
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
        .def("get_array",
             [](SharedLibraryModel<T> &s, const std::string &varName, size_t bytes)
             {
                 return pybind11::memoryview::from_memory(*static_cast<void**>(s.getSymbol(varName)), bytes);
             });
        // Declare memory view getters for standard sized types
        /*DECLARE_GET_STDINT_VAR(int8);
        DECLARE_GET_STDINT_VAR(int16);
        DECLARE_GET_STDINT_VAR(int32);
        DECLARE_GET_STDINT_VAR(int64);
        DECLARE_GET_STDINT_VAR(uint8);
        DECLARE_GET_STDINT_VAR(uint16);
        DECLARE_GET_STDINT_VAR(uint32);
        DECLARE_GET_STDINT_VAR(uint64);
        
        // Declare memory view getters for standard types
        DECLARE_GET_VAR(bool);
        DECLARE_GET_VAR(float);
        DECLARE_GET_VAR(double);*/
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