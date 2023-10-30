// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// GeNN includes
#include "runtime/runtime.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

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
        .def(pybind11::init<const std::string&, const GeNN::CodeGenerator::ModelSpecMerged&, 
                            const GeNN::CodeGenerator::BackendBase&>())

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
        .def("custom_update", &Runtime::customUpdate)
        
        .def("get_array", pybind11::overload_cast<const GeNN::CurrentSource&, const std::string&>(&Runtime::getArray, pybind11::const_), pybind11::return_value_policy::reference)
        .def("get_array", pybind11::overload_cast<const GeNN::NeuronGroup&, const std::string&>(&Runtime::getArray, pybind11::const_), pybind11::return_value_policy::reference)
        .def("get_array", pybind11::overload_cast<const GeNN::SynapseGroup&, const std::string&>(&Runtime::getArray, pybind11::const_), pybind11::return_value_policy::reference)
        .def("get_array", pybind11::overload_cast<const GeNN::CustomUpdateBase&, const std::string&>(&Runtime::getArray, pybind11::const_), pybind11::return_value_policy::reference)
        .def("get_array", pybind11::overload_cast<const GeNN::CustomConnectivityUpdate&, const std::string&>(&Runtime::getArray, pybind11::const_), pybind11::return_value_policy::reference)
        
        .def("allocate_array", pybind11::overload_cast<const GeNN::CurrentSource&, const std::string&, size_t>(&Runtime::allocateArray))
        .def("allocate_array", pybind11::overload_cast<const GeNN::NeuronGroup&, const std::string&, size_t>(&Runtime::allocateArray))
        .def("allocate_array", pybind11::overload_cast<const GeNN::SynapseGroup&, const std::string&, size_t>(&Runtime::allocateArray))
        .def("allocate_array", pybind11::overload_cast<const GeNN::CustomUpdateBase&, const std::string&, size_t>(&Runtime::allocateArray))
        .def("allocate_array", pybind11::overload_cast<const GeNN::CustomConnectivityUpdate&, const std::string&, size_t>(&Runtime::allocateArray))
        
        
        .def("pull_recording_buffers_from_device", &Runtime::pullRecordingBuffersFromDevice)

        .def("get_custom_update_time", &Runtime::getCustomUpdateTime)
        .def("get_custom_update_transpose_time", &Runtime::getCustomUpdateTransposeTime)
        
        .def("get_recorded_spikes", 
             [](const Runtime &r, const GeNN::NeuronGroup &group)
             {
                 const auto spikes = r.getRecordedSpikes(group);
                 std::vector<std::pair<pybind11::array_t<double>, pybind11::array_t<int>>> npSpikes;
                 std::transform(spikes.cbegin(), spikes.cend(), std::back_inserter(npSpikes),
                                [](const auto &s)
                                {
                                    const pybind11::array_t<double> times = pybind11::cast(s.first);
                                    const pybind11::array_t<int> ids = pybind11::cast(s.second);
                                    return std::make_pair(times, ids);
                                });
                 return npSpikes;
             })
        .def("get_recorded_spike_events", 
             [](const Runtime &r, const GeNN::NeuronGroup &group)
             {
                 const auto spikes = r.getRecordedSpikeEvents(group);
                 std::vector<std::pair<pybind11::array_t<double>, pybind11::array_t<int>>> npSpikes;
                 std::transform(spikes.cbegin(), spikes.cend(), std::back_inserter(npSpikes),
                                [](const auto &s)
                                {
                                    const pybind11::array_t<double> times = pybind11::cast(s.first);
                                    const pybind11::array_t<int> ids = pybind11::cast(s.second);
                                    return std::make_pair(times, ids);
                                });
                 return npSpikes;
             });
        /*.def("get_nccl_unique_id", 
            [](SharedLibraryModel<T> &s) 
            { 
               return pybind11::memoryview::from_memory(s.ncclGetUniqueID(), 
                                                        s.ncclGetUniqueIDBytes(), true);
            })*/

}