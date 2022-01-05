// PyBind11 includes
#include <pybind11/pybind11.h>

// GeNN includes
#include "currentSource.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// genn
//----------------------------------------------------------------------------
PYBIND11_MODULE(genn, m) 
{
    //------------------------------------------------------------------------
    // pygenn::ModelSpec
    //------------------------------------------------------------------------
    pybind11::class_<ModelSpec>(m, "ModelSpec")
        .def(pybind11::init<>())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property("name", &ModelSpec::getName, &ModelSpec::setName)
        .def_property("precision", &ModelSpec::getPrecision, &ModelSpec::setPrecision)
        .def_property("time_precision", &ModelSpec::getTimePrecision, &ModelSpec::setTimePrecision)
        .def_property("dt", &ModelSpec::getDT, &ModelSpec::setDT)
        .def_property("timing_enabled", &ModelSpec::isTimingEnabled, &ModelSpec::setTiming)
        .def_property("batch_size", &ModelSpec::getBatchSize, &ModelSpec::setBatchSize)
        .def_property("seed", &ModelSpec::getSeed, &ModelSpec::setSeed)
        
        .def_property("default_var_location", nullptr, &ModelSpec::setDefaultVarLocation)
        .def_property("default_sparse_connectivity_location", nullptr, &ModelSpec::setDefaultSparseConnectivityLocation)
        .def_property("default_narrow_sparse_ind_enabled", nullptr, &ModelSpec::setDefaultNarrowSparseIndEnabled)
        .def_property("fuse_postsynaptic_models", nullptr, &ModelSpec::setFusePostsynapticModels)
        .def_property("fuse_pre_post_weight_update_models", nullptr, &ModelSpec::setFusePrePostWeightUpdateModels)
        
        .def_property_readonly("num_neurons", &ModelSpec::getNumNeurons);
    
    //------------------------------------------------------------------------
    // pygenn::CurrentSource
    //------------------------------------------------------------------------
    pybind11::class_<CurrentSource>(m, "CurrentSource")
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &CurrentSource::getName)
        
        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("set_var_location", &CurrentSource::setVarLocation)
        .def("get_var_location", pybind11::overload_cast<const std::string&>(&CurrentSource::getVarLocation, pybind11::const_));
}
