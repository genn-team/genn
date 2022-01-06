// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Severity.h>

// GeNN includes
#include "currentSource.h"
#include "modelSpecInternal.h"
#include "snippet.h"
#include "models.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

// PyGeNN includes
#include "trampolines.h"

//----------------------------------------------------------------------------
// genn
//----------------------------------------------------------------------------
PYBIND11_MODULE(genn, m) 
{
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    pybind11::enum_<plog::Severity>(m, "PlogSeverity")
        .value("NONE", plog::Severity::none)
        .value("FATAL", plog::Severity::fatal)
        .value("ERROR", plog::Severity::error)
        .value("WARNING", plog::Severity::warning)
        .value("INFO", plog::Severity::info)
        .value("DEBUG", plog::Severity::debug)
        .value("VERBOSE", plog::Severity::verbose);

    pybind11::enum_<SynapseMatrixConnectivity>(m, "SynapseMatrixConnectivity")
        .value("DENSE", SynapseMatrixConnectivity::DENSE)
        .value("BITMASK", SynapseMatrixConnectivity::BITMASK)
        .value("SPARSE", SynapseMatrixConnectivity::SPARSE)
        .value("PROCEDURAL", SynapseMatrixConnectivity::PROCEDURAL)
        .value("TOEPLITZ", SynapseMatrixConnectivity::TOEPLITZ);

    pybind11::enum_<SynapseMatrixWeight>(m, "SynapseMatrixWeight")
        .value("GLOBAL", SynapseMatrixWeight::GLOBAL)
        .value("INDIVIDUAL", SynapseMatrixWeight::INDIVIDUAL)
        .value("PROCEDURAL", SynapseMatrixWeight::PROCEDURAL)
        .value("KERNEL", SynapseMatrixWeight::KERNEL);

    pybind11::enum_<SynapseMatrixType>(m, "SynapseMatrixType")
        .value("DENSE_GLOBALG", SynapseMatrixType::DENSE_GLOBALG)
        .value("DENSE_INDIVIDUALG", SynapseMatrixType::DENSE_INDIVIDUALG)
        .value("DENSE_PROCEDURALG", SynapseMatrixType::DENSE_PROCEDURALG)
        .value("BITMASK_GLOBALG", SynapseMatrixType::BITMASK_GLOBALG)
        .value("SPARSE_GLOBALG", SynapseMatrixType::SPARSE_GLOBALG)
        .value("SPARSE_INDIVIDUALG", SynapseMatrixType::SPARSE_INDIVIDUALG)
        .value("PROCEDURAL_GLOBALG", SynapseMatrixType::PROCEDURAL_GLOBALG)
        .value("PROCEDURAL_PROCEDURALG", SynapseMatrixType::PROCEDURAL_PROCEDURALG)
        .value("PROCEDURAL_KERNELG", SynapseMatrixType::PROCEDURAL_KERNELG)
        .value("TOEPLITZ_KERNELG", SynapseMatrixType::TOEPLITZ_KERNELG)

        .def("__and__", [](SynapseMatrixType a, SynapseMatrixConnectivity b){ return a & b; }, 
             pybind11::is_operator())
        .def("__and__", [](SynapseMatrixType a, SynapseMatrixWeight b){ return a & b; }, 
             pybind11::is_operator());

    pybind11::enum_<VarAccessModeAttribute>(m, "VarAccessModeAttribute")
        .value("READ_ONLY", VarAccessModeAttribute::READ_ONLY)
        .value("READ_WRITE", VarAccessModeAttribute::READ_WRITE)
        .value("REDUCE", VarAccessModeAttribute::REDUCE)
        .value("SUM", VarAccessModeAttribute::SUM)
        .value("MAX", VarAccessModeAttribute::MAX);

    //! Supported combination of VarAccessModeAttribute
    pybind11::enum_<VarAccessMode>(m, "VarAccessMode")
        .value("READ_WRITE", VarAccessMode::READ_WRITE)
        .value("READ_ONLY", VarAccessMode::READ_ONLY)
        .value("REDUCE_SUM", VarAccessMode::REDUCE_SUM)
        .value("REDUCE_MAX", VarAccessMode::REDUCE_MAX)

        .def("__and__", [](VarAccessMode a, VarAccessModeAttribute b){ return a & b; }, 
             pybind11::is_operator());

    //! Flags defining how variables should be duplicated across multiple batches
    pybind11::enum_<VarAccessDuplication>(m, "VarAccessDuplication")
        .value("DUPLICATE", VarAccessDuplication::DUPLICATE)
        .value("SHARED", VarAccessDuplication::SHARED);

    //! Supported combinations of VarAccessMode and VarAccessDuplication
    pybind11::enum_<VarAccess>(m, "VarAccess")
        .value("READ_WRITE", VarAccess::READ_WRITE)
        .value("READ_ONLY", VarAccess::READ_ONLY)
        .value("READ_ONLY_DUPLICATE", VarAccess::READ_ONLY_DUPLICATE)
        .value("REDUCE_BATCH_SUM", VarAccess::REDUCE_BATCH_SUM)
        .value("REDUCE_BATCH_MAX", VarAccess::REDUCE_BATCH_MAX)

        .def("__and__", [](VarAccess a, VarAccessModeAttribute b){ return a & b; }, 
             pybind11::is_operator())
        .def("__and__", [](VarAccess a, VarAccessMode b){ return a & b; }, 
             pybind11::is_operator())
        .def("__and__", [](VarAccess a, VarAccessDuplication b){ return a & b; }, 
             pybind11::is_operator());

    //------------------------------------------------------------------------
    // genn.ModelSpec
    //------------------------------------------------------------------------
    pybind11::class_<ModelSpecInternal>(m, "ModelSpecInternal")
        .def(pybind11::init<>())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property("name", &ModelSpecInternal::getName, &ModelSpecInternal::setName)
        .def_property("precision", &ModelSpecInternal::getPrecision, &ModelSpecInternal::setPrecision)
        .def_property("time_precision", &ModelSpecInternal::getTimePrecision, &ModelSpecInternal::setTimePrecision)
        .def_property("dt", &ModelSpecInternal::getDT, &ModelSpecInternal::setDT)
        .def_property("timing_enabled", &ModelSpecInternal::isTimingEnabled, &ModelSpecInternal::setTiming)
        .def_property("batch_size", &ModelSpecInternal::getBatchSize, &ModelSpecInternal::setBatchSize)
        .def_property("seed", &ModelSpecInternal::getSeed, &ModelSpecInternal::setSeed)

        .def_property("default_var_location", nullptr, &ModelSpecInternal::setDefaultVarLocation)
        .def_property("default_sparse_connectivity_location", nullptr, &ModelSpecInternal::setDefaultSparseConnectivityLocation)
        .def_property("default_narrow_sparse_ind_enabled", nullptr, &ModelSpecInternal::setDefaultNarrowSparseIndEnabled)
        .def_property("fuse_postsynaptic_models", nullptr, &ModelSpecInternal::setFusePostsynapticModels)
        .def_property("fuse_pre_post_weight_update_models", nullptr, &ModelSpecInternal::setFusePrePostWeightUpdateModels)

        .def_property_readonly("num_neurons", &ModelSpecInternal::getNumNeurons)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("finalize", &ModelSpecInternal::finalize);

    //------------------------------------------------------------------------
    // genn.CurrentSource
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

    //------------------------------------------------------------------------
    // genn.NeuronGroup
    //------------------------------------------------------------------------
    pybind11::class_<NeuronGroup>(m, "NeuronGroup")
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &NeuronGroup::getName)
        .def_property_readonly("num_neurons", &NeuronGroup::getNumNeurons)

        .def_property("spike_location", &NeuronGroup::getSpikeLocation, &NeuronGroup::setSpikeLocation)
        .def_property("spike_event_location", &NeuronGroup::getSpikeEventLocation, &NeuronGroup::setSpikeEventLocation)
        .def_property("spike_time_location", &NeuronGroup::getSpikeTimeLocation, &NeuronGroup::setSpikeTimeLocation)
        .def_property("prev_spike_time_location", &NeuronGroup::getPrevSpikeTimeLocation, &NeuronGroup::setPrevSpikeTimeLocation)
        .def_property("spike_event_time_location", &NeuronGroup::getSpikeEventTimeLocation, &NeuronGroup::setSpikeEventTimeLocation)
        .def_property("prev_spike_event_time_location", &NeuronGroup::getPrevSpikeEventTimeLocation, &NeuronGroup::setPrevSpikeEventTimeLocation)
        .def_property("spike_recording_enabled", &NeuronGroup::isSpikeRecordingEnabled, &NeuronGroup::setSpikeRecordingEnabled)
        .def_property("spike_event_recording_enabled", &NeuronGroup::isSpikeEventRecordingEnabled, &NeuronGroup::setSpikeEventRecordingEnabled)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("set_var_location", &NeuronGroup::setVarLocation)
        .def("get_var_location", pybind11::overload_cast<const std::string&>(&NeuronGroup::getVarLocation, pybind11::const_));

    //------------------------------------------------------------------------
    // genn.PreferencesBase
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::PreferencesBase>(m, "PreferencesBase")
        .def_readwrite("optimize_code", &CodeGenerator::PreferencesBase::optimizeCode)
        .def_readwrite("debug_code", &CodeGenerator::PreferencesBase::debugCode)
        .def_readwrite("enable_bitmask_optimisations", &CodeGenerator::PreferencesBase::enableBitmaskOptimisations)
        .def_readwrite("generate_extra_global_param_pull", &CodeGenerator::PreferencesBase::generateExtraGlobalParamPull)
        .def_readwrite("log_level", &CodeGenerator::PreferencesBase::logLevel);

    //------------------------------------------------------------------------
    // genn.SnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base, PySnippet<>>(m, "SnippetBase")
        .def("get_param_names", &Snippet::Base::getParamNames)
        .def("get_derived_params", &Snippet::Base::getDerivedParams)
        .def("get_extra_global_params", &Snippet::Base::getExtraGlobalParams);

    //------------------------------------------------------------------------
    // genn.ModelBase
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base, Snippet::Base, PyModel<>>(m, "ModelBase")
        .def("get_vars", &Models::Base::getVars);
}
