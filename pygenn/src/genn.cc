// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Severity.h>

// GeNN includes
#include "currentSource.h"
#include "initSparseConnectivitySnippet.h"
#include "initToeplitzConnectivitySnippet.h"
#include "modelSpecInternal.h"
#include "snippet.h"
#include "models.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/generateMakefile.h"
#include "code_generator/generateModules.h"
#include "code_generator/generateMSBuild.h"

// PyGeNN includes
#include "trampolines.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
CodeGenerator::MemAlloc generateCode(ModelSpecInternal &model, CodeGenerator::BackendBase &backend, 
                                     const std::string &sharePathStr, const std::string &outputPathStr, bool forceRebuild)
{
    const filesystem::path outputPath(outputPathStr);

    // Generate code, returning list of module names that must be build
    const auto output = CodeGenerator::generateAll(
        model, backend, 
        filesystem::path(sharePathStr), outputPath, 
        forceRebuild);

#ifdef _WIN32
    // Create MSBuild project to compile and link all generated modules
    std::ofstream makefile((outputPath / "runner.vcxproj").str());
    CodeGenerator::generateMSBuild(makefile, model, backend, "", output.first);
#else
    // Create makefile to compile and link all generated modules
    std::ofstream makefile((outputPath / "Makefile").str());
    CodeGenerator::generateMakefile(makefile, backend, output.first);
#endif
    return output.second;
}
}

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
    
    //! Locations of variables
    pybind11::enum_<VarLocation>(m, "VarLocation")
        .value("HOST", VarLocation::HOST)
        .value("DEVICE", VarLocation::DEVICE)
        .value("ZERO_COPY", VarLocation::ZERO_COPY)
        .value("HOST_DEVICE", VarLocation::HOST_DEVICE)
        .value("HOST_DEVICE_ZERO_COPY", VarLocation::HOST_DEVICE_ZERO_COPY)
        
        .def("__and__", [](VarLocation a, VarLocation b){ return a & b; }, 
             pybind11::is_operator());
        
    //! Paralllelism hints for synapse groups
    pybind11::enum_<SynapseGroup::SpanType>(m, "SpanType")
        .value("POSTSYNAPTIC", SynapseGroup::SpanType::POSTSYNAPTIC)
        .value("PRESYNAPTIC", SynapseGroup::SpanType::PRESYNAPTIC);
        
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("generate_code", &generateCode, pybind11::return_value_policy::move);

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
        .def_property_readonly("recording_in_use", &ModelSpecInternal::isRecordingInUse)
    
        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("add_current_source",  
             static_cast<CurrentSource* (ModelSpecInternal::*)(
                const std::string&, const CurrentSourceModels::Base*, const std::string&, 
                const ParamValues&, const VarValues&)>(&ModelSpecInternal::addCurrentSource), 
            pybind11::return_value_policy::reference)
        .def("add_neuron_population",  
             static_cast<NeuronGroup* (ModelSpecInternal::*)(
                const std::string&, unsigned int, const NeuronModels::Base*, 
                const ParamValues&, const VarValues&)>(&ModelSpecInternal::addNeuronPopulation), 
            pybind11::return_value_policy::reference)
        .def("add_synapse_population",
            static_cast<SynapseGroup* (ModelSpecInternal::*)(
                const std::string&, SynapseMatrixType, unsigned int, const std::string&, const std::string&,
                const WeightUpdateModels::Base*, const ParamValues&, const VarValues&, const VarValues&, const VarValues&,
                const PostsynapticModels::Base*, const ParamValues&, const VarValues&,
                const InitSparseConnectivitySnippet::Init&)>(&ModelSpecInternal::addSynapsePopulation),
            pybind11::return_value_policy::reference)
        .def("add_synapse_population",
            static_cast<SynapseGroup* (ModelSpecInternal::*)(
                const std::string&, SynapseMatrixType, unsigned int, const std::string&, const std::string&,
                const WeightUpdateModels::Base*, const ParamValues&, const VarValues&, const VarValues&, const VarValues&,
                const PostsynapticModels::Base*, const ParamValues&, const VarValues&,
                const InitToeplitzConnectivitySnippet::Init&)>(&ModelSpecInternal::addSynapsePopulation), 
            pybind11::return_value_policy::reference)

        .def("finalize", &ModelSpecInternal::finalize);

    //------------------------------------------------------------------------
    // genn.CurrentSource
    //------------------------------------------------------------------------
    pybind11::class_<CurrentSource>(m, "CurrentSource", pybind11::dynamic_attr())
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
    pybind11::class_<NeuronGroup>(m, "NeuronGroup", pybind11::dynamic_attr())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &NeuronGroup::getName)
        .def_property_readonly("num_neurons", &NeuronGroup::getNumNeurons)
        .def_property_readonly("neuron_model", &NeuronGroup::getNeuronModel)
        .def_property_readonly("params", &NeuronGroup::getParams)
        .def_property_readonly("var_initialisers", &NeuronGroup::getVarInitialisers)
        
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
    // genn.SynapseGroup
    //------------------------------------------------------------------------
    pybind11::class_<SynapseGroup>(m, "SynapseGroup", pybind11::dynamic_attr())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &SynapseGroup::getName)
        .def_property("ps_target_var", &SynapseGroup::getPSTargetVar, &SynapseGroup::setPSTargetVar)
        .def_property("pre_target_var", &SynapseGroup::getPreTargetVar, &SynapseGroup::setPreTargetVar)
         .def_property("in_syn_location", &SynapseGroup::getInSynLocation, &SynapseGroup::setInSynVarLocation)
         .def_property("sparse_connectivity_location", &SynapseGroup::getSparseConnectivityLocation, &SynapseGroup::setSparseConnectivityLocation)
         .def_property("dendritic_delay_location",&SynapseGroup::getDendriticDelayLocation, &SynapseGroup::setDendriticDelayLocation)
         .def_property("max_connections",&SynapseGroup::getMaxConnections, &SynapseGroup::setMaxConnections)
         .def_property("max_source_connections",&SynapseGroup::getMaxSourceConnections, &SynapseGroup::setMaxSourceConnections)
         .def_property("max_dendritic_delay_timesteps",&SynapseGroup::getMaxDendriticDelayTimesteps, &SynapseGroup::setMaxDendriticDelayTimesteps)
         .def_property("span_type",&SynapseGroup::getSpanType, &SynapseGroup::setSpanType)
         .def_property("num_threads_per_spike",&SynapseGroup::getNumThreadsPerSpike, &SynapseGroup::setNumThreadsPerSpike)
         .def_property("back_prop_delay_steps",&SynapseGroup::getBackPropDelaySteps, &SynapseGroup::setBackPropDelaySteps)
         .def_property("narrow_sparse_ind_enabled",nullptr, &SynapseGroup::setNarrowSparseIndEnabled)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("set_wu_var_location", &SynapseGroup::setWUVarLocation)
        .def("set_wu_pre_var_location", &SynapseGroup::setWUPreVarLocation)
        .def("set_wu_post_var_location", &SynapseGroup::setWUPostVarLocation)
        .def("set_ps_var_location", &SynapseGroup::setPSVarLocation);
        
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
    
    //------------------------------------------------------------------------
    // genn.VarInit
    //------------------------------------------------------------------------
    pybind11::class_<Models::VarInit>(m, "VarInit")
        .def(pybind11::init<const InitVarSnippet::Base*, const std::unordered_map<std::string, double>&>())
        .def(pybind11::init<double>());
    
    //------------------------------------------------------------------------
    // genn.ToeplitzConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitToeplitzConnectivitySnippet::Init>(m, "ToeplitzConnectivityInit")
        .def(pybind11::init<const InitToeplitzConnectivitySnippet::Base*, const std::unordered_map<std::string, double>&>());
    
    //------------------------------------------------------------------------
    // genn.SparseConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitSparseConnectivitySnippet::Init>(m, "SparseConnectivityInit")
        .def(pybind11::init<const InitSparseConnectivitySnippet::Base*, const std::unordered_map<std::string, double>&>());
    
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
    // genn.BackendBase
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::BackendBase>(m, "BackendBase");
    
    //------------------------------------------------------------------------
    // genn.MemAlloc
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::MemAlloc>(m, "MemAlloc")
        .def_property_readonly("host_bytes", &CodeGenerator::MemAlloc::getHostBytes)
        .def_property_readonly("device_bytes", &CodeGenerator::MemAlloc::getDeviceBytes)
        .def_property_readonly("zero_copy_bytes", &CodeGenerator::MemAlloc::getZeroCopyBytes)
        .def_property_readonly("host_mbytes", &CodeGenerator::MemAlloc::getHostMBytes)
        .def_property_readonly("device_mbytes", &CodeGenerator::MemAlloc::getDeviceMBytes)
        .def_property_readonly("zero_copy_mbytes", &CodeGenerator::MemAlloc::getZeroCopyMBytes);

}
