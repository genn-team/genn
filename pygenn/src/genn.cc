// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Severity.h>
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "currentSource.h"
#include "currentSourceModels.h"
#include "initSparseConnectivitySnippet.h"
#include "initToeplitzConnectivitySnippet.h"
#include "initVarSnippet.h"
#include "modelSpecInternal.h"
#include "neuronModels.h"
#include "postsynapticModels.h"
#include "weightUpdateModels.h"
#include "snippet.h"
#include "models.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/generateMakefile.h"
#include "code_generator/generateModules.h"
#include "code_generator/generateMSBuild.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// PySnippet
//----------------------------------------------------------------------------
// 'Trampoline' base class to wrap classes derived off Snippet::Base
template <class SnippetBase = Snippet::Base> 
class PySnippet : public SnippetBase 
{   
public: 
    virtual Snippet::Base::StringVec getParamNames() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::StringVec, SnippetBase, "get_param_names", getParamNames); }
    virtual Snippet::Base::DerivedParamVec getDerivedParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::DerivedParamVec, SnippetBase, "get_derived_params", getDerivedParams); }
    virtual Snippet::Base::EGPVec getExtraGlobalParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::EGPVec, SnippetBase, "get_extra_global_params", getExtraGlobalParams); }    
};

//----------------------------------------------------------------------------
// PyModel
//----------------------------------------------------------------------------
// 'Trampoline' base class to wrap classes derived off Models::Base
template <class ModelBase = Models::Base> 
class PyModel : public PySnippet<ModelBase> 
{
public:
    virtual Models::Base::VarVec getVars() const override{ PYBIND11_OVERRIDE_NAME(Models::Base::VarVec, ModelBase, "get_vars", getVars); }
};

//----------------------------------------------------------------------------
// PyInitSparseConnectivitySnippetBase
//----------------------------------------------------------------------------
// 'Trampoline' class for sparse connectivity initialisation snippets
class PyInitSparseConnectivitySnippetBase : public PySnippet<InitSparseConnectivitySnippet::Base> 
{
    using Base = InitSparseConnectivitySnippet::Base;
public:
    virtual std::string getRowBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_row_build_code", getRowBuildCode); }
    virtual ParamValVec getRowBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(Snippet::Base::ParamValVec, Base, "get_row_build_state_vars", getRowBuildStateVars); }
    virtual std::string getColBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_col_build_code", getColBuildCode); }
    virtual ParamValVec getColBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(Snippet::Base::ParamValVec, Base, "get_col_build_state_vars", getColBuildStateVars); }
    virtual std::string getHostInitCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_host_init_code", getHostInitCode); }
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_row_length_func", getCalcMaxRowLengthFunc); }
    virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_col_length_func", getCalcMaxColLengthFunc); }
    virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override { PYBIND11_OVERRIDE_NAME(CalcKernelSizeFunc, Base, "get_calc_kernel_size_func", getCalcKernelSizeFunc); }
};

//----------------------------------------------------------------------------
// PyInitToeplitzConnectivitySnippetBase
//----------------------------------------------------------------------------
// 'Trampoline' class for toeplitz connectivity initialisation snippets
class PyInitToeplitzConnectivitySnippetBase : public PySnippet<InitToeplitzConnectivitySnippet::Base> 
{
    using Base = InitToeplitzConnectivitySnippet::Base;
public:
    virtual std::string getDiagonalBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_diagonal_build_code", getDiagonalBuildCode); }
    virtual ParamValVec getDiagonalBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(ParamValVec, Base, "get_diagonal_build_state_vars", getDiagonalBuildStateVars); }
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_row_length_func", getCalcMaxRowLengthFunc); }
    virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override { PYBIND11_OVERRIDE_NAME(CalcKernelSizeFunc, Base, "get_calc_kernel_size_func", getCalcKernelSizeFunc); }
};

//----------------------------------------------------------------------------
// PyInitVarSnippetBase
//----------------------------------------------------------------------------
// 'Trampoline' class for variable initialisation snippets
class PyInitVarSnippetBase : public PySnippet<InitVarSnippet::Base> 
{
    using Base = InitVarSnippet::Base;
public:
    virtual std::string getCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_code", getCode); }
};

//----------------------------------------------------------------------------
// PyCurrentSourceModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for current source models
class PyCurrentSourceModelBase : public PyModel<CurrentSourceModels::Base> 
{
    using Base = CurrentSourceModels::Base;
public:
    virtual std::string getInjectionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_injection_code", getInjectionCode); }
};

//----------------------------------------------------------------------------
// PyNeuronModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for neuron models
class PyNeuronModelBase : public PyModel<NeuronModels::Base> 
{
    using Base = NeuronModels::Base;
public:
    virtual std::string getSimCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_sim_code", getSimCode); }
    virtual std::string getThresholdConditionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_threshold_condition_code", getThresholdConditionCode); }
    virtual std::string getResetCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_reset_code", getResetCode); }
    virtual std::string getSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_support_code", getSupportCode); }

    virtual Models::Base::ParamValVec getAdditionalInputVars() const override { PYBIND11_OVERRIDE_NAME(Models::Base::ParamValVec, Base, "get_additional_input_vars", getAdditionalInputVars); }

    virtual bool isAutoRefractoryRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_auto_refractory_required", isAutoRefractoryRequired); }
};

//----------------------------------------------------------------------------
// PyPostsynapticModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for postsynaptic models
class PyPostsynapticModelBase : public PyModel<PostsynapticModels::Base> 
{
    using Base = PostsynapticModels::Base;
public:
    virtual std::string getDecayCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_decay_code", getDecayCode); }
    virtual std::string getApplyInputCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_apply_input_code", getApplyInputCode); }
    virtual std::string getSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_support_code", getSupportCode); }
};

//----------------------------------------------------------------------------
// PyWeightUpdateModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for weight update models
class PyWeightUpdateModelBase : public PyModel<WeightUpdateModels::Base> 
{
    using Base = WeightUpdateModels::Base;
public:
    virtual std::string getSimCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_sim_code", getSimCode); }
    virtual std::string getEventCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_event_code", getEventCode); }
    virtual std::string getLearnPostCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_learn_post_code", getLearnPostCode); }
    virtual std::string getSynapseDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_synapse_dynamics_code", getSynapseDynamicsCode); }
    virtual std::string getEventThresholdConditionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_event_threshold_condition_code", getEventThresholdConditionCode); }
    virtual std::string getSimSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_learn_post_support_code", getSimSupportCode); }
    virtual std::string getLearnPostSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_learn_post_support_code", getLearnPostSupportCode); }
    virtual std::string getSynapseDynamicsSuppportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_synapse_dynamics_support_code", getSynapseDynamicsSuppportCode); }
    virtual std::string getPreSpikeCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_pre_spike_code", getPreSpikeCode); }
    virtual std::string getPostSpikeCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_post_spike_code", getPostSpikeCode); }
    virtual std::string getPreDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_pre_dynamics_code", getPreDynamicsCode); }
    virtual std::string getPostDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_post_dynamics_code", getPostDynamicsCode); }
    virtual VarVec getPreVars() const override { PYBIND11_OVERRIDE_NAME(Models::Base::VarVec, Base, "get_pre_vars", getPreVars); }
    virtual VarVec getPostVars() const override { PYBIND11_OVERRIDE_NAME(Models::Base::VarVec, Base, "get_post_vars", getPostVars); }
    virtual bool isPreSpikeTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_pre_spike_time_required", isPreSpikeTimeRequired); }
    virtual bool isPostSpikeTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_post_spike_time_required", isPostSpikeTimeRequired); }
    virtual bool isPreSpikeEventTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_pre_spike_event_time_required", isPreSpikeEventTimeRequired); }
    virtual bool isPrevPreSpikeTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_prev_pre_spike_time_required", isPrevPreSpikeTimeRequired); }
    virtual bool isPrevPostSpikeTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_prev_post_spike_time_required", isPrevPostSpikeTimeRequired); }
    virtual bool isPrevPreSpikeEventTimeRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_prev_pre_spike_event_time_required", isPrevPreSpikeEventTimeRequired); }
};

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

void initLogging(plog::Severity gennLevel, plog::Severity codeGeneratorLevel)
{
    auto *consoleAppender = new plog::ConsoleAppender<plog::TxtFormatter>;
    Logging::init(gennLevel, codeGeneratorLevel, consoleAppender, consoleAppender);
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
    
    //! Precision to use for scalar type variables 
    pybind11::enum_<ScalarPrecision>(m, "ScalarPrecision")
        .value("FLOAT", ScalarPrecision::FLOAT)
        .value("DOUBLE", ScalarPrecision::DOUBLE)
        .value("LONG_DOUBLE", ScalarPrecision::LONG_DOUBLE);

    //! Precision to use for variables which store time
    pybind11::enum_<TimePrecision>(m, "TimePrecision")
        .value("DEFAULT", TimePrecision::DEFAULT)
        .value("FLOAT", TimePrecision::FLOAT)
        .value("DOUBLE", TimePrecision::DOUBLE);
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("generate_code", &generateCode, pybind11::return_value_policy::move);
    m.def("init_logging", &initLogging);

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
        .def_property_readonly("current_source_model", &CurrentSource::getCurrentSourceModel, pybind11::return_value_policy::reference)
        .def_property_readonly("params", &CurrentSource::getParams)
        .def_property_readonly("var_initialisers", &CurrentSource::getVarInitialisers)

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
        .def_property_readonly("neuron_model", &NeuronGroup::getNeuronModel, pybind11::return_value_policy::reference)
        .def_property_readonly("params", &NeuronGroup::getParams)
        .def_property_readonly("var_initialisers", &NeuronGroup::getVarInitialisers)
        .def_property_readonly("num_delay_slots", &NeuronGroup::getNumDelaySlots)
        
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
        .def_property_readonly("wu_model", &SynapseGroup::getWUModel)
        .def_property_readonly("wu_params", &SynapseGroup::getWUParams)
        .def_property_readonly("wu_var_initialisers", &SynapseGroup::getWUVarInitialisers)
        .def_property_readonly("wu_pre_var_initialisers", &SynapseGroup::getWUPreVarInitialisers)
        .def_property_readonly("wu_post_var_initialisers", &SynapseGroup::getWUPostVarInitialisers)
        .def_property_readonly("ps_model", &SynapseGroup::getPSModel)
        .def_property_readonly("ps_params", &SynapseGroup::getPSParams)
        .def_property_readonly("ps_var_initialisers", &SynapseGroup::getPSVarInitialisers)
        .def_property_readonly("kernel_size", &SynapseGroup::getKernelSize)
        .def_property_readonly("matrix_type", &SynapseGroup::getMatrixType)
        .def_property_readonly("sparse_connectivity_initialiser", &SynapseGroup::getConnectivityInitialiser)
        .def_property_readonly("toeplitz_connectivity_initialiser", &SynapseGroup::getToeplitzConnectivityInitialiser)
    
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
        // **NOTE** we use the 'publicist' pattern to expose some protected properties
        .def_property_readonly("_ps_model_fused", &SynapseGroupInternal::isPSModelFused)
        .def_property_readonly("_wu_pre_model_fused", &SynapseGroupInternal::isWUPreModelFused)
        .def_property_readonly("_wu_post_model_fused", &SynapseGroupInternal::isWUPostModelFused)
        .def_property_readonly("_sparse_ind_type", &SynapseGroupInternal::getSparseIndType)
        
        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("get_wu_var_location", &SynapseGroup::getWUVarLocation)
        .def("set_wu_var_location", &SynapseGroup::setWUVarLocation)
        .def("get_wu_pre_var_location", &SynapseGroup::getWUPreVarLocation)
        .def("set_wu_pre_var_location", &SynapseGroup::setWUPreVarLocation)
        .def("get_wu_post_var_location", &SynapseGroup::getWUPostVarLocation)
        .def("set_wu_post_var_location", &SynapseGroup::setWUPostVarLocation)
        .def("get_ps_var_location", &SynapseGroup::getPSVarLocation)
        .def("set_ps_var_location", &SynapseGroup::setPSVarLocation);
    
    //------------------------------------------------------------------------
    // genn.EGP
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::EGP>(m, "EGP")
        .def(pybind11::init<const std::string&, const std::string&>())
        .def_readonly("name", &Snippet::Base::EGP::name)
        .def_readonly("type", &Snippet::Base::EGP::type);
    
    //------------------------------------------------------------------------
    // genn.ParamVal
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::ParamVal>(m, "ParamVal")
        .def(pybind11::init<const std::string&, const std::string&, const std::string&>())
        .def(pybind11::init<const std::string&, const std::string&, double>())
        .def_readonly("name", &Snippet::Base::ParamVal::name)
        .def_readonly("type", &Snippet::Base::ParamVal::type)
        .def_readonly("value", &Snippet::Base::ParamVal::value);

    //------------------------------------------------------------------------
    // genn.SnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base, PySnippet<>>(m, "SnippetBase")
        .def("get_param_names", &Snippet::Base::getParamNames)
        .def("get_derived_params", &Snippet::Base::getDerivedParams)
        .def("get_extra_global_params", &Snippet::Base::getExtraGlobalParams);
    
    //------------------------------------------------------------------------
    // genn.InitSparseConnectivitySnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<InitSparseConnectivitySnippet::Base, Snippet::Base, PyInitSparseConnectivitySnippetBase>(m, "InitSparseConnectivitySnippetBase")
        .def(pybind11::init<>())
        .def("get_row_build_code", &InitSparseConnectivitySnippet::Base::getRowBuildCode)
        .def("get_row_build_state_vars", &InitSparseConnectivitySnippet::Base::getRowBuildStateVars)
        .def("get_col_build_code", &InitSparseConnectivitySnippet::Base::getColBuildCode)
        .def("get_col_build_state_vars", &InitSparseConnectivitySnippet::Base::getColBuildStateVars)
        .def("get_host_init_code", &InitSparseConnectivitySnippet::Base::getHostInitCode)
        .def("get_calc_max_row_length_func", &InitSparseConnectivitySnippet::Base::getCalcMaxRowLengthFunc)
        .def("get_calc_max_col_length_func", &InitSparseConnectivitySnippet::Base::getCalcMaxColLengthFunc)
        .def("get_calc_kernel_size_func", &InitSparseConnectivitySnippet::Base::getCalcKernelSizeFunc);
    
    //------------------------------------------------------------------------
    // genn.InitToeplitzConnectivitySnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<InitToeplitzConnectivitySnippet::Base, Snippet::Base, PyInitToeplitzConnectivitySnippetBase>(m, "InitToeplitzConnectivitySnippetBase")
        .def(pybind11::init<>())
        .def("get_diagonal_build_code", &InitToeplitzConnectivitySnippet::Base::getDiagonalBuildCode)
        .def("get_diagonal_build_state_vars", &InitToeplitzConnectivitySnippet::Base::getDiagonalBuildStateVars)
        .def("get_calc_max_row_length_func", &InitToeplitzConnectivitySnippet::Base::getCalcMaxRowLengthFunc)
        .def("get_calc_kernel_size_func", &InitToeplitzConnectivitySnippet::Base::getCalcKernelSizeFunc);
    
    //------------------------------------------------------------------------
    // genn.InitVarSnippetBaseBase
    //------------------------------------------------------------------------
    pybind11::class_<InitVarSnippet::Base, Snippet::Base, PyInitVarSnippetBase>(m, "InitVarSnippetBase")
        .def(pybind11::init<>())

        .def("get_code", &InitVarSnippet::Base::getCode);
    
    //------------------------------------------------------------------------
    // genn.Var
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base::Var>(m, "Var")
        .def(pybind11::init<const std::string&, const std::string&, VarAccess>())
        .def(pybind11::init<const std::string&, const std::string&>())
        .def_readonly("name", &Models::Base::Var::name)
        .def_readonly("type", &Models::Base::Var::type)
        .def_readonly("access", &Models::Base::Var::access);
        
    //------------------------------------------------------------------------
    // genn.ModelBase
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base, Snippet::Base, PyModel<>>(m, "ModelBase")
        .def("get_vars", &Models::Base::getVars);
    
    //------------------------------------------------------------------------
    // genn.CurrentSourceModelBase
    //------------------------------------------------------------------------
    pybind11::class_<CurrentSourceModels::Base, Models::Base, PyCurrentSourceModelBase>(m, "CurrentSourceModelBase")
        .def(pybind11::init<>())

        .def("get_inection_code", &CurrentSourceModels::Base::getInjectionCode);
    
    //------------------------------------------------------------------------
    // genn.NeuronModelBase
    //------------------------------------------------------------------------
    pybind11::class_<NeuronModels::Base, Models::Base, PyNeuronModelBase>(m, "NeuronModelBase")
        .def(pybind11::init<>())

        .def("get_sim_code", &NeuronModels::Base::getSimCode)
        .def("get_threshold_condition_code", &NeuronModels::Base::getThresholdConditionCode)
        .def("get_reset_code", &NeuronModels::Base::getResetCode)
        .def("get_support_code", &NeuronModels::Base::getSupportCode)
        .def("get_additional_input_vars", &NeuronModels::Base::getAdditionalInputVars)
        .def("is_auto_refractory_required", &NeuronModels::Base::isAutoRefractoryRequired);
        
    //------------------------------------------------------------------------
    // genn.PostsynapticModelBase
    //------------------------------------------------------------------------
    pybind11::class_<PostsynapticModels::Base, Models::Base, PyPostsynapticModelBase>(m, "PostsynapticModelBase")
        .def(pybind11::init<>())

        .def("get_decay_code", &PostsynapticModels::Base::getDecayCode)
        .def("get_apply_input_code", &PostsynapticModels::Base::getApplyInputCode)
        .def("get_support_code", &PostsynapticModels::Base::getSupportCode);
    
    //------------------------------------------------------------------------
    // genn.WeightUpdateModelBase
    //------------------------------------------------------------------------
    pybind11::class_<WeightUpdateModels::Base, Models::Base, PyWeightUpdateModelBase>(m, "WeightUpdateModelBase")
        .def(pybind11::init<>())
        
        .def("get_sim_code", &WeightUpdateModels::Base::getSimCode)
        .def("get_event_code", &WeightUpdateModels::Base::getEventCode)
        .def("get_learn_post_code", &WeightUpdateModels::Base::getLearnPostCode)
        .def("get_synapse_dynamics_code", &WeightUpdateModels::Base::getSynapseDynamicsCode)
        .def("get_event_threshold_condition_code", &WeightUpdateModels::Base::getEventThresholdConditionCode)
        .def("get_sim_support_cde", &WeightUpdateModels::Base::getSimSupportCode)
        .def("get_learn_post_support_code", &WeightUpdateModels::Base::getLearnPostSupportCode)
        .def("get_synapse_dynamics_support_code", &WeightUpdateModels::Base::getSynapseDynamicsSuppportCode)
        .def("get_pre_spike_code", &WeightUpdateModels::Base::getPreSpikeCode)
        .def("get_post_spike_code", &WeightUpdateModels::Base::getPostSpikeCode)
        .def("get_pre_dynamics_code", &WeightUpdateModels::Base::getPreDynamicsCode)
        .def("get_post_dynamics_code", &WeightUpdateModels::Base::getPostDynamicsCode)
        .def("get_pre_vars", &WeightUpdateModels::Base::getPreVars)
        .def("get_post_vars", &WeightUpdateModels::Base::getPostVars)
        .def("is_pre_spike_time_required", &WeightUpdateModels::Base::isPreSpikeTimeRequired)
        .def("is_post_spike_time_required", &WeightUpdateModels::Base::isPostSpikeTimeRequired)
        .def("is_pre_spike_event_time_required", &WeightUpdateModels::Base::isPreSpikeEventTimeRequired)
        .def("is_prev_pre_spike_time_required", &WeightUpdateModels::Base::isPrevPreSpikeTimeRequired)
        .def("is_prev_post_spike_time_required", &WeightUpdateModels::Base::isPrevPostSpikeTimeRequired)
        .def("is_prev_pre_spike_event_time_required", &WeightUpdateModels::Base::isPrevPreSpikeEventTimeRequired);

    //------------------------------------------------------------------------
    // genn.SparseConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitSparseConnectivitySnippet::Init>(m, "SparseConnectivityInit")
        .def(pybind11::init<const InitSparseConnectivitySnippet::Base*, const std::unordered_map<std::string, double>&>())
        .def_property_readonly("snippet", &InitSparseConnectivitySnippet::Init::getSnippet, pybind11::return_value_policy::reference);
        
    //------------------------------------------------------------------------
    // genn.ToeplitzConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitToeplitzConnectivitySnippet::Init>(m, "ToeplitzConnectivityInit")
        .def(pybind11::init<const InitToeplitzConnectivitySnippet::Base*, const std::unordered_map<std::string, double>&>())
        .def_property_readonly("snippet", &InitToeplitzConnectivitySnippet::Init::getSnippet, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.VarInit
    //------------------------------------------------------------------------
    pybind11::class_<Models::VarInit>(m, "VarInit")
        .def(pybind11::init<const InitVarSnippet::Base*, const std::unordered_map<std::string, double>&>())
        .def(pybind11::init<double>())
        .def_property_readonly("snippet", &Models::VarInit::getSnippet, pybind11::return_value_policy::reference);

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
