// PyBind11 includes
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Plog includes
#include <plog/Severity.h>
#include <plog/Appenders/ConsoleAppender.h>

// GeNN includes
#include "currentSource.h"
#include "currentSourceModels.h"
#include "customConnectivityUpdate.h"
#include "customConnectivityUpdateModels.h"
#include "customUpdate.h"
#include "customUpdateModels.h"
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
#include "code_generator/modelSpecMerged.h"

// GeNN runtime includes
#include "runtime/runtime.h"

using namespace GeNN;
using namespace pybind11::literals;

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
    virtual Snippet::Base::ParamVec getParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::ParamVec, SnippetBase, "get_params", getParams); }
    virtual Snippet::Base::DerivedParamVec getDerivedParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::DerivedParamVec, SnippetBase, "get_derived_params", getDerivedParams); }
    virtual Snippet::Base::EGPVec getExtraGlobalParams() const override{ PYBIND11_OVERRIDE_NAME(Snippet::Base::EGPVec, SnippetBase, "get_extra_global_params", getExtraGlobalParams); }    
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
    virtual std::string getColBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_col_build_code", getColBuildCode); }
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
class PyCurrentSourceModelBase : public PySnippet<CurrentSourceModels::Base> 
{
    using Base = CurrentSourceModels::Base;
public:
    virtual std::vector<Models::Base::Var> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_vars", getVars); }
    virtual VarRefVec getNeuronVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_neuron_var_refs", getNeuronVarRefs); }

    virtual std::string getInjectionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_injection_code", getInjectionCode); }
};

//----------------------------------------------------------------------------
// PyCustomConnectivityUpdateModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for custom connectivity update models
class PyCustomConnectivityUpdateModelBase : public PySnippet<CustomConnectivityUpdateModels::Base> 
{
    using Base = CustomConnectivityUpdateModels::Base;
public:
    virtual std::vector<Models::Base::Var> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_vars", getVars); }
    virtual std::vector<Models::Base::Var> getPreVars() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_pre_vars", getPreVars); }
    virtual std::vector<Models::Base::Var> getPostVars() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_post_vars", getPostVars); }
    
    virtual VarRefVec getVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_var_refs", getVarRefs); }
    virtual VarRefVec getPreVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_pre_var_refs", getPreVarRefs); }
    virtual VarRefVec getPostVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_post_var_refs", getPostVarRefs); }
    
    virtual std::string getRowUpdateCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_row_update_code", getRowUpdateCode); }
    virtual std::string getHostUpdateCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_host_update_code", getHostUpdateCode); }
};

//----------------------------------------------------------------------------
// PyCustomUpdateModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for custom update models
class PyCustomUpdateModelBase : public PySnippet<CustomUpdateModels::Base> 
{
    using Base = CustomUpdateModels::Base;
public:
    virtual std::vector<Models::Base::CustomUpdateVar> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::CustomUpdateVar>, Base, "get_vars", getVars); }
    virtual VarRefVec getVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_var_refs", getVarRefs); }
    virtual EGPRefVec getExtraGlobalParamRefs() const override { PYBIND11_OVERRIDE_NAME(EGPRefVec, Base, "get_extra_global_param_refs", getExtraGlobalParamRefs); }
    
    virtual std::string getUpdateCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_update_code", getUpdateCode); }
};

//----------------------------------------------------------------------------
// PyNeuronModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for neuron models
class PyNeuronModelBase : public PySnippet<NeuronModels::Base> 
{
    using Base = NeuronModels::Base;
public:
    virtual std::string getSimCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_sim_code", getSimCode); }
    virtual std::string getThresholdConditionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_threshold_condition_code", getThresholdConditionCode); }
    virtual std::string getResetCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_reset_code", getResetCode); }

    virtual std::vector<Models::Base::Var> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_vars", getVars); }
    virtual Models::Base::ParamValVec getAdditionalInputVars() const override { PYBIND11_OVERRIDE_NAME(Models::Base::ParamValVec, Base, "get_additional_input_vars", getAdditionalInputVars); }

    virtual bool isAutoRefractoryRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_auto_refractory_required", isAutoRefractoryRequired); }
};

//----------------------------------------------------------------------------
// PyPostsynapticModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for postsynaptic models
class PyPostsynapticModelBase : public PySnippet<PostsynapticModels::Base> 
{
    using Base = PostsynapticModels::Base;
public:
    virtual std::vector<Models::Base::Var> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_vars", getVars); }
    virtual VarRefVec getNeuronVarRefs() const override { PYBIND11_OVERRIDE_NAME(VarRefVec, Base, "get_neuron_var_refs", getNeuronVarRefs); }
    
    virtual std::string getDecayCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_decay_code", getDecayCode); }
    virtual std::string getApplyInputCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_apply_input_code", getApplyInputCode); }
};

//----------------------------------------------------------------------------
// PyWeightUpdateModelBase
//----------------------------------------------------------------------------
// 'Trampoline' class for weight update models
class PyWeightUpdateModelBase : public PySnippet<WeightUpdateModels::Base> 
{
    using Base = WeightUpdateModels::Base;
public:
    virtual std::string getSimCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_sim_code", getSimCode); }
    virtual std::string getEventCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_event_code", getEventCode); }
    virtual std::string getLearnPostCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_learn_post_code", getLearnPostCode); }
    virtual std::string getSynapseDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_synapse_dynamics_code", getSynapseDynamicsCode); }
    virtual std::string getEventThresholdConditionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_event_threshold_condition_code", getEventThresholdConditionCode); }
    virtual std::string getPreSpikeCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_pre_spike_code", getPreSpikeCode); }
    virtual std::string getPostSpikeCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_post_spike_code", getPostSpikeCode); }
    virtual std::string getPreDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_pre_dynamics_code", getPreDynamicsCode); }
    virtual std::string getPostDynamicsCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_post_dynamics_code", getPostDynamicsCode); }
    
    virtual std::vector<Models::Base::Var> getVars() const override{ PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_vars", getVars); }
    virtual std::vector<Models::Base::Var> getPreVars() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_pre_vars", getPreVars); }
    virtual std::vector<Models::Base::Var> getPostVars() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::Var>, Base, "get_post_vars", getPostVars); }
    
    virtual std::vector<Models::Base::VarRef> getPreNeuronVarRefs() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::VarRef> , Base, "get_pre_neuron_var_refs", getPreNeuronVarRefs); }
    virtual std::vector<Models::Base::VarRef> getPostNeuronVarRefs() const override { PYBIND11_OVERRIDE_NAME(std::vector<Models::Base::VarRef> , Base, "get_post_neuron_var_refs", getPostNeuronVarRefs); }
};

const CodeGenerator::ModelSpecMerged *generateCode(ModelSpecInternal &model, CodeGenerator::BackendBase &backend, 
                                                   const std::string &sharePathStr, const std::string &outputPathStr, bool forceRebuild)
{
    const filesystem::path outputPath(outputPathStr);

    // Create merged model and generate code
    auto *modelMerged = new CodeGenerator::ModelSpecMerged(backend, model);
    const auto output = CodeGenerator::generateAll(
        *modelMerged, backend, 
        filesystem::path(sharePathStr), outputPath, 
        forceRebuild);

#ifdef _WIN32
    // Create MSBuild project to compile and link all generated modules
    std::ofstream makefile((outputPath / "runner.vcxproj").str());
    CodeGenerator::generateMSBuild(makefile, model, backend, "", output);
#else
    // Create makefile to compile and link all generated modules
    std::ofstream makefile((outputPath / "Makefile").str());
    CodeGenerator::generateMakefile(makefile, backend, output);
#endif
    return modelMerged;
}

void initLogging(plog::Severity gennLevel, plog::Severity codeGeneratorLevel, 
                 plog::Severity runtimeLevel, plog::Severity transpilerLevel)
{
    auto *consoleAppender = new plog::ConsoleAppender<plog::TxtFormatter>;
    Logging::init(gennLevel, codeGeneratorLevel, runtimeLevel, transpilerLevel,
                  consoleAppender, consoleAppender, consoleAppender, consoleAppender);
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
        .value("INDIVIDUAL", SynapseMatrixWeight::INDIVIDUAL)
        .value("PROCEDURAL", SynapseMatrixWeight::PROCEDURAL)
        .value("KERNEL", SynapseMatrixWeight::KERNEL);

    pybind11::enum_<SynapseMatrixType>(m, "SynapseMatrixType")
        .value("DENSE", SynapseMatrixType::DENSE)
        .value("DENSE_PROCEDURALG", SynapseMatrixType::DENSE_PROCEDURALG)
        .value("BITMASK", SynapseMatrixType::BITMASK)
        .value("SPARSE", SynapseMatrixType::SPARSE)
        .value("PROCEDURAL", SynapseMatrixType::PROCEDURAL)
        .value("PROCEDURAL_KERNELG", SynapseMatrixType::PROCEDURAL_KERNELG)
        .value("TOEPLITZ", SynapseMatrixType::TOEPLITZ)

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
    
    //! Flags defining dimensions this variables has
    pybind11::enum_<VarAccessDim>(m, "VarAccessDim")
        .value("ELEMENT", VarAccessDim::ELEMENT)
        .value("BATCH", VarAccessDim::BATCH)
        
        .def("__and__", [](VarAccessDim a, VarAccessDim b){ return a & b; }, 
             pybind11::is_operator());
    
    //! Supported combinations of access mode and dimension for neuron variables
    pybind11::enum_<VarAccess>(m, "VarAccess")
        .value("READ_WRITE", VarAccess::READ_WRITE)
        .value("READ_ONLY", VarAccess::READ_ONLY)
        .value("READ_ONLY_DUPLICATE", VarAccess::READ_ONLY_DUPLICATE)
        .value("READ_ONLY_SHARED_NEURON", VarAccess::READ_ONLY_SHARED_NEURON);

    //! Supported combinations of access mode and dimension for custom update variables
    /*! The axes are defined 'subtractively' ie VarAccessDim::BATCH indicates that this axis should be removed */
    pybind11::enum_<CustomUpdateVarAccess>(m, "CustomUpdateVarAccess")
        .value("READ_WRITE", CustomUpdateVarAccess::READ_WRITE)
        .value("READ_ONLY", CustomUpdateVarAccess::READ_ONLY)
        .value("READ_ONLY_SHARED", CustomUpdateVarAccess::READ_ONLY_SHARED)
        .value("READ_ONLY_SHARED_NEURON", CustomUpdateVarAccess::READ_ONLY_SHARED_NEURON)
        .value("REDUCE_BATCH_SUM", CustomUpdateVarAccess::REDUCE_BATCH_SUM)
        .value("REDUCE_BATCH_MAX", CustomUpdateVarAccess::REDUCE_BATCH_MAX)
        .value("REDUCE_NEURON_SUM", CustomUpdateVarAccess::REDUCE_NEURON_SUM)
        .value("REDUCE_NEURON_MAX", CustomUpdateVarAccess::REDUCE_NEURON_MAX);

    //! Locations of variables
    pybind11::enum_<VarLocation>(m, "VarLocation")
        .value("HOST", VarLocation::HOST)
        .value("DEVICE", VarLocation::DEVICE)
        .value("ZERO_COPY", VarLocation::ZERO_COPY)
        .value("HOST_DEVICE", VarLocation::HOST_DEVICE)
        .value("HOST_DEVICE_ZERO_COPY", VarLocation::HOST_DEVICE_ZERO_COPY)
        
        .def("__and__", [](VarLocation a, VarLocation b){ return a & b; }, 
             pybind11::is_operator());
        
    //! Parallelism hints for synapse groups
    pybind11::enum_<SynapseGroup::SpanType>(m, "SpanType")
        .value("POSTSYNAPTIC", SynapseGroup::SpanType::POSTSYNAPTIC)
        .value("PRESYNAPTIC", SynapseGroup::SpanType::PRESYNAPTIC);

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("generate_code", &generateCode, pybind11::return_value_policy::take_ownership);
    m.def("init_logging", &initLogging);
    m.def("create_var_ref", pybind11::overload_cast<NeuronGroup*, const std::string&>(&createVarRef), pybind11::return_value_policy::move);
    m.def("create_var_ref", pybind11::overload_cast<CurrentSource*, const std::string&>(&createVarRef), pybind11::return_value_policy::move);
    m.def("create_var_ref", pybind11::overload_cast<CustomUpdate*, const std::string&>(&createVarRef), pybind11::return_value_policy::move);
    m.def("create_psm_var_ref", &createPSMVarRef, pybind11::return_value_policy::move);
    m.def("create_wu_pre_var_ref", &createWUPreVarRef, pybind11::return_value_policy::move);
    m.def("create_wu_post_var_ref", &createWUPostVarRef, pybind11::return_value_policy::move);
    m.def("create_pre_var_ref", &createPreVarRef, pybind11::return_value_policy::move);
    m.def("create_post_var_ref", &createPostVarRef, pybind11::return_value_policy::move);
    m.def("create_wu_var_ref", pybind11::overload_cast<SynapseGroup*, const std::string&, SynapseGroup*, const std::string&>(&createWUVarRef),
          "sg"_a, "var_name"_a, "transpose_sg"_a = nullptr, "transpose_var_name"_a = "", pybind11::return_value_policy::move);
    m.def("create_wu_var_ref", pybind11::overload_cast<CustomUpdateWU*, const std::string&>(&createWUVarRef), pybind11::return_value_policy::move);
    m.def("create_wu_var_ref", pybind11::overload_cast<CustomConnectivityUpdate*, const std::string&>(&createWUVarRef), pybind11::return_value_policy::move);
    m.def("create_egp_ref", pybind11::overload_cast<NeuronGroup*, const std::string&>(&createEGPRef), pybind11::return_value_policy::move);
    m.def("create_egp_ref", pybind11::overload_cast<CurrentSource*, const std::string&>(&createEGPRef), pybind11::return_value_policy::move);
    m.def("create_egp_ref", pybind11::overload_cast<CustomUpdate*, const std::string&>(&createEGPRef), pybind11::return_value_policy::move);
    m.def("create_egp_ref", pybind11::overload_cast<CustomUpdateWU*, const std::string&>(&createEGPRef), pybind11::return_value_policy::move);
    m.def("create_psm_egp_ref", pybind11::overload_cast<SynapseGroup*, const std::string&>(&createPSMEGPRef), pybind11::return_value_policy::move);
    m.def("create_wu_egp_ref", pybind11::overload_cast<SynapseGroup*, const std::string&>(&createWUEGPRef), pybind11::return_value_policy::move);
    m.def("get_var_access_dim", pybind11::overload_cast<VarAccess>(&getVarAccessDim));
    m.def("get_var_access_dim", pybind11::overload_cast<CustomUpdateVarAccess, VarAccessDim>(&getVarAccessDim));

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
        .def_property_readonly("type_context", &ModelSpecInternal::getTypeContext)
    
        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("add_current_source",  
             static_cast<CurrentSource* (ModelSpecInternal::*)(
                const std::string&, const CurrentSourceModels::Base*, const std::string&, 
                const ParamValues&, const VarValues&, const VarReferences&)>(&ModelSpecInternal::addCurrentSource),
            pybind11::return_value_policy::reference)
        .def("add_custom_connectivity_update",  
             static_cast<CustomConnectivityUpdate* (ModelSpecInternal::*)(
                const std::string&, const std::string&, const std::string&, const CustomConnectivityUpdateModels::Base*, 
                const ParamValues&, const VarValues&, const VarValues&, const VarValues&, 
                const WUVarReferences&, const VarReferences&, const VarReferences&)>(&ModelSpecInternal::addCustomConnectivityUpdate),
            pybind11::return_value_policy::reference)
        .def("add_custom_update",  
             static_cast<CustomUpdate* (ModelSpecInternal::*)(
                const std::string&, const std::string&, const CustomUpdateModels::Base*, 
                const ParamValues&, const VarValues&, const VarReferences&, const EGPReferences&)>(&ModelSpecInternal::addCustomUpdate),
            pybind11::return_value_policy::reference)
        .def("add_custom_update",  
             static_cast<CustomUpdateWU* (ModelSpecInternal::*)(
                const std::string&, const std::string&, const CustomUpdateModels::Base*, 
                const ParamValues&, const VarValues&, const WUVarReferences&, const EGPReferences&)>(&ModelSpecInternal::addCustomUpdate),
            pybind11::return_value_policy::reference)
        .def("add_neuron_population",  
             static_cast<NeuronGroup* (ModelSpecInternal::*)(
                const std::string&, unsigned int, const NeuronModels::Base*, 
                const ParamValues&, const VarValues&)>(&ModelSpecInternal::addNeuronPopulation), 
            pybind11::return_value_policy::reference)
        .def("add_synapse_population",
            static_cast<SynapseGroup* (ModelSpecInternal::*)(
                const std::string&, SynapseMatrixType, unsigned int, const std::string&, const std::string&,
                const WeightUpdateModels::Init&, const PostsynapticModels::Init&,
                const InitSparseConnectivitySnippet::Init&)>(&ModelSpecInternal::addSynapsePopulation),
            pybind11::return_value_policy::reference)
        .def("add_synapse_population",
            static_cast<SynapseGroup* (ModelSpecInternal::*)(
                const std::string&, SynapseMatrixType, unsigned int, const std::string&, const std::string&,
                const WeightUpdateModels::Init&, const PostsynapticModels::Init&,
                const InitToeplitzConnectivitySnippet::Init&)>(&ModelSpecInternal::addSynapsePopulation), 
            pybind11::return_value_policy::reference)

        .def("finalise", &ModelSpecInternal::finalise);

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
    // genn.CustomConnectivityUpdate
    //------------------------------------------------------------------------
    pybind11::class_<CustomConnectivityUpdate>(m, "CustomConnectivityUpdate", pybind11::dynamic_attr())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &CustomConnectivityUpdate::getName)
        .def_property_readonly("update_group_name", &CustomConnectivityUpdate::getUpdateGroupName)
        .def_property_readonly("model", &CustomConnectivityUpdate::getCustomConnectivityUpdateModel, pybind11::return_value_policy::reference)
        .def_property_readonly("params", &CustomConnectivityUpdate::getParams)
        
        .def_property_readonly("var_initialisers", &CustomConnectivityUpdate::getVarInitialisers)
        .def_property_readonly("pre_var_initialisers", &CustomConnectivityUpdate::getPreVarInitialisers)
        .def_property_readonly("post_var_initialisers", &CustomConnectivityUpdate::getPostVarInitialisers)

        .def_property_readonly("var_references", &CustomConnectivityUpdate::getVarReferences)
        .def_property_readonly("pre_var_references", &CustomConnectivityUpdate::getPreVarReferences)
        .def_property_readonly("post_var_references", &CustomConnectivityUpdate::getPostVarReferences)
        
        .def_property_readonly("synapse_group", 
            [](const CustomConnectivityUpdate &cu)
            {
                const auto &cuInternal = static_cast<const CustomConnectivityUpdateInternal&>(cu);
                return static_cast<const SynapseGroup*>(cuInternal.getSynapseGroup());
            })

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("set_var_location", &CustomConnectivityUpdate::setVarLocation)
        .def("set_pre_var_location", &CustomConnectivityUpdate::setPreVarLocation)
        .def("set_post_var_location", &CustomConnectivityUpdate::setPostVarLocation)
        .def("get_var_location", &CustomConnectivityUpdate::getVarLocation)
        .def("get_pre_var_location", &CustomConnectivityUpdate::getPreVarLocation)
        .def("get_post_var_location", &CustomConnectivityUpdate::getPostVarLocation);


    //------------------------------------------------------------------------
    // genn.CustomUpdateBase
    //------------------------------------------------------------------------
    pybind11::class_<CustomUpdateBase>(m, "CustomUpdateBase")
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &CustomUpdateBase::getName)
        .def_property_readonly("update_group_name", &CustomUpdateBase::getUpdateGroupName)
        .def_property_readonly("custom_update_model", &CustomUpdateBase::getCustomUpdateModel, pybind11::return_value_policy::reference)
        .def_property_readonly("params", &CustomUpdateBase::getParams)
        .def_property_readonly("var_initialisers", &CustomUpdateBase::getVarInitialisers)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        .def("set_var_location", &CustomUpdateBase::setVarLocation)
        .def("get_var_location", &CustomUpdateBase::getVarLocation);
    
    //------------------------------------------------------------------------
    // genn.CustomUpdate
    //------------------------------------------------------------------------
    pybind11::class_<CustomUpdate, CustomUpdateBase>(m, "CustomUpdate", pybind11::dynamic_attr())
        .def_property_readonly("size", &CustomUpdate::getSize)
        .def_property_readonly("var_references", &CustomUpdate::getVarReferences)

        // **NOTE** we use the 'publicist' pattern to expose some protected properties
        .def_property_readonly("_dims", &CustomUpdateInternal::getDims);

    //------------------------------------------------------------------------
    // genn.CustomUpdateWU
    //------------------------------------------------------------------------
    pybind11::class_<CustomUpdateWU, CustomUpdateBase>(m, "CustomUpdateWU", pybind11::dynamic_attr())
        .def_property_readonly("var_references", &CustomUpdateWU::getVarReferences)
        .def_property_readonly("synapse_group", 
            [](const CustomUpdateWU &cu)
            {
                const auto &cuInternal = static_cast<const CustomUpdateWUInternal&>(cu);
                return static_cast<const SynapseGroup*>(cuInternal.getSynapseGroup());
            })

        // **NOTE** we use the 'publicist' pattern to expose some protected properties
        .def_property_readonly("_dims", &CustomUpdateWUInternal::getDims);

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
        .def("get_var_location", pybind11::overload_cast<const std::string&>(&NeuronGroup::getVarLocation, pybind11::const_))
        // **NOTE** we use the 'publicist' pattern to expose some protected methods
        .def("_is_var_queue_required", &NeuronGroupInternal::isVarQueueRequired);
    
    //------------------------------------------------------------------------
    // genn.SynapseGroup
    //------------------------------------------------------------------------
    pybind11::class_<SynapseGroup>(m, "SynapseGroup", pybind11::dynamic_attr())
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        .def_property_readonly("name", &SynapseGroup::getName)
        .def_property_readonly("delay_steps", &SynapseGroupInternal::getDelaySteps)
        .def_property_readonly("ps_initialiser", &SynapseGroup::getPSInitialiser)
        .def_property_readonly("wu_initialiser", &SynapseGroup::getWUInitialiser)
        .def_property_readonly("kernel_size", &SynapseGroup::getKernelSize)
        .def_property_readonly("matrix_type", &SynapseGroup::getMatrixType)
        .def_property_readonly("sparse_connectivity_initialiser", &SynapseGroup::getConnectivityInitialiser)
        .def_property_readonly("toeplitz_connectivity_initialiser", &SynapseGroup::getToeplitzConnectivityInitialiser)
    
        .def_property("post_target_var", &SynapseGroup::getPostTargetVar, &SynapseGroup::setPostTargetVar)
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
    // genn.NumericValue
    //------------------------------------------------------------------------
    pybind11::class_<Type::NumericValue>(m, "NumericValue")
        .def(pybind11::init<double>())
        .def(pybind11::init<uint64_t>())
        .def(pybind11::init<int64_t>())
        .def(pybind11::init<int>())
        .def(pybind11::init<unsigned int>());

    //------------------------------------------------------------------------
    // genn.ResolvedType
    //------------------------------------------------------------------------
    pybind11::class_<Type::ResolvedType>(m, "ResolvedType")
        .def("__hash__",
             [](const Type::ResolvedType &a)
             {
                 // Calculate hash digest
                 boost::uuids::detail::sha1 shaHash;
                 Type::updateHash(a, shaHash);
                 const auto shaDigest = shaHash.get_digest();
                
                 // Return size-t worth of hash
                 size_t hash;
                 memcpy(&hash, &shaDigest[0], sizeof(size_t));
                 return hash;
             })
        .def("__copy__",
             [](const Type::ResolvedType &a) { return Type::ResolvedType(a); })
        .def("__eq__", 
             [](const Type::ResolvedType &a, Type::ResolvedType b) { return a == b; });

    //------------------------------------------------------------------------
    // genn.UnresolvedType
    //------------------------------------------------------------------------
    pybind11::class_<Type::UnresolvedType>(m, "UnresolvedType")
        .def(pybind11::init<const std::string&>())
        .def(pybind11::init<const Type::ResolvedType&>())
        .def("resolve", &Type::UnresolvedType::resolve)

        .def("__copy__",
             [](const Type::UnresolvedType &a) { return Type::UnresolvedType(a); })
        .def("__eq__", 
             [](const Type::UnresolvedType &a, Type::UnresolvedType b) { return a == b; });

    //------------------------------------------------------------------------
    // genn.Param
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::Param>(m, "Param")
        .def(pybind11::init<const std::string&, const std::string&>(),
             pybind11::arg("name"), pybind11::arg("type") = "scalar")
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Snippet::Base::Param::name)
        .def_readonly("type", &Snippet::Base::Param::type);

    //------------------------------------------------------------------------
    // genn.DerivedParam
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::DerivedParam>(m, "DerivedParam")
        .def(pybind11::init<const std::string&, Snippet::Base::DerivedParam::Func, const std::string&>(),
             pybind11::arg("name"), pybind11::arg("func"), pybind11::arg("type") = "scalar")
        .def(pybind11::init<const std::string&, Snippet::Base::DerivedParam::Func, const Type::ResolvedType&>())
        .def_readonly("name", &Snippet::Base::DerivedParam::name)
        .def_readonly("func", &Snippet::Base::DerivedParam::func)
        .def_readonly("type", &Snippet::Base::DerivedParam::type);
        
    //------------------------------------------------------------------------
    // genn.EGP
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::EGP>(m, "EGP")
        .def(pybind11::init<const std::string&, const std::string&>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Snippet::Base::EGP::name)
        .def_readonly("type", &Snippet::Base::EGP::type);
    
    //------------------------------------------------------------------------
    // genn.ParamVal
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base::ParamVal>(m, "ParamVal")
        .def(pybind11::init<const std::string&, const std::string&, double>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&, double>())
        .def_readonly("name", &Snippet::Base::ParamVal::name)
        .def_readonly("type", &Snippet::Base::ParamVal::type)
        .def_readonly("value", &Snippet::Base::ParamVal::value);

    //------------------------------------------------------------------------
    // genn.SnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<Snippet::Base, PySnippet<>>(m, "SnippetBase")
        .def("get_params", &Snippet::Base::getParams)
        .def("get_derived_params", &Snippet::Base::getDerivedParams)
        .def("get_extra_global_params", &Snippet::Base::getExtraGlobalParams);
    
    //------------------------------------------------------------------------
    // genn.InitSparseConnectivitySnippetBase
    //------------------------------------------------------------------------
    pybind11::class_<InitSparseConnectivitySnippet::Base, Snippet::Base, PyInitSparseConnectivitySnippetBase>(m, "InitSparseConnectivitySnippetBase")
        .def(pybind11::init<>())
        .def("get_row_build_code", &InitSparseConnectivitySnippet::Base::getRowBuildCode)
        .def("get_col_build_code", &InitSparseConnectivitySnippet::Base::getColBuildCode)
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
        .def("get_calc_max_row_length_func", &InitToeplitzConnectivitySnippet::Base::getCalcMaxRowLengthFunc)
        .def("get_calc_kernel_size_func", &InitToeplitzConnectivitySnippet::Base::getCalcKernelSizeFunc);

    //------------------------------------------------------------------------
    // genn.InitVarSnippetBase
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
        .def(pybind11::init<const std::string&, const Type::ResolvedType&, VarAccess>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Models::Base::Var::name)
        .def_readonly("type", &Models::Base::Var::type)
        .def_readonly("access", &Models::Base::Var::access);

    //------------------------------------------------------------------------
    // genn.CustomUpdateVar
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base::CustomUpdateVar>(m, "CustomUpdateVar")
        .def(pybind11::init<const std::string&, const std::string&, CustomUpdateVarAccess>())
        .def(pybind11::init<const std::string&, const std::string&>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&, CustomUpdateVarAccess>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Models::Base::CustomUpdateVar::name)
        .def_readonly("type", &Models::Base::CustomUpdateVar::type)
        .def_readonly("access", &Models::Base::CustomUpdateVar::access);

    //------------------------------------------------------------------------
    // genn.VarRef
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base::VarRef>(m, "VarRef")
        .def(pybind11::init<const std::string&, const std::string&, VarAccessMode>())
        .def(pybind11::init<const std::string&, const std::string&>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&, VarAccessMode>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Models::Base::VarRef::name)
        .def_readonly("type", &Models::Base::VarRef::type)
        .def_readonly("access", &Models::Base::VarRef::access);

    //------------------------------------------------------------------------
    // genn.EGPRef
    //------------------------------------------------------------------------
    pybind11::class_<Models::Base::EGPRef>(m, "EGPRef")
        .def(pybind11::init<const std::string&, const std::string&>())
        .def(pybind11::init<const std::string&, const Type::ResolvedType&>())
        .def_readonly("name", &Models::Base::EGPRef::name)
        .def_readonly("type", &Models::Base::EGPRef::type);

    //------------------------------------------------------------------------
    // genn.CurrentSourceModelBase
    //------------------------------------------------------------------------
    pybind11::class_<CurrentSourceModels::Base, Snippet::Base, PyCurrentSourceModelBase>(m, "CurrentSourceModelBase")
        .def(pybind11::init<>())

        .def("get_vars", &CurrentSourceModels::Base::getVars)
        .def("get_neuron_var_refs", &CurrentSourceModels::Base::getNeuronVarRefs)

        .def("get_injection_code", &CurrentSourceModels::Base::getInjectionCode);

    //------------------------------------------------------------------------
    // genn.CustomConnectivityUpdateModelBase
    //------------------------------------------------------------------------
    pybind11::class_<CustomConnectivityUpdateModels::Base, Snippet::Base, PyCustomConnectivityUpdateModelBase>(m, "CustomConnectivityUpdateModelBase")
        .def(pybind11::init<>())

        .def("get_vars", &CustomConnectivityUpdateModels::Base::getVars)
        .def("get_pre_vars", &CustomConnectivityUpdateModels::Base::getPreVars)
        .def("get_post_vars", &CustomConnectivityUpdateModels::Base::getPostVars)
        
        .def("get_var_refs", &CustomConnectivityUpdateModels::Base::getVarRefs)
        .def("get_pre_var_refs", &CustomConnectivityUpdateModels::Base::getPreVarRefs)
        .def("get_post_var_refs", &CustomConnectivityUpdateModels::Base::getPostVarRefs)
        
        .def("get_row_update_code", &CustomConnectivityUpdateModels::Base::getRowUpdateCode)
        .def("get_host_update_code", &CustomConnectivityUpdateModels::Base::getHostUpdateCode);
    
    //------------------------------------------------------------------------
    // genn.CustomUpdateModelBase
    //------------------------------------------------------------------------
    pybind11::class_<CustomUpdateModels::Base, Snippet::Base, PyCustomUpdateModelBase>(m, "CustomUpdateModelBase")
        .def(pybind11::init<>())
        
        .def("get_vars", &CustomUpdateModels::Base::getVars)
        .def("get_var_refs", &CustomUpdateModels::Base::getVarRefs)
        .def("get_extra_global_param_refs", &CustomUpdateModels::Base::getExtraGlobalParamRefs)

        .def("get_update_code", &CustomUpdateModels::Base::getUpdateCode);
    
    //------------------------------------------------------------------------
    // genn.NeuronModelBase
    //------------------------------------------------------------------------
    pybind11::class_<NeuronModels::Base, Snippet::Base, PyNeuronModelBase>(m, "NeuronModelBase")
        .def(pybind11::init<>())
        
        .def("get_vars", &NeuronModels::Base::getVars)

        .def("get_sim_code", &NeuronModels::Base::getSimCode)
        .def("get_threshold_condition_code", &NeuronModels::Base::getThresholdConditionCode)
        .def("get_reset_code", &NeuronModels::Base::getResetCode)
        .def("get_additional_input_vars", &NeuronModels::Base::getAdditionalInputVars)
        
        .def("is_auto_refractory_required", &NeuronModels::Base::isAutoRefractoryRequired);
        
    //------------------------------------------------------------------------
    // genn.PostsynapticModelBase
    //------------------------------------------------------------------------
    pybind11::class_<PostsynapticModels::Base, Snippet::Base, PyPostsynapticModelBase>(m, "PostsynapticModelBase")
        .def(pybind11::init<>())
        
        .def("get_vars", &PostsynapticModels::Base::getVars)
        .def("get_neuron_var_refs", &PostsynapticModels::Base::getNeuronVarRefs)
        
        .def("get_decay_code", &PostsynapticModels::Base::getDecayCode)
        .def("get_apply_input_code", &PostsynapticModels::Base::getApplyInputCode);
    
    //------------------------------------------------------------------------
    // genn.WeightUpdateModelBase
    //------------------------------------------------------------------------
    pybind11::class_<WeightUpdateModels::Base, Snippet::Base, PyWeightUpdateModelBase>(m, "WeightUpdateModelBase")
        .def(pybind11::init<>())
        
        .def("get_sim_code", &WeightUpdateModels::Base::getSimCode)
        .def("get_event_code", &WeightUpdateModels::Base::getEventCode)
        .def("get_learn_post_code", &WeightUpdateModels::Base::getLearnPostCode)
        .def("get_synapse_dynamics_code", &WeightUpdateModels::Base::getSynapseDynamicsCode)
        .def("get_event_threshold_condition_code", &WeightUpdateModels::Base::getEventThresholdConditionCode)
        .def("get_pre_spike_code", &WeightUpdateModels::Base::getPreSpikeCode)
        .def("get_post_spike_code", &WeightUpdateModels::Base::getPostSpikeCode)
        .def("get_pre_dynamics_code", &WeightUpdateModels::Base::getPreDynamicsCode)
        .def("get_post_dynamics_code", &WeightUpdateModels::Base::getPostDynamicsCode)
        .def("get_vars", &WeightUpdateModels::Base::getVars)
        .def("get_pre_vars", &WeightUpdateModels::Base::getPreVars)
        .def("get_post_vars", &WeightUpdateModels::Base::getPostVars)
        .def("get_pre_neuron_var_refs", &WeightUpdateModels::Base::getPreNeuronVarRefs)
        .def("get_post_neuron_var_refs", &WeightUpdateModels::Base::getPostNeuronVarRefs);

    //------------------------------------------------------------------------
    // genn.SparseConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitSparseConnectivitySnippet::Init>(m, "SparseConnectivityInit")
        .def(pybind11::init<const InitSparseConnectivitySnippet::Base*, const std::unordered_map<std::string, Type::NumericValue>&>())
        .def_property_readonly("snippet", &InitSparseConnectivitySnippet::Init::getSnippet, pybind11::return_value_policy::reference);
        
    //------------------------------------------------------------------------
    // genn.ToeplitzConnectivityInit
    //------------------------------------------------------------------------
    pybind11::class_<InitToeplitzConnectivitySnippet::Init>(m, "ToeplitzConnectivityInit")
        .def(pybind11::init<const InitToeplitzConnectivitySnippet::Base*, const std::unordered_map<std::string, Type::NumericValue>&>())
        .def_property_readonly("snippet", &InitToeplitzConnectivitySnippet::Init::getSnippet, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.VarInit
    //------------------------------------------------------------------------
    pybind11::class_<InitVarSnippet::Init>(m, "VarInit")
        .def(pybind11::init<const InitVarSnippet::Base*, const std::unordered_map<std::string, Type::NumericValue>&>())
        .def(pybind11::init<double>())
        .def_property_readonly("snippet", &InitVarSnippet::Init::getSnippet, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.WeightUpdateInit
    //------------------------------------------------------------------------
    pybind11::class_<WeightUpdateModels::Init>(m, "WeightUpdateInit")
        .def(pybind11::init<const WeightUpdateModels::Base*, const std::unordered_map<std::string, Type::NumericValue>&, const std::unordered_map<std::string, InitVarSnippet::Init>&, const std::unordered_map<std::string, InitVarSnippet::Init>&, const std::unordered_map<std::string, InitVarSnippet::Init>&, const std::unordered_map<std::string, Models::VarReference>&, const std::unordered_map<std::string, Models::VarReference>&>())
        .def_property_readonly("snippet", &WeightUpdateModels::Init::getSnippet, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.PostsynapticInit
    //------------------------------------------------------------------------
    pybind11::class_<PostsynapticModels::Init>(m, "PostsynapticInit")
        .def(pybind11::init<const PostsynapticModels::Base*, const std::unordered_map<std::string, Type::NumericValue>&, const std::unordered_map<std::string, InitVarSnippet::Init>&, const std::unordered_map<std::string, Models::VarReference>&>())
        .def_property_readonly("snippet", &PostsynapticModels::Init::getSnippet, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.VarReference
    //------------------------------------------------------------------------
    pybind11::class_<Models::VarReference>(m, "VarReference");

    //------------------------------------------------------------------------
    // genn.WUVarReference
    //------------------------------------------------------------------------
    pybind11::class_<Models::WUVarReference>(m, "WUVarReference")
        .def_property_readonly("synapse_group", &Models::WUVarReference::getSynapseGroup, pybind11::return_value_policy::reference);
    
    //------------------------------------------------------------------------
    // genn.EGPReference
    //------------------------------------------------------------------------
    pybind11::class_<Models::EGPReference>(m, "EGPReference");

    //------------------------------------------------------------------------
    // genn.PreferencesBase
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::PreferencesBase>(m, "PreferencesBase")
        .def_readwrite("optimize_code", &CodeGenerator::PreferencesBase::optimizeCode)
        .def_readwrite("debug_code", &CodeGenerator::PreferencesBase::debugCode)
        .def_readwrite("enable_bitmask_optimisations", &CodeGenerator::PreferencesBase::enableBitmaskOptimisations)
        .def_readwrite("genn_log_level", &CodeGenerator::PreferencesBase::gennLogLevel)
        .def_readwrite("code_generator_log_level", &CodeGenerator::PreferencesBase::codeGeneratorLogLevel)
        .def_readwrite("transpiler_log_level", &CodeGenerator::PreferencesBase::transpilerLogLevel)
        .def_readwrite("runtime_log_level", &CodeGenerator::PreferencesBase::runtimeLogLevel);

    //------------------------------------------------------------------------
    // genn.BackendBase
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::BackendBase>(m, "BackendBase");
    
    //------------------------------------------------------------------------
    // genn.BackendBase
    //------------------------------------------------------------------------
    pybind11::class_<CodeGenerator::ModelSpecMerged>(m, "ModelSpecMerged");
    
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
