// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "weightUpdateModels.h"

// PyGeNN includes
#include "trampolines.h"

using namespace WeightUpdateModels;

namespace
{
//----------------------------------------------------------------------------
// PyWeightUpdateModelBase
//----------------------------------------------------------------------------
class PyWeightUpdateModelBase : public PyModel<Base> 
{
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

template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// weight_update_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(weight_update_models, m) 
{
    pybind11::module_::import("pygenn.genn");

    //------------------------------------------------------------------------
    // neuron_models.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Models::Base, PyWeightUpdateModelBase>(m, "Base")
        .def(pybind11::init<>())
        
        .def("get_sim_code", &Base::getSimCode)
        .def("get_event_code", &Base::getEventCode)
        .def("get_learn_post_code", &Base::getLearnPostCode)
        .def("get_synapse_dynamics_code", &Base::getSynapseDynamicsCode)
        .def("get_event_threshold_condition_code", &Base::getEventThresholdConditionCode)
        .def("get_sim_support_cde", &Base::getSimSupportCode)
        .def("get_learn_post_support_code", &Base::getLearnPostSupportCode)
        .def("get_synapse_dynamics_support_code", &Base::getSynapseDynamicsSuppportCode)
        .def("get_pre_spike_code", &Base::getPreSpikeCode)
        .def("get_post_spike_code", &Base::getPostSpikeCode)
        .def("get_pre_dynamics_code", &Base::getPreDynamicsCode)
        .def("get_post_dynamics_code", &Base::getPostDynamicsCode)
        .def("get_pre_vars", &Base::getPreVars)
        .def("get_post_vars", &Base::getPostVars)
        .def("is_pre_spike_time_required", &Base::isPreSpikeTimeRequired)
        .def("is_post_spike_time_required", &Base::isPostSpikeTimeRequired)
        .def("is_pre_spike_event_time_required", &Base::isPreSpikeEventTimeRequired)
        .def("is_prev_pre_spike_time_required", &Base::isPrevPreSpikeTimeRequired)
        .def("is_prev_post_spike_time_required", &Base::isPrevPostSpikeTimeRequired)
        .def("is_prev_pre_spike_event_time_required", &Base::isPrevPreSpikeEventTimeRequired);


    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("StaticPulse", &getBaseInstance<StaticPulse>, pybind11::return_value_policy::reference);
    m.def("StaticPulseDendriticDelay", &getBaseInstance<StaticPulseDendriticDelay>, pybind11::return_value_policy::reference);
    m.def("StaticGraded", &getBaseInstance<StaticGraded>, pybind11::return_value_policy::reference);
    m.def("PiecewiseSTDP", &getBaseInstance<PiecewiseSTDP>, pybind11::return_value_policy::reference);
}
