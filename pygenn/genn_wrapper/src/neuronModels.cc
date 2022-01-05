// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "neuronModels.h"

// PyGeNN includes
#include "trampolines.h"

using namespace NeuronModels;

namespace
{
//----------------------------------------------------------------------------
// PyNeuronBase
//----------------------------------------------------------------------------
class PyNeuronBase : public PyModel<Base> 
{
public:
    virtual std::string getSimCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_sim_code", getSimCode); }
    virtual std::string getThresholdConditionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_threshold_condition_code", getThresholdConditionCode); }
    virtual std::string getResetCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_reset_code", getResetCode); }
    virtual std::string getSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_support_code", getSupportCode); }

    virtual Models::Base::ParamValVec getAdditionalInputVars() const override { PYBIND11_OVERRIDE_NAME(Models::Base::ParamValVec, Base, "get_additional_input_vars", getAdditionalInputVars); }

    virtual bool isAutoRefractoryRequired() const override { PYBIND11_OVERRIDE_NAME(bool, Base, "is_auto_refractory_required", isAutoRefractoryRequired); }
};

template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// neuron_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(neuron_models, m) 
{
    pybind11::module_::import("genn_wrapper.genn");
    
    //------------------------------------------------------------------------
    // neuron_models.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Models::Base, PyNeuronBase>(m, "Base")
        .def(pybind11::init<>())
        
        .def("get_sim_code", &Base::getSimCode)
        .def("get_threshold_condition_code", &Base::getThresholdConditionCode)
        .def("get_reset_code", &Base::getResetCode)
        .def("get_support_code", &Base::getSupportCode)
        .def("get_additional_input_vars", &Base::getAdditionalInputVars)
        .def("is_auto_refractory_required", &Base::isAutoRefractoryRequired);
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    m.def("RulkovMap", &getBaseInstance<RulkovMap>, pybind11::return_value_policy::reference);
    m.def("Izhikevich", &getBaseInstance<Izhikevich>, pybind11::return_value_policy::reference);
    m.def("IzhikevichVariable", &getBaseInstance<IzhikevichVariable>, pybind11::return_value_policy::reference);
    m.def("LIF", &getBaseInstance<LIF>, pybind11::return_value_policy::reference);
    m.def("SpikeSource", &getBaseInstance<SpikeSource>, pybind11::return_value_policy::reference);
    m.def("SpikeSourceArray", &getBaseInstance<SpikeSourceArray>, pybind11::return_value_policy::reference);
    m.def("Poisson", &getBaseInstance<Poisson>, pybind11::return_value_policy::reference);
    m.def("PoissonNew", &getBaseInstance<PoissonNew>, pybind11::return_value_policy::reference);
    m.def("TraubMiles", &getBaseInstance<TraubMiles>, pybind11::return_value_policy::reference);
    m.def("TraubMilesFast", &getBaseInstance<TraubMilesFast>, pybind11::return_value_policy::reference);
    m.def("TraubMilesAlt", &getBaseInstance<TraubMilesAlt>, pybind11::return_value_policy::reference);
    m.def("TraubMilesNStep", &getBaseInstance<TraubMilesNStep>, pybind11::return_value_policy::reference);
}
