// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "postsynapticModels.h"

// PyGeNN includes
#include "trampolines.h"

using namespace PostsynapticModels;

namespace
{
//----------------------------------------------------------------------------
// PyPostsynapticModelBase
//----------------------------------------------------------------------------
class PyPostsynapticModelBase : public PyModel<Base> 
{
public:
    virtual std::string getDecayCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_decay_code", getDecayCode); }
    virtual std::string getApplyInputCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_apply_input_code", getApplyInputCode); }
    virtual std::string getSupportCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_support_code", getSupportCode); }
};

template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// postsynaptic_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(postsynaptic_models, m) 
{
    pybind11::module_::import("genn_wrapper.genn");

    //------------------------------------------------------------------------
    // postsynaptic_models.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Models::Base, PyPostsynapticModelBase>(m, "Base")
        .def(pybind11::init<>())

        .def("get_decay_code", &Base::getDecayCode)
        .def("get_apply_input_code", &Base::getApplyInputCode)
        .def("get_support_code", &Base::getSupportCode);

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("ExpCurr", &getBaseInstance<ExpCurr>, pybind11::return_value_policy::reference);
    m.def("ExpCond", &getBaseInstance<ExpCond>, pybind11::return_value_policy::reference);
    m.def("DeltaCurr", &getBaseInstance<DeltaCurr>, pybind11::return_value_policy::reference);
}
