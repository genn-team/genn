// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "currentSourceModels.h"

// PyGeNN includes
#include "trampolines.h"

using namespace CurrentSourceModels;

namespace
{
//----------------------------------------------------------------------------
// PyCurrentSourceModelBase
//----------------------------------------------------------------------------
class PyCurrentSourceModelBase : public PyModel<Base> 
{
public:
    virtual std::string getInjectionCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_injection_code", getInjectionCode); }
};

template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// current_source_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(current_source_models, m) 
{
    pybind11::module_::import("genn_wrapper.genn");

    //------------------------------------------------------------------------
    // neuron_models.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Models::Base, PyCurrentSourceModelBase>(m, "Base")
        .def(pybind11::init<>())

        .def("get_sim_code", &Base::getInjectionCode);

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("DC", &getBaseInstance<DC>, pybind11::return_value_policy::reference);
    m.def("GaussianNoise", &getBaseInstance<GaussianNoise>, pybind11::return_value_policy::reference);
    m.def("PoissonExp", &getBaseInstance<PoissonExp>, pybind11::return_value_policy::reference);
}
