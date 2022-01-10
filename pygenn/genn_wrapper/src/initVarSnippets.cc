// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initVarSnippet.h"

// PyGeNN includes
#include "trampolines.h"

using namespace InitVarSnippet;

namespace
{
//----------------------------------------------------------------------------
// PyInitVarSnippetBase
//----------------------------------------------------------------------------
class PyInitVarSnippetBase : public PySnippet<Base> 
{
public:
    virtual std::string getCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_code", getCode); }
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
PYBIND11_MODULE(init_var_snippets, m) 
{
    pybind11::module_::import("genn_wrapper.genn");

    //------------------------------------------------------------------------
    // neuron_models.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Snippet::Base, PyInitVarSnippetBase>(m, "Base")
        .def(pybind11::init<>())

        .def("get_code", &Base::getCode);

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Uninitialised", &getBaseInstance<Uninitialised>, pybind11::return_value_policy::reference);
    m.def("Constant", &getBaseInstance<Constant>, pybind11::return_value_policy::reference);
    m.def("Kernel", &getBaseInstance<Kernel>, pybind11::return_value_policy::reference);
    m.def("Uniform", &getBaseInstance<Uniform>, pybind11::return_value_policy::reference);
    m.def("Normal", &getBaseInstance<Normal>, pybind11::return_value_policy::reference);
    m.def("NormalClipped", &getBaseInstance<NormalClipped>, pybind11::return_value_policy::reference);
    m.def("NormalClippedDelay", &getBaseInstance<NormalClippedDelay>, pybind11::return_value_policy::reference);
    m.def("Exponential", &getBaseInstance<Exponential>, pybind11::return_value_policy::reference);
    m.def("Gamma", &getBaseInstance<Gamma>, pybind11::return_value_policy::reference);
}
