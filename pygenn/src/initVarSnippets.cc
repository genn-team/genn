// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initVarSnippet.h"

using namespace GeNN::InitVarSnippet;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// init_var_snippets
//----------------------------------------------------------------------------
PYBIND11_MODULE(init_var_snippets, m) 
{
    pybind11::module_::import("pygenn.genn");

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
	m.def("HalfNormal", &getBaseInstance<HalfNormal>, pybind11::return_value_policy::reference);
    m.def("NormalClipped", &getBaseInstance<NormalClipped>, pybind11::return_value_policy::reference);
    m.def("NormalClippedDelay", &getBaseInstance<NormalClippedDelay>, pybind11::return_value_policy::reference);
    m.def("Exponential", &getBaseInstance<Exponential>, pybind11::return_value_policy::reference);
    m.def("Gamma", &getBaseInstance<Gamma>, pybind11::return_value_policy::reference);
    m.def("Binomial", &getBaseInstance<Binomial>, pybind11::return_value_policy::reference);
}
