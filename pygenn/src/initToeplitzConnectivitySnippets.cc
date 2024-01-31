// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initToeplitzConnectivitySnippet.h"

using namespace GeNN::InitToeplitzConnectivitySnippet;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// init_toeplitz_connectivity_snippets
//----------------------------------------------------------------------------
PYBIND11_MODULE(init_toeplitz_connectivity_snippets, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Uninitialised", &getBaseInstance<Uninitialised>, pybind11::return_value_policy::reference);
    m.def("Conv2D", &getBaseInstance<Conv2D>, pybind11::return_value_policy::reference);
    m.def("AvgPoolConv2D", &getBaseInstance<AvgPoolConv2D>, pybind11::return_value_policy::reference);
}
