// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "postsynapticModels.h"

using namespace PostsynapticModels;

namespace
{
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
    pybind11::module_::import("pygenn.genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("ExpCurr", &getBaseInstance<ExpCurr>, pybind11::return_value_policy::reference);
    m.def("ExpCond", &getBaseInstance<ExpCond>, pybind11::return_value_policy::reference);
    m.def("DeltaCurr", &getBaseInstance<DeltaCurr>, pybind11::return_value_policy::reference);
}
