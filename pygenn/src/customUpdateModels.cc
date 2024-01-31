// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "customUpdateModels.h"

using namespace GeNN::CustomUpdateModels;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// custom_update_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(custom_update_models, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Transpose", &getBaseInstance<Transpose>, pybind11::return_value_policy::reference);
}
