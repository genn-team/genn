// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "customConnectivityUpdateModels.h"

using namespace GeNN::CustomConnectivityUpdateModels;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// custom_connectivity_update_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(custom_connectivity_update_models, m) 
{
    pybind11::module_::import("pygenn.genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
}
