// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "customUpdateModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::CustomUpdateModels;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(CustomUpdateModels, NAME))

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
    WRAP(Transpose);
}
