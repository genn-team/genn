// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "postsynapticModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::PostsynapticModels;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(PostsynapticModels, NAME))

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
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    WRAP(ExpCurr);
    WRAP(ExpCond);
    WRAP(DeltaCurr);
}
