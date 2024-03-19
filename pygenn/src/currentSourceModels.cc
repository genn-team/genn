// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "currentSourceModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::CurrentSourceModels;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(CurrentSourceModels, NAME))

namespace
{
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
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    WRAP(DC);
    WRAP(GaussianNoise);
    WRAP(PoissonExp);
}
