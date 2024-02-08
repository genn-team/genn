// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "currentSourceModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::CurrentSourceModels;

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
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("DC", &getBaseInstance<DC>, pybind11::return_value_policy::reference, DOC(CurrentSourceModels, DC));
    m.def("GaussianNoise", &getBaseInstance<GaussianNoise>, pybind11::return_value_policy::reference, DOC(CurrentSourceModels, GaussianNoise));
    m.def("PoissonExp", &getBaseInstance<PoissonExp>, pybind11::return_value_policy::reference, DOC(CurrentSourceModels, PoissonExp));
}
