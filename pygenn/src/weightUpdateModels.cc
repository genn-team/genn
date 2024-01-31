// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "weightUpdateModels.h"

using namespace GeNN::WeightUpdateModels;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// weight_update_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(weight_update_models, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("StaticPulse", &getBaseInstance<StaticPulse>, pybind11::return_value_policy::reference);
    m.def("StaticPulseConstantWeight", &getBaseInstance<StaticPulseConstantWeight>, pybind11::return_value_policy::reference);
    m.def("StaticPulseDendriticDelay", &getBaseInstance<StaticPulseDendriticDelay>, pybind11::return_value_policy::reference);
    m.def("StaticGraded", &getBaseInstance<StaticGraded>, pybind11::return_value_policy::reference);
    m.def("PiecewiseSTDP", &getBaseInstance<PiecewiseSTDP>, pybind11::return_value_policy::reference);
}
