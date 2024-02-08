// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "neuronModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::NeuronModels;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// neuron_models
//----------------------------------------------------------------------------
PYBIND11_MODULE(neuron_models, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("RulkovMap", &getBaseInstance<RulkovMap>, pybind11::return_value_policy::reference, DOC(NeuronModels, RulkovMap));
    m.def("Izhikevich", &getBaseInstance<Izhikevich>, pybind11::return_value_policy::reference, DOC(NeuronModels, Izhikevich));
    m.def("IzhikevichVariable", &getBaseInstance<IzhikevichVariable>, pybind11::return_value_policy::reference, DOC(NeuronModels, IzhikevichVariable));
    m.def("LIF", &getBaseInstance<LIF>, pybind11::return_value_policy::reference, DOC(NeuronModels, LIF));
    m.def("SpikeSource", &getBaseInstance<SpikeSource>, pybind11::return_value_policy::reference, DOC(NeuronModels, SpikeSource));
    m.def("SpikeSourceArray", &getBaseInstance<SpikeSourceArray>, pybind11::return_value_policy::reference, DOC(NeuronModels, SpikeSourceArray));
    m.def("Poisson", &getBaseInstance<Poisson>, pybind11::return_value_policy::reference, DOC(NeuronModels, Poisson));
    m.def("PoissonNew", &getBaseInstance<PoissonNew>, pybind11::return_value_policy::reference, DOC(NeuronModels, PoissonNew));
    m.def("TraubMiles", &getBaseInstance<TraubMiles>, pybind11::return_value_policy::reference, DOC(NeuronModels, TraubMiles));
    m.def("TraubMilesFast", &getBaseInstance<TraubMilesFast>, pybind11::return_value_policy::reference, DOC(NeuronModels, TraubMilesFast));
    m.def("TraubMilesAlt", &getBaseInstance<TraubMilesAlt>, pybind11::return_value_policy::reference, DOC(NeuronModels, TraubMilesAlt));
    m.def("TraubMilesNStep", &getBaseInstance<TraubMilesNStep>, pybind11::return_value_policy::reference, DOC(NeuronModels, TraubMilesNStep));
}
