// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "neuronModels.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::NeuronModels;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(NeuronModels, NAME))

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
    WRAP(RulkovMap);
    WRAP(Izhikevich);
    WRAP(IzhikevichVariable);
    WRAP(LIF);
    WRAP(SpikeSourceArray);
    WRAP(Poisson);
    WRAP(TraubMiles);
}
