// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initSparseConnectivitySnippet.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::InitSparseConnectivitySnippet;

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// init_sparse_connectivity_snippets
//----------------------------------------------------------------------------
PYBIND11_MODULE(init_sparse_connectivity_snippets, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Uninitialised", &getBaseInstance<Uninitialised>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, Uninitialised));
    m.def("OneToOne", &getBaseInstance<OneToOne>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, OneToOne));
    m.def("FixedProbability", &getBaseInstance<FixedProbability>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, FixedProbability));
    m.def("FixedProbabilityNoAutapse", &getBaseInstance<FixedProbabilityNoAutapse>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, FixedProbabilityNoAutapse));
    m.def("FixedNumberPostWithReplacement", &getBaseInstance<FixedNumberPostWithReplacement>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, FixedNumberPostWithReplacement));
    m.def("FixedNumberTotalWithReplacement", &getBaseInstance<FixedNumberTotalWithReplacement>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, FixedNumberTotalWithReplacement));
    m.def("FixedNumberPreWithReplacement", &getBaseInstance<FixedNumberPreWithReplacement>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, FixedNumberPreWithReplacement));
    m.def("Conv2D", &getBaseInstance<Conv2D>, pybind11::return_value_policy::reference, DOC(InitSparseConnectivitySnippet, Conv2D));
}
