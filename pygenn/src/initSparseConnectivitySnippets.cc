// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initSparseConnectivitySnippet.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::InitSparseConnectivitySnippet;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(InitSparseConnectivitySnippet, NAME))

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
    WRAP(Uninitialised);
    WRAP(OneToOne);
    WRAP(FixedProbability);
    WRAP(FixedProbabilityNoAutapse);
    WRAP(FixedNumberPostWithReplacement);
    WRAP(FixedNumberTotalWithReplacement);
    WRAP(FixedNumberPreWithReplacement);
    WRAP(Conv2D);
}
