// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initToeplitzConnectivitySnippet.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::InitToeplitzConnectivitySnippet;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(InitToeplitzConnectivitySnippet, NAME))

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// init_toeplitz_connectivity_snippets
//----------------------------------------------------------------------------
PYBIND11_MODULE(init_toeplitz_connectivity_snippets, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    WRAP(Uninitialised);
    WRAP(Conv2D);
    WRAP(AvgPoolConv2D);
}
