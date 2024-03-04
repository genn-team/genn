// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initVarSnippet.h"

// Doc strings
#include "docStrings.h"

using namespace GeNN::InitVarSnippet;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define WRAP(NAME) m.def(#NAME, &getBaseInstance<NAME>,\
                         pybind11::return_value_policy::reference,\
                         DOC(InitVarSnippet, NAME))

namespace
{
template<typename T>
const Base *getBaseInstance()
{
    return static_cast<const Base*>(T::getInstance());
}
}

//----------------------------------------------------------------------------
// init_var_snippets
//----------------------------------------------------------------------------
PYBIND11_MODULE(init_var_snippets, m) 
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    WRAP(Uninitialised);
    WRAP(Constant);
    WRAP(Uniform);
    WRAP(Normal);
    WRAP(NormalClipped);
    WRAP(NormalClippedDelay);
    WRAP(Exponential);
    WRAP(Gamma);
    WRAP(Binomial);
}
