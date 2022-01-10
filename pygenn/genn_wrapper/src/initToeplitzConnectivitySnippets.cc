// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initToeplitzConnectivitySnippet.h"

// PyGeNN includes
#include "trampolines.h"

using namespace InitToeplitzConnectivitySnippet;

namespace
{
//----------------------------------------------------------------------------
// PyInitToeplitzConnectivitySnippetBase
//----------------------------------------------------------------------------
class PyInitToeplitzConnectivitySnippetBase : public PySnippet<Base> 
{
public:
    virtual std::string getDiagonalBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_diagonal_build_code", getDiagonalBuildCode); }
    virtual ParamValVec getDiagonalBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(ParamValVec, Base, "get_diagonal_build_state_vars", getDiagonalBuildStateVars); }
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_row_length_func", getCalcMaxRowLengthFunc); }
    virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override { PYBIND11_OVERRIDE_NAME(CalcKernelSizeFunc, Base, "get_calc_kernel_size_func", getCalcKernelSizeFunc); }
};

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
    pybind11::module_::import("genn_wrapper.genn");

    //------------------------------------------------------------------------
    // init_toeplitz_connectivity_snippets.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Snippet::Base, PyInitToeplitzConnectivitySnippetBase>(m, "Base")
        .def(pybind11::init<>())
        .def("get_diagonal_build_code", &Base::getDiagonalBuildCode)
        .def("get_diagonal_build_state_vars", &Base::getDiagonalBuildStateVars)
        .def("get_calc_max_row_length_func", &Base::getCalcMaxRowLengthFunc)
        .def("get_calc_kernel_size_func", &Base::getCalcKernelSizeFunc);
    
    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Uninitialised", &getBaseInstance<Uninitialised>, pybind11::return_value_policy::reference);
    m.def("Conv2D", &getBaseInstance<Conv2D>, pybind11::return_value_policy::reference);
    m.def("AvgPoolConv2D", &getBaseInstance<AvgPoolConv2D>, pybind11::return_value_policy::reference);
}
