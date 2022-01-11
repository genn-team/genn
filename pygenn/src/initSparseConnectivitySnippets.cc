// PyBind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN includes
#include "initSparseConnectivitySnippet.h"

// PyGeNN includes
#include "trampolines.h"

using namespace InitSparseConnectivitySnippet;

namespace
{
//----------------------------------------------------------------------------
// PyInitSparseConnectivitySnippetBase
//----------------------------------------------------------------------------
class PyInitSparseConnectivitySnippetBase : public PySnippet<Base> 
{
public:
    virtual std::string getRowBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_row_build_code", getRowBuildCode); }
    virtual ParamValVec getRowBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(Snippet::Base::ParamValVec, Base, "get_row_build_state_vars", getRowBuildStateVars); }
    virtual std::string getColBuildCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_col_build_code", getColBuildCode); }
    virtual ParamValVec getColBuildStateVars() const override { PYBIND11_OVERRIDE_NAME(Snippet::Base::ParamValVec, Base, "get_col_build_state_vars", getColBuildStateVars); }
    virtual std::string getHostInitCode() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "get_host_init_code", getHostInitCode); }
    virtual CalcMaxLengthFunc getCalcMaxRowLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_row_length_func", getCalcMaxRowLengthFunc); }
    virtual CalcMaxLengthFunc getCalcMaxColLengthFunc() const override { PYBIND11_OVERRIDE_NAME(CalcMaxLengthFunc, Base, "get_calc_max_col_length_func", getCalcMaxColLengthFunc); }
    virtual CalcKernelSizeFunc getCalcKernelSizeFunc() const override { PYBIND11_OVERRIDE_NAME(CalcKernelSizeFunc, Base, "get_calc_kernel_size_func", getCalcKernelSizeFunc); }
};

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
    pybind11::module_::import("genn_wrapper.genn");

    //------------------------------------------------------------------------
    // init_sparse_connectivity_snippets.Base
    //------------------------------------------------------------------------
    pybind11::class_<Base, Snippet::Base, PyInitSparseConnectivitySnippetBase>(m, "Base")
        .def(pybind11::init<>())
        .def("get_row_build_code", &Base::getRowBuildCode)
        .def("get_row_build_state_vars", &Base::getRowBuildStateVars)
        .def("get_col_build_code", &Base::getColBuildCode)
        .def("get_col_build_state_vars", &Base::getColBuildStateVars)
        .def("get_host_init_code", &Base::getHostInitCode)
        .def("get_calc_max_row_length_func", &Base::getCalcMaxRowLengthFunc)
        .def("get_calc_max_col_length_func", &Base::getCalcMaxColLengthFunc)
        .def("get_calc_kernel_size_func", &Base::getCalcKernelSizeFunc);

    //------------------------------------------------------------------------
    // Free functions
    //------------------------------------------------------------------------
    // **THINK** with some cunning, standard macros could maybe populate
    // an array with instance pointers that we could loop over
    m.def("Uninitialised", &getBaseInstance<Uninitialised>, pybind11::return_value_policy::reference);
    m.def("OneToOne", &getBaseInstance<OneToOne>, pybind11::return_value_policy::reference);
    m.def("FixedProbability", &getBaseInstance<FixedProbability>, pybind11::return_value_policy::reference);
    m.def("FixedProbabilityNoAutapse", &getBaseInstance<FixedProbabilityNoAutapse>, pybind11::return_value_policy::reference);
    m.def("FixedNumberPostWithReplacement", &getBaseInstance<FixedNumberPostWithReplacement>, pybind11::return_value_policy::reference);
    m.def("FixedNumberTotalWithReplacement", &getBaseInstance<FixedNumberTotalWithReplacement>, pybind11::return_value_policy::reference);
    m.def("FixedNumberPreWithReplacement", &getBaseInstance<FixedNumberPreWithReplacement>, pybind11::return_value_policy::reference);
    m.def("Conv2D", &getBaseInstance<Conv2D>, pybind11::return_value_policy::reference);
}
