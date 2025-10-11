// PyBind11 includes
#include <pybind11/pybind11.h>

// GeNN includes
#include "type.h"

using namespace GeNN::Type;

//----------------------------------------------------------------------------
// types
//----------------------------------------------------------------------------
PYBIND11_MODULE(types, m)
{
    pybind11::module_::import("pygenn._genn");

    //------------------------------------------------------------------------
    // Attributes
    //------------------------------------------------------------------------
    m.attr("Bool") = pybind11::cast(Bool);

    m.attr("Int8") = pybind11::cast(Int8);
    m.attr("Int16") = pybind11::cast(Int16);
    m.attr("Int32") = pybind11::cast(Int32);
    m.attr("Int64") = pybind11::cast(Int64);
    
    m.attr("Uint8") = pybind11::cast(Uint8);
    m.attr("Uint16") = pybind11::cast(Uint16);
    m.attr("Uint32") = pybind11::cast(Uint32);
    m.attr("Uint64") = pybind11::cast(Uint64);

    m.attr("Half") = pybind11::cast(Half);
    m.attr("Float") = pybind11::cast(Float);
    m.attr("Double") = pybind11::cast(Double);
}