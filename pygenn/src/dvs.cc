// PyBind11 includes
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// GeNN runtime includes
#include "runtime/runtime.h"

// DVS includes
#include "dvs.h"

// Doc strings
#include "dvsDocStrings.h"

using namespace GeNN::DVS;

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define DOC_DVS(...) DOC(DVS, __VA_ARGS__)
#define WRAP_NS_ENUM(ENUM, NS, VAL) .value(#VAL, NS::ENUM::VAL, DOC_DVS(NS, ENUM, VAL))
#define WRAP_METHOD(NAME, CLASS, METH) .def(NAME, &CLASS::METH, DOC_DVS(CLASS, METH))
//#define WRAP_STATIC_METHOD(NAME, CLASS, METH) .def_static(NAME, &CLASS::METH, DOC_DVS(CLASS, METH))
#define WRAP_PROPERTY_RO(NAME, CLASS, METH_STEM) .def_property_readonly(NAME, &CLASS::get##METH_STEM, DOC_DVS(CLASS, m_##METH_STEM))

//----------------------------------------------------------------------------
// dvs
//----------------------------------------------------------------------------
PYBIND11_MODULE(dvs, m) 
{
    pybind11::module_::import("pygenn.runtime");

    //------------------------------------------------------------------------
    // dvs.Polarity
    //------------------------------------------------------------------------
    pybind11::enum_<DVS::Polarity>(m, "Polarity")
        WRAP_NS_ENUM(Polarity, DVS, ON_ONLY)
        WRAP_NS_ENUM(Polarity, DVS, OFF_ONLY)
        WRAP_NS_ENUM(Polarity, DVS, SEPERATE)
        WRAP_NS_ENUM(Polarity, DVS, MERGE);

    //------------------------------------------------------------------------
    // dvs.CropRect
    //------------------------------------------------------------------------
    /*pybind11::class_<DVS::CropRect>(m, "CropRect")
        //.def(pybind11::init<const std::string&, Snippet::Base::DerivedParam::Func, const std::string&>(),
        //     pybind11::arg("name"), pybind11::arg("func"), pybind11::arg("type") = "scalar")
        
        WRAP_NS_ATTR("left", DVS, CropRect, left)
        WRAP_NS_ATTR("top", DVS, CropRect, top)
        WRAP_NS_ATTR("right", DVS, CropRect, right)
        WRAP_NS_ATTR("bottom", DVS, CropRect, bottom);*/

    //------------------------------------------------------------------------
    // DVS.DVS
    //------------------------------------------------------------------------
    pybind11::class_<DVS>(m, "DVS")
        //--------------------------------------------------------------------
        // Properties
        //--------------------------------------------------------------------
        WRAP_PROPERTY_RO("width", DVS, Width)
        WRAP_PROPERTY_RO("height", DVS, Height)

        //--------------------------------------------------------------------
        // Methods
        //--------------------------------------------------------------------
        WRAP_METHOD("start", DVS, start)
        WRAP_METHOD("stop", DVS, stop)
        WRAP_METHOD("read_events", DVS, readEvents)

        //--------------------------------------------------------------------
        // Static methods
        //--------------------------------------------------------------------
        .def_static("create_davis", &DVS::create<libcaer::devices::davis>, pybind11::return_value_policy::reference)
        .def_static("create_dvs128", &DVS::create<libcaer::devices::dvs128>, pybind11::return_value_policy::reference)
        .def_static("create_dvxplorer", &DVS::create<libcaer::devices::dvXplorer>, pybind11::return_value_policy::reference);
}
