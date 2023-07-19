// PyBind11 includes
#include <pybind11/pybind11.h>

// GeNN includes
#include "type.h"

using namespace GeNN::Type;

//----------------------------------------------------------------------------
// type
//----------------------------------------------------------------------------
PYBIND11_MODULE(type, m)
{
    //------------------------------------------------------------------------
    // Attributes
    //------------------------------------------------------------------------
    m.attr("Bool") = pybind11::cast(Bool);
}
/*Bool = CREATE_NUMERIC(bool, 0, "");
inline static const ResolvedType Int8 = CREATE_NUMERIC(int8_t, 10, "");
inline static const ResolvedType Int16 = CREATE_NUMERIC(int16_t, 20, "");
inline static const ResolvedType Int32 = CREATE_NUMERIC(int32_t, 30, "");
inline static const ResolvedType Int64 = CREATE_NUMERIC(int64_t, 40, "");

inline static const ResolvedType Uint8 = CREATE_NUMERIC(uint8_t, 10, "u");
inline static const ResolvedType Uint16 = CREATE_NUMERIC(uint16_t, 20, "u");
inline static const ResolvedType Uint32 = CREATE_NUMERIC(uint32_t, 30, "u");
inline static const ResolvedType Uint64 = CREATE_NUMERIC(uint64_t, 40, "u");

inline static const ResolvedType Float = CREATE_NUMERIC(float, 50, "f");
inline static const ResolvedType Double*/
