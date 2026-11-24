// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "type.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Type, writeNumeric)
{
    // Float to float
    // Float to double
    // Float to int
    // Int to float
    // Int to int
    // Int to unsigned int

    ASSERT_EQ(Type::writeNumeric(12.8, Type::Int32), "12");
    ASSERT_EQ(Type::writeNumeric(-12.8, Type::Int32), "-12");
    ASSERT_EQ(Type::writeNumeric(13.1, Type::Uint32), "13u");
    ASSERT_EQ(Type::writeNumeric(0xFFFFFFFFu, Type::Uint32), "4294967295u");

    // Too small
    try {
        ASSERT_EQ(Type::writeNumeric(-12, Type::Uint32), "-12u");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }

    // Too big
    try {
        ASSERT_EQ(Type::writeNumeric(0xFFFFFFFFu, Type::Int32), "4294967295");
        FAIL();
    }
    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(Type, FFIType)
{
    ASSERT_EQ(Type::Int8.getFFIType(), &ffi_type_sint8);
    ASSERT_EQ(Type::Int16.getFFIType(), &ffi_type_sint16);
    ASSERT_EQ(Type::Int32.getFFIType(), &ffi_type_sint32);
    ASSERT_EQ(Type::Int64.getFFIType(), &ffi_type_sint64);

    ASSERT_EQ(Type::Uint8.getFFIType(), &ffi_type_uint8);
    ASSERT_EQ(Type::Uint16.getFFIType(), &ffi_type_uint16);
    ASSERT_EQ(Type::Uint32.getFFIType(), &ffi_type_uint32);
    ASSERT_EQ(Type::Uint64.getFFIType(), &ffi_type_uint64);

    ASSERT_EQ(Type::Float.getFFIType(), &ffi_type_float);
    ASSERT_EQ(Type::Double.getFFIType(), &ffi_type_double);
}
