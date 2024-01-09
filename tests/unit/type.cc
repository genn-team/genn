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