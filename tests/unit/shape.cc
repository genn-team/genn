// Google test includes
#include "gtest/gtest.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

using namespace CodeGenerator;
/* 

    //! Multiply-together size of all dimensions
    size_t getFlattenedSize() const;

    //--------------------------------------------------------------------------
    // Operators
    //--------------------------------------------------------------------------
    size_t operator [] (size_t i) const
    {
        return m_Dims.at(i);
    }

    //! Concatenate shapes
    Shape operator + (const Shape &other) const;*/

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Shape, Construct)
{
    const Shape fromScalar(size_t{12});
    const Shape fromVector(std::vector<size_t>{1, 2, 4});

    ASSERT_EQ(fromScalar.getNumDims(), 1);
    ASSERT_EQ(fromVector.getNumDims(), 3);

    ASSERT_EQ(fromScalar[0], 12);
    ASSERT_EQ(fromVector[0], 1);
    ASSERT_EQ(fromVector[0], 2);
    ASSERT_EQ(fromVector[0], 4);
}

TEST(Shape, RemoveInner)
{
    const Shape fromScalar(std::vector<size_t>{1, 2, 4, 8});
    const auto remove2Inner = fromScalar.removeInner(2);

    ASSERT_EQ(remove2Inner.getNumDims(), 2);
    ASSERT_EQ(remove2Inner[0], 1);
    ASSERT_EQ(remove2Inner[1], 2);
}

TEST(Shape, PadInner)
{
    const Shape fromScalar(std::vector<size_t>{1, 2, 4, 8});
    const auto pad1Inner = fromScalar.padInner(1);
    const auto pad3Inner = fromScalar.padInner(3);

    // Check dimensionality is unchanged
    ASSERT_EQ(pad1Inner.getNumDims(), 4);
    ASSERT_EQ(pad3Inner.getNumDims(), 4);

    // Check padding by one doesn't change shape
    ASSERT_EQ(pad1Inner[0], 1);
    ASSERT_EQ(pad1Inner[1], 2);
    ASSERT_EQ(pad1Inner[2], 4);
    ASSERT_EQ(pad1Inner[3], 8);

    // Check padding by three rounds inner dimension to 9
    ASSERT_EQ(pad3Inner[0], 1);
    ASSERT_EQ(pad3Inner[1], 2);
    ASSERT_EQ(pad3Inner[2], 4);
    ASSERT_EQ(pad3Inner[3], 9);
}

TEST(Shape, DivideInner)
{
    const Shape fromScalar(std::vector<size_t>{1, 2, 4, 8});
    const auto divide1Inner = fromScalar.padInner(1);
    const auto divide3Inner = fromScalar.padInner(3);

    // Check dimensionality is unchanged
    ASSERT_EQ(divide1Inner.getNumDims(), 4);
    ASSERT_EQ(divide3Inner.getNumDims(), 4);

    // Check dividing by one doesn't change shape
    ASSERT_EQ(divide1Inner[0], 1);
    ASSERT_EQ(divide1Inner[1], 2);
    ASSERT_EQ(divide1Inner[2], 4);
    ASSERT_EQ(divide1Inner[3], 8);

    // Check dividing by three rounds inner dimension to 3
    ASSERT_EQ(divide3Inner[0], 1);
    ASSERT_EQ(divide3Inner[1], 2);
    ASSERT_EQ(divide3Inner[2], 4);
    ASSERT_EQ(divide3Inner[3], 3);
}

TEST(Shape, SqueezeOuter)
{
    const Shape fromScalar(std::vector<size_t>{1, 2, 4, 8});
    const auto squeezeOuter = fromScalar.squeezeOuter();

    // Check dimensionality is reduced
    ASSERT_EQ(squeezeOuter.getNumDims(), 2);

    ASSERT_EQ(squeezeOuter[0], 8);
    ASSERT_EQ(squeezeOuter[1], 8);
}

TEST(Shape, GetFlattenedSize)
{
    const Shape fromScalar(std::vector<size_t>{1, 2, 4, 8});
    
    ASSERT_EQ(fromScalar.getFlattenedSize(), 16);
}

TEST(Shape, Concatenate)
{
    const auto concatenated = Shape(std::vector<size_t>{1, 2}) + Shape(std::vector<size_t>{4, 8, 16});
    
    ASSERT_EQ(concatenated.getNumDims(), 5);
    
    ASSERT_EQ(concatenated[0], 1);
    ASSERT_EQ(concatenated[1], 2);
    ASSERT_EQ(concatenated[2], 4);
    ASSERT_EQ(concatenated[3], 8);
    ASSERT_EQ(concatenated[4], 16);
}