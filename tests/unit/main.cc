// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

int main(int argc, char **argv) {
    initGeNN();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}