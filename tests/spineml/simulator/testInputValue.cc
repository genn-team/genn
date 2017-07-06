// Standard C++ includes
#include <bitset>

// Filesystem includes
#include "filesystem/path.h"

// Google test includes
#include "gtest/gtest.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "sparseProjection.h"

// SpineML simulator includes
#include "inputValue.h"

using namespace SpineMLSimulator;

//------------------------------------------------------------------------
// Constant tests
//------------------------------------------------------------------------
TEST(ConstantTest, All) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantInput value=\"0.5\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantInput");

    // Parse XML and create input value
    auto inputValue = InputValue::create(1.0, 10, input);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           EXPECT_EQ(v, 0.5);
                           valuesUpdate.set(i);
                       });

    // Check all values have been updated
    EXPECT_TRUE(valuesUpdate.all());

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 1; t < 100; t++) {
        inputValue->update(1.0, t,
                           [](unsigned int, double)
                           {
                               FAIL();
                           });
    }
}
//------------------------------------------------------------------------
TEST(ConstantTest, Indices) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantInput value=\"0.5\" target_indices=\"0,2,3\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantInput");

    // Parse XML and create input value
    auto inputValue = InputValue::create(1.0, 10, input);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           EXPECT_EQ(v, 0.5);
                           valuesUpdate.set(i);
                       });

    // Check that only 3 values have been set and these are correct
    EXPECT_EQ(valuesUpdate.count(), 3);
    EXPECT_TRUE(valuesUpdate.test(0) && valuesUpdate.test(2) && valuesUpdate.test(3));

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 1; t < 100; t++) {
        inputValue->update(1.0, t,
                           [](unsigned int, double)
                           {
                               FAIL();
                           });
    }
}

//------------------------------------------------------------------------
// ConstantArray tests
//------------------------------------------------------------------------
TEST(ConstantArrayTest, AllArraySizeDeath) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantArrayInput array_value=\"0.5,0.2,0.3,0.35\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantArrayInput");


    // Parse XML
    try
    {
        InputValue::create(1.0, 10, input);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }

}
//------------------------------------------------------------------------
TEST(ConstantArrayTest, All) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantArrayInput array_value=\"0.5,0.2,0.3,0.35\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantArrayInput");

    // Parse XML and create input value
    auto inputValue = InputValue::create(1.0, 4, input);

    // Update for first timestep
    std::bitset<4> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           if(i == 0) {
                               EXPECT_EQ(v, 0.5);
                           }
                           else if(i == 1) {
                               EXPECT_EQ(v, 0.2);
                           }
                           else if(i == 2) {
                               EXPECT_EQ(v, 0.3);
                           }
                           else if(i == 3) {
                               EXPECT_EQ(v, 0.35);
                           }
                           valuesUpdate.set(i);
                       });

    // Check all values have been updated
    EXPECT_TRUE(valuesUpdate.all());

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 1; t < 100; t++) {
        inputValue->update(1.0, t,
                           [](unsigned int, double)
                           {
                               FAIL();
                           });
    }
}
//------------------------------------------------------------------------
TEST(ConstantArrayTest, IndicesSizeDeath) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantArrayInput array_value=\"0.5,0.2,0.3,0.35\" target_indices=\"0,2,3\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantArrayInput");


    // Parse XML
    try
    {
        InputValue::create(1.0, 10, input);
        FAIL();
    }
    catch(const std::runtime_error &)
    {
    }

}
//------------------------------------------------------------------------
TEST(ConstantArrayTest, Indices) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantArrayInput array_value=\"0.5,0.2,0.3,0.35\" target_indices=\"0,2,3,7\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("ConstantArrayInput");

    // Parse XML and create input value
    auto inputValue = InputValue::create(1.0, 10, input);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           if(i == 0) {
                               EXPECT_EQ(v, 0.5);
                           }
                           else if(i == 2) {
                               EXPECT_EQ(v, 0.2);
                           }
                           else if(i == 3) {
                               EXPECT_EQ(v, 0.3);
                           }
                           else if(i == 7) {
                               EXPECT_EQ(v, 0.35);
                           }
                           valuesUpdate.set(i);
                       });

    // Check that only 4 values have been set
    EXPECT_EQ(valuesUpdate.count(), 4);

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 1; t < 100; t++) {
        inputValue->update(1.0, t,
                           [](unsigned int, double)
                           {
                               FAIL();
                           });
    }
}