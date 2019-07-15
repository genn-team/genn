// Standard C++ includes
#include <bitset>
#include <map>

// Standard C includes
#include <cmath>

// Filesystem includes
#include "path.h"

// Google test includes
#include "gtest/gtest.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML simulator includes
#include "inputValue.h"

using namespace SpineMLSimulator;

//------------------------------------------------------------------------
// ConstantInput tests
//------------------------------------------------------------------------
TEST(ConstantTest, All) {
    // XML fragment specifying input
    const char *inputXML = "<ConstantInput value=\"0.5\"/>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);

    auto input = inputDocument.child("ConstantInput");

    // Parse XML and create input value
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           ASSERT_DOUBLE_EQ(v, 0.5);
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
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           ASSERT_DOUBLE_EQ(v, 0.5);
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
// ConstantArrayInput tests
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
        std::map<std::string, InputValue::External*> externalInputs;
        InputValue::create(1.0, 10, input, externalInputs);
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
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 4, input, externalInputs);

    // Update for first timestep
    std::bitset<4> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           if(i == 0) {
                               ASSERT_DOUBLE_EQ(v, 0.5);
                           }
                           else if(i == 1) {
                               ASSERT_DOUBLE_EQ(v, 0.2);
                           }
                           else if(i == 2) {
                               ASSERT_DOUBLE_EQ(v, 0.3);
                           }
                           else if(i == 3) {
                               ASSERT_DOUBLE_EQ(v, 0.35);
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
        std::map<std::string, InputValue::External*> externalInputs;
        InputValue::create(1.0, 10, input, externalInputs);
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
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Update for first timestep
    std::bitset<10> valuesUpdate;
    inputValue->update(1.0, 0,
                       [&valuesUpdate](unsigned int i, double v)
                       {
                           EXPECT_FALSE(valuesUpdate.test(i));
                           if(i == 0) {
                               ASSERT_DOUBLE_EQ(v, 0.5);
                           }
                           else if(i == 2) {
                               ASSERT_DOUBLE_EQ(v, 0.2);
                           }
                           else if(i == 3) {
                               ASSERT_DOUBLE_EQ(v, 0.3);
                           }
                           else if(i == 7) {
                               ASSERT_DOUBLE_EQ(v, 0.35);
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

//------------------------------------------------------------------------
// TimeVaryingInput tests
//------------------------------------------------------------------------
TEST(TimeVaryingInput, All) {
    // XML fragment specifying input
    const char *inputXML =
        "<TimeVaryingInput>\n"
        "   <TimePointValue time=\"0\" value=\"0.0\"/>\n"
        "   <TimePointValue time=\"10\" value=\"0.1\"/>\n"
        "   <TimePointValue time=\"20\" value=\"0.2\"/>\n"
        "   <TimePointValue time=\"30\" value=\"0.3\"/>\n"
        "   <TimePointValue time=\"40\" value=\"0.4\"/>\n"
        "</TimeVaryingInput>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("TimeVaryingInput");

    // Parse XML and create input value
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 0; t < 50; t++) {
        if(t % 10 == 0) {
            std::bitset<10> valuesUpdate;
            inputValue->update(1.0, t,
                               [t, &valuesUpdate](unsigned int i, double v)
                               {
                                   EXPECT_FALSE(valuesUpdate.test(i));
                                   ASSERT_DOUBLE_EQ(v, 0.01 * (double)t);
                                   valuesUpdate.set(i);
                               });

            // Check all values have been updated
            EXPECT_TRUE(valuesUpdate.all());
        }
        else {
            inputValue->update(1.0, t,
                               [](unsigned int, double)
                               {
                                   FAIL();
                               });
        }
    }
}
//------------------------------------------------------------------------
TEST(TimeVaryingInput, ShuffleAll) {
    // XML fragment specifying input
    const char *inputXML =
        "<TimeVaryingInput>\n"
        "   <TimePointValue time=\"20\" value=\"0.2\"/>\n"
        "   <TimePointValue time=\"0\" value=\"0.0\"/>\n"
        "   <TimePointValue time=\"10\" value=\"0.1\"/>\n"
        "   <TimePointValue time=\"40\" value=\"0.4\"/>\n"
        "   <TimePointValue time=\"30\" value=\"0.3\"/>\n"
        "</TimeVaryingInput>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("TimeVaryingInput");

    // Parse XML and create input value
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 0; t < 50; t++) {
        if(t % 10 == 0) {
            std::bitset<10> valuesUpdate;
            inputValue->update(1.0, t,
                               [t, &valuesUpdate](unsigned int i, double v)
                               {
                                   EXPECT_FALSE(valuesUpdate.test(i));
                                   ASSERT_DOUBLE_EQ(v, 0.01 * (double)t);
                                   valuesUpdate.set(i);
                               });

            // Check all values have been updated
            EXPECT_TRUE(valuesUpdate.all());
        }
        else {
            inputValue->update(1.0, t,
                               [](unsigned int, double)
                               {
                                   FAIL();
                               });
        }
    }
}

//------------------------------------------------------------------------
// TimeVaryingArrayInput tests
//------------------------------------------------------------------------
TEST(TimeVaryingArrayInput, All) {
    // XML fragment specifying input
    const char *inputXML =
        "<TimeVaryingArrayInput>\n"
        "   <TimePointArrayValue index=\"0\" array_time=\"0,10,20,30\" array_value=\"0.1,0.2,0.3,0.4\"/>\n"
        "   <TimePointArrayValue index=\"2\" array_time=\"2,12,22,32\" array_value=\"2.1,2.2,2.3,2.4\"/>\n"
        "   <TimePointArrayValue index=\"4\" array_time=\"4,14,24,34\" array_value=\"4.1,4.2,4.3,4.4\"/>\n"
        "   <TimePointArrayValue index=\"6\" array_time=\"6,16,26,36\" array_value=\"6.1,6.2,6.3,6.4\"/>\n"
        "</TimeVaryingArrayInput>\n";

    // Load XML and get root LL:Synapse element
    pugi::xml_document inputDocument;
    inputDocument.load_string(inputXML);
    auto input = inputDocument.child("TimeVaryingArrayInput");

    // Parse XML and create input value
    std::map<std::string, InputValue::External*> externalInputs;
    auto inputValue = InputValue::create(1.0, 10, input, externalInputs);

    // Check nothing gets updated in subsequent updates
    for(unsigned int t = 0; t < 50; t++) {
        inputValue->update(1.0, t,
                           [t](unsigned int i, double v)
                           {
                               auto p = div(t - i, 10);
                               if(p.rem == 0) {
                                   ASSERT_DOUBLE_EQ(v, (double)i + (0.1 * (double)(1 + p.quot)));
                               }
                               else {
                                   FAIL();
                               }
                           });

    }
}
