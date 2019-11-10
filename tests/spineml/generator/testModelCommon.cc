// Standard C++ include
#include <regex>
#include <sstream>
#include <unordered_map>

// Google test includes
#include "gtest/gtest.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// SpineML generator includes
#include "modelCommon.h"

using namespace SpineMLGenerator;

//------------------------------------------------------------------------
// FixedProbabilityConnection tests
//------------------------------------------------------------------------
TEST(Aliases, Recurrency) {
    // XML fragment specifying connector
    const char *aliasXML = R"(
        <?xml version="1.0"?>
        <ComponentClass name="calc all1" type="neuron_body">
        <Dynamics initial_regime="integration">
        <Alias name="ratio" dimension="?">
            <MathInline>a/(a_slow+r_f)</MathInline>
        </Alias>
        <Alias name="ratio_rev" dimension="?">
            <MathInline>a_rev/(a_slow_rev+r_f)</MathInline>
        </Alias>
        <Alias name="rhd_prog" dimension="?">
            <MathInline>(a_rev+a_slow_rev)</MathInline>
        </Alias>
        <Alias name="rhd_reg" dimension="?">
            <MathInline>(a+a_slow)</MathInline>
        </Alias>
        <Alias name="with_rhd_prog" dimension="?">
            <MathInline>(ratio-f*(rhd_diff)*(rhd_diff>0))*(ratio-f*(rhd_diff)*(rhd_diff>0)>0)</MathInline>
        </Alias>
        <Alias name="with_rhd_reg" dimension="?">
            <MathInline>(ratio_rev-f*(-rhd_diff)*(rhd_diff&lt;0))*(ratio_rev-f*(-rhd_diff)*(rhd_diff&lt;0)>0)</MathInline>
        </Alias>
        <Alias name="diff" dimension="?">
            <MathInline>with_rhd_prog_smooth-with_rhd_reg_smooth</MathInline>
        </Alias>
        <Alias name="rhd_diff" dimension="?">
            <MathInline>rhd_prog-rhd_reg</MathInline>
        </Alias>
        </Dynamics>
        </ComponentClass>)";

    // Initialise log channels, appending all to console
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Load XML and get root LL:Synapse element
    pugi::xml_document aliasDocument;
    aliasDocument.load_string(aliasXML);
    auto componentClass = aliasDocument.child("ComponentClass");

    // Parse aliases
    Aliases aliases(componentClass);

    // Generate aliases required for a state variable update
    std::stringstream os;
    std::string simCode = "scalar lwith_rhd_reg_smooth += DT * ((with_rhd_reg+0*(0.01+with_rhd_reg)-with_rhd_reg_smooth)/tau_rhd)";
    aliases.genAliases(os, {simCode});

    std::unordered_map<std::string, bool> aliasesDeclared = {
        {"ratio", false},
        {"ratio_rev", false},
        {"rhd_prog", false},
        {"rhd_reg", false},
        {"with_rhd_prog", false},
        {"with_rhd_reg", false},
        {"diff", false},
        {"rhd_diff", false}};

    // Loop through generated lines
    std::istringstream is(os.str());
    for (std::string line; std::getline(is, line); ) {
        // Skip comments and empty lines
        if(line.empty() || line.find_first_of("//") == 0) {
            continue;
        }

        // Find assignment operator
        const auto assignPos = line.find_first_of("=");
        ASSERT_NE(assignPos, std::string::npos);

        // Extract alias name
        const std::string aliasName = line.substr(13, assignPos - 14);

        // Loop through aliases
        bool varFound = false;
        for(auto &a : aliasesDeclared) {
            // If this is the alias this line is declaring
            if(a.first == aliasName) {
                // Mark it as declared
                a.second = true;

                // Set flag signifying our variable is found
                varFound = true;
            }
            else {
                // Build a regex to find alias name with at least one character that
                // can't be in a variable name on either side (or an end/beginning of string)
                // **NOTE** the suffix is non-capturing so two instances of variables separated by a single character are matched e.g. a*a
                const std::regex regex("(^|[^0-9a-zA-Z_])" + a.first + "(?=$|[^a-zA-Z0-9_])");

                // If this line references this alias, check that this alias has been declared
                if(std::regex_search(line, regex)) {
                    ASSERT_TRUE(a.second);
                }
            }
        }

        ASSERT_TRUE(varFound);
    }
}
