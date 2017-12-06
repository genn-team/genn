#include "experimentParser.h"

// Standard C++ includes
#include <iostream>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineMLCommon includes
#include "spineMLUtils.h"

using namespace SpineMLCommon;

//------------------------------------------------------------------------
// SpineMLGenerator
//------------------------------------------------------------------------
void SpineMLGenerator::parseExperiment(const std::string &experimentFilename,
                                       std::map<std::string, std::set<std::string>> &externalInputs)
{
    std::cout << "Parsing experiment '" << experimentFilename << "'" << std::endl;

    // Load experiment document
    pugi::xml_document experimentDoc;
    auto experimentResult = experimentDoc.load_file(experimentFilename.c_str());
    if(!experimentResult) {
        throw std::runtime_error("Unable to load experiment XML file:" + experimentFilename + ", error:" + experimentResult.description());
    }

    // Get SpineML root
    auto experimentSpineML = experimentDoc.child("SpineML");
    if(!experimentSpineML) {
        throw std::runtime_error("XML file:" + experimentFilename+ " is not a SpineML experiment - it has no root SpineML node");
    }

    // Get experiment node
    auto experiment = experimentSpineML.child("Experiment");
    if(!experiment) {
        throw std::runtime_error("No 'Experiment' node found");
    }

    // Loop through inputs
    for(auto input : experiment.select_nodes(SpineMLUtils::xPathNodeHasSuffix("Input").c_str())) {
        // Read target and port
        const std::string target = SpineMLUtils::getSafeName(input.node().attribute("target").value());
        const std::string port = input.node().attribute("port").value();

        // Add to map
        std::cout << "\tInput targetting '" << target << "':'" << port << "'" << std::endl;
        if(!externalInputs[target].emplace(port).second) {
            throw std::runtime_error("Multiple inputs targetting " + target + ":" + port);
        }
    }

    // **TODO** lesions
}