#pragma once

// Standard C++ include
#include <map>
#include <set>
#include <string>

//------------------------------------------------------------------------
// SpineMLGenerator
//------------------------------------------------------------------------
namespace SpineMLGenerator
{
// Parse SpineML experiment file to determine properties related to model
void parseExperiment(const std::string &experimentFilename,
                     std::map<std::string, std::set<std::string>> &externalInputs);
}