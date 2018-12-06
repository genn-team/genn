#pragma once

// Standard C++ includes
#include <string>

// Forward declarations
class NeuronGroup;
class SynapseGroup;

namespace NewModels
{
    class VarInit;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
// **TODO** move all of these SOMWHERE else. Into NeuronGroup and SynapseGroup?
namespace CodeGenerator
{
void applyNeuronModelSubstitutions(std::string &code, const NeuronGroup &ng,
                                   const std::string &varPrefix, const std::string &varSuffix = "", const std::string &varExt = "");
void applyPostsynapticModelSubstitutions(std::string &code, const SynapseGroup &sg, const std::string &varPrefix);

void applyWeightUpdateModelSubstitutions(std::string &code, const SynapseGroup &sg,
                                         const std::string &varPrefix, const std::string &varSuffix = "", const std::string &varExt = "");

void applyVarInitSnippetSubstitutions(std::string &code, const NewModels::VarInit &varInit);

void applySparsConnectInitSnippetSubstitutions(std::string &code, const SynapseGroup &sg);
}   // namespace CodeGenerator