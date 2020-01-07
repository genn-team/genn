#include "code_generator/generateSupportCode.h"

// Standard C++ includes
#include <string>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSupportCode(CodeStream &os, const ModelSpecMerged &modelMerged)
{
    os << "#pragma once" << std::endl;
    os << std::endl;

    os << "// support code for neuron update groups" << std::endl;
    modelMerged.genNeuronUpdateGroupSupportCode(os);
    os << std::endl;

    os << "// support code for postsynaptic dynamics" << std::endl;
    modelMerged.genPostsynapticDynamicsSupportCode(os);
    os << std::endl;

    os << "// support code for presynaptic update" << std::endl;
    modelMerged.genPresynapticUpdateSupportCode(os);
    os << std::endl;

    os << "// support code for postsynaptic update groups" << std::endl;
    modelMerged.genPostsynapticUpdateSupportCode(os);
    os << std::endl;

    os << "// support code for synapse dynamics update groups" << std::endl;
    modelMerged.genSynapseDynamicsSupportCode(os);
    os << std::endl;
}
