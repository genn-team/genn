#include "code_generator/generateSupportCode.h"

// Standard C++ includes
#include <fstream>
#include <string>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSupportCode(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged)
{
    std::ofstream supportCodeStream((outputPath / "supportCode.h").str());
    CodeStream supportCode(supportCodeStream);

    supportCode << "#pragma once" << std::endl;
    supportCode << std::endl;

    supportCode << "// support code for neuron update groups" << std::endl;
    modelMerged.genNeuronUpdateGroupSupportCode(supportCode);
    supportCode << std::endl;

    supportCode << "// support code for postsynaptic dynamics" << std::endl;
    modelMerged.genPostsynapticDynamicsSupportCode(supportCode);
    supportCode << std::endl;

    supportCode << "// support code for presynaptic update" << std::endl;
    modelMerged.genPresynapticUpdateSupportCode(supportCode);
    supportCode << std::endl;

    supportCode << "// support code for postsynaptic update groups" << std::endl;
    modelMerged.genPostsynapticUpdateSupportCode(supportCode);
    supportCode << std::endl;

    supportCode << "// support code for synapse dynamics update groups" << std::endl;
    modelMerged.genSynapseDynamicsSupportCode(supportCode);
    supportCode << std::endl;
}
