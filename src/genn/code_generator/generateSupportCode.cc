#include "code_generator/generateSupportCode.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateSupportCode(CodeStream &os, const ModelSpec &model)
{
    os << "#pragma once" << std::endl;
    os << std::endl;

    os << "// support code for neuron groups" << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        if (!n.second.getNeuronModel()->getSupportCode().empty()) {
            os << "namespace " << n.first << "_neuron";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(n.second.getNeuronModel()->getSupportCode(), model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;
    os << "// support code for synapse groups" << std::endl;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        if (!wu->getSimSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simCode";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getSimSupportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!wu->getLearnPostSupportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_simLearnPost";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getLearnPostSupportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
            os << "namespace " << s.first << "_weightupdate_synapseDynamics";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(wu->getSynapseDynamicsSuppportCode(), model.getPrecision()) << std::endl;
            }
        }
        if (!psm->getSupportCode().empty()) {
            os << "namespace " << s.first << "_postsyn";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(psm->getSupportCode(), model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;
}
