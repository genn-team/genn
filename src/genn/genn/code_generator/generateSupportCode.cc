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
    const ModelSpecInternal &model = modelMerged.getModel();
    for(const auto &n : modelMerged.getMergedNeuronUpdateGroups()) {
        const std::string supportCode = n.getArchetype().getNeuronModel()->getSupportCode();
        if (!supportCode.empty()) {
            os << "namespace merged" << n.getIndex() << "_neuron";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(supportCode, model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;

    os << "// support code for presynaptic update groups" << std::endl;
    for(const auto &s : modelMerged.getMergedPresynapticUpdateGroups()) {
        const std::string supportCode = s.getArchetype().getWUModel()->getSimSupportCode();
        if (!supportCode.empty()) {
            os << "namespace merged" << s.getIndex() << "_weightupdate_simCode";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(supportCode, model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;

    os << "// support code for postsynaptic update groups" << std::endl;
    for(const auto &s : modelMerged.getMergedPostsynapticUpdateGroups()) {
        const std::string supportCode = s.getArchetype().getWUModel()->getLearnPostSupportCode();
        if (!supportCode.empty()) {
            os << "namespace merged" << s.getIndex() << "_weightupdate_simLearnPost";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(supportCode, model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;

    os << "// support code for synapse dynamics update groups" << std::endl;
    for(const auto &s : modelMerged.getMergedSynapseDynamicsGroups()) {
        const std::string supportCode = s.getArchetype().getWUModel()->getSynapseDynamicsSuppportCode();
        if (!supportCode.empty()) {
            os << "namespace merged" << s.getIndex() << "_weightupdate_synapseDynamics";
            {
                CodeStream::Scope b(os);
                os << ensureFtype(supportCode, model.getPrecision()) << std::endl;
            }
        }
    }
    os << std::endl;

    /*os << "// support code for synapse groups" << std::endl;
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();


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
    os << std::endl;*/
}
