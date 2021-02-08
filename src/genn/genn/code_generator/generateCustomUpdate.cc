#include "code_generator/generateCustomUpdate.h"

// Standard C++ includes
#include <iostream>
#include <string>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"
#include "code_generator/teeStream.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename C>
void genCustomUpdate(CodeStream &os, Substitutions &baseSubs, const C &cg, 
                                  const ModelSpecMerged &modelMerged, const std::string &index)
{
    Substitutions updateSubs(&baseSubs);

    const CustomUpdateModels::Base *cm = cg.getArchetype().getCustomUpdateModel();

    // Read variables into registers
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type << " l" << v.name << " = " << "group->" << v.name << "[" << updateSubs[index] << "];" << std::endl;
    }

    // Read variables references into registers
    for(const auto &v : cm->getVarRefs()) {
        if(v.access == VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type << " l" << v.name << " = " << "group->" << v.name << "[" << updateSubs[index] << "];" << std::endl;
    }
    
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "l");
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "l");
    updateSubs.addParamValueSubstitution(cm->getParamNames(), cg.getArchetype().getParams(),
                                         [&cg](size_t i) { return cg.isParamHeterogeneous(i);  },
                                         "", "group->");
    updateSubs.addVarValueSubstitution(cm->getDerivedParams(), cg.getArchetype().getDerivedParams(),
                                       [&cg](size_t i) { return cg.isDerivedParamHeterogeneous(i);  },
                                       "", "group->");
    updateSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");

    std::string code = cm->getUpdateCode();
    updateSubs.applyCheckUnreplaced(code, "custom update : merged" + cg.getIndex());
    code = ensureFtype(code, modelMerged.getModel().getPrecision());
    os << code;

    // Write read/write variables back to global memory
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << "["  << updateSubs[index] << "] = l" << v.name << ";" << std::endl;
        }
    }

    // Write read/write variable references back to global memory
    for(const auto &v : cm->getVarRefs()) {
        if(v.access == VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << "["  << updateSubs[index] << "] = l" << v.name << ";" << std::endl;
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateCustomUpdate(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                                         const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;
    os << std::endl;

    // Neuron update kernel
    backend.genCustomUpdate(os, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            // Generate functions to push merged neuron group structures
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomUpdateWUGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups(), backend);
        },
        // Custom update handler
        [&modelMerged](CodeStream &os, const CustomUpdateGroupMerged &cg, Substitutions &popSubs)
        {
            genCustomUpdate(os, popSubs, cg, modelMerged, "id");
        },
        // Custom weight update handler
        [&modelMerged](CodeStream &os, const CustomUpdateWUGroupMerged &cg, Substitutions &popSubs)
        {
            genCustomUpdate(os, popSubs, cg, modelMerged, "id_syn");
        },
        // Custom transpose weight update handler
        [&modelMerged](CodeStream &os, const CustomUpdateTransposeWUGroupMerged &cg, Substitutions &popSubs)
        {
            genCustomUpdate(os, popSubs, cg, modelMerged, "id_syn");
        },
        // Push EGP handler
        // **TODO** this needs to be per-update group
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<CustomUpdateGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<CustomUpdateWUGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<CustomUpdateTransposeWUGroupMerged>(os, backend);
        });
}