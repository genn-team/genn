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
void addCustomUpdateSubstitutions(CodeStream &os, Substitutions &baseSubs, 
                                  const CustomUpdateGroupMerged &cg, const ModelSpecMerged &modelMerged)
{
    Substitutions updateSubs(&baseSubs);

    const CustomUpdateModels::Base *cm = cg.getArchetype().getCustomUpdateModel();
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "group->", "[" + updateSubs["id"] + "]");
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "group->", "[" + updateSubs["id"] + "]");
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
        },
        // Custom neuron update handler
        [&modelMerged](CodeStream &os, const CustomUpdateGroupMerged &cg, Substitutions &popSubs)
        {
            addCustomUpdateSubstitutions(os, popSubs, cg, modelMerged);
        },
        // Push EGP handler
        // **TODO** this needs to be per-update group
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush(os, "NeuronCustomUpdate", backend);
        });
}