#include "code_generator/generateSupportCode.h"

// Standard C++ includes
#include <fstream>
#include <string>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

namespace
{
template<typename G>
void generatePopulationMacros(CodeStream &os, const std::map<std::string, G> &groups, const MergedRunnerMap &map, const std::string &name)
{
    // Loop through custom WU updates
    for(const auto &g : groups) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        os << "#define " << g.second.getName() << "_MERGED_GROUP " << name << "RunnerGroup" << std::get<0>(groupIndices) << std::endl;
        os << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
    }
    os << std::endl;
}
}
//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
void generateMacroLookup(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged)
{
    std::ofstream macroLookupStream((outputPath / ("macroLookup.h")).str());
    CodeStream macroLookup(macroLookupStream);
    
    macroLookup << "#pragma once" << std::endl;
    
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// helper macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    // Generate boost preprocessor style implicit concatentation macros
    // **NOTE** the two level macro expansion is required so macros referring to other macros get expanded
    macroLookup << "#define GENN_CAT_I(A, B) A ## B" << std::endl;
    macroLookup << "#define GENN_CAT(A, B) GENN_CAT_I(A, B)" << std::endl;
    macroLookup << std::endl;

    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// field getter macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    // Using these, generate macros to call functions on group fields
    macroLookup << "#define GET_FIELD(POP, VAR) GENN_CAT(get, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP)))(GENN_CAT(POP, _GROUP))" << std::endl;
    macroLookup << "#define PUSH_FIELD(POP, VAR) GENN_CAT(GENN_CAT(push, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP))), ToDevice)(GENN_CAT(POP, _GROUP))" << std::endl;
    macroLookup << "#define PULL_FIELD(POP, VAR) GENN_CAT(GENN_CAT(pull, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP))), FromDevice)(GENN_CAT(POP, _GROUP))" << std::endl;
    macroLookup << std::endl;

    // Genererate macros
    const auto &model = modelMerged.getModel();
    const auto &map = modelMerged.getMergedRunnerGroups();
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// neuron group macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    generatePopulationMacros(macroLookup, model.getNeuronGroups(), map, "Neuron");

    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// synapse group macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    generatePopulationMacros(macroLookup, model.getSynapseGroups(), map, "Synapse");

    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// current source macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    generatePopulationMacros(macroLookup, model.getLocalCurrentSources(), map, "CurrentSource");

    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// custom update macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    generatePopulationMacros(macroLookup, model.getCustomUpdates(), map, "CustomUpdate");
    generatePopulationMacros(macroLookup, model.getCustomWUUpdates(), map, "CustomUpdateWU");
}
}
