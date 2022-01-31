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

    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// field getter macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    // Using these, generate macros to call functions on group fields
    macroLookup << "#define GET_FIELD(POP, VAR) GENN_CAT(get, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP)))(GENN_CAT(POP, _GROUP))" << std::endl;
    macroLookup << "#define PUSH_FIELD(POP, VAR) GENN_CAT(GENN_CAT(push, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP))), ToDevice)(GENN_CAT(POP, _GROUP))" << std::endl;
    macroLookup << "#define PULL_FIELD(POP, VAR) GENN_CAT(GENN_CAT(pull, GENN_CAT(VAR, GENN_CAT(POP, _MERGED_GROUP))), FromDevice)(GENN_CAT(POP, _GROUP))" << std::endl;
    
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// group macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    
    // Loop through neuron groups
    const auto &model = modelMerged.getModel();
    const auto &map = modelMerged.getMergedRunnerGroups();
    for(const auto &g : model.getNeuronGroups()) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        macroLookup << "#define " << g.second.getName() << "_MERGED_GROUP NeuronRunnerGroup" << std::get<0>(groupIndices) << std::endl;
        macroLookup << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
        macroLookup << std::endl;
    }
    
    // Loop through synapse groups
    for(const auto &g : model.getSynapseGroups()) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        macroLookup << "#define " << g.second.getName() << "_MERGED_GROUP SynapseRunnerGroup" << std::get<0>(groupIndices) << std::endl;
        macroLookup << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
        macroLookup << std::endl;
    }
    
    // Loop through current sources
    for(const auto &g : model.getLocalCurrentSources()) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        macroLookup << "#define " << g.second.getName() << "_MERGED_GROUP CurrentSourceRunnerGroup" << std::get<0>(groupIndices) << std::endl;
        macroLookup << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
        macroLookup << std::endl;
    }

    // Loop through custom updates
    for(const auto &g : model.getCustomUpdates()) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        macroLookup << "#define " << g.second.getName() << "_MERGED_GROUP CustomUpdateRunnerGroup" << std::get<0>(groupIndices) << std::endl;
        macroLookup << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
        macroLookup << std::endl;
    }

    // Loop through custom WU updates
    for(const auto &g : model.getCustomWUUpdates()) {
        // Get indices
        const auto groupIndices = map.getIndices(g.second.getName());
        
        // Write out 
        macroLookup << "#define " << g.second.getName() << "_MERGED_GROUP CustomUpdateWURunnerGroup" << std::get<0>(groupIndices) << std::endl;
        macroLookup << "#define " << g.second.getName() << "_GROUP " << std::get<1>(groupIndices) << std::endl;
        macroLookup << std::endl;
    }
}
}
