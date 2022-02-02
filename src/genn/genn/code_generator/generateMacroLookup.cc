#include "code_generator/generateSupportCode.h"

// Standard C++ includes
#include <fstream>
#include <string>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

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
    macroLookup << "#define GET_FIELD(POP, VAR) GENN_CAT(get, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP)))(GENN_CAT(GROUP_, POP))" << std::endl;
    macroLookup << "#define PUSH_FIELD(POP, VAR) GENN_CAT(GENN_CAT(push, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP))), ToDevice)(GENN_CAT(GROUP_, POP))" << std::endl;
    macroLookup << "#define PULL_FIELD(POP, VAR) GENN_CAT(GENN_CAT(pull, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP))), FromDevice)(GENN_CAT(GROUP_, POP))" << std::endl;
    macroLookup << "#define PUSH_EGP_FIELD(POP, VAR, COUNT) GENN_CAT(GENN_CAT(push, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP))), ToDevice)(GENN_CAT(GROUP_, POP), COUNT)" << std::endl;
    macroLookup << "#define PULL_EGP_FIELD(POP, VAR, COUNT) GENN_CAT(GENN_CAT(pull, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP))), FromDevice)(GENN_CAT(GROUP_, POP), COUNT)" << std::endl;
    macroLookup << "#define ALLOCATE_EGP_FIELD(POP, VAR, COUNT) GENN_CAT(allocate, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP)))(GENN_CAT(GROUP_, POP), COUNT)" << std::endl;
    macroLookup << "#define FREE_EGP_FIELD(POP, VAR) GENN_CAT(free, GENN_CAT(VAR, GENN_CAT(MERGED_GROUP_, POP)))(GENN_CAT(GROUP_, POP))" << std::endl;
    macroLookup << std::endl;

    // Generate macros for resolving population names to merged groups
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// group macros" << std::endl;
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &g : modelMerged.getMergedRunnerGroups().getGroups()) {
        macroLookup << "#define MERGED_GROUP_" << g.first << " " << std::get<2>(g.second) << "Group" << std::get<0>(g.second) << std::endl;
        macroLookup << "#define GROUP_" << g.first << " " << std::get<1>(g.second) << std::endl;
    }
}
}
