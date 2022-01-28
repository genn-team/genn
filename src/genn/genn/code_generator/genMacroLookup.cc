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
void CodeGenerator::generateMacroLookup(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged)
{
    std::ofstream macroLookupStream((outputPath / ("macroLookup.h")).str());
    CodeStream macroLookup(macroLookupStream);
    
    macroLookup << "#pragma once" << std::endl;
    
    macroLookup << "// ------------------------------------------------------------------------" << std::endl;
    macroLookup << "// helper macros
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
    for(const auto &g : modelMerged.get
}
