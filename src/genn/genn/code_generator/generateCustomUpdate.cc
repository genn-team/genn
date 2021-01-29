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
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateCustomUpdate(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                                         const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;

    os << std::endl;
}