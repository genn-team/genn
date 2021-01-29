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

    // Neuron update kernel
    backend.genCustomUpdate(os, modelMerged, memorySpaces,
                            // Preamble handler
                            [&modelMerged, &backend](CodeStream &os)
                            {
                                // Generate functions to push merged neuron group structures
                                modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomNeuronUpdateGroups(), backend);
                            },
                            // Custom neuron update handler
                            [&modelMerged](CodeStream &os, const CustomUpdateGroupMerged<NeuronVarReference> &cg, Substitutions &popSubs)
                            {

                            },
                            // Push EGP handler
                            // **TODO** this needs to be per-update group
                            [&backend, &modelMerged](CodeStream &os)
                            {
                                modelMerged.genScalarEGPPush(os, "NeuronCustomUpdate", backend);
                            });
}