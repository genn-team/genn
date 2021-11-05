#include "code_generator/generateNeuronUpdate.h"

// Standard C++ includes
#include <fstream>
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
void CodeGenerator::generateNeuronUpdate(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                         const BackendBase &backend, const std::string &suffix)
{
    // Create output stream to write to file and wrap in CodeStream
    std::ofstream neuronUpdateStream((outputPath / ("neuronUpdate" + suffix + ".cc")).str());
    CodeStream neuronUpdate(neuronUpdateStream);

    neuronUpdate << "#include \"definitionsInternal" << suffix << ".h\"" << std::endl;
    if (backend.supportsNamespace()) {
        neuronUpdate << "#include \"supportCode" << suffix << ".h\"" << std::endl;
    }
    neuronUpdate << std::endl;

    // Neuron update kernel
    backend.genNeuronUpdate(neuronUpdate, modelMerged,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            // Generate functions to push merged neuron group structures
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedNeuronUpdateGroups(), backend);
        },
        // Push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<NeuronUpdateGroupMerged>(os, backend);
        });
}
