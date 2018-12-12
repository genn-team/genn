#include "generateRunner.h"

// Standard C++ includes
#include <sstream>
#include <string>

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

// NuGeNN includes
#include "tee_stream.h"
#include "backends/base.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void writeTypeRange(CodeStream &os, const std::string &precision, const std::string &prefix)
{
    os << "#define " << prefix << "_MIN ";
    if (precision == "float") {
        writePreciseString(os, std::numeric_limits<float>::min());
        os << "f" << std::endl;
    }
    else {
        writePreciseString(os, std::numeric_limits<double>::min());
        os << std::endl;
    }

    os << "#define " << prefix << "_MAX ";
    if (precision == "float") {
        writePreciseString(os, std::numeric_limits<float>::max());
        os << "f" << std::endl;
    }
    else {
        writePreciseString(os, std::numeric_limits<double>::max());
        os << std::endl;
    }
    os << std::endl;
}
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateRunner(CodeStream &definitions, CodeStream &runner, const NNmodel &model,
                                   const Backends::Base &backend, int localHostID)
{
    // Write definitions pre-amble
    definitions << "#pragma once" << std::endl;

    // Write runner pre-amble
    runner << "#include \"definitions.h\"" << std::endl << std::endl;
    backend.genRunnerPreamble(runner);

    // Create codestreams to generate different sections of runner
    std::stringstream runnerVarDeclStream;
    std::stringstream runnerAllocStream;
    std::stringstream runnerFreeStream;
    CodeStream runnerVarDecl(runnerVarDeclStream);
    CodeStream runnerAlloc(runnerAllocStream);
    CodeStream runnerFree(runnerFreeStream);

    // Create a teestream to allow simultaneous writing to both streams
    TeeStream allStreams(definitions, runnerVarDecl, runnerAlloc, runnerFree);

    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif


    // write DT macro
    if (model.getTimePrecision() == "float") {
        definitions << "#define DT " << std::to_string(model.getDT()) << "f" << std::endl;
    } else {
        definitions << "#define DT " << std::to_string(model.getDT()) << std::endl;
    }

    // Typedefine scalar type
    definitions << "typedef " << model.getPrecision() << " scalar;" << std::endl;

    // Write ranges of scalar and time types
    writeTypeRange(definitions, model.getPrecision(), "SCALAR");
    writeTypeRange(definitions, model.getTimePrecision(), "TIME");

    // Begin extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        runnerVarDecl << "extern \"C\" {" << std::endl;
    }

    //---------------------------------
    // REMOTE NEURON GROUPS
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// remote neuron groups" << std::endl;
    allStreams << std::endl;

    // Loop through remote neuron groups
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // Write macro so whether a neuron group is remote or not can be determined at compile time
        // **NOTE** we do this for REMOTE groups so #ifdef GROUP_NAME_REMOTE is backward compatible
        definitions << "#define " << n.first << "_REMOTE" << std::endl;

        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            // Check that, whatever variable mode is set for these variables,
            // they are instantiated on host so they can be copied using MPI
            if(!(n.second.getSpikeVarMode() & VarLocation::HOST)) {
                gennError("Remote neuron group '" + n.first + "' has its spike variable mode set so it is not instantiated on the host - this is not supported");
            }

            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCnt"+n.first, n.second.getSpikeVarMode(),
                                   n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpk"+n.first, n.second.getSpikeVarMode(),
                                   n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
    }
    allStreams << std::endl;

    //---------------------------------
    // LOCAL NEURON VARIABLES
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// local neuron groups" << std::endl;
    allStreams << std::endl;

    for(const auto &n : model.getLocalNeuronGroups()) {
        backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCnt"+n.first, n.second.getSpikeVarMode(),
                               n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1);
        backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpk"+n.first, n.second.getSpikeVarMode(),
                               n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());

        if (n.second.isSpikeEventRequired()) {
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkCntEvnt"+n.first, n.second.getSpikeEventVarMode(),
                                   n.second.getNumDelaySlots());
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, "unsigned int", "glbSpkEvnt"+n.first, n.second.getSpikeEventVarMode(),
                                   n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }
        if (n.second.isDelayRequired()) {
            //**FIXME**
            definitions << varExportPrefix << " unsigned int spkQuePtr" << n.first << ";" << std::endl;
            runnerVarDecl << "unsigned int spkQuePtr" << n.first << ";" << std::endl;
#ifndef CPU_ONLY
            runnerVarDecl << "__device__ volatile unsigned int dd_spkQuePtr" << n.first << ";" << std::endl;
#endif
        }
        if (n.second.isSpikeTimeRequired()) {
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getTimePrecision()+" *", "sT"+n.first, n.second.getSpikeTimeVarMode(),
                                   n.second.getNumNeurons() * n.second.getNumDelaySlots());
        }
#ifndef CPU_ONLY
        //**FIXME**
        if(n.second.isSimRNGRequired()) {
            definitions << "extern curandState *d_rng" << n.first << ";" << std::endl;
            runnerVarDecl << "curandState *d_rng" << n.first << ";" << std::endl;
            runnerVarDecl << "__device__ curandState *dd_rng" << n.first << ";" << std::endl;
        }
#endif
        auto neuronModel = n.second.getNeuronModel();
        for(auto const &v : neuronModel->getVars()) {
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + n.first, n.second.getVarMode(v.first),
                                   n.second.isVarQueueRequired(v.first) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons());
        }
        for(auto const &v : neuronModel->getExtraGlobalParams()) {
            definitions << "extern " << v.second << " " << v.first + n.first << ";" << std::endl;
            runnerVarDecl << v.second << " " <<  v.first << n.first << ";" << std::endl;
        }

        if(!n.second.getCurrentSources().empty()) {
            allStreams << "// current source variables" << std::endl;
        }
        for (auto const *cs : n.second.getCurrentSources()) {
            auto csModel = cs->getCurrentSourceModel();
            for(auto const &v : csModel->getVars()) {
                backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + cs->getName(), cs->getVarMode(v.first),
                                       n.second.getNumNeurons());
            }
            for(auto const &v : csModel->getExtraGlobalParams()) {
                definitions << "extern " << v.second << " " <<  v.first << cs->getName() << ";" << std::endl;
                runnerVarDecl << v.second << " " <<  v.first << cs->getName() << ";" << std::endl;
            }
        }
    }
    allStreams << std::endl;

    //----------------------------------
    // POSTSYNAPTIC VARIABLES
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// postsynaptic variables" << std::endl;
    allStreams << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Loop through incoming synaptic populations
        for(const auto &m : n.second.getMergedInSyn()) {
            const auto *sg = m.first;

            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getPrecision(), "inSyn" + sg->getPSModelTargetName(), sg->getInSynVarMode(),
                                   sg->getTrgNeuronGroup()->getNumNeurons());

            if (sg->isDendriticDelayRequired()) {
                backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, model.getPrecision(), "denDelay" + sg->getPSModelTargetName(), sg->getDendriticDelayVarMode(),
                                       sg->getMaxDendriticDelayTimesteps() * sg->getTrgNeuronGroup()->getNumNeurons());

                //**FIXME**
                runnerVarDecl << varExportPrefix << " unsigned int denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
                runnerVarDecl << "unsigned int denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
#ifndef CPU_ONLY
                runnerVarDecl << "__device__ volatile unsigned int dd_denDelayPtr" << sg->getPSModelTargetName() << ";" << std::endl;
#endif
            }

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : sg->getPSModel()->getVars()) {
                    backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree, v.second, v.first + sg->getPSModelTargetName(), sg->getPSVarMode(v.first),
                                           sg->getTrgNeuronGroup()->getNumNeurons());
                }
            }
        }
    }
    allStreams << std::endl;

    //----------------------------------
    // SYNAPSE VARIABLE
    allStreams << "// ------------------------------------------------------------------------" << std::endl;
    allStreams << "// synapse variables" << std::endl;
    allStreams << std::endl;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                    "uint32_t", "gp" + s.first, s.second.getSparseConnectivityVarMode(), gpSize);
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            const VarMode varMode = s.second.getSparseConnectivityVarMode();
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections();

            // Row lengths
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   "unsigned int", "rowLength" + s.first, varMode, s.second.getSrcNeuronGroup()->getNumNeurons());

            // Target indices
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   "unsigned int", "ind" + s.first, varMode, size);

            // **TODO** remap is not always required
            if(!s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
                // Allocate synRemap
                // **THINK** this is over-allocating
                backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "synRemap" + s.first, varMode, size + 1);
            }

            // **TODO** remap is not always required
            if(!s.second.getWUModel()->getLearnPostCode().empty()) {
                const size_t postSize = (size_t)s.second.getTrgNeuronGroup()->getNumNeurons() * (size_t)s.second.getMaxSourceConnections();

                // Allocate column lengths
                backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "colLength" + s.first, varMode, s.second.getTrgNeuronGroup()->getNumNeurons());

                // Allocate remap
                backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                       "unsigned int", "remap" + s.first, varMode, postSize);

            }

            // If weight update variables should be individual
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : wu->getVars()) {
                    backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                           v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                }
            }
        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) {
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons();

            // If weight update variables should be individual
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : wu->getVars()) {
                    backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                           v.second, v.first + s.first, s.second.getWUVarMode(v.first), size);
                }
            }

        }

         const size_t preSize = (s.second.getDelaySteps() == NO_DELAY)
                ? s.second.getSrcNeuronGroup()->getNumNeurons()
                : s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getSrcNeuronGroup()->getNumDelaySlots();
        for(const auto &v : wu->getPreVars()) {
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   v.second, v.first + s.first, s.second.getWUPreVarMode(v.first), preSize);
        }

        const size_t postSize = (s.second.getBackPropDelaySteps() == NO_DELAY)
                ? s.second.getTrgNeuronGroup()->getNumNeurons()
                : s.second.getTrgNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumDelaySlots();
        for(const auto &v : wu->getPostVars()) {
            backend.genArray(definitions, runnerVarDecl, runnerAlloc, runnerFree,
                                   v.second, v.first + s.first, s.second.getWUPostVarMode(v.first), postSize);
        }

        for(const auto &v : wu->getExtraGlobalParams()) {
            definitions << "extern " << v.second << " " << v.first + s.first << ";" << std::endl;
            runnerVarDecl << v.second << " " <<  v.first << s.first << ";" << std::endl;
        }

        for(auto const &p : s.second.getConnectivityInitialiser().getSnippet()->getExtraGlobalParams()) {
            definitions << "extern " << p.second << " initSparseConn" << p.first + s.first << ";" << std::endl;
            runnerVarDecl << p.second << " initSparseConn" << p.first + s.first << ";" << std::endl;
        }
    }
    allStreams << std::endl;
    // End extern C block around variable declarations
    if(GENN_PREFERENCES::buildSharedLibrary) {
        runnerVarDecl << "}\t// extern \"C\"" << std::endl;
    }

    // Write variable declarations to runner
    runner << runnerVarDeclStream.str();

    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
    runner << "void allocateMem()";
    {
        CodeStream::Scope b(runner);
#ifndef CPU_ONLY
        // **TODO** move to code generator
        runner << "CHECK_CUDA_ERRORS(cudaSetDevice(" << theDevice << "));" << std::endl;

        // If the model requires zero-copy
        if(model.zeroCopyInUse()) {
            // If device doesn't support mapping host memory error
            if(!deviceProp[theDevice].canMapHostMemory) {
                gennError("Device does not support mapping CPU host memory!");
            }

            // set appropriate device flags
            runner << "CHECK_CUDA_ERRORS(cudaSetDeviceFlags(cudaDeviceMapHost));" << std::endl;
        }

        // If RNG is required, allocate memory for global philox RNG
        if(model.isDeviceRNGRequired()) {
            //allocate_device_variable(os, "curandStatePhilox4_32_10_t", "rng", VarMode::LOC_DEVICE_INIT_DEVICE, 1);
        }
#endif
        runner << runnerAllocStream.str();
    }
    runner << std::endl;

    // ------------------------------------------------------------------------
    // Function to free all global memory structures
    runner << "void freeMem()";
    {
        CodeStream::Scope b(runner);

        runner << runnerFreeStream.str();
    }

    // ---------------------------------------------------------------------
    // Function definitions
    definitions << "// Runner functions" << std::endl;
    definitions << "void allocateMem();" << std::endl;
    definitions << "void freeMem();" << std::endl;
    definitions << std::endl;
    definitions << "// Neuron update functions" << std::endl;
    definitions << "void updateNeurons(float t);" << std::endl;

}