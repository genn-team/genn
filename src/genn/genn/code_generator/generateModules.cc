#include "code_generator/generateModules.h"

// Standard C++ includes
#include <fstream>
#include <string>
#include <vector>

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "modelSpecInternal.h"

// Code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/generateRunner.h"
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void writeStringStream(const filesystem::path &file, const std::ostringstream &stream)
{
    std::ofstream fileStream(file.str());
    fileStream << stream.str();
}
//--------------------------------------------------------------------------
bool shouldRebuildModel(const filesystem::path &outputPath, const boost::uuids::detail::sha1::digest_type &hashDigest)
{
    try
    {
        // Open file
        std::ifstream is((outputPath / "model.sha").str());

        // Throw exceptions in case of all errors
        is.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

        // Read previous hash digest as hex
        boost::uuids::detail::sha1::digest_type previousHashDigest; 
        is >> std::hex;
        for(auto &d : previousHashDigest) {
            is >> d;
        }

        // If hash matches
        if(previousHashDigest == hashDigest) {
            LOGD_CODE_GEN << "Model unchanged - skipping code generation";
            return false;
        }
        else {
            LOGD_CODE_GEN << "Model changed - re-generating code";
        }
    }
    catch(const std::ios_base::failure&) {
        LOGD_CODE_GEN << "Unable to read previous model hash - re-generating code";
    }

    return true;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
std::vector<std::string> generateAll(ModelSpecMerged &modelMerged, const BackendBase &backend,
                                     const filesystem::path &outputPath, bool alwaysRebuild, 
                                     bool neverRebuild)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Get memory spaces available to this backend
    // **NOTE** Memory spaces are given out on a first-come, first-serve basis so subsequent groups are in preferential order
    auto memorySpaces = backend.getMergedGroupMemorySpaces(modelMerged);
    
    // Create vector to hold file streams and lambda function to create them
    std::vector<std::tuple<std::string, std::string, std::ostringstream>> fileStreams;
    auto fileStreamCreator =
        [&fileStreams](const std::string &title, const std::string &extension) -> std::ostream&
        {
            fileStreams.emplace_back(title, extension, std::ostringstream());
            return std::get<2>(fileStreams.back());
        };

    generateSynapseUpdate(fileStreamCreator, modelMerged, backend, memorySpaces);
    generateNeuronUpdate(fileStreamCreator, modelMerged, backend, memorySpaces);
    generateCustomUpdate(fileStreamCreator, modelMerged, backend, memorySpaces);
    generateInit(fileStreamCreator, modelMerged, backend, memorySpaces);

    // If force rebuild flag is set or model should be rebuilt
    const auto hashDigest = modelMerged.getHashDigest(backend);
    if(!neverRebuild && (alwaysRebuild || shouldRebuildModel(outputPath, hashDigest))) {
        // Generate runner
        generateRunner(outputPath, modelMerged, backend);

        // Write file streams to disk
        for(const auto &s : fileStreams) {
            std::ofstream fileStream((outputPath / (std::get<0>(s) + "." + std::get<1>(s))).str());
            fileStream << std::get<2>(s).str();
        }

        // Open file
        std::ofstream os((outputPath / "model.sha").str());
    
        // Write digest as hex with each word seperated by a space
        os << std::hex;
        for(const auto d : hashDigest) {
            os << d << " ";
        }
        os << std::endl;
    }

    // Output summary to log
    LOGI_CODE_GEN << "Merging model with " << modelMerged.getModel().getNeuronGroups().size() << " neuron groups and " << modelMerged.getModel().getSynapseGroups().size() << " synapse groups results in:";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronUpdateGroups().size() << " merged neuron update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedPresynapticUpdateGroups().size() << " merged presynaptic update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedPostsynapticUpdateGroups().size() << " merged postsynaptic update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDynamicsGroups().size() << " merged synapse dynamics groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateGroups().size() << " merged custom update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateWUGroups().size() << " merged custom weight update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateTransposeWUGroups().size() << " merged custom weight transpose update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomConnectivityUpdateGroups().size() << " merged custom connectivity update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomConnectivityHostUpdateGroups().size() << " merged custom connectivity host update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronInitGroups().size() << " merged neuron init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateInitGroups().size() << " merged custom update init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomWUUpdateInitGroups().size() << " merged custom WU update init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomConnectivityUpdatePreInitGroups().size() << " merged custom connectivity update presynaptic init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomConnectivityUpdatePostInitGroups().size() << " merged custom connectivity update postsynaptic init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseInitGroups().size() << " merged synapse init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityInitGroups().size() << " merged synapse connectivity init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseSparseInitGroups().size() << " merged synapse sparse init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomWUUpdateSparseInitGroups().size() << " merged custom WU update sparse init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups().size() << " merged custom connectivity update sparse init groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronSpikeQueueUpdateGroups().size() << " merged neuron spike queue update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDendriticDelayUpdateGroups().size() << " merged synapse dendritic delay update groups";
    LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityHostInitGroups().size() << " merged synapse connectivity host init groups";

    // Return list of modules and memory usage
    return std::vector<std::string>{"customUpdate", "neuronUpdate", "synapseUpdate", "init", "runner"};
}
//--------------------------------------------------------------------------
void generateNeuronUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                          BackendBase::MemorySpaces &memorySpaces, const std::string &suffix)
{
    // Wrap output stream in CodeStream
    CodeStream neuronUpdate(streamCreator("neuronUpdate", "cc"));

    neuronUpdate << "#include \"definitions" << suffix << ".h\"" << std::endl;
    neuronUpdate << std::endl;

    // Neuron update kernel
    backend.genNeuronUpdate(neuronUpdate, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            // Generate functions to push merged neuron group structures
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedNeuronSpikeQueueUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedNeuronUpdateGroups(), backend);
        });
}
//--------------------------------------------------------------------------
void generateCustomUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                          BackendBase::MemorySpaces &memorySpaces, const std::string &suffix)
{
    // Wrap output stream in CodeStream
    CodeStream customUpdate(streamCreator("customUpdate", "cc"));

    customUpdate << "#include \"definitions" << suffix << ".h\"" << std::endl;
    customUpdate << std::endl;

    // Neuron update kernel
    backend.genCustomUpdate(customUpdate, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            // Generate functions to push merged neuron group structures
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomUpdateWUGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomUpdateTransposeWUGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomConnectivityUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomConnectivityHostUpdateGroups(), backend, true);
        });
}
//--------------------------------------------------------------------------
void generateSynapseUpdate(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                           BackendBase::MemorySpaces &memorySpaces, const std::string &suffix)
{
    // Wrap output stream in CodeStream
    CodeStream synapseUpdate(streamCreator("synapseUpdate", "cc"));

    synapseUpdate << "#include \"definitions" << suffix << ".h\"" << std::endl;
    synapseUpdate << std::endl;

    // Synaptic update kernels
    backend.genSynapseUpdate(synapseUpdate, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedSynapseDendriticDelayUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedPresynapticUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedPostsynapticUpdateGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedSynapseDynamicsGroups(), backend);
        });
}
//--------------------------------------------------------------------------
void generateInit(FileStreamCreator streamCreator, ModelSpecMerged &modelMerged, const BackendBase &backend,
                  BackendBase::MemorySpaces &memorySpaces, const std::string &suffix)
{
    // Wrap output stream in CodeStream
    CodeStream init(streamCreator("init", "cc"));

    init << "#include \"definitions" << suffix << ".h\"" << std::endl;

    backend.genInit(init, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &memorySpaces, &backend](CodeStream &os)
        {
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedNeuronInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomUpdateInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomConnectivityUpdatePreInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomConnectivityUpdatePostInitGroups(), backend);

            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomWUUpdateInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedSynapseInitGroups(), backend);

            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedSynapseConnectivityInitGroups(), backend);
            
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedSynapseSparseInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomWUUpdateSparseInitGroups(), backend);
            modelMerged.genDynamicFieldPush(os, modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups(), backend);

            // Generate merged synapse connectivity host init code
            // **NOTE** this needs to be done before generating the runner because this configures the required fields BUT
            // needs to be done into a seperate stream because it actually needs to be RUN afterwards so valid pointers 
            // get copied straight into subsequent structures and merged EGP system isn't required
            // Generate stream with neuron update code
            std::ostringstream initStream;
            CodeStream init(initStream);
            init << "void initializeHost()";
            {
                CodeStream::Scope b(init);
                modelMerged.genMergedSynapseConnectivityHostInitGroups(
                    backend, memorySpaces,
                    [&backend, &init](auto &sg)
                    {
                        EnvironmentExternal env(init);
                        sg.generateInit(backend, env);
                    });
            }

            // Generate host connectivity init structures and write function afterwards
            modelMerged.genMergedSynapseConnectivityHostInitStructs(os, backend);

            // Generate arrays
            modelMerged.genMergedSynapseConnectivityHostInitStructArrayPush(os, backend);

            // Insert generated initialisation function
            os << initStream.str();
        });
}
}   // namespace GeNN::CodeGenerator
