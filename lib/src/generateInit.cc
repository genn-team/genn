#include "generateInit.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>

// Standard C includes
#include <cmath>
#include <cstdlib>

// GeNN includes
#include "codeStream.h"
#include "global.h"
#include "modelSpec.h"
#include "standardSubstitutions.h"

// ------------------------------------------------------------------------
// Anonymous namespace
// ------------------------------------------------------------------------
namespace
{
class PaddedSizeScope
{
public:
    PaddedSizeScope(CodeStream &codeStream, unsigned int count, unsigned int blockSize, unsigned int &startThread)
    :   m_CodeStream(codeStream), m_Level(s_NextLevel++), m_StartThread(startThread),
        m_EndThread(m_StartThread + (unsigned int)(ceil((double)count / (double)blockSize) * (double)blockSize))
    {
        // Write if block to determine if this thread should be used for this neuron group
        if(m_StartThread == 0) {
            m_CodeStream << "if (id < " << m_EndThread << ")";
        }
        else {
            m_CodeStream << "if ((id >= " << m_StartThread << ") && (id < " << m_EndThread << "))";
        }
        m_CodeStream << CodeStream::OB(m_Level);
        m_CodeStream << "const unsigned int lid = id - " << m_StartThread << ";" << std::endl;
    }

    ~PaddedSizeScope()
    {
        m_CodeStream << CodeStream::CB(m_Level);
        m_StartThread = m_EndThread;
    }
private:
    //------------------------------------------------------------------------
    // Static members
    //------------------------------------------------------------------------
    static unsigned int s_NextLevel;

    CodeStream &m_CodeStream;

    const unsigned int m_Level;

    unsigned int &m_StartThread;
    const unsigned int m_EndThread;
};
unsigned int PaddedSizeScope::s_NextLevel = 0;

bool shouldInitOnHost(VarMode varMode)
{
#ifndef CPU_ONLY
    return (varMode & VarInit::HOST);
#else
    USE(varMode);
    return true;
#endif
}
// ------------------------------------------------------------------------
template<typename I, typename M>
void genHostInitNeuronVarCode(CodeStream &os, const NewModels::Base::StringPairVec &vars, size_t count, const std::string &popName, const std::string &ftype,
                              I getVarInitialiser, M getVarMode)
{
    for (size_t k= 0, l= vars.size(); k < l; k++) {
        const auto &varInit = getVarInitialiser(k);
        const VarMode varMode = getVarMode(k);

        // If this variable should be initialised on the host and has any initialisation code
        if(shouldInitOnHost(varMode) && !varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);
            os << "for (int i = 0; i < " << count << "; i++)";
            {
                CodeStream::Scope b(os);
                os << StandardSubstitutions::initNeuronVariable(varInit, vars[k].first + popName + "[i]",
                                                                cpuFunctions, "i", ftype, "rng") << std::endl;
            }
        }
    }
}
// ------------------------------------------------------------------------
template<typename I, typename M>
void genDeviceNeuronInitVarCode(CodeStream &os, const NewModels::Base::StringPairVec &vars, const std::string &idx, const std::string &popName, const std::string &ftype,
                                I getVarInitialiser, M getVarMode)
{
    for(size_t j = 0; j < vars.size(); j++) {
        const auto &varInit = getVarInitialiser(j);
        const VarMode varMode = getVarMode(j);

        // Initialise directly into device variable
        if((varMode & VarInit::DEVICE) && !varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);
            os << StandardSubstitutions::initNeuronVariable(varInit, "dd_" + vars[j].first + popName + "[" + idx + "]",
                                                            cudaFunctions, idx, ftype, "&initRNG") << std::endl;
        }
    }
}
// ------------------------------------------------------------------------
template<typename I, typename M>
void genDeviceWeightUpdateInitVarCode(CodeStream &os, const NewModels::Base::StringPairVec &vars, const std::string &idx, const std::string &preIdx, const std::string &postIdx, const std::string &popName, const std::string &ftype,
                                      I getVarInitialiser, M getVarMode)
{
    for(size_t j = 0; j < vars.size(); j++) {
        const auto &varInit = getVarInitialiser(j);
        const VarMode varMode = getVarMode(j);

        // Initialise directly into device variable
        if((varMode & VarInit::DEVICE) && !varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);
            os << StandardSubstitutions::initWeightUpdateVariable(varInit, "dd_" + vars[j].first + popName + "[" + idx + "]",
                                                                  cudaFunctions, preIdx, postIdx, ftype, "&initRNG") << std::endl;
        }
    }
}
// ------------------------------------------------------------------------
void genHostInitSpikeCode(CodeStream &os, const NeuronGroup &ng, bool spikeEvent)
{
    // Get variable mode
    const VarMode varMode = spikeEvent ? ng.getSpikeEventVarMode() : ng.getSpikeVarMode();

    // Is host initialisation required at all
    const bool hostInitRequired = spikeEvent ?
        (ng.isSpikeEventRequired() && shouldInitOnHost(varMode))
        : shouldInitOnHost(varMode);

    // Is delay required
    const bool delayRequired = spikeEvent ?
        ng.isDelayRequired() :
        (ng.isTrueSpikeRequired() && ng.isDelayRequired());

    const char *spikeCntPrefix = spikeEvent ? "glbSpkCntEvnt" : "glbSpkCnt";
    const char *spikePrefix = spikeEvent ? "glbSpkEvnt" : "glbSpk";

    if(hostInitRequired) {
        if (delayRequired) {
            {
                CodeStream::Scope b(os);
                os << "for (int i = 0; i < " << ng.getNumDelaySlots() << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << spikeCntPrefix << ng.getName() << "[i] = 0;" << std::endl;
                }
            }

            {
                CodeStream::Scope b(os);
                os << "for (int i = 0; i < " << ng.getNumNeurons() * ng.getNumDelaySlots() << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << spikePrefix << ng.getName() << "[i] = 0;" << std::endl;
                }
            }
        }
        else {
            os << spikeCntPrefix << ng.getName() << "[0] = 0;" << std::endl;
            {
                CodeStream::Scope b(os);
                os << "for (int i = 0; i < " << ng.getNumNeurons() << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << spikePrefix << ng.getName() << "[i] = 0;" << std::endl;
                }
            }
        }
    }
}
// ------------------------------------------------------------------------
#ifndef CPU_ONLY
unsigned int genInitializeDeviceKernel(CodeStream &os, const NNmodel &model, int localHostID)
{
    // init kernel header
    os << "extern \"C\" __global__ void initializeDevice()";

    // initialization kernel code
    unsigned int startThread = 0;
    {
        // common variables for all cases
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << initBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;

        // If RNG is required
        if(model.isDeviceRNGRequired()) {
            os << "// Initialise global GPU RNG" << std::endl;
            os << "if(id == 0)";
            {
                CodeStream::Scope b(os);
                os << "curand_init(" << model.getSeed() << ", 0, 0, &dd_rng[0]);" << std::endl;
            }
        }
        // Loop through remote neuron groups
        for(const auto &n : model.getRemoteNeuronGroups()) {
            if(n.second.hasOutputToHost(localHostID) && n.second.getSpikeVarMode() & VarInit::DEVICE) {
                os << "// remote neuron group " << n.first << std::endl;
                PaddedSizeScope p(os, n.second.getNumNeurons(), initBlkSz, startThread);

                os << "if(lid == 0)";
                {
                    CodeStream::Scope b(os);

                    // If delay is required, loop over delay bins
                    if(n.second.isDelayRequired()) {
                        os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(14);
                    }

                    if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                        os << "dd_glbSpkCnt" << n.first << "[i] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                    }

                    // If delay was required, close loop brace
                    if(n.second.isDelayRequired()) {
                        os << CodeStream::CB(14);
                    }
                }


                os << "// only do this for existing neurons" << std::endl;
                os << "if (lid < " << n.second.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If delay is required, loop over delay bins
                    if(n.second.isDelayRequired()) {
                        os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(16);
                    }

                    // Zero spikes
                    if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                        os << "dd_glbSpk" << n.first << "[(i * " + std::to_string(n.second.getNumNeurons()) + ") + lid] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpk" << n.first << "[lid] = 0;" << std::endl;
                    }

                    // If delay was required, close loop brace
                    if(n.second.isDelayRequired()) {
                        os << CodeStream::CB(16) << std::endl;
                    }
                }
            }
        }

        // Loop through local neuron groups
        for(const auto &n : model.getLocalNeuronGroups()) {
            // If this neuron group require any kind of device initialisation
            if(n.second.isDeviceInitRequired()) {
                os << "// local neuron group " << n.first << std::endl;
                PaddedSizeScope p(os, n.second.getNumNeurons(), initBlkSz, startThread);

                // Determine which built in variables should be initialised on device
                const bool shouldInitSpikeVar = (n.second.getSpikeVarMode() & VarInit::DEVICE);
                const bool shouldInitSpikeEventVar = n.second.isSpikeEventRequired() && (n.second.getSpikeEventVarMode() & VarInit::DEVICE);
                const bool shouldInitSpikeTimeVar = n.second.isSpikeTimeRequired() && (n.second.getSpikeTimeVarMode() & VarInit::DEVICE);

                // If per-population spike variables should be initialised on device
                // **NOTE** could optimise here and use getNumDelaySlots threads if getNumDelaySlots < numthreads
                if(shouldInitSpikeVar || shouldInitSpikeEventVar)
                {
                    os << "if(lid == 0)";
                    {
                        CodeStream::Scope b(os);

                        // If delay is required, loop over delay bins
                        if(n.second.isDelayRequired()) {
                            os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(22);
                        }

                        // Zero spike count
                        if(shouldInitSpikeVar) {
                            if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                                os << "dd_glbSpkCnt" << n.first << "[i] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                            }
                        }

                        // Zero spike event count
                        if(shouldInitSpikeEventVar) {
                            if(n.second.isDelayRequired()) {
                                os << "dd_glbSpkCntEvnt" << n.first << "[i] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
                            }
                        }

                        // If delay was required, close loop brace
                        if(n.second.isDelayRequired()) {
                            os << CodeStream::CB(22);
                        }
                    }
                }

                os << "// only do this for existing neurons" << std::endl;
                os << "if (lid < " << n.second.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If this neuron is going to require a simulation RNG, initialise one using thread id for sequence
                    if(n.second.isSimRNGRequired()) {
                        os << "curand_init(" << model.getSeed() << ", id, 0, &dd_rng" << n.first << "[lid]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(n.second.isInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;
                    }

                    // Build string to use for delayed variable index
                    const std::string delayedIndex = "(i * " + std::to_string(n.second.getNumNeurons()) + ") + lid";

                    // If spike variables are initialised on device
                    if(shouldInitSpikeVar || shouldInitSpikeEventVar || shouldInitSpikeTimeVar) {
                        // If delay is required, loop over delay bins
                        if(n.second.isDelayRequired()) {
                            os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(31);
                        }

                        // Zero spikes
                        if(shouldInitSpikeVar) {
                            if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
                                os << "dd_glbSpk" << n.first << "[" << delayedIndex << "] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpk" << n.first << "[lid] = 0;" << std::endl;
                            }
                        }

                        // Zero spike events
                        if(shouldInitSpikeEventVar) {
                            if(n.second.isDelayRequired()) {
                                os << "dd_glbSpkEvnt" << n.first << "[" << delayedIndex << "] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCnt" << n.first << "[lid] = 0;" << std::endl;
                            }
                        }

                        // Reset spike times
                        if(shouldInitSpikeTimeVar) {
                            if(n.second.isDelayRequired()) {
                                os << "dd_sT" << n.first << "[" << delayedIndex << "] = -SCALAR_MAX;" << std::endl;
                            }
                            else {
                                os << "dd_sT" << n.first << "[lid] = -SCALAR_MAX;" << std::endl;
                            }
                        }

                        // If delay was required, close loop brace
                        if(n.second.isDelayRequired()) {
                            os << CodeStream::CB(31) << std::endl;
                        }
                    }

                    // Loop through neuron variables
                    auto neuronModelVars = n.second.getNeuronModel()->getVars();
                    for (size_t j = 0; j < neuronModelVars.size(); j++) {
                        const auto &varInit = n.second.getVarInitialisers()[j];
                        const VarMode varMode = n.second.getVarMode(j);

                        // If this variable should be initialised on the device and has any initialisation code
                        if((varMode & VarInit::DEVICE) && !varInit.getSnippet()->getCode().empty()) {
                            CodeStream::Scope b(os);

                            // If variable requires a queue
                            if (n.second.isVarQueueRequired(j)) {
                                // Generate initial value into temporary variable
                                os << neuronModelVars[j].second << " initVal;" << std::endl;
                                os << StandardSubstitutions::initNeuronVariable(varInit, "initVal", cudaFunctions, "lid",
                                                                                model.getPrecision(), "&initRNG") << std::endl;

                                // Copy this into all delay slots
                                os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)";
                                {
                                    CodeStream::Scope b(os);
                                    os << "dd_" << neuronModelVars[j].first << n.first << "[" << delayedIndex << "] = initVal;" << std::endl;
                                }
                            }
                            // Otherwise, initialise directly into device variable
                            else {
                                os << StandardSubstitutions::initNeuronVariable(varInit, "dd_" + neuronModelVars[j].first + n.first + "[lid]",
                                                                                cudaFunctions, "lid", model.getPrecision(), "&initRNG") << std::endl;
                            }
                        }
                    }
                    // Loop through incoming synaptic populations
                    for(const auto &m : n.second.getMergedInSyn()) {
                        const auto *sg = m.first;

                        // If this synapse group's input variable should be initialised on device
                        if(sg->getInSynVarMode() & VarInit::DEVICE) {
                            os << "dd_inSyn" << sg->getPSModelTargetName() << "[lid] = " << model.scalarExpr(0.0) << ";" << std::endl;
                        }

                        // If dendritic delays are required and these should be initialised on device
                        if(sg->isDendriticDelayRequired() && (sg->getDendriticDelayVarMode() & VarInit::DEVICE)) {
                            os << "for (int i = 0; i < " << sg->getMaxDendriticDelayTimesteps() << "; i++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(i * " + std::to_string(n.second.getNumNeurons()) + ") + lid";
                                os << "dd_denDelay" << sg->getPSModelTargetName() << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
                        }

                        // If postsynaptic model variables should be individual
                        if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                            genDeviceNeuronInitVarCode(os, sg->getPSModel()->getVars(), "lid", sg->getName(), model.getPrecision(),
                                                       [sg](size_t i){ return sg->getPSVarInitialisers()[i]; },
                                                       [sg](size_t i){ return sg->getPSVarMode(i); });
                        }
                    }
                    
                    // Loop through incoming synaptic populations
                    for(const auto *s : n.second.getInSyn()) {
                        genDeviceNeuronInitVarCode(os, s->getWUModel()->getPostVars(), "lid", s->getName(), model.getPrecision(),
                                                   [&s](size_t i){ return s->getWUPostVarInitialisers()[i]; },
                                                   [&s](size_t i){ return s->getWUPostVarMode(i); });
                    }

                    // Loop through outgoing synaptic populations
                    for(const auto *s : n.second.getOutSyn()) {
                        genDeviceNeuronInitVarCode(os, s->getWUModel()->getPreVars(), "lid", s->getName(), model.getPrecision(),
                                                   [&s](size_t i){ return s->getWUPreVarInitialisers()[i]; },
                                                   [&s](size_t i){ return s->getWUPreVarMode(i); });
                    }

                    // Loop through current sources
                    os << "// current source variables" << std::endl;
                    for (auto const *cs : n.second.getCurrentSources()) {
                        genDeviceNeuronInitVarCode(os, cs->getCurrentSourceModel()->getVars(), "lid", cs->getName(), model.getPrecision(),
                                                   [cs](size_t i){ return cs->getVarInitialisers()[i]; },
                                                   [cs](size_t i){ return cs->getVarMode(i); });
                    }
                }
            }
        }


        // Loop through synapse groups
        for(const auto &s : model.getLocalSynapseGroups()) {
            // If this group has dense connectivity with individual synapse variables
            // and it's weight update has variables that require initialising on GPU
            if((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) 
                && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) 
                && s.second.isWUDeviceVarInitRequired())
            {
                os << "// synapse group " << s.first << std::endl;

                PaddedSizeScope p(os, s.second.getTrgNeuronGroup()->getNumNeurons(), initBlkSz, startThread);

                os << "// only do this for existing synapses" << std::endl;
                os << "if (lid < " << s.second.getTrgNeuronGroup()->getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If this post synapse requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(s.second.isWUInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;
                    }

                    // Loop through rows of matrix
                    os << "unsigned int idx = lid;" << std::endl;
                    os << "for(unsigned int i = 0; i < " << s.second.getSrcNeuronGroup()->getNumNeurons() << "; i++)";
                    {
                        CodeStream::Scope b(os);

                        genDeviceWeightUpdateInitVarCode(os, s.second.getWUModel()->getVars(), "idx", "i", "lid", s.first, model.getPrecision(),
                                                         [&s](size_t i){ return s.second.getWUVarInitialisers()[i]; },
                                                         [&s](size_t i){ return s.second.getWUVarMode(i); });
                        // Advance to next row
                        os << "idx += " << s.second.getTrgNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }
                }
            }

            // If we should initialise this synapse group's connectivity on the
            // device and it has a connectivity initialisation snippet
            if(s.second.isDeviceSparseConnectivityInitRequired()) {
                const auto &connectInit = s.second.getConnectivityInitialiser();
                const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
                const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

                os << "// synapse group " << s.first << std::endl;
                PaddedSizeScope p(os, numSrcNeurons, initBlkSz, startThread);

                os << "// only do this for existing synapses" << std::endl;
                os << "if (lid < " << numSrcNeurons << ")";
                {
                    CodeStream::Scope b(os);

                    // If this connectivity requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(::isRNGRequired(connectInit.getSnippet()->getRowBuildCode())) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;
                    }

                    // If the synapse group has bitmask connectivity
                    if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Calculate indices of bits at start and end of row
                        os << "// Calculate indices" << std::endl;
                        const size_t maxSynapses = numSrcNeurons * numTrgNeurons;
                        if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                            os << "const uint64_t rowStartGID = lid * " << numTrgNeurons << "ull;" << std::endl;
                        }
                        else {
                            os << "const unsigned int rowStartGID = lid * " << numTrgNeurons << ";" << std::endl;
                        }

                        // Build function template to set correct bit in bitmask
                        const std::string addSynapseTemplate = "atomicOr(&dd_gp" + s.first + "[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))";

                        // Loop through synapses in row and generate code to initialise sparse connectivity
                        os << "// Build sparse connectivity" << std::endl;
                        os << "for(int prevJ = -1;;)";
                        {
                            CodeStream::Scope b(os);

                            os << StandardSubstitutions::initSparseConnectivity(connectInit, addSynapseTemplate, numTrgNeurons, "lid",
                                                                                cudaFunctions, model.getPrecision(), "&initRNG");
                        }
                    }
                    // Otherwise, if synapse group has ragged connectivity
                    else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                        const std::string rowLength = "dd_rowLength" + s.first + "[lid]";
                        const std::string ind = "dd_ind" + s.first;

                        // Zero row length
                        os << rowLength << " = 0;" << std::endl;

                        // Build function template to increment row length and insert synapse into ind array
                        const std::string addSynapseTemplate = ind + "[(lid * " + std::to_string(s.second.getMaxConnections()) + ") + (" + rowLength + "++)] = $(0)";

                        // Loop through synapses in row
                        os << "// Build sparse connectivity" << std::endl;
                        os << "for(int prevJ = -1;;)";
                        {
                            CodeStream::Scope b(os);

                            os << StandardSubstitutions::initSparseConnectivity(connectInit, addSynapseTemplate, numTrgNeurons, "lid",
                                                                                cudaFunctions, model.getPrecision(), "&initRNG");
                        }
                    }
                    // Otherwise, give an error
                    else {
                        gennError("Only BITMASK and RAGGED format connectivity can be generated using a connectivity initialiser");
                    }
                }
            }
        }
    }   // end initialization kernel code
    os << std::endl;

    // Return maximum of last thread and 1
    // **NOTE** start thread may be zero if only device RNG is being initialised
    return std::max<unsigned int>(1, startThread);
}
//----------------------------------------------------------------------------
unsigned int genInitializeSparseDeviceKernel(unsigned int numStaticInitThreads, CodeStream &os, const NNmodel &model)
{
    // init kernel header
    os << "extern \"C\" __global__ void initializeSparseDevice()";
    
    // initialization kernel code
    unsigned int startThread = 0;
    {
        CodeStream::Scope b(os);

        // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
        // **TODO** check actually required
        os << "__shared__ unsigned int shRowLength[" << initSparseBlkSz << "];" << std::endl;
        os << "__shared__ unsigned int shRowStart[" << initSparseBlkSz + 1 << "];" << std::endl;

        // common variables for all cases
        os << "const unsigned int id = " << initSparseBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Loop through local synapse groups
        for(const auto &s : model.getLocalSynapseGroups()) {
            // If this group requires sparse initialisation
            if(s.second.isDeviceSparseInitRequired()) {
                // Get padded size of group and hence it's end thread
                const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
                const unsigned int paddedSize = (unsigned int)(ceil((double)s.second.getMaxConnections() / (double)initSparseBlkSz) * (double)initSparseBlkSz);
                const unsigned int endThread = startThread + paddedSize;

                if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    os << "// ragged synapse group " << s.first << std::endl;
                }
                else {
                    os << "// yale-format synapse group " << s.first << std::endl;
                }
                if(startThread == 0) {
                    os << "if (id < " << endThread << ")";
                }
                else {
                    os << "if ((id >= " << startThread << ") && (id < " << endThread <<  "))";
                }
                {
                    CodeStream::Scope b(os);
                    if(startThread == 0) {
                        os << "const unsigned int lid = id;" << std::endl;
                    }
                    else {
                        os << "const unsigned int lid = id - " << startThread << ";" << std::endl;
                    }

                    // If this weight update requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(s.second.isWUInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)" << numStaticInitThreads << " + id, &initRNG);" << std::endl;
                    }

                    if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                        os << "unsigned int idx = lid;" << std::endl;
                    }

                    // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
                    const unsigned int numBlocks = (numSrcNeurons + initSparseBlkSz - 1) / initSparseBlkSz;

                    // Loop through blocks
                    os << "for(unsigned int r = 0; r < " << numBlocks << "; r++)";
                    {
                        CodeStream::Scope b(os);

                        // Calculate number of rows to process in this block
                        os << "const unsigned numRowsInBlock = (r == " << numBlocks - 1 << ")";
                        os << " ? " << ((numSrcNeurons - 1) % initSparseBlkSz) + 1;
                        os << " : " << initSparseBlkSz << ";" << std::endl;

                        // Use threads to copy block of sparse structure into shared memory
                        os << "__syncthreads();" << std::endl;
                        os << "if (threadIdx.x < numRowsInBlock)";
                        {
                            CodeStream::Scope b(os);
                            if(s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                os << "const unsigned int rowStart = dd_indInG" << s.first << "[(r * " << initSparseBlkSz << ") + threadIdx.x];" << std::endl;
                                os << "shRowStart[threadIdx.x] = rowStart;" << std::endl;
                                os << "shRowLength[threadIdx.x] = dd_indInG" << s.first << "[(r * " << initSparseBlkSz << ") + threadIdx.x + 1] - rowStart;" << std::endl;
                            }
                            else {
                                os << "shRowLength[threadIdx.x] = dd_rowLength" << s.first << "[(r * " << initSparseBlkSz << ") + threadIdx.x];" << std::endl;
                            }
                        }

                        // If this synapse projection has ragged connectivity initialised on device and has synapse dynamics
                        if(s.second.isDeviceSparseConnectivityInitRequired()
                            && (s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
                            && model.isSynapseGroupDynamicsRequired(s.first))
                        {
                            // Use first thread to generate cumulative sum
                            os << "if (threadIdx.x == 0)";
                            {
                                CodeStream::Scope b(os);

                                // Get index of last row in resultant synapse dynamics structure
                                // **NOTE** if there IS a previous block, it will always have had initSparseBlkSz rows in it
                                os << "unsigned int rowStart = (r == 0) ? 0 : shRowStart[" << initSparseBlkSz << "];" << std::endl;
                                os << "shRowStart[0] = rowStart;" << std::endl;

                                // Loop through rows in block
                                os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                                {
                                    CodeStream::Scope b(os);

                                    // Add this row's length to cumulative sum and write this to this row's end
                                    os << "rowStart += shRowLength[i];" << std::endl;
                                    os << "shRowStart[i + 1] = rowStart;" << std::endl;
                                }

                                // If this is the first thread block and the last block of rows,
                                // write the total cumulative sum to the first entry of the remap structure
                                os << "if(blockIdx.x == 0 && (r == " << numBlocks - 1 << "))";
                                {
                                    CodeStream::Scope b(os);
                                    os << "dd_synRemap" << s.first << "[0] = shRowStart[numRowsInBlock];" << std::endl;
                                }

                            }
                        }

                        os << "__syncthreads();" << std::endl;

                        // Loop through rows
                        os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                        {
                            CodeStream::Scope b(os);

                            // If there is a synapse for this thread to initialise
                            os << "if(lid < shRowLength[i])";
                            {
                                CodeStream::Scope b(os);

                                // If this matrix is sparse calculate index from start index of row and thread id
                                if(s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                    os << "const unsigned idx = shRowStart[i] + lid;" << std::endl;
                                }

                                // If this synapse group has individual variables
                                if(s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                    const std::string preIdx = "((r * " + std::to_string(initSparseBlkSz) + ") + i)";
                                    const std::string postIdx = "dd_ind" + s.first + "[idx]";
                                    genDeviceWeightUpdateInitVarCode(os, s.second.getWUModel()->getVars(), "idx", preIdx, postIdx, s.first, model.getPrecision(),
                                                                     [&s](size_t i){ return s.second.getWUVarInitialisers()[i]; },
                                                                     [&s](size_t i){ return s.second.getWUVarMode(i); });
                                }

                                // If matrix is ragged, connectivity is initialised on device and postsynaptic learning is required
                                if((s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
                                    && s.second.isDeviceSparseConnectivityInitRequired())
                                {
                                    // If postsynaptic learning is required
                                    if(model.isSynapseGroupPostLearningRequired(s.first)) {
                                        CodeStream::Scope b(os);

                                        // Extract index of synapse's postsynaptic target
                                        os << "const unsigned int postIndex = dd_ind" << s.first << "[idx];" << std::endl;

                                        // Atomically increment length of column of connectivity associated with this target
                                        // **NOTE** this returns previous length i.e. where to insert new entry
                                        os << "const unsigned int colLocation = atomicAdd(&dd_colLength" << s.first << "[postIndex], 1);" << std::endl;

                                        // From this calculate index into column-major matrix
                                        os << "const unsigned int colMajorIndex = (postIndex * " << s.second.getMaxSourceConnections() << ") + colLocation;" << std::endl;

                                        // Add remapping entry at this location poining back to row-major index
                                        os << "dd_remap" << s.first << "[colMajorIndex] = idx;" << std::endl;
                                    }

                                    // If synapse dynamics are required, copy idx into syn remap structure
                                    if(model.isSynapseGroupDynamicsRequired(s.first)) {
                                        CodeStream::Scope b(os);
                                        os << "dd_synRemap" << s.first << "[shRowStart[i] + lid + 1] = idx;" << std::endl;
                                    }
                                }
                            }

                            // If matrix is ragged, advance index to next row by adding stride
                            if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                                os << "idx += " << s.second.getMaxConnections() << ";" << std::endl;
                            }
                        }
                    }
                }
                
                // Update start thread
                startThread = endThread;
            }
        }
    }
    
    // Return number of threads used
    return startThread;
}
#endif  // CPU_ONLY
}   // Anonymous namespace

void genInit(const NNmodel &model,      //!< Model description
             const std::string &path,   //!< Path for code generation
             int localHostID)           //!< ID of local host
{
    const std::string runnerName= model.getGeneratedCodePath(path, "init.cc");
    std::ofstream fs;
    fs.open(runnerName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

#ifndef CPU_ONLY
    // If required, insert kernel to initialize neurons and dense matrices
    const unsigned int numInitThreads = model.isDeviceInitRequired(localHostID) ? genInitializeDeviceKernel(os, model, localHostID) : 0;

    // If required, insert kernel to initialize sparse matrices i.e. those that need structure creating between calls to initialize() and init_MODEL()
    const unsigned int numSparseInitThreads = model.isDeviceSparseInitRequired() ? genInitializeSparseDeviceKernel(numInitThreads, os, model) : 0;
    
#endif  // CPU_ONLY

    // ------------------------------------------------------------------------
    // initializing variables
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to (re)set all model variables to their compile-time, homogeneous initial values." << std::endl;
    os << " Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "void initialize()";
    {
        CodeStream::Scope b(os);

        // Extra braces around Windows for loops to fix https://support.microsoft.com/en-us/kb/315481
#ifdef _WIN32
        std::string oB = "{", cB = "}";
#else
        std::string oB = "", cB = "";
#endif // _WIN32

        if (model.isTimingEnabled()) {
            os << "initHost_timer.startTimer();" << std::endl;
        }

        // **NOTE** if we are using GCC on x86_64, bugs in some version of glibc can cause bad performance issues.
        // Best solution involves setting LD_BIND_NOW=1 so check whether this has been applied
        os << "#if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__) && __GLIBC__ == 2 && (__GLIBC_MINOR__ == 23 || __GLIBC_MINOR__ == 24)" << std::endl;
        os << "if(std::getenv(\"LD_BIND_NOW\") == NULL)";
        {
            CodeStream::Scope b(os);
            os << "fprintf(stderr, \"Warning: a bug has been found in glibc 2.23 or glibc 2.24 (https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280) \"" << std::endl;
            os << "                \"which results in poor CPU maths performance. We recommend setting the environment variable LD_BIND_NOW=1 to work around this issue.\\n\");" << std::endl;
        }
        os << "#endif" << std::endl;

#ifdef MPI_ENABLE
        os << "MPI_Init(NULL, NULL);" << std::endl;
        os << "int localHostID;" << std::endl;
        os << "MPI_Comm_rank(MPI_COMM_WORLD, &localHostID);" << std::endl;
        os << "printf(\"MPI initialized - host ID:%d\\n\", localHostID);" << std::endl;
#endif

        // Seed legacy RNG
        if (model.getSeed() == 0) {
            os << "srand((unsigned int) time(NULL));" << std::endl;
        }
        else {
            os << "srand((unsigned int) " << model.getSeed() << ");" << std::endl;
        }

        // If model requires a host RNG
        if(model.isHostRNGRequired()) {
            // If no seed is specified, use system randomness to generate seed sequence
            CodeStream::Scope b(os);
            if (model.getSeed() == 0) {
                os << "uint32_t seedData[std::mt19937::state_size];" << std::endl;
                os << "std::random_device seedSource;" << std::endl;
                {
                    CodeStream::Scope b(os);
                    os << "for(int i = 0; i < std::mt19937::state_size; i++)";
                    {
                        CodeStream::Scope b(os);
                        os << "seedData[i] = seedSource();" << std::endl;
                    }
                }
                os << "std::seed_seq seeds(std::begin(seedData), std::end(seedData));" << std::endl;
            }
            // Otherwise, create a seed sequence from model seed
            // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
            else {
                os << "std::seed_seq seeds{" << model.getSeed() << "};" << std::endl;
            }

            // Seed RNG from seed sequence
            os << "rng.seed(seeds);" << std::endl;
        }
        os << std::endl;

        // INITIALISE REMOTE NEURON SPIKE VARIABLES
        os << "// remote neuron spike variables" << std::endl;
        for(const auto &n : model.getRemoteNeuronGroups()) {
            // If this neuron group has outputs to local host
            if(n.second.hasOutputToHost(localHostID)) {
                genHostInitSpikeCode(os, n.second, false);

                if (n.second.isDelayRequired()) {
                    os << "spkQuePtr" << n.first << " = 0;" << std::endl;
#ifndef CPU_ONLY
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtr" << n.first;
                    os << ", &spkQuePtr" << n.first;
                    os << ", sizeof(unsigned int), 0, cudaMemcpyHostToDevice));" << std::endl;
#endif
                }
            }
        }

        // INITIALISE NEURON VARIABLES
        os << "// neuron variables" << std::endl;
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (n.second.isDelayRequired()) {
                os << "spkQuePtr" << n.first << " = 0;" << std::endl;
#ifndef CPU_ONLY
                os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtr" << n.first;
                os << ", &spkQuePtr" << n.first;
                os << ", sizeof(unsigned int), 0, cudaMemcpyHostToDevice));" << std::endl;
#endif
            }

            // Generate code to intialise spike and spike event variables
            genHostInitSpikeCode(os, n.second, false);
            genHostInitSpikeCode(os, n.second, true);

            if (n.second.isSpikeTimeRequired() && shouldInitOnHost(n.second.getSpikeTimeVarMode())) {
                CodeStream::Scope b(os);
                os << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << "sT" <<  n.first << "[i] = -SCALAR_MAX;" << std::endl;
                }
            }

            auto neuronModelVars = n.second.getNeuronModel()->getVars();
            for (size_t j = 0; j < neuronModelVars.size(); j++) {
                const auto &varInit = n.second.getVarInitialisers()[j];
                const VarMode varMode = n.second.getVarMode(j);

                // If this variable should be initialised on the host and has any initialisation code
                if(shouldInitOnHost(varMode) && !varInit.getSnippet()->getCode().empty()) {
                    CodeStream::Scope b(os);
                    if (n.second.isVarQueueRequired(j)) {
                        os << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)";
                    }
                    else {
                        os << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)";
                    }
                    {
                        CodeStream::Scope b(os);
                        os << StandardSubstitutions::initNeuronVariable(varInit, neuronModelVars[j].first + n.first + "[i]",
                                                                        cpuFunctions, "i", model.getPrecision(), "rng") << std::endl;
                    }
                }
            }

            if (n.second.getNeuronModel()->isPoisson()) {
                CodeStream::Scope b(os);
                os << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)";
                {
                    CodeStream::Scope b(os);
                    os << "seed" << n.first << "[i] = rand();" << std::endl;
                }
            }

            // Loop through current sources injecting into neuron model
            os << "// current source variables" << std::endl;
            for (auto const *cs : n.second.getCurrentSources()) {
                genHostInitNeuronVarCode(os, cs->getCurrentSourceModel()->getVars(), n.second.getNumNeurons(), cs->getName(), model.getPrecision(),
                                         [cs](size_t i){ return cs->getVarInitialisers()[i]; },
                                         [cs](size_t i){ return cs->getVarMode(i); });

            }

            /*if ((model.neuronType[i] == IZHIKEVICH) && (model.getDT() != 1.0)) {
                os << "    fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); " << std::endl;
            }*/

            // Loop through incoming synaptic populations
            for(const auto &m : n.second.getMergedInSyn()) {
                const auto *sg = m.first;

                // If insyn variables should be initialised on the host
                if(shouldInitOnHost(sg->getInSynVarMode())) {
                    CodeStream::Scope b(os);
                    os << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)";
                    {
                        CodeStream::Scope b(os);
                        os << "inSyn" << sg->getPSModelTargetName() << "[i] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    }
                }

                if(sg->isDendriticDelayRequired()) {
                    os << "denDelayPtr" << sg->getPSModelTargetName() << " = 0;" << std::endl;
#ifndef CPU_ONLY
                    os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_denDelayPtr" << sg->getPSModelTargetName();
                    os << ", &denDelayPtr" << sg->getPSModelTargetName();
                    os << ", sizeof(unsigned int), 0, cudaMemcpyHostToDevice));" << std::endl;
#endif

                    // If dendritic delay buffer should be initialised on the host
                    if(shouldInitOnHost(sg->getDendriticDelayVarMode())) {
                        CodeStream::Scope b(os);
                        os << "for (int i = 0; i < " << n.second.getNumNeurons() * sg->getMaxDendriticDelayTimesteps() << "; i++)";
                        {
                            CodeStream::Scope b(os);
                            os << "denDelay" << sg->getPSModelTargetName() << "[i] = " << model.scalarExpr(0.0) << ";" << std::endl;
                        }
                    }
                }

                // If matrix has individual postsynaptic variables
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    genHostInitNeuronVarCode(os, sg->getPSModel()->getVars(), n.second.getNumNeurons(), sg->getName(), model.getPrecision(),
                                             [sg](size_t i){ return sg->getPSVarInitialisers()[i]; },
                                             [sg](size_t i){ return sg->getPSVarMode(i); });
                }
            }

        }
        os << std::endl;

        // INITIALISE SYNAPSE VARIABLES
        os << "// synapse variables" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            const auto *wu = s.second.getWUModel();

            const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

            // Generate code to initialise pre and postsynaptic weight update variables on host if necessary
            genHostInitNeuronVarCode(os, wu->getPreVars(), numSrcNeurons, s.first, model.getPrecision(),
                                     [&s](size_t i){ return s.second.getWUPreVarInitialisers()[i]; },
                                     [&s](size_t i){ return s.second.getWUPreVarMode(i); });

            genHostInitNeuronVarCode(os, wu->getPostVars(), numTrgNeurons, s.first, model.getPrecision(),
                                     [&s](size_t i){ return s.second.getWUPostVarInitialisers()[i]; },
                                     [&s](size_t i){ return s.second.getWUPostVarMode(i); });
            // If we should initialise this synapse group's connectivity on the
            // host and it has a connectivity initialisation snippet
            const auto &connectInit = s.second.getConnectivityInitialiser();
            if(shouldInitOnHost(s.second.getSparseConnectivityVarMode())
                && !connectInit.getSnippet()->getRowBuildCode().empty())
            {
                CodeStream::Scope b(os);

                // If matrix connectivity is ragged
                if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    const std::string rowLength = "C" + s.first + ".rowLength";
                    const std::string ind = "C" + s.first + ".ind";

                    // Zero row lengths
                    os << "memset(" << rowLength << ", 0, " << numSrcNeurons << " * sizeof(unsigned int));" << std::endl;

                    // Loop through source neurons
                    os << "for (int i = 0; i < " << numSrcNeurons << "; i++)";
                    {
                        CodeStream::Scope b(os);

                        // Build function template to increment row length and insert synapse into ind array
                        const std::string addSynapseTemplate = ind + "[(i * " + std::to_string(s.second.getMaxConnections()) + ") + (" + rowLength + "[i]++)] = $(0)";

                        // Loop through synapses in row
                        os << "for(int prevJ = -1;;)";
                        {
                            CodeStream::Scope b(os);

                            os << StandardSubstitutions::initSparseConnectivity(connectInit, addSynapseTemplate, numTrgNeurons, "i",
                                                                                cpuFunctions, model.getPrecision(), "rng");
                        }
                    }

                }
                // Otherwise, if matrix connectivity is a bitmask
                else if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Zero memory before setting sparse bits
                    os << "memset(gp" << s.first << ", 0, " << (numSrcNeurons * numTrgNeurons) / 32 + 1 << " * sizeof(uint32_t));" << std::endl;

                    // Loop through source neurons
                    os << "for (int i = 0; i < " << numSrcNeurons << "; i++)";
                    {
                        // Calculate index of bit at start of this row
                        CodeStream::Scope b(os);
                        os << "const int64_t rowStartGID = i * " << numTrgNeurons << "ll;" << std::endl;

                        // Build function template to set correct bit in bitmask
                        const std::string addSynapseTemplate = "setB(gp" + s.first + "[(rowStartGID + $(0)) / 32], (rowStartGID + $(0)) & 31)";

                        // Loop through synapses in row
                        os << "for(int prevJ = -1;;)";
                        {
                            CodeStream::Scope b(os);

                            os << StandardSubstitutions::initSparseConnectivity(connectInit, addSynapseTemplate, numTrgNeurons, "i",
                                                                                cpuFunctions, model.getPrecision(), "rng");
                        }
                    }
                }
                else {
                    gennError("Only BITMASK and RAGGED format connectivity can be generated using a connectivity initialiser");
                }
            }
            // Otherwise, if this synapse group has connectivity that should be initialised on device
            else if(s.second.isDeviceSparseConnectivityInitRequired()) {
                // If this synapse population has BITMASK connectivity, insert a call to cudaMemset to zero the whole bitmask
                if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t gpSize = ((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
                    os << "cudaMemset(d_gp" << s.first << ", 0, " << gpSize << " * sizeof(uint32_t));" << std::endl;
                }
                // If this synapse population has RAGGED connectivity and has postsynaptic learning, insert a call to cudaMemset to zero column lengths
                else if((s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
                    && model.isSynapseGroupPostLearningRequired(s.first))
                {
                    os << "cudaMemset(d_colLength" << s.first << ", 0, " << s.second.getTrgNeuronGroup()->getNumNeurons() << " * sizeof(unsigned int));" << std::endl;
                }
            }

            // If matrix is dense (i.e. can be initialised here) and each synapse has individual values (i.e. needs initialising at all)
            if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                auto wuVars = wu->getVars();
                for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                    const auto &varInit = s.second.getWUVarInitialisers()[k];
                    const VarMode varMode = s.second.getWUVarMode(k);

                    // If this variable should be initialised on the host and has any initialisation code
                    if(shouldInitOnHost(varMode) && !varInit.getSnippet()->getCode().empty()) {
                        CodeStream::Scope b(os);
                        os << "for (int i = 0; i < " << numSrcNeurons << "; i++)";
                        {
                            CodeStream::Scope b(os);
                            os << "for (int j = 0; j < " << numTrgNeurons << "; j++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string idx = "(i * " + std::to_string(numTrgNeurons) + ") + j";
                                os << StandardSubstitutions::initWeightUpdateVariable(varInit, wuVars[k].first + s.first + "[" + idx + "]",
                                                                                      cpuFunctions, "i", "j", model.getPrecision(), "rng") << std::endl;
                            }
                        }
                    }
                }
            }
        }
        os << std::endl << std::endl;
        if (model.isTimingEnabled()) {
            os << "initHost_timer.stopTimer();" << std::endl;
            os << "initHost_tme+= initHost_timer.getElapsedTime();" << std::endl;
        }

#ifndef CPU_ONLY
        if(!GENN_PREFERENCES::autoInitSparseVars) {
            os << "copyStateToDevice(true);" << std::endl << std::endl;
        }

        // If any init threads were required, perform init kernel launch
        if(numInitThreads > 0) {
            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(initDeviceStart);" << std::endl;
            }

            os << "// perform on-device init" << std::endl;
            os << "dim3 iThreads(" << initBlkSz << ", 1);" << std::endl;
            os << "dim3 iGrid(" << numInitThreads / initBlkSz << ", 1);" << std::endl;
            os << "initializeDevice <<<iGrid, iThreads>>>();" << std::endl;

            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(initDeviceStop);" << std::endl;
                os << "cudaEventSynchronize(initDeviceStop);" << std::endl;
                os << "float tmp;" << std::endl;
                if (!model.getLocalSynapseGroups().empty()) {
                    os << "cudaEventElapsedTime(&tmp, initDeviceStart, initDeviceStop);" << std::endl;
                    os << "initDevice_tme+= tmp/1000.0;" << std::endl;
                }
            }
        }
#endif
    }
    os << std::endl;

     // ------------------------------------------------------------------------
    // initializing sparse arrays
#ifndef CPU_ONLY
    os << "void initializeAllSparseArrays()";
    {
        CodeStream::Scope b(os);
        for(const auto &s : model.getLocalSynapseGroups()) {
            if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                if (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE){
                    os << "initializeSparseArray(C" << s.first << ", ";
                    os << "d_ind" << s.first << ", ";
                    os << "d_indInG" << s.first << ", ";
                    os << s.second.getSrcNeuronGroup()->getNumNeurons() <<");" << std::endl;

                    if (model.isSynapseGroupDynamicsRequired(s.first)) {
                        os << "initializeSparseArrayPreInd(C" << s.first << ", ";
                        os << "d_preInd" << s.first << ");" << std::endl;
                    }
                    if (model.isSynapseGroupPostLearningRequired(s.first)) {
                        os << "initializeSparseArrayRev(C" << s.first << ", ";
                        os << "d_revInd" << s.first << ",";
                        os << "d_revIndInG" << s.first << ",";
                        os << "d_remap" << s.first << ",";
                        os << s.second.getTrgNeuronGroup()->getNumNeurons() << ");" << std::endl;
                    }
                }
                else if(s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    // If sparse connectivity was initialised on host, upload to device
                    // **TODO** this may well be the wrong check i.e. zero copy
                    if(shouldInitOnHost(s.second.getSparseConnectivityVarMode())) {
                        os << "initializeRaggedArray(C" << s.first << ", ";
                        os << "d_ind" << s.first << ", ";
                        os << "d_rowLength" << s.first << ", ";
                        os << s.second.getSrcNeuronGroup()->getNumNeurons() << ");" << std::endl;

                        if (model.isSynapseGroupDynamicsRequired(s.first)) {
                            os << "initializeRaggedArraySynRemap(C" << s.first << ", ";
                            os << "d_synRemap" << s.first << ");" << std::endl;
                        }
                        if (model.isSynapseGroupPostLearningRequired(s.first)) {
                            os << "initializeRaggedArrayRev(C" << s.first << ", ";
                            os << "d_colLength" << s.first << ",";
                            os << "d_remap" << s.first << ",";
                            os << s.second.getTrgNeuronGroup()->getNumNeurons() << ");" << std::endl;
                        }
                    }
                }

                // **LEGACY** if sparse variables aren't automatically initialised - this code used to copy their state
                if (!GENN_PREFERENCES::autoInitSparseVars && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                    // Get number of per-synapse variables to copy (as a string)
                    const std::string count = (s.second.getMatrixType() & SynapseMatrixConnectivity::YALE)
                        ? "C" + s.first + ".connN"
                        : to_string(s.second.getMaxConnections() * s.second.getSrcNeuronGroup()->getNumNeurons());

                    for(const auto &v : s.second.getWUModel()->getVars()) {
                        const VarMode varMode = s.second.getWUVarMode(v.first);

                        // If variable is located on both host and device;
                        // and it isn't zero-copied, copy state variables to device
                        if((varMode & VarLocation::HOST) && (varMode & VarLocation::DEVICE) &&
                            !(varMode & VarLocation::ZERO_COPY))
                        {
                            os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << v.first << s.first << ", ";
                            os << v.first << s.first << ", ";
                            os << "sizeof(" << v.second << ") * " << count << " , cudaMemcpyHostToDevice));" << std::endl;
                        }
                    }
                }
            }
        }
    }
    os << std::endl;
#endif

    // ------------------------------------------------------------------------
    // initialization of variables, e.g. reverse sparse arrays etc.
    // that the user would not want to worry about

    os << "void init" << model.getName() << "()";
    {
        CodeStream::Scope b(os);
        if (model.isTimingEnabled()) {
            os << "sparseInitHost_timer.startTimer();" << std::endl;
        }
        bool anySparse = false;
        for(const auto &s : model.getLocalSynapseGroups()) {
            const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                anySparse = true;

                // If we should initialise sparse connectivity on the host
                if(shouldInitOnHost(s.second.getSparseConnectivityVarMode())) {
                    if (model.isSynapseGroupDynamicsRequired(s.first)) {
                        os << "createPreIndices(" << numSrcNeurons << ", " << numTrgNeurons << ", &C" << s.first << ");" << std::endl;
                    }
                    if (model.isSynapseGroupPostLearningRequired(s.first)) {
                        os << "createPosttoPreArray(" << numSrcNeurons << ", " << numTrgNeurons << ", &C" << s.first << ");" << std::endl;
                    }
                }

                // If synapses in this population have individual variables
                if(s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL && GENN_PREFERENCES::autoInitSparseVars) {
                    auto wuVars = s.second.getWUModel()->getVars();
                    for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                        const auto &varInit = s.second.getWUVarInitialisers()[k];
                        const VarMode varMode = s.second.getWUVarMode(k);

                        // If this variable should be initialised on the host and has any initialisation code
                        if(shouldInitOnHost(varMode) && !varInit.getSnippet()->getCode().empty()) {
                            CodeStream::Scope b(os);
                            os << "for (int i = 0; i < " << numSrcNeurons << "; i++)";
                            {
                                CodeStream::Scope b(os);
                                if(s.second.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                    os << "for (int j = C" << s.first << ".indInG[i]; j < C" << s.first << ".indInG[i + 1]; j++)";
                                    {
                                        CodeStream::Scope b(os);
                                        os << StandardSubstitutions::initWeightUpdateVariable(varInit, wuVars[k].first + s.first + "[j]",
                                                                                              cpuFunctions, "i", "C" + s.first + ".ind[j]",
                                                                                              model.getPrecision(), "rng") << std::endl;
                                    }
                                }
                                else {
                                    os << "for (int j = 0; j < C" << s.first << ".rowLength[i]; j++)";
                                    {
                                        CodeStream::Scope b(os);
                                        const std::string synIndex = "(i * " + std::to_string(s.second.getMaxConnections()) + ") + j";
                                        os << StandardSubstitutions::initWeightUpdateVariable(varInit,
                                                                                              wuVars[k].first + s.first + "[" + synIndex + "]",
                                                                                              cpuFunctions, "i", "C" + s.first + ".ind[" + synIndex + "]", 
                                                                                              model.getPrecision(), "rng") << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }


            }
        }

        os << std::endl << std::endl;
        if (model.isTimingEnabled()) {
            os << "sparseInitHost_timer.stopTimer();" << std::endl;
            os << "sparseInitHost_tme+= sparseInitHost_timer.getElapsedTime();" << std::endl;
        }

#ifndef CPU_ONLY
        if(GENN_PREFERENCES::autoInitSparseVars) {
            os << "copyStateToDevice(true);" << std::endl << std::endl;
        }

        // If there are any sparse synapse projections, initialise them
        if (anySparse) {
            os << "initializeAllSparseArrays();" << std::endl;
        }

        // If there are any sparse initialisation 
        if(numSparseInitThreads > 0) {
            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(sparseInitDeviceStart);" << std::endl;
            }

            os << "// perform on-device sparse init" << std::endl;
            os << "dim3 iThreads(" << initSparseBlkSz << ", 1);" << std::endl;
            os << "dim3 iGrid(" << numSparseInitThreads / initSparseBlkSz << ", 1);" << std::endl;
            os << "initializeSparseDevice <<<iGrid, iThreads>>>();" << std::endl;

            if (model.isTimingEnabled()) {
                os << "cudaEventRecord(sparseInitDeviceStop);" << std::endl;
                os << "cudaEventSynchronize(sparseInitDeviceStop);" << std::endl;
                os << "float tmp;" << std::endl;
                if (!model.getLocalSynapseGroups().empty()) {
                    os << "cudaEventElapsedTime(&tmp, sparseInitDeviceStart, sparseInitDeviceStop);" << std::endl;
                    os << "sparseInitDevice_tme+= tmp/1000.0;" << std::endl;
                }
            }
        }
#else
        USE(anySparse);
#endif
    }
    os << std::endl;
}