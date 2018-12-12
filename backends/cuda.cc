#include "cuda.h"

// Standard C++ includes
#include <algorithm>

// CUDA includes
#include <cuda_runtime.h>

// GeNN includes
#include "codeGenUtils.h"
#include "codeStream.h"
#include "global.h"
#include "modelSpec.h"

// NuGeNN includes
#include "../substitution_stack.h"
#include "../tee_stream.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}
//--------------------------------------------------------------------------
size_t padSize(size_t size, size_t blockSize)
{
    return ceilDivide(size, blockSize) * blockSize;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator::Backends::CUDA
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace Backends
{
CUDA::CUDA(size_t neuronUpdateBlockSize, size_t presynapticUpdateBlockSize, size_t initBlockSize, size_t initSparseBlockSize,
           int localHostID, const Base &hostBackend)
:   m_HostBackend(hostBackend), m_NeuronUpdateBlockSize(neuronUpdateBlockSize), m_PresynapticUpdateBlockSize(presynapticUpdateBlockSize),
    m_InitBlockSize(initBlockSize), m_InitSparseBlockSize(initSparseBlockSize), m_LocalHostID(localHostID), m_ChosenDevice(-1)
{
    // Get number of CUDA devices and reserve memory
    int numDevices;
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&numDevices));
    
    // If any devices were found
    if(numDevices > 0) {
        m_Devices.reserve(numDevices);
        
        std::cout << numDevices << " CUDA device found" << std::endl;
        for (int i = 0; i < numDevices; i++) {
            CHECK_CUDA_ERRORS(cudaSetDevice(i));
            
            // Get device properties and add to devices
            m_Devices.push_back(cudaDeviceProp());
            CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&m_Devices.back(), i));
        }
        
        m_ChosenDevice = 0;
    }
}
//--------------------------------------------------------------------------
void CUDA::genNeuronUpdate(CodeStream &os, const NNmodel &model, NeuronGroupHandler handler) const
{
    os << "#include \"definitions.h\"" << std::endl;

    size_t idStart = 0;
    os << "__global__ void updateNeuronsKernel(";
    for(const auto &p : model.getNeuronKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    for(const auto &p : model.getCurrentSourceKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getTimePrecision() << " t)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_NeuronUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

        Substitutions kernelSubs(cudaFunctions);
        kernelSubs.addVarSubstitution("t", "t");

        // If any neuron groups emit spike events
        if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
            [](const NNmodel::NeuronGroupValueType &n){ return n.second.isSpikeEventRequired(); }))
        {
            os << "__shared__ volatile unsigned int shSpkEvnt[" << m_NeuronUpdateBlockSize << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpkEvnt;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkEvntCount;" << std::endl;
            os << std::endl;
            os << "if (threadIdx.x == 1);";
            {
                CodeStream::Scope b(os);
                os << "shSpkEvntCount = 0;" << std::endl;
            }
            os << std::endl;
        }

        // If any neuron groups emit true spikes
        if(std::any_of(model.getLocalNeuronGroups().cbegin(), model.getLocalNeuronGroups().cend(),
            [](const NNmodel::NeuronGroupValueType &n){ return !n.second.getNeuronModel()->getThresholdConditionCode().empty(); }))
        {
            os << "__shared__ volatile unsigned int shSpk[" << m_NeuronUpdateBlockSize << "];" << std::endl;
            os << "__shared__ volatile unsigned int shPosSpk;" << std::endl;
            os << "__shared__ volatile unsigned int shSpkCount;" << std::endl;
            os << "if (threadIdx.x == 0);";
            {
                CodeStream::Scope b(os);
                os << "shSpkCount = 0;" << std::endl;
            }
            os << std::endl;
        }
            
        os << "__syncthreads();" << std::endl;

        // Parallelise over neuron groups
        genParallelGroup<NeuronGroup>(os, kernelSubs, model.getLocalNeuronGroups(), idStart,
            [this](const NeuronGroup &ng){ return padSize(ng.getNumNeurons(), m_NeuronUpdateBlockSize); },
            [&model, handler, this](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
            {
                // Get name of rng to use for this neuron
                popSubs.addVarSubstitution("rng", "&dd_rng" + ng.getName() + "[" + popSubs.getVarSubstitution("id") + "]");
                
                // Call handler to generate generic neuron code
                handler(os, ng, popSubs);

                os << "__syncthreads();" << std::endl;

                if (ng.isSpikeEventRequired()) {
                    os << "if (threadIdx.x == 1)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkEvntCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << ng.getName();
                            if (ng.isDelayRequired()) {
                                os << "[dd_spkQuePtr" << ng.getName() << "], shSpkEvntCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkEvntCount);" << std::endl;
                            }
                        }
                    } // end if (threadIdx.x == 0)
                    os << "__syncthreads();" << std::endl;
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (shSpkCount > 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "shPosSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << ng.getName();
                            if (ng.isDelayRequired() && ng.isTrueSpikeRequired()) {
                                os << "[dd_spkQuePtr" << ng.getName() << "], shSpkCount);" << std::endl;
                            }
                            else {
                                os << "[0], shSpkCount);" << std::endl;
                            }
                        }
                    } // end if (threadIdx.x == 1)
                    os << "__syncthreads();" << std::endl;
                }

                const std::string queueOffset = ng.isDelayRequired() ? "writeDelayOffset + " : "";
                if (ng.isSpikeEventRequired()) {
                    os << "if (threadIdx.x < shSpkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpkEvnt" << ng.getName() << "[" << queueOffset << "shPosSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
                    }
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    const std::string queueOffsetTrueSpk = ng.isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (threadIdx.x < shSpkCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpk" << ng.getName() << "[" << queueOffsetTrueSpk << "shPosSpk + threadIdx.x] = shSpk[threadIdx.x];" << std::endl;
                        if (ng.isSpikeTimeRequired()) {
                            os << "dd_sT" << ng.getName() << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << std::endl;
                        }
                    }
                }
            }
        );
    }

    os << "void updateNeurons(float t)";
    {
        CodeStream::Scope b(os);
        if(idStart > 0) {
            const size_t gridSize = ceilDivide(idStart, m_NeuronUpdateBlockSize);
            os << "const dim3 threads(" << neuronBlkSz << ", 1);" << std::endl;
            if (gridSize < getChosenCUDADevice().maxGridSize[1]) {
                os << "const dim3 grid(" << gridSize << ", 1);" << std::endl;
            }
            else {
                // **TODO** this needs to be implemented in genParallelGroup
                assert(false);
                const size_t squareGridSize = (size_t)std::ceil(std::sqrt(gridSize));
                os << "const dim3 grid(" << squareGridSize << ", "<< squareGridSize <<");" << std::endl;
            }

            // Launch kernel
            os << "updateNeuronsKernel<<<grid, threads>>>(";
            for(const auto &p : model.getNeuronKernelParameters()) {
                os << p.first << ", ";
            }
            for(const auto &p : model.getCurrentSourceKernelParameters()) {
                os << p.first << ", ";
            }
            os << "t);" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void CUDA::genSynapseUpdate(CodeStream &os, const NNmodel &model,
                            SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getPrecision() << " t)" << std::endl; // end of synapse kernel header
    {
        CodeStream::Scope b(os);
        
        Substitutions kernelSubs(cudaFunctions);
        kernelSubs.addVarSubstitution("t", "t");

        os << "const unsigned int id = " << m_PresynapticUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

        // We need shLg if any synapse groups accumulate into shared memory
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [this](const NNmodel::SynapseGroupValueType &s){ return this->shouldAccumulateInSharedMemory(s.second); }))
        {
            os << "__shared__ " << model.getPrecision() << " shLg[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [&model](const NNmodel::SynapseGroupValueType &s)
            { 
                return (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first));
            }))
        {
            os << "__shared__ unsigned int shSpk[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [](const NNmodel::SynapseGroupValueType &s){ return (s.second.isSpikeEventRequired()); }))
        {
            os << "__shared__ unsigned int shSpkEvnt[" << m_PresynapticUpdateBlockSize << "];" << std::endl;
        }
        
        // Parallelise over synapse groups
        size_t idStart = 0;
        genParallelGroup<SynapseGroup>(os, kernelSubs, model.getLocalSynapseGroups(), idStart,
            [this](const SynapseGroup &sg){ return getPresynapticUpdateKernelSize(sg); },
            [wumThreshHandler, wumSimHandler, &model, this](CodeStream &os, const SynapseGroup &sg, const Substitutions &popSubs)
            {
                if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int delaySlot = (dd_spkQuePtr" <<sg.getSrcNeuronGroup()->getName();
                    os << " + " << (sg.getSrcNeuronGroup()->getNumDelaySlots() - sg.getDelaySteps());
                    os << ") % " << sg.getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                }

                // If we are going to accumulate postsynaptic input into a register, copy current value into register from global memory
                if (shouldAccumulateInLinSyn(sg)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << model.getPrecision() << " linSyn;" << std::endl;
                    os << "if(" << popSubs.getVarSubstitution("id") << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "linSyn = dd_inSyn" << sg.getName() << "[" << popSubs.getVarSubstitution("id") << "];" << std::endl;
                    }
                }
                // Otherwise, if we are going to accumulate into shared memory, copy current value into correct array index
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(sg)) {
                    os << "if(threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "shLg[threadIdx.x] = dd_inSyn" << sg.getName() << "[threadIdx.x];"<< std::endl;
                    }
                    os << "__syncthreads();" << std::endl;
                }

                if (sg.isSpikeEventRequired()) {
                    os << "const unsigned int spkCntEvent = dd_glbSpkCntEvnt" << sg.getSrcNeuronGroup()->getName();
                    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[delaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                }

                if (sg.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(sg.getName())) {
                    os << "const unsigned int spkCnt = dd_glbSpkCnt" << sg.getSrcNeuronGroup()->getName();
                    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[delaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                }
            
                // If spike events should be processed
                if (sg.isSpikeEventRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                        assert(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
                        genPresynapticUpdatePreSpan(os, model, sg, popSubs, false,
                                                    wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdatePostSpan(os, model, sg, popSubs, false,
                                                     wumThreshHandler, wumSimHandler);
                    }
                }

                // If true spikes should be processed
                if (sg.isTrueSpikeRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                        assert(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
                        genPresynapticUpdatePreSpan(os, model, sg, popSubs, true,
                                                    wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdatePostSpan(os, model, sg, popSubs, true,
                                                     wumThreshHandler, wumSimHandler);
                    }
                }
                
                os << std::endl;

                // If we have been accumulating into a register, write value back to global memory
                if (shouldAccumulateInLinSyn(sg)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << "if (" << popSubs.getVarSubstitution("id") << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << sg.getName() << "[" << popSubs.getVarSubstitution("id") << "] = linSyn;" << std::endl;
                    }
                }
                // Otherwise, if we have been accumulating into shared memory, write value back to global memory
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(sg)) {
                    os << "__syncthreads();" << std::endl;
                    os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << sg.getName() << "[threadIdx.x] = shLg[threadIdx.x];"<< std::endl;
                    }
                }
            }
        );
    }
}

//--------------------------------------------------------------------------
void CUDA::genInit(CodeStream &os, const NNmodel &model,
                   NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                   SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                   SynapseGroupHandler sgSparseInitHandler) const
{
    // Create codestreams to generate different sections of runner
    /*std::stringstream initHostStream;
    std::stringstream initDeviceStream;
    CodeStream initHost(initHostStream);
    CodeStream initDevice(initDeviceStream);*/
    CodeStream &initDevice = os;
    
    // init kernel header
    initDevice << "extern \"C\" __global__ void initializeDevice(";
    const auto &params = model.getInitKernelParameters();
    for(auto p = params.cbegin(); p != params.cend(); p++) {
        initDevice << p->second << " " << p->first;
        if (std::next(p) != params.cend()) {
            initDevice  << ", ";
        }
    }
    initDevice << ")";

    // initialization kernel code
    size_t idStart = 0;
    {
        Substitutions kernelSubs(cudaFunctions);

        // common variables for all cases
        CodeStream::Scope b(initDevice);

        initDevice << "const unsigned int id = " << m_InitBlockSize << " * blockIdx.x + threadIdx.x;" << std::endl;

        // If RNG is required
        // **TODO** move into seperate kernel
        if(model.isDeviceRNGRequired()) {
            initDevice << "// Initialise global GPU RNG" << std::endl;
            initDevice << "if(id == 0)";
            {
                CodeStream::Scope b(initDevice);
                initDevice << "curand_init(" << model.getSeed() << ", 0, 0, &dd_rng[0]);" << std::endl;
            }
        }

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Remote neuron groups" << std::endl;
        genParallelGroup<NeuronGroup>(initDevice, kernelSubs, model.getRemoteNeuronGroups(), idStart,
            [this](const NeuronGroup &ng){ return padSize(ng.getNumNeurons(), m_InitBlockSize); },
            [this](const NeuronGroup &ng){ return (ng.hasOutputToHost(m_LocalHostID) && ng.getSpikeVarMode() & VarInit::DEVICE); },
            [this, remoteNGHandler](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    remoteNGHandler(os, ng, popSubs);
                }
            });
        os << std::endl;
   
        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Local neuron groups" << std::endl;
        genParallelGroup<NeuronGroup>(os, kernelSubs, model.getLocalNeuronGroups(), idStart,
            [this](const NeuronGroup &ng){ return padSize(ng.getNumNeurons(), m_InitBlockSize); },
            [this](const NeuronGroup &ng){ return ng.isDeviceInitRequired(); },
            [this, &model, localNGHandler](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
            {
                os << "// only do this for existing neurons" << std::endl;
                os << "if(" << popSubs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);
                    // If this neuron is going to require a simulation RNG, initialise one using GLOBALthread id for sequence
                    if(ng.isSimRNGRequired()) {
                        os << "curand_init(" << model.getSeed() << ", id, 0, &dd_rng" << ng.getName() << "[" << popSubs.getVarSubstitution("id") << "]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    // **NOTE** not LOCAL id
                    if(ng.isInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "initRNG");
                    }

                    localNGHandler(os, ng, popSubs);
                }
            });
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with dense connectivity" << std::endl;
        genParallelGroup<SynapseGroup>(os, kernelSubs, model.getLocalSynapseGroups(), idStart, 
            [this](const SynapseGroup &sg){ return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_InitBlockSize); },
            [](const SynapseGroup &sg){ return (sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && sg.isWUDeviceVarInitRequired(); },
            [sgDenseInitHandler](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
            {
                // If this post synapse requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(sg.isWUInitRNGRequired(VarInit::DEVICE)) {
                    os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                    os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                    // Add substitution for RNG
                    popSubs.addVarSubstitution("rng", "initRNG");
                }

                popSubs.addVarSubstitution("id_post", popSubs.getVarSubstitution("id"));
                sgDenseInitHandler(os, sg, popSubs);
            });
        os << std::endl;

        os << "// ------------------------------------------------------------------------" << std::endl;
        os << "// Synapse groups with sparse connectivity" << std::endl;
        genParallelGroup<SynapseGroup>(os, kernelSubs, model.getLocalSynapseGroups(), idStart,
            [this](const SynapseGroup &sg){ return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_InitBlockSize); },
            [](const SynapseGroup &sg){ return sg.isDeviceSparseConnectivityInitRequired(); },
            [sgSparseConnectHandler](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
            {
                const size_t numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                const size_t numTrgNeurons = sg.getTrgNeuronGroup()->getNumNeurons();

                // If this connectivity requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(::isRNGRequired(sg.getConnectivityInitialiser().getSnippet()->getRowBuildCode())) {
                    os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                    os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                    // Add substitution for RNG
                    popSubs.addVarSubstitution("rng", "initRNG");
                }

                // If the synapse group has bitmask connectivity
                if(sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    // Calculate indices of bits at start and end of row
                    os << "// Calculate indices" << std::endl;
                    const size_t maxSynapses = numSrcNeurons * numTrgNeurons;
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t rowStartGID = " << popSubs.getVarSubstitution("id") << " * " << numTrgNeurons << "ull;" << std::endl;
                    }
                    else {
                        os << "const unsigned int rowStartGID = " << popSubs.getVarSubstitution("id") << " * " << numTrgNeurons << ";" << std::endl;
                    }

                    // Build function template to set correct bit in bitmask
                    popSubs.addFuncSubstitution("addSynapse", 1,
                                                "atomicOr(&dd_gp" + sg.getName() + "[(rowStartGID + $(0)) / 32], 0x80000000 >> ((rowStartGID + $(0)) & 31))");
                }
                // Otherwise, if synapse group has ragged connectivity
                else if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    const std::string rowLength = "dd_rowLength" + sg.getName() + "[" + popSubs.getVarSubstitution("id") + "]";
                    const std::string ind = "dd_ind" + sg.getName();

                    // Zero row length
                    os << rowLength << " = 0;" << std::endl;

                    // Build function template to increment row length and insert synapse into ind array
                    popSubs.addFuncSubstitution("addSynapse", 1,
                                                ind + "[(" + popSubs.getVarSubstitution("id") + " * " + std::to_string(sg.getMaxConnections()) + ") + (" + rowLength + "++)] = $(0)");
                }
                else {
                    assert(false);
                }

                sgSparseConnectHandler(os, sg, popSubs);
            });
        os << std::endl;
    }
    const unsigned int numStaticInitThreads = idStart;

    // initialization kernel code
    initDevice << "extern \"C\" __global__ void initializeSparseDevice()";
    {
        CodeStream::Scope b(initDevice);

        // common variables for all cases
        Substitutions kernelSubs(cudaFunctions);

        initDevice << "const unsigned int id = " << m_InitSparseBlockSize << " * blockIdx.x + threadIdx.x;" << std::endl;

        // Shared memory array so row lengths don't have to be read by EVERY postsynaptic thread
        // **TODO** check actually required
        initDevice << "__shared__ unsigned int shRowLength[" << m_InitSparseBlockSize << "];" << std::endl;
        initDevice << "__shared__ unsigned int shRowStart[" << m_InitSparseBlockSize + 1 << "];" << std::endl;

        // Initialise weight update variables for synapse groups with dense connectivity
        genParallelGroup<SynapseGroup>(os, kernelSubs, model.getLocalSynapseGroups(), idStart,
            [this](const SynapseGroup &sg){ return padSize(sg.getMaxConnections(), m_InitSparseBlockSize); },
            [](const SynapseGroup &sg){ return sg.isDeviceSparseInitRequired(); },
            [this, &model, sgSparseInitHandler, numStaticInitThreads](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
            {
                // If this post synapse requires an RNG for initialisation,
                // make copy of global phillox RNG and skip ahead by thread id
                // **NOTE** not LOCAL id
                if(sg.isWUInitRNGRequired(VarInit::DEVICE)) {
                    os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                    os << "skipahead_sequence((unsigned long long)" << numStaticInitThreads << " + id, &initRNG);" << std::endl;
                }

                os << "unsigned int idx = " << popSubs.getVarSubstitution("id") << ";" << std::endl;

                // Calculate how many blocks rows need to be processed in (in order to store row lengths in shared memory)
                const unsigned int numSrcNeurons = sg.getSrcNeuronGroup()->getNumNeurons();
                const unsigned int numBlocks = padSize(numSrcNeurons, m_InitSparseBlockSize);

                // Loop through blocks
                os << "for(unsigned int r = 0; r < " << numBlocks << "; r++)";
                {
                    CodeStream::Scope b(os);

                    // Calculate number of rows to process in this block
                    os << "const unsigned numRowsInBlock = (r == " << numBlocks - 1 << ")";
                    os << " ? " << ((numSrcNeurons - 1) % m_InitSparseBlockSize) + 1;
                    os << " : " << m_InitSparseBlockSize << ";" << std::endl;

                    // Use threads to copy block of sparse structure into shared memory
                    os << "__syncthreads();" << std::endl;
                    os << "if (threadIdx.x < numRowsInBlock)";
                    {
                        CodeStream::Scope b(os);
                        os << "shRowLength[threadIdx.x] = dd_rowLength" << sg.getName() << "[(r * " << m_InitSparseBlockSize << ") + threadIdx.x];" << std::endl;
                    }

                    // If this synapse projection has ragged connectivity initialised on device and has synapse dynamics
                    if(sg.isDeviceSparseConnectivityInitRequired()
                        && (sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
                        && model.isSynapseGroupDynamicsRequired(sg.getName()))
                    {
                        // Use first thread to generate cumulative sum
                        os << "if (threadIdx.x == 0)";
                        {
                            CodeStream::Scope b(os);

                            // Get index of last row in resultant synapse dynamics structure
                            // **NOTE** if there IS a previous block, it will always have had initSparseBlkSz rows in it
                            os << "unsigned int rowStart = (r == 0) ? 0 : shRowStart[" << m_InitSparseBlockSize << "];" << std::endl;
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
                                os << "dd_synRemap" << sg.getName() << "[0] = shRowStart[numRowsInBlock];" << std::endl;
                            }

                        }
                    }

                    os << "__syncthreads();" << std::endl;

                    // Loop through rows
                    os << "for(unsigned int i = 0; i < numRowsInBlock; i++)";
                    {
                        CodeStream::Scope b(os);

                        // If there is a synapse for this thread to initialise
                        os << "if(" << popSubs.getVarSubstitution("id") << " < shRowLength[i])";
                        {
                            CodeStream::Scope b(os);

                            popSubs.addVarSubstitution("id_syn", "idx");
                            popSubs.addVarSubstitution("id_pre", "((r * " + std::to_string(m_InitSparseBlockSize) + ") + i)");
                            popSubs.addVarSubstitution("id_post", "dd_ind" + sg.getName() + "[idx]");
                            sgSparseInitHandler(os, sg, popSubs);

                            // If matrix is ragged, connectivity is initialised on device and postsynaptic learning is required
                            if((sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
                                && sg.isDeviceSparseConnectivityInitRequired())
                            {
                                // If postsynaptic learning is required
                                if(model.isSynapseGroupPostLearningRequired(sg.getName())) {
                                    CodeStream::Scope b(os);

                                    // Extract index of synapse's postsynaptic target
                                    os << "const unsigned int postIndex = dd_ind" << sg.getName() << "[idx];" << std::endl;

                                    // Atomically increment length of column of connectivity associated with this target
                                    // **NOTE** this returns previous length i.e. where to insert new entry
                                    os << "const unsigned int colLocation = atomicAdd(&dd_colLength" << sg.getName() << "[postIndex], 1);" << std::endl;

                                    // From this calculate index into column-major matrix
                                    os << "const unsigned int colMajorIndex = (postIndex * " << sg.getMaxSourceConnections() << ") + colLocation;" << std::endl;

                                    // Add remapping entry at this location poining back to row-major index
                                    os << "dd_remap" << sg.getName() << "[colMajorIndex] = idx;" << std::endl;
                                }

                                // If synapse dynamics are required, copy idx into syn remap structure
                                if(model.isSynapseGroupDynamicsRequired(sg.getName())) {
                                    CodeStream::Scope b(os);
                                    os << "dd_synRemap" << sg.getName() << "[shRowStart[i] + lid + 1] = idx;" << std::endl;
                                }
                            }
                        }

                        // If matrix is ragged, advance index to next row by adding stride
                        os << "idx += " << sg.getMaxConnections() << ";" << std::endl;
                    }
                }
            });
    }
}
//--------------------------------------------------------------------------
void CUDA::genRunnerPreamble(CodeStream &os) const
{
    // Allow host backend to generate any of it's own preamble
    m_HostBackend.genRunnerPreamble(os);

    // Include header files
    os << "#include <string>" << std::endl;
    os << "#include <stdexcept>" << std::endl;

    os << std::endl;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << std::endl;
    os << "    cudaError_t error = call;\\" << std::endl;
    os << "    if (error != cudaSuccess) {\\" << std::endl;
    os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": cuda error \" + std::to_string(error) + \": \" + cudaGetErrorString(error));\\" << std::endl;
    os << "    }\\" << std::endl;
    os << "}" << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for allocating memory blocks on the GPU device" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)";
    {
        CodeStream::Scope b(os);
        os << "void *devptr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    }
    os << std::endl;

    os << "// ------------------------------------------------------------------------" << std::endl;
    os << "// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol" << std::endl;
    os << std::endl;
    os << "template<class T>" << std::endl;
    os << "void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)";
    {
        CodeStream::Scope b(os);
        os << "CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));" << std::endl;
        os << "void *devSymbolPtr;" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));" << std::endl;
        os << "CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));" << std::endl;
    }
    os << std::endl;
}
//--------------------------------------------------------------------------
void CUDA::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    if(mode & VarLocation::HOST) {
        m_HostBackend.genVariableDefinition(os, type, name, mode);
    }
    if(mode & VarLocation::DEVICE) {
        os << getVarExportPrefix() << " " << type << " d_" << name << ";" << std::endl;
        os << getVarExportPrefix() << " __device__ " << type << " dd_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CUDA::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    if(mode & VarLocation::HOST) {
        m_HostBackend.genVariableImplementation(os, type, name, mode);
    }
    if(mode & VarLocation::DEVICE) {
        os << type << " d_" << name << ";" << std::endl;
        os << "__device__ " << type << " dd_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CUDA::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const
{
    if(mode & VarLocation::HOST) {
        // **NOTE** because we want out memory to be pinned for faster copying to GPU, DON'T use host code generator
        const char *flags = (mode & VarLocation::ZERO_COPY) ? "cudaHostAllocMapped" : "cudaHostAllocPortable";
        os << "cudaHostAlloc(&" << name << ", " << count << " * sizeof(" << type << "), " << flags << ");" << std::endl;
    }

    // If variable is present on device at all
    if(mode & VarLocation::DEVICE) {
        // Insert call to correct helper depending on whether variable should be allocated in zero-copy mode or not
        if(mode & VarLocation::ZERO_COPY) {
            os << "deviceZeroCopy(" << name << ", &d_" << name << ", dd_" << name << ");" << std::endl;
        }
        else {
            os << "deviceMemAllocate(&d_" << name << ", dd_" << name << ", " << count << " * sizeof(" << type << "));" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void CUDA::genVariableFree(CodeStream &os, const std::string &name, VarMode mode) const
{
    // **NOTE** because we pinned the variable we need to free it with cudaFreeHost rather than use the host code generator
    if(mode & VarLocation::HOST) {
        os << "CHECK_CUDA_ERRORS(cudaFreeHost(" << name << "));" << std::endl;
    }

    // If this variable wasn't allocated in zero-copy mode, free it
    if(mode & VarLocation::DEVICE) {
        os << "CHECK_CUDA_ERRORS(cudaFree(d_" << name << "));" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CUDA::genPopVariableInit(CodeStream &os, VarMode mode, const Substitutions &kernelSubs, Handler handler) const
{
    Substitutions varSubs(&kernelSubs);

    // If variable should be initialised on device
    if(mode & VarInit::DEVICE) {
        os << "if(" << varSubs.getVarSubstitution("id") << " == 0)";
        {
            CodeStream::Scope b(os);
            handler(os, varSubs);
        }
    }
}
//--------------------------------------------------------------------------
void CUDA::genVariableInit(CodeStream &os, VarMode mode, size_t, const std::string &countVarName,
                           const Substitutions &kernelSubs, Handler handler) const
{
    // Variable should already be provided via parallelism
    assert(kernelSubs.hasVarSubstitution(countVarName));

    // If variable should be initialised on device
    if(mode & VarInit::DEVICE) {
        Substitutions varSubs(&kernelSubs);
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void CUDA::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs.getVarSubstitution("id") << ";" << std::endl;
}
//--------------------------------------------------------------------------
void CUDA::genPresynapticUpdatePreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                       SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "evnt";
    const auto *wu = sg.getWUModel();

    os << "if (" << popSubs.getVarSubstitution("id") << " < " ;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[delaySlot])";
    }
    else {
        os << "dd_glbSpkCnt" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[0])";
    }
    {
        CodeStream::Scope b(os);

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
        }

        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "const unsigned int preInd = dd_glbSpk"  << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[(delaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + " << popSubs.getVarSubstitution("id") << "];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = dd_glbSpk"  << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[" << popSubs.getVarSubstitution("id") << "];" << std::endl;
        }

        if(sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << "unsigned int synAddress = dd_indInG" << sg.getName() << "[preInd];" << std::endl;
            os << "const unsigned int npost = dd_indInG" << sg.getName() << "[preInd + 1] - prePos;" << std::endl;
        }
        else if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            os << "unsigned int synAddress = preInd * " << std::to_string(sg.getMaxConnections()) << ";" << std::endl;
            os << "const unsigned int npost = dd_rowLength" << sg.getName() << "[preInd];" << std::endl;
        }

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << "if(";
 
            Substitutions threshSubs(&popSubs);
            threshSubs.addVarSubstitution("id_pre", "preInd");
            threshSubs.addVarSubstitution("id_post", "i");

            // Generate weight update threshold condition
            wumThreshHandler(os, sg, threshSubs);
            
            // end code substitutions ----
            os << ")";

            os << CodeStream::OB(130);
        }

        os << "for(unsigned int i = 0; i < npost; i++, synAddress++)";
        {
            CodeStream::Scope b(os);

            // **TODO** pretty sure __ldg will boost performance here - basically will bring whole row into cache
            os << "const unsigned int ipost = dd_ind" <<  sg.getName() << "[prePos];" << std::endl;

            // Code substitutions ----------------------------------------------------------------------------------
            string wCode = trueSpike ? wu->getSimCode() : wu->getEventCode();

            Substitutions synSubs(&popSubs);
            synSubs.addVarSubstitution("id_pre", "preInd");
            synSubs.addVarSubstitution("id_post", "ipost");
            synSubs.addVarSubstitution("id_syn", "synAddress");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.isDendriticDelayRequired()) {
                synSubs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
            }
            // Otherwise
            else {
                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(shouldAccumulateInSharedMemory(sg)) {
                    synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                }
            }

            wumSimHandler(os, sg, synSubs);
        }

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//--------------------------------------------------------------------------
void CUDA::genPresynapticUpdatePostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                        SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler) const
{
     // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "evnt";
    const auto *wu = sg.getWUModel();
    os << "for (unsigned int r = 0; r < numSpikeSubsets" << eventSuffix << "; r++)";
    {
        CodeStream::Scope b(os);
        os << "const unsigned int lmax = (r == numSpikeSubsets" << eventSuffix << " - 1) ? ((lscnt" << eventSuffix << " - 1) % " << m_PresynapticUpdateBlockSize << ") + 1 : " << m_PresynapticUpdateBlockSize << ";" << std::endl;
        
        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < lmax)";
        {
            CodeStream::Scope b(os);
            const string offsetTrueSpkPost = (sg.getTrgNeuronGroup()->isTrueSpikeRequired() && sg.getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
            os << "const unsigned int spk = dd_glbSpk" << eventSuffix << sg.getSrcNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << m_PresynapticUpdateBlockSize << ") + threadIdx.x];" << std::endl;
            os << "shSpk" << eventSuffix << "[threadIdx.x] = spk;" << std::endl;
            if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                os << "shRowLength" << eventSuffix << "[threadIdx.x] = dd_rowLength" << sg.getName() << "[spk];" << std::endl;
            }
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (unsigned int j = 0; j < lmax; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << popSubs.getVarSubstitution("id") << " < " << sg.getMaxConnections() << ")";
            {
                CodeStream::Scope b(os);
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t maxSynapses = (size_t)sg.getTrgNeuronGroup()->getNumNeurons() * (size_t)sg.getSrcNeuronGroup()->getNumNeurons();
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + " << popSubs.getVarSubstitution("id") << ");" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << popSubs.getVarSubstitution("id") << ");" << std::endl;
                    }
                }

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
                    os << "if(";
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        os << "(B(dd_gp" << sg.getName() << "[gid / 32], gid & 31)) && ";
                    }

                    Substitutions threshSubs(&popSubs);
                    threshSubs.addVarSubstitution("id_pre", "preInd");
                    threshSubs.addVarSubstitution("id_post", "ipost");
                   
                    // Generate weight update threshold condition
                    wumThreshHandler(os, sg, threshSubs);

                    // end code substitutions ----
                    os << ")";
                    os << CodeStream::OB(130);
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(dd_gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(135);
                }


                if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                        os << "unsigned int synAddress = dd_indInG" << sg.getName() << "[shSpk" << eventSuffix << "[j]];" << std::endl;
                        os << "const unsigned int npost = dd_indInG" << sg.getName() << "[shSpk" << eventSuffix << "[j] + 1] - synAddress;" << std::endl;
                    }
                    else {
                        os << "unsigned int synAddress = shSpk" << eventSuffix << "[j] * " << to_string(sg.getMaxConnections()) << ";" << std::endl;
                        os << "const unsigned int npost = shRowLength" << eventSuffix << "[j];" << std::endl;
                    }

                    os << "if (" << popSubs.getVarSubstitution("id") << " < npost)" << CodeStream::OB(140);
                    os << "synAddress += " << popSubs.getVarSubstitution("id") << ";" << std::endl;
                    os << "const unsigned int ipost = dd_ind" << sg.getName() << "[synAddress];" << std::endl;
                }
                else { // DENSE
                    os << "ipost = " << popSubs.getVarSubstitution("id") << ";" << std::endl;
                }

                Substitutions synSubs(&popSubs);
                synSubs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                synSubs.addVarSubstitution("id_post", "ipost");
                synSubs.addVarSubstitution("id_syn", "synAddress");

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.isDendriticDelayRequired()) {
                    synSubs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
                }
                // Otherwise
                else {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                        if (shouldAccumulateInSharedMemory(sg)) {
                            synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                        }
                        else {
                            synSubs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                        }
                    }
                    else {
                        synSubs.addFuncSubstitution("addToInSyn", 1, "linSyn += $(0)");
                    }
                }

                wumSimHandler(os, sg, synSubs);

                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if (!trueSpike && sg.isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
size_t CUDA::getPresynapticUpdateKernelSize(const SynapseGroup &sg) const
{
     if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if (sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
            return padSize(sg.getSrcNeuronGroup()->getNumNeurons(), m_PresynapticUpdateBlockSize);
        }
        else {
            // paddedSize is the lowest multiple of blockSize >= maxConn[i]
            // **TODO** integer ceil trick
            return padSize(sg.getMaxConnections(), m_PresynapticUpdateBlockSize);
        }
    }
    else {
        // paddedSize is the lowest multiple of blockSize >= neuronN[synapseTarget[i]]
        return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_PresynapticUpdateBlockSize);
    }
}
//--------------------------------------------------------------------------
bool CUDA::shouldAccumulateInLinSyn(const SynapseGroup &sg) const
{
    // We should accumulate each postsynaptic neuron's input in a register if matrix is dense or bitfield (where each thread represents an individual neuron)
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK));
}
//--------------------------------------------------------------------------
bool CUDA::shouldAccumulateInSharedMemory(const SynapseGroup &sg) const
{
    // If parallelism is presynaptic i.e. atomics are required and device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC && getChosenCUDADevice().major < 5) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.getTrgNeuronGroup()->getNumNeurons() <= m_PresynapticUpdateBlockSize);
    }
}
//--------------------------------------------------------------------------
std::string CUDA::getFloatAtomicAdd(const std::string &ftype) const
{
    USE(ftype);
    int version;
    cudaRuntimeGetVersion(&version);
    if (((getChosenCUDADevice().major < 2) && (ftype == "float"))
        || (((getChosenCUDADevice().major < 6) || (version < 8000)) && (ftype == "double"))) {
        return "atomicAddSW";
    }
    else {
        return "atomicAdd";
    }
}
}   // namespace Backends
}   // namespace CodeGenerator
