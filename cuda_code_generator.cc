#include "cuda_code_generator.h"

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
#include "substitution_stack.h"
#include "tee_stream.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
size_t padSize(size_t size, size_t blockSize)
{
    return ((size + blockSize - 1) / blockSize) * blockSize;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CUDA::CodeGenerator
//--------------------------------------------------------------------------
namespace CUDA
{
CodeGenerator::CodeGenerator(size_t neuronUpdateBlockSize, size_t presynapticUpdateBlockSize, size_t initBlockSize, int localHostID,
                             const ::CodeGenerator::Base &hostCodeGenerator)
:   m_HostCodeGenerator(hostCodeGenerator), m_NeuronUpdateBlockSize(neuronUpdateBlockSize), m_PresynapticUpdateBlockSize(presynapticUpdateBlockSize), 
    m_InitBlockSize(initBlockSize), m_LocalHostID(localHostID), m_ChosenDevice(-1)
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
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, 
                                          NeuronGroupHandler handler) const
{
    os << "extern \"C\" __global__ void calcNeurons(float time)" << std::endl;
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
        size_t idStart = 0;
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
                        os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << ng.getName();
                        if (ng.isDelayRequired()) {
                            os << "[dd_spkQuePtr" << ng.getName() << "], spkEvntCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkEvntCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 0)
                    os << "__syncthreads();" << std::endl;
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << ng.getName();
                        if (ng.isDelayRequired() && ng.isTrueSpikeRequired()) {
                            os << "[dd_spkQuePtr" << ng.getName() << "], spkCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 1)
                    os << "__syncthreads();" << std::endl;
                }

                const std::string queueOffset = ng.isDelayRequired() ? "writeDelayOffset + " : "";
                if (ng.isSpikeEventRequired()) {
                    os << "if (threadIdx.x < spkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpkEvnt" << ng.getName() << "[" << queueOffset << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
                    }   // end if (threadIdx.x < spkEvntCount)
                }

                if (!ng.getNeuronModel()->getThresholdConditionCode().empty()) {
                    const std::string queueOffsetTrueSpk = ng.isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (threadIdx.x < spkCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpk" << ng.getName() << "[" << queueOffsetTrueSpk << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << std::endl;
                        if (ng.isSpikeTimeRequired()) {
                            os << "dd_sT" << ng.getName() << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << std::endl;
                        }
                    }   // end if (threadIdx.x < spkCount)
                }
            }
        );
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genPresynapticUpdateKernel(CodeStream &os, const NNmodel &model,
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
                        genPresynapticUpdateKernelPreSpan(os, model, sg, popSubs, false, 
                                                          wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdateKernelPostSpan(os, model, sg, popSubs, false,
                                                           wumThreshHandler, wumSimHandler);
                    }
                }

                // If true spikes should be processed
                if (sg.isTrueSpikeRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                        assert(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
                        genPresynapticUpdateKernelPreSpan(os, model, sg, popSubs, true, 
                                                          wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdateKernelPostSpan(os, model, sg, popSubs, true,
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
void CodeGenerator::genInitKernel(CodeStream &os, const NNmodel &model,
                                  NeuronGroupHandler localNGHandler, SynapseGroupHandler sgHandler) const
{
    // Create codestreams to generate different sections of runner
    /*std::stringstream initHostStream;
    std::stringstream initDeviceStream;
    CodeStream initHost(initHostStream);
    CodeStream initDevice(initDeviceStream);*/
    CodeStream &initDevice = os;
    
    /*primitives
     1) callbacks for per-population init and per-neuron init for both remote and local ngs
     2) per-population 
     
     os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)";
    {
        CodeStream::Scope b(os);
        os << "dd_glbSpkCnt" << n.first << "[i] = 0;" << std::endl;
    }
    
    os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
    
    3) per-neuron
    // Zero spikes
    if(n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
        os << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)";
        {
            CodeStream::Scope b(os);
            os << "dd_glbSpk" << n.first << "[(i * " + std::to_string(n.second.getNumNeurons()) + ") + lid] = 0;" << std::endl;
        }
    }
    else {
        os << "dd_glbSpk" << n.first << "[lid] = 0;" << std::endl;
    }
     */
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

    Substitutions kernelSubs(cudaFunctions);

    // initialization kernel code
    size_t idStart = 0;
    {
        // common variables for all cases
        CodeStream::Scope b(initDevice);

        initDevice << "const unsigned int id = " << m_InitBlockSize << " * blockIdx.x + threadIdx.x;" << std::endl;

        // If RNG is required
        if(model.isDeviceRNGRequired()) {
            initDevice << "// Initialise global GPU RNG" << std::endl;
            initDevice << "if(id == 0)";
            {
                CodeStream::Scope b(initDevice);
                initDevice << "curand_init(" << model.getSeed() << ", 0, 0, &dd_rng[0]);" << std::endl;
            }
        }

        // Parallelise over remote neuron groups
        genParallelGroup<NeuronGroup>(initDevice, kernelSubs, model.getRemoteNeuronGroups(), idStart,
            [this](const NeuronGroup &ng){ return padSize(ng.getNumNeurons(), m_InitBlockSize); },
            [this](const NeuronGroup &ng){ return (ng.hasOutputToHost(m_LocalHostID) && ng.getSpikeVarMode() & VarInit::DEVICE); },
            [this, &model](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
            {
                os << "if(" << popSubs.getVarSubstitution("id") << " == 0)";
                {
                    CodeStream::Scope b(os);

                    
                }


                os << "// only do this for existing neurons" << std::endl;
                os << "if (" << popSubs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    
                }
            });

   
        // Parallelise over local neuron groups
        genParallelGroup<NeuronGroup>(os, kernelSubs, model.getLocalNeuronGroups(), idStart,
            [this](const NeuronGroup &ng){ return padSize(ng.getNumNeurons(), m_InitBlockSize); },
            [this](const NeuronGroup &ng){ return ng.isDeviceInitRequired(); },
            [this, &model, localNGHandler](CodeStream &os, const NeuronGroup &ng, Substitutions &popSubs)
            {
                os << "if(" << popSubs.getVarSubstitution("id") << " == 0)";
                {
                    CodeStream::Scope b(os);
                }
                
                os << "// only do this for existing neurons" << std::endl;
                os << "if (" << popSubs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If this neuron is going to require a simulation RNG, initialise one using GLOBALthread id for sequence
                    if(ng.isSimRNGRequired()) {
                        os << "curand_init(" << model.getSeed() << ", id, 0, &dd_rng" << ng.getName() << "[" << popSubs.getVarSubstitution("id") << "]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(ng.isInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;

                        // Add substitution for RNG
                        popSubs.addVarSubstitution("rng", "initRNG");
                    }
                    
                    localNGHandler(os, ng, popSubs);
                }
            });

        // Initialise weight update variables for dense 
        genParallelGroup<SynapseGroup>(os, kernelSubs, model.getLocalSynapseGroups(), idStart, 
            [this](const SynapseGroup &sg){ return padSize(sg.getTrgNeuronGroup()->getNumNeurons(), m_InitBlockSize); },
            [](const SynapseGroup &sg){ return (sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && sg.isWUDeviceVarInitRequired(); },
            [](CodeStream &os, const SynapseGroup &sg, Substitutions &popSubs)
            {
                
            });
        // 3) genParallelSynapseGroup for local synapse groups **TODO** if((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUDeviceVarInitRequired())
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    if(mode & VarLocation::HOST) {
        m_HostCodeGenerator.genVariableDefinition(os, type, name, mode);
    }
    if(mode & VarLocation::DEVICE) {
        os << getVarExportPrefix() << " " << type << " d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    if(mode & VarLocation::HOST) {
        m_HostCodeGenerator.genVariableImplementation(os, type, name, mode);
    }
    if(mode & VarLocation::DEVICE) {
        os << type << " d_" << name << ";" << std::endl;
        os << "__device__ " << type << " dd_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode, size_t count) const
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
void CodeGenerator::genVariableInit(CodeStream &os, VarMode mode, size_t count, const Substitutions &kernelSubs, Handler handler) const
{
    // If variable should be initialised on device
    if(mode & VarInit::DEVICE) {
        Substitutions varSubs(&kernelSubs);
        handler(os, varSubs);
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs.getVarSubstitution("id") << ";" << std::endl;
}
//--------------------------------------------------------------------------
void CodeGenerator::genPresynapticUpdateKernelPreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
                                                      GroupHandler<SynapseGroup> wumThreshHandler, 
                                                      GroupHandler<SynapseGroup> wumSimHandler) const
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
            synSubs.addVarSubstitution("syn_address", "synAddress");

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
void CodeGenerator::genPresynapticUpdateKernelPostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &popSubs, bool trueSpike,
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
                synSubs.addVarSubstitution("syn_address", "synAddress");

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
size_t CodeGenerator::getPresynapticUpdateKernelSize(const SynapseGroup &sg) const
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
bool CodeGenerator::shouldAccumulateInLinSyn(const SynapseGroup &sg) const
{
    // We should accumulate each postsynaptic neuron's input in a register if matrix is dense or bitfield (where each thread represents an individual neuron)
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK));
}
//--------------------------------------------------------------------------
bool CodeGenerator::shouldAccumulateInSharedMemory(const SynapseGroup &sg) const
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
std::string CodeGenerator::getFloatAtomicAdd(const std::string &ftype) const
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

}   // namespace CUDA
