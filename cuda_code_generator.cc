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

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
size_t padSize(size_t size, size_t blockSize)
{
    return ((size + blockSize - 1) / blockSize) * blockSize;
}

std::string getFloatAtomicAdd(const std::string &ftype)
{
    /*int version;
    cudaRuntimeGetVersion(&version);
    if (((deviceProp[theDevice].major < 2) && (ftype == "float"))
        || (((deviceProp[theDevice].major < 6) || (version < 8000)) && (ftype == "double"))) {
        return "atomicAddSW";
    }
    else {*/
        return "atomicAdd";
    //}
}


}   // Anonymous namespace

//--------------------------------------------------------------------------
// CUDA::CodeGenerator
//--------------------------------------------------------------------------
namespace CUDA
{
void CodeGenerator::genNeuronUpdateKernel(CodeStream &os, const NNmodel &model, 
                                          std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &ng, const Substitutions &)> handler) const
{
    os << "extern \"C\" __global__ void calcNeurons(float time)" << std::endl;
    {
        CodeStream::Scope b(os);
        os << "const unsigned int id = " << m_NeuronUpdateBlockSize << " * blockIdx.x + threadIdx.x; " << std::endl;

        Substitutions baseSubs;
        baseSubs.addVarSubstitution("t", "t");

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
        genParallelNeuronGroup(os, baseSubs, model.getLocalNeuronGroups(), 
            [&model, handler](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NeuronGroup &ng, const Substitutions &baseSubs)
            {
                // Get name of rng to use for this neuron
                Substitutions subs(&baseSubs);
                subs.addVarSubstitution("rng", "&dd_rng" + ng.getName() + "[" + subs.getVarSubstitution("id") + "]");
                
                // Call handler to generate generic neuron code
                handler(os, codeGenerator, model, ng, subs);

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
                                               std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> wumThreshHandler,
                                               std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumSimHandler) const
{
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getPrecision() << " t)" << std::endl; // end of synapse kernel header
    {
        CodeStream::Scope b(os);
        
        Substitutions baseSubs;
        baseSubs.addVarSubstitution("t", "t");

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
        genParallelSynapseGroup(os, baseSubs, model, 
            [this](const SynapseGroup &sg){ return getPresynapticUpdateKernelSize(sg); },
            [wumThreshHandler, wumSimHandler, this](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NNmodel &model, const SynapseGroup &sg, const Substitutions &subs)
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
                    os << "if(" << subs.getVarSubstitution("id") << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "linSyn = dd_inSyn" << sg.getName() << "[" << subs.getVarSubstitution("id") << "];" << std::endl;
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
                        genPresynapticUpdateKernelPreSpan(os, model, sg, subs, false, 
                                                          wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdateKernelPostSpan(os, model, sg, subs, false,
                                                           wumThreshHandler, wumSimHandler);
                    }
                }

                // If true spikes should be processed
                if (sg.isTrueSpikeRequired()) {
                    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
                        assert(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE);
                        genPresynapticUpdateKernelPreSpan(os, model, sg, subs, true, 
                                                          wumThreshHandler, wumSimHandler);
                    }
                    else {
                        genPresynapticUpdateKernelPostSpan(os, model, sg, subs, true,
                                                           wumThreshHandler, wumSimHandler);
                    }
                }
                
                os << std::endl;

                // If we have been accumulating into a register, write value back to global memory
                if (shouldAccumulateInLinSyn(sg)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << "if (" << subs.getVarSubstitution("id") << " < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << sg.getName() << "[" << subs.getVarSubstitution("id") << "] = linSyn;" << std::endl;
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
                                  std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const NeuronGroup &, const Substitutions&)> ngHandler,
                                  std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel&, const SynapseGroup &, const Substitutions&)> sgHandler) const
{
    // init kernel header
    /*os << "extern \"C\" __global__ void initializeDevice(";
    const auto &params = model.getInitKernelParameters();
    for(auto p = params.cbegin(); p != params.cend(); p++) {
        os << p->second << " " << p->first;
        if (std::next(p) != params.cend()) {
            os  << ", ";
        }
    }
    os << ")";

    Substitutions baseSubs;

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

        // Parallelise over remote neuron groups
        genParallelNeuronGroup(os, model.getRemoteNeuronGroups(),  baseSubs,
            [this](const NeuronGroup &ng){ return (ng.hasOutputToHost(m_LocalHostID) && ng.getSpikeVarMode() & VarInit::DEVICE); },
            [this, &model](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NeuronGroup &ng, const Substitutions &subs)
            {
                os << "if(" << subs.getVarSubstitution("id") << " == 0)";
                {
                    CodeStream::Scope b(os);

                    // If delay is required, loop over delay bins
                    if(ng.isDelayRequired()) {
                        os << "for (int i = 0; i < " << ng.getNumDelaySlots() << "; i++)" << CodeStream::OB(14);
                    }

                    if(ng.isTrueSpikeRequired() && ng.isDelayRequired()) {
                        os << "dd_glbSpkCnt" << ng.getName() << "[i] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
                    }

                    // If delay was required, close loop brace
                    if(ng.isDelayRequired()) {
                        os << CodeStream::CB(14);
                    }
                }


                os << "// only do this for existing neurons" << std::endl;
                os << "if (" << subs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If delay is required, loop over delay bins
                    if(ng.isDelayRequired()) {
                        os << "for (int i = 0; i < " << ng.getNumDelaySlots() << "; i++)" << CodeStream::OB(16);
                    }

                    // Zero spikes
                    if(ng.isTrueSpikeRequired() && ng.isDelayRequired()) {
                        os << "dd_glbSpk" << ng.getName() << "[(i * " + std::to_string(ng.getNumNeurons()) + ") + lid] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpk" << ng.getName() << "[lid] = 0;" << std::endl;
                    }

                    // If delay was required, close loop brace
                    if(ng.isDelayRequired()) {
                        os << CodeStream::CB(16) << std::endl;
                    }
                }
            });

   
        // Parallelise over remote neuron groups
        genParallelNeuronGroup(os, model.getLocalNeuronGroups(), baseSubs,
            [this](const NeuronGroup &ng){ return ng.isDeviceInitRequired(); },
            [this, &model](CodeStream &os, const ::CodeGenerator::Base &codeGenerator, const NeuronGroup &ng, const Substitutions &subs)
            {
                // Determine which built in variables should be initialised on device
                const bool shouldInitSpikeVar = (ng.getSpikeVarMode() & VarInit::DEVICE);
                const bool shouldInitSpikeEventVar = ng.isSpikeEventRequired() && (ng.getSpikeEventVarMode() & VarInit::DEVICE);
                const bool shouldInitSpikeTimeVar = ng.isSpikeTimeRequired() && (ng.getSpikeTimeVarMode() & VarInit::DEVICE);

                // If per-population spike variables should be initialised on device
                // **NOTE** could optimise here and use getNumDelaySlots threads if getNumDelaySlots < numthreads
                if(shouldInitSpikeVar || shouldInitSpikeEventVar)
                {
                    os << "if(" << subs.getVarSubstitution("id") << " == 0)";
                    {
                        CodeStream::Scope b(os);

                        // If delay is required, loop over delay bins
                        if(ng.isDelayRequired()) {
                            os << "for (int i = 0; i < " << ng.getNumDelaySlots() << "; i++)" << CodeStream::OB(22);
                        }

                        // Zero spike count
                        if(shouldInitSpikeVar) {
                            if(ng.isTrueSpikeRequired() && ng.isDelayRequired()) {
                                os << "dd_glbSpkCnt" << ng.getName() << "[i] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
                            }
                        }

                        // Zero spike event count
                        if(shouldInitSpikeEventVar) {
                            if(ng.isDelayRequired()) {
                                os << "dd_glbSpkCntEvnt" << ng.getName() << "[i] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCntEvnt" << ng.getName() << "[0] = 0;" << std::endl;
                            }
                        }

                        // If delay was required, close loop brace
                        if(ng.isDelayRequired()) {
                            os << CodeStream::CB(22);
                        }
                    }
                }
                os << "// only do this for existing neurons" << std::endl;
                os << "if (" << subs.getVarSubstitution("id") << " < " << ng.getNumNeurons() << ")";
                {
                    CodeStream::Scope b(os);

                    // If this neuron is going to require a simulation RNG, initialise one using GLOBALthread id for sequence
                    if(ng.isSimRNGRequired()) {
                        os << "curand_init(" << model.getSeed() << ", id, 0, &dd_rng" << ng.getName() << "[" << subs.getVarSubstitution("id") << "]);" << std::endl;
                    }

                    // If this neuron requires an RNG for initialisation,
                    // make copy of global phillox RNG and skip ahead by thread id
                    if(ng.isInitRNGRequired(VarInit::DEVICE)) {
                        os << "curandStatePhilox4_32_10_t initRNG = dd_rng[0];" << std::endl;
                        os << "skipahead_sequence((unsigned long long)id, &initRNG);" << std::endl;
                    }

                    // Build string to use for delayed variable index
                    const std::string delayedIndex = "(i * " + std::to_string(ng.getNumNeurons()) + ") + lid";

                    // If spike variables are initialised on device
                    if(shouldInitSpikeVar || shouldInitSpikeEventVar || shouldInitSpikeTimeVar) {
                        // If delay is required, loop over delay bins
                        if(ng.isDelayRequired()) {
                            os << "for (int i = 0; i < " << ng.getNumDelaySlots() << "; i++)" << CodeStream::OB(31);
                        }

                        // Zero spikes
                        if(shouldInitSpikeVar) {
                            if(ng.isTrueSpikeRequired() && ng.isDelayRequired()) {
                                os << "dd_glbSpk" << ng.getName() << "[" << delayedIndex << "] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpk" << ng.getName() << "[lid] = 0;" << std::endl;
                            }
                        }

                        // Zero spike events
                        if(shouldInitSpikeEventVar) {
                            if(ng.isDelayRequired()) {
                                os << "dd_glbSpkEvnt" << ng.getName() << "[" << delayedIndex << "] = 0;" << std::endl;
                            }
                            else {
                                os << "dd_glbSpkCnt" << ng.getName() << "[lid] = 0;" << std::endl;
                            }
                        }

                        // Reset spike times
                        if(shouldInitSpikeTimeVar) {
                            if(ng.isDelayRequired()) {
                                os << "dd_sT" << ng.getName() << "[" << delayedIndex << "] = -SCALAR_MAX;" << std::endl;
                            }
                            else {
                                os << "dd_sT" << ng.getName() << "[lid] = -SCALAR_MAX;" << std::endl;
                            }
                        }

                        // If delay was required, close loop brace
                        if(ng.isDelayRequired()) {
                            os << CodeStream::CB(31) << std::endl;
                        }
                    }

                    ngHandler(os, *this, model, ng, subs);
                }
            });

        // 3) genParallelSynapseGroup for local synapse groups **TODO** if((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUDeviceVarInitRequired())
    }*/
}
//--------------------------------------------------------------------------
void CodeGenerator::genVariableDefinition(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    // In windows making variables extern isn't enough to export then as DLL symbols - you need to add __declspec(dllexport)
#ifdef _WIN32
    const std::string varExportPrefix = GENN_PREFERENCES::buildSharedLibrary ? "__declspec(dllexport) extern" : "extern";
#else
    const std::string varExportPrefix = "extern";
#endif

    if(mode & VarLocation::HOST) {
        // **TODO** this should be a call to CPUCodeGenerator::genVariableDefinition
        os << varExportPrefix << " " << type << " " << name << ";" << std::endl;
    }
    if(mode & VarLocation::DEVICE) {
        os << varExportPrefix << " " << type << " d_" << name << ";" << std::endl;
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarMode mode) const
{
    if(mode & VarLocation::HOST) {
         // **TODO** this should be a call to CPUCodeGenerator::genVariableImplementation
        os << type << " " << name << ";" << std::endl;
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
void CodeGenerator::genParallelNeuronGroup(CodeStream &os, const Substitutions &baseSubs,
                                           const std::map<std::string, NeuronGroup> &ngs, std::function<bool(const NeuronGroup &)> filter,
                                           std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NeuronGroup&, const Substitutions &)> handler) const
{
    // Populate neuron update groups
    Substitutions subs(&baseSubs);
    size_t idStart = 0;
    for (const auto &ng : ngs) {
        // If this neuron group should be processed
        if(filter(ng.second)) {
            const size_t paddedSize = padSize(ng.second.getNumNeurons(), m_NeuronUpdateBlockSize);

            os << "// Neuron group " << ng.first << std::endl;

            // If this is the first  group
            if (idStart == 0) {
                os << "if(id < " << paddedSize << ")" << CodeStream::OB(1);
                subs.addVarSubstitution("id", "id");
            }
            else {
                os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")" << CodeStream::OB(1);
                os << "const unsigned int lid = id - " << idStart << ";" << std::endl;
                subs.addVarSubstitution("id", "lid");
            }

            handler(os, *this, ng.second, subs);

            idStart += paddedSize;
            os << CodeStream::CB(1) << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genParallelSynapseGroup(CodeStream &os, const Substitutions &baseSubs, const NNmodel &model,
                                            std::function<size_t(const SynapseGroup&)> getPaddedSizeFunc, 
                                            std::function<bool(const SynapseGroup &)> filter,
                                            std::function<void(CodeStream &, const ::CodeGenerator::Base &, const NNmodel &, const SynapseGroup&, const Substitutions &)> handler) const
{
    // Populate neuron update groups
    Substitutions subs(&baseSubs);
    size_t idStart = 0;
    for (const auto &sg : model.getLocalSynapseGroups()) {
        // If this synapse group should be processed
        if(filter(sg.second)) {
            const size_t paddedSize = getPaddedSizeFunc(sg.second);

            os << "// Synapse group " << sg.first << std::endl;

            // If this is the first  group
            if (idStart == 0) {
                os << "if(id < " << paddedSize << ")" << CodeStream::OB(1);
                subs.addVarSubstitution("id", "id");
            }
            else {
                os << "if(id >= " << idStart << " && id < " << idStart + paddedSize << ")" << CodeStream::OB(1);
                os << "const unsigned int lid = id - " << idStart << ";" << std::endl;
                subs.addVarSubstitution("id", "lid");
            }

            handler(os, *this, model, sg.second, subs);

            idStart += paddedSize;
            os << CodeStream::CB(1) << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genEmitSpike(CodeStream &os, const Substitutions &subs, const std::string &suffix) const
{
    os << "const unsigned int spk" << suffix << "Idx = atomicAdd((unsigned int *) &shSpk" << suffix << "Count, 1);" << std::endl;
    os << "shSpk" << suffix << "[spk" << suffix << "Idx] = " << subs.getVarSubstitution("id") << ";" << std::endl;
}
//--------------------------------------------------------------------------
void CodeGenerator::genPresynapticUpdateKernelPreSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &baseSubs, bool trueSpike,
                                                      std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumThreshHandler,
                                                      std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumSimHandler) const
{
    // Get suffix based on type of events
    const std::string eventSuffix = trueSpike ? "" : "evnt";
    const auto *wu = sg.getWUModel();

    os << "if (" << baseSubs.getVarSubstitution("id") << " < " ;
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
            os << "[(delaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + " << baseSubs.getVarSubstitution("id") << "];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = dd_glbSpk"  << eventSuffix << sg.getSrcNeuronGroup()->getName();
            os << "[" << baseSubs.getVarSubstitution("id") << "];" << std::endl;
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
 
            Substitutions subs(&baseSubs);
            subs.addVarSubstitution("id_pre", "preInd");
            subs.addVarSubstitution("id_post", "i");

            // Generate weight update threshold condition
            wumThreshHandler(os, *this, model, sg, subs);
            
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

            Substitutions subs(&baseSubs);
            subs.addVarSubstitution("id_pre", "preInd");
            subs.addVarSubstitution("id_post", "ipost");
            subs.addVarSubstitution("syn_address", "synAddress");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.isDendriticDelayRequired()) {
                subs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
            }
            // Otherwise
            else {
                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(shouldAccumulateInSharedMemory(sg)) {
                    subs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    subs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                }
            }

            wumSimHandler(os, *this, model, sg, subs);
        }

        if (!trueSpike && sg.isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
    }
}
//--------------------------------------------------------------------------
void CodeGenerator::genPresynapticUpdateKernelPostSpan(CodeStream &os, const NNmodel &model, const SynapseGroup &sg, const Substitutions &baseSubs, bool trueSpike,
                                                       std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumThreshHandler,
                                                       std::function<void(CodeStream&, const::CodeGenerator::Base&, const NNmodel&, const SynapseGroup&, const Substitutions&)> wumSimHandler) const
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
            os << "if (" << baseSubs.getVarSubstitution("id") << " < " << sg.getMaxConnections() << ")";
            {
                CodeStream::Scope b(os);
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t maxSynapses = (size_t)sg.getTrgNeuronGroup()->getNumNeurons() * (size_t)sg.getSrcNeuronGroup()->getNumNeurons();
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "const uint64_t gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + " << baseSubs.getVarSubstitution("id") << ");" << std::endl;
                    }
                    else {
                        os << "const unsigned int gid = (shSpk" << eventSuffix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << baseSubs.getVarSubstitution("id") << ");" << std::endl;
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

                    Substitutions subs(&baseSubs);
                    subs.addVarSubstitution("id_pre", "preInd");
                    subs.addVarSubstitution("id_post", "ipost");
                   
                    // Generate weight update threshold condition
                    wumThreshHandler(os, *this, model, sg, subs);

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

                    os << "if (" << baseSubs.getVarSubstitution("id") << " < npost)" << CodeStream::OB(140);
                    os << "synAddress += " << baseSubs.getVarSubstitution("id") << ";" << std::endl;
                    os << "const unsigned int ipost = dd_ind" << sg.getName() << "[synAddress];" << std::endl;
                }
                else { // DENSE
                    os << "ipost = " << baseSubs.getVarSubstitution("id") << ";" << std::endl;
                }

                Substitutions subs(&baseSubs);
                subs.addVarSubstitution("id_pre", "shSpk" + eventSuffix + "[j]");
                subs.addVarSubstitution("id_post", "ipost");
                subs.addVarSubstitution("syn_address", "synAddress");

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.isDendriticDelayRequired()) {
                    subs.addFuncSubstitution("addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
                }
                // Otherwise
                else {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                        if (shouldAccumulateInSharedMemory(sg)) {
                            subs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&shLg[ipost], $(0))");
                        }
                        else {
                            subs.addFuncSubstitution("addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");
                        }
                    }
                    else {
                        subs.addFuncSubstitution("addToInSyn", 1, "linSyn += $(0)");
                    }
                }

                wumSimHandler(os, *this, model, sg, subs);

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
            // paddedSize is the lowest multiple of blockSize >= neuronN[synapseSource[i]
            // **TODO** integer ceil trick
            return ceil((double)sg.getSrcNeuronGroup()->getNumNeurons() / (double)m_PresynapticUpdateBlockSize) * (double)m_PresynapticUpdateBlockSize;
        }
        else {
            // paddedSize is the lowest multiple of blockSize >= maxConn[i]
            // **TODO** integer ceil trick
            return ceil((double)sg.getMaxConnections() / (double) m_PresynapticUpdateBlockSize) * (double) m_PresynapticUpdateBlockSize;
        }
    }
    else {
        // paddedSize is the lowest multiple of blockSize >= neuronN[synapseTarget[i]]
        return ceil((double)sg.getTrgNeuronGroup()->getNumNeurons() / (double) m_PresynapticUpdateBlockSize) * (double) m_PresynapticUpdateBlockSize;
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
    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC/* && deviceProp[theDevice].major < 5*/) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.getTrgNeuronGroup()->getNumNeurons() <= m_PresynapticUpdateBlockSize);
    }
}
}   // namespace CUDA
