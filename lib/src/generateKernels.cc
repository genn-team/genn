/*--------------------------------------------------------------------------
   Author: Thomas Nowotny

   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK 

   email to:  T.Nowotny@sussex.ac.uk

   initial version: 2010-02-07

--------------------------------------------------------------------------*/

//------------------------------------------------------------------------
/*! \file generateKernels.cc

  \brief Contains functions that generate code for CUDA kernels. Part of the code generation section.

*/
//-------------------------------------------------------------------------

#include "generateKernels.h"
#include "global.h"
#include "utils.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "codeGenUtils.h"
#include "codeStream.h"

#include <algorithm>


// The CPU_ONLY version does not need any of this
#ifndef CPU_ONLY

//-------------------------------------------------------------------------
// Anonymous namespace
//-------------------------------------------------------------------------
namespace
{
string getFloatAtomicAdd(const string &ftype)
{
    int version;
    cudaRuntimeGetVersion(&version);
    if (((deviceProp[theDevice].major < 2) && (ftype == "float"))
        || (((deviceProp[theDevice].major < 6) || (version < 8000)) && (ftype == "double"))) {
        return "atomicAddSW";
    }
    else {
        return "atomicAdd";
    }
}

bool shouldAccumulateInLinSyn(const SynapseGroup &sg)
{
    // We should accumulate each postsynaptic neuron's input in a register if matrix is dense or bitfield (where each thread represents an individual neuron)
    return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) || (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK));
}

bool shouldAccumulateInSharedMemory(const SynapseGroup &sg)
{
    // If parallelism is presynaptic i.e. atomics are required and device is older than Maxwell, we shouldn't use shared memory as atomics are emulated
    // and actually slower than global memory (see https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
    if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC && deviceProp[theDevice].major < 5) {
        return false;
    }
    // Otherwise, if dendritic delays are required, shared memory approach cannot be used so return false
    else if(sg.isDendriticDelayRequired()) {
        return false;
    }
    // Otherwise, we should accumulate each postsynaptic neuron's input in shared menory if matrix is sparse
    // and the output population is small enough that input to it can be stored in a shared memory array
    else {
        return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.getTrgNeuronGroup()->getNumNeurons() <= synapseBlkSz);
    }
}

// parallelisation along pre-synaptic spikes, looped over post-synaptic neurons
void generatePreParallelisedSparseCode(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    const string &localID, //!< the variable name of the local ID of the thread within the synapse group
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    const bool evnt = (postfix == "Evnt");
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    os << "if (" << localID << " < " ;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[preReadDelaySlot])";
    }
    else {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[0])";
    }
    {
        CodeStream::Scope b(os);

        if (!wu->getSimSupportCode().empty()) {
            os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
        }

        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "const unsigned int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
            os << "[(preReadDelaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + " << localID << "];" << std::endl;
        }
        else {
            os << "const unsigned int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
            os << "[" << localID << "];" << std::endl;
        }

        if(sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
            os << "prePos = dd_indInG" << sg.getName() << "[preInd];" << std::endl;
            os << "npost = dd_indInG" << sg.getName() << "[preInd + 1] - prePos;" << std::endl;
        }
        else if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
            os << "prePos = preInd * " << to_string(sg.getMaxConnections()) << ";" << std::endl;
            os << "npost = dd_rowLength" << sg.getName() << "[preInd];" << std::endl;
        }
        else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << "unsigned int gid = (dd_glbSpkCnt" << postfix << "[" << localID << "] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + i);" << std::endl;
        }

        if (evnt && sg.isEventThresholdReTestRequired()) {
            os << "if ";
            if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                os << "((B(dd_gp" << sg.getName() << "[gid / 32], gid & 31)) && ";
            }

            // code substitutions ----
            string eCode = wu->getEventThresholdConditionCode();
            StandardSubstitutions::weightUpdateThresholdCondition(eCode, sg,
                                                                wuDerivedParams, wuExtraGlobalParams,
                                                                "preInd", "i", "dd_", cudaFunctions, ftype);
            // end code substitutions ----
            os << "(" << eCode << ")";

            if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << ")";
            }
            os << CodeStream::OB(130);
        }
        else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << "if (B(dd_gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(135);
        }

        os << "for(unsigned int i = 0; i < npost; ++i)";
        {
            CodeStream::Scope b(os);

            // **TODO** pretty sure __ldg will boost performance here - basically will bring whole row into cache
            os << "ipost = dd_ind" <<  sg.getName() << "[prePos];" << std::endl;

            // Code substitutions ----------------------------------------------------------------------------------
            string wCode = evnt ? wu->getEventCode() : wu->getSimCode();
            substitute(wCode, "$(t)", "t");

            // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
            if(sg.isDendriticDelayRequired()) {
                functionSubstitute(wCode, "addToInSynDelay", 2, getFloatAtomicAdd(ftype) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
            }
            // Otherwise
            else {
                // Use atomic operation to update $(inSyn)
                substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");

                // If postsynaptic input should be accumulated in shared memory, substitute shared memory array for $(inSyn)
                if(shouldAccumulateInSharedMemory(sg)) {
                    functionSubstitute(wCode, "addToInSyn", 1, getFloatAtomicAdd(ftype) + "(&shLg[ipost], $(0))");

                    substitute(wCode, "$(inSyn)", "shLg[ipost]");
                }
                // Otherwise, substitute global memory array for $(inSyn)
                else {
                    functionSubstitute(wCode, "addToInSyn", 1, getFloatAtomicAdd(ftype) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");

                    substitute(wCode, "$(inSyn)", "dd_inSyn" + sg.getPSModelTargetName() + "[ipost]");
                }
            }

            if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[prePos]");
            }

            StandardSubstitutions::weightUpdateSim(wCode, sg,
                                                   wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                   "preInd", "ipost", "dd_", cudaFunctions, ftype);
            // end code substitutions -------------------------------------------------------------------------

            os << wCode << std::endl;

            os << "prePos += 1;" << std::endl;
        }

        if (evnt && sg.isEventThresholdReTestRequired()) {
            os << CodeStream::CB(130);
        }
        else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << CodeStream::CB(135);
        }
    }
}

// classical parallelisation of post-synaptic neurons in parallel and spikes in a loop
void generatePostParallelisedCode(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    const string &localID, //!< the variable name of the local ID of the thread within the synapse group
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    const bool evnt = (postfix == "Evnt");
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << std::endl;
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)";
    {
        CodeStream::Scope b(os);
        os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = ((lscnt" << postfix << "-1) % BLOCKSZ_SYN) +1;" << std::endl;
        os << "else lmax = BLOCKSZ_SYN;" << std::endl;
        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < lmax)";
        {
            CodeStream::Scope b(os);
            const std::string queueOffset = sg.getSrcNeuronGroup()->isDelayRequired() ? "preReadDelayOffset + " : "";
            os << "shSpk" << postfix << "[threadIdx.x] = dd_glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << queueOffset << "(r * BLOCKSZ_SYN) + threadIdx.x];" << std::endl;
        }
        os << "__syncthreads();" << std::endl;

        os << "// loop through all incoming spikes" << std::endl;
        os << "for (j = 0; j < lmax; j++)";
        {
            CodeStream::Scope b(os);
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << localID << " < " << sg.getMaxConnections() << ")";
            {
                CodeStream::Scope b(os);
                if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    const size_t maxSynapses = (size_t)sg.getTrgNeuronGroup()->getNumNeurons() * (size_t)sg.getSrcNeuronGroup()->getNumNeurons();
                    if((maxSynapses & 0xFFFFFFFF00000000ULL) != 0) {
                        os << "uint64_t gid = (shSpk" << postfix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << "ull + " << localID << ");" << std::endl;
                    }
                    else {
                        os << "unsigned int gid = (shSpk" << postfix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << localID << ");" << std::endl;
                    }
                }

                if (!wu->getSimSupportCode().empty()) {
                    os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
                }
                if (evnt && sg.isEventThresholdReTestRequired()) {
                    os << "if ";
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
                        os << "((B(dd_gp" << sg.getName() << "[gid / 32], gid & 31)) && ";
                    }

                    // code substitutions ----
                    string eCode = wu->getEventThresholdConditionCode();
                    StandardSubstitutions::weightUpdateThresholdCondition(eCode, sg, wuDerivedParams, wuExtraGlobalParams,
                                                                        "shSpkEvnt[j]", "ipost", "dd_",
                                                                        cudaFunctions, ftype);
                    // end code substitutions ----
                    os << "(" << eCode << ")";

                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        os << ")";
                    }
                    os << CodeStream::OB(130);
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(dd_gp" << sg.getName() << "[gid / 32], gid & 31))" << CodeStream::OB(135);
                }


                if(sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                        os << "prePos = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j]];" << std::endl;
                        os << "npost = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j] + 1] - prePos;" << std::endl;
                    }
                    else {
                        os << "prePos = shSpk" << postfix << "[j] * " << to_string(sg.getMaxConnections()) << ";" << std::endl;
                        os << "npost = dd_rowLength" << sg.getName() << "[shSpk" << postfix << "[j]];" << std::endl;
                    }

                    os << "if (" << localID << " < npost)" << CodeStream::OB(140);
                    os << "prePos += " << localID << ";" << std::endl;
                    os << "ipost = dd_ind" << sg.getName() << "[prePos];" << std::endl;
                }
                else { // DENSE
                    os << "ipost = " << localID << ";" << std::endl;
                }

                // Code substitutions ----------------------------------------------------------------------------------
                string wCode = (evnt ? wu->getEventCode() : wu->getSimCode());
                substitute(wCode, "$(t)", "t");

                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                if(sg.isDendriticDelayRequired()) {
                    functionSubstitute(wCode, "addToInSynDelay", 2, getFloatAtomicAdd(ftype) + "(&dd_denDelay" + sg.getPSModelTargetName() + "[" + sg.getDendriticDelayOffset("dd_", "$(1)") + "ipost], $(0))");
                }
                // Otherwise
                else {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        // **THINK** this is only correct if there are no multapses i.e. there is only one synapse between any pair of pre and postsynaptic neurons
                        if (shouldAccumulateInSharedMemory(sg)) {
                            functionSubstitute(wCode, "addToInSyn", 1, getFloatAtomicAdd(ftype) + "(&shLg[ipost], $(0))");

                            // **DEPRECATED**
                            substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                            substitute(wCode, "$(inSyn)", "shLg[ipost]");
                        }
                        else {
                            functionSubstitute(wCode, "addToInSyn", 1, getFloatAtomicAdd(ftype) + "(&dd_inSyn" + sg.getPSModelTargetName() + "[ipost], $(0))");

                            // **DEPRECATED**
                            substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");
                            substitute(wCode, "$(inSyn)", "dd_inSyn" + sg.getPSModelTargetName() + "[ipost]");
                        }
                    }
                    else {
                        functionSubstitute(wCode, "addToInSyn", 1, "linSyn += $(0)");

                        // **DEPRECATED**
                        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                        substitute(wCode, "$(inSyn)", "linSyn");
                    }
                }

                if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                        name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd,
                                           sg.getName() + "[prePos]");
                    }
                    else {
                        name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd,
                                           sg.getName() + "[shSpk" + postfix + "[j] * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + "+ ipost]");
                    }
                }

                StandardSubstitutions::weightUpdateSim(wCode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                    "shSpk" + postfix + "[j]", "ipost", "dd_",
                                                    cudaFunctions, ftype);
                // end Code substitutions -------------------------------------------------------------------------
                os << wCode << std::endl;

                if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                    os << CodeStream::CB(140); // end if (id < npost)
                }

                if (evnt && sg.isEventThresholdReTestRequired()) {
                    os << CodeStream::CB(130); // end if (eCode)
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid / 32], gid
                }
            }
        }
    }
}
//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic
  spikes or spike type events

*/
//-------------------------------------------------------------------------
void generate_process_presynaptic_events_code(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    const string &localID, //!< the variable name of the local ID of the thread within the synapse group
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    const bool evnt = (postfix == "Evnt");

     if ((evnt && sg.isSpikeEventRequired()) || (!evnt && sg.isTrueSpikeRequired())) {
        // parallelisation along pre-synaptic spikes, looped over post-synaptic neurons
        if(sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
            generatePreParallelisedSparseCode(os, sg, localID, postfix, ftype);
        }
        // classical parallelisation of post-synaptic neurons in parallel and spikes in a loop
        else {
            generatePostParallelisedCode(os, sg, localID, postfix, ftype);
        }
    }
}
}   // Anonymous namespace

//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA kernel that simulates all neurons in the model.

  The code generated upon execution of this function is for defining GPU side global variables that will hold model state in the GPU global memory and for the actual kernel function for simulating the neurons for one time step.
*/
//-------------------------------------------------------------------------

void genNeuronKernel(const NNmodel &model, //!< Model description
                     const string &path)  //!< Path for code generation
{
    string localID;
    ofstream fs;
    string name = model.getGeneratedCodePath(path, "neuronKrnl.cc");
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // write header content
    writeHeader(os);
    os << std::endl;

    // compiler/include control (include once)
    os << "#ifndef _" << model.getName() << "_neuronKrnl_cc" << std::endl;
    os << "#define _" << model.getName() << "_neuronKrnl_cc" << std::endl;
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file neuronKrnl.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing the neuron kernel function." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    //os << "__device__ __host__ float exp(int i) { return exp((float) i); }" << endl;

    os << "// include the support codes provided by the user for neuron or synaptic models" << std::endl;
    os << "#include \"support_code.h\"" << std::endl << std::endl;

    // kernel header
    os << "extern \"C\" __global__ void calcNeurons(";
    for(const auto &p : model.getNeuronKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    for(const auto &p : model.getCurrentSourceKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getTimePrecision() << " t)" << std::endl;
    {
        // kernel code
        CodeStream::Scope b(os);
        unsigned int neuronGridSz = model.getNeuronGridSize();
        neuronGridSz = neuronGridSz / neuronBlkSz;
        if (neuronGridSz < (unsigned int)deviceProp[theDevice].maxGridSize[1]) {
            os << "unsigned int id = " << neuronBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;
        }
        else {
            os << "unsigned int id = " << neuronBlkSz << " * (blockIdx.x * " << ceil(sqrt((float) neuronGridSz)) << " + blockIdx.y) + threadIdx.x;" << std::endl;
        }

        // these variables deal with high V "spike type events"
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (n.second.isSpikeEventRequired()) {
                os << "__shared__ volatile unsigned int posSpkEvnt;" << std::endl;
                os << "__shared__ unsigned int shSpkEvnt[" << neuronBlkSz << "];" << std::endl;
                os << "unsigned int spkEvntIdx;" << std::endl;
                os << "__shared__ volatile unsigned int spkEvntCount;" << std::endl;
                break;
            }
        }

        // these variables now deal only with true spikes, not high V "events"
        for(const auto &n : model.getLocalNeuronGroups()) {
            if(!n.second.getNeuronModel()->getThresholdConditionCode().empty()) {
                os << "__shared__ unsigned int shSpk[" << neuronBlkSz << "];" << std::endl;
                os << "__shared__ volatile unsigned int posSpk;" << std::endl;
                os << "unsigned int spkIdx;" << std::endl;
                os << "__shared__ volatile unsigned int spkCount;" << std::endl;
                break;
            }
        }
        os << std::endl;

        // Reset global spike counting vars here if there are no synapses at all
        if (model.getResetKernel() == GENN_FLAGS::calcNeurons) {
            os << "if (id == 0)";
            {
                CodeStream::Scope b(os);
                for(const auto &n : model.getLocalNeuronGroups()) {
                    StandardGeneratedSections::neuronOutputInit(os, n.second, "dd_");
                }
            }
            os << "__threadfence();" << std::endl << std::endl;
        }

        // Initialise shared spike count vars
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (!n.second.getNeuronModel()->getThresholdConditionCode().empty()) {
                os << "if (threadIdx.x == 0)";
                {
                    CodeStream::Scope b(os);
                    os << "spkCount = 0;" << std::endl;
                }
                break;
            }
        }
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (n.second.isSpikeEventRequired()) {
                os << "if (threadIdx.x == 1)";
                {
                    CodeStream::Scope b(os);
                    os << "spkEvntCount = 0;" << std::endl;
                }
                break;
            }
        }
        os << "__syncthreads();" << std::endl;
        os << std::endl;

        for(auto n = model.getLocalNeuronGroups().cbegin(); n != model.getLocalNeuronGroups().cend(); ++n) {
            os << "// neuron group " << n->first << std::endl;
            const auto &groupIDRange = n->second.getPaddedIDRange();
            if(n == model.getLocalNeuronGroups().cbegin()) {
                os << "if (id < " << groupIDRange.second << ")";
                localID = "id";
            }
            else {
                os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))";
                localID = "lid";
            }
            {
                CodeStream::Scope b(os);
                if(n != model.getLocalNeuronGroups().cbegin()) {
                    os << "unsigned int lid = id - " << groupIDRange.first << ";" << std::endl;
                }

                // If axonal delays are required
                if (n->second.isDelayRequired()) {
                    // We should READ from delay slot before spkQuePtr
                    os << "const unsigned int readDelayOffset = " << n->second.getPrevQueueOffset("dd_") << ";" << std::endl;
                    
                    // And we should WRITE to delay slot pointed to be spkQuePtr
                    os << "const unsigned int writeDelayOffset = " << n->second.getCurrentQueueOffset("dd_") << ";" << std::endl;
                }
                os << std::endl;

                os << "// only do this for existing neurons" << std::endl;
                os << "if (" << localID << " < " << n->second.getNumNeurons() << ")" << CodeStream::OB(20);

                os << "// pull neuron variables in a coalesced access" << std::endl;

                const auto *nm = n->second.getNeuronModel();

                // Get name of rng to use for this neuron
                // **TODO** Phillox option
                const std::string rngName = "&dd_rng" + n->first + "[" + localID + "]";

                // Create iteration context to iterate over the variables; derived and extra global parameters
                VarNameIterCtx nmVars(nm->getVars());
                DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
                ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

                // Generate code to copy neuron state into local variables
                StandardGeneratedSections::neuronLocalVarInit(os, n->second, nmVars, "dd_", localID, model.getTimePrecision());

                if ((nm->getSimCode().find("$(sT)") != string::npos)
                    || (nm->getThresholdConditionCode().find("$(sT)") != string::npos)
                    || (nm->getResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
                    os << model.getPrecision() << " lsT = dd_sT" <<  n->first << "[";
                    if (n->second.isDelayRequired()) {
                        os << "readDelayOffset + ";
                    }
                    os << localID << "];" << std::endl;
                }
                os << std::endl;

                if (!n->second.getMergedInSyn().empty() || (nm->getSimCode().find("Isyn") != string::npos)) {
                    os << model.getPrecision() << " Isyn = 0;" << std::endl;
                }

                // Initialise any additional input variables supported by neuron model
                for(const auto &a : nm->getAdditionalInputVars()) {
                    os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
                }

                 for(const auto &m : n->second.getMergedInSyn()) {
                    const auto *sg = m.first;
                    const auto *psm = sg->getPSModel();

                    os << "// pull inSyn values in a coalesced access" << std::endl;
                    os << model.getPrecision() << " linSyn" << sg->getPSModelTargetName() << " = dd_inSyn" << sg->getPSModelTargetName() << "[" << localID << "];" << std::endl;

                    // If dendritic delay is required
                    if(sg->isDendriticDelayRequired()) {
                        // Get reference to dendritic delay buffer input for this timestep
                        os << model.getPrecision() << " &denDelayFront" << sg->getPSModelTargetName() << " = dd_denDelay" + sg->getPSModelTargetName() + "[" + sg->getDendriticDelayOffset("dd_") + localID + "];" << std::endl;

                        // Add delayed input from buffer into inSyn
                        os << "linSyn" + sg->getPSModelTargetName() + " += denDelayFront" << sg->getPSModelTargetName() << ";" << std::endl;

                        // Zero delay buffer slot
                        os << "denDelayFront" << sg->getPSModelTargetName() << " = " << model.scalarExpr(0.0) << ";" << std::endl;
                    }

                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                        for(const auto &v : psm->getVars()) {
                            os << v.second << " lps" << v.first << sg->getPSModelTargetName();
                            os << " = dd_" <<  v.first << sg->getPSModelTargetName() << "[" << localID << "];" << std::endl;
                        }
                    }
                    string psCode = psm->getApplyInputCode();
                    substitute(psCode, "$(id)", localID);
                    substitute(psCode, "$(inSyn)", "linSyn" + sg->getPSModelTargetName());
                    StandardSubstitutions::postSynapseApplyInput(psCode, sg, n->second,
                        nmVars, nmDerivedParams, nmExtraGlobalParams,
                        cudaFunctions, model.getPrecision(), rngName);

                    if (!psm->getSupportCode().empty()) {
                        os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                    }
                    os << psCode << std::endl;
                    if (!psm->getSupportCode().empty()) {
                        os << CodeStream::CB(29) << " // namespace bracket closed" << std::endl;
                    }
                }

                if (!nm->getSupportCode().empty()) {
                    os << " using namespace " << n->first << "_neuron;" << std::endl;
                }
                string thCode = nm->getThresholdConditionCode();
                if (thCode.empty()) { // no condition provided
                    cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << n->first << "\" was provided. There will be no spikes detected in this population!" << endl;
                }
                else {
                    os << "// test whether spike condition was fulfilled previously" << std::endl;
                    substitute(thCode, "$(id)", localID);
                    StandardSubstitutions::neuronThresholdCondition(thCode, n->second,
                                                                    nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                                    cudaFunctions, model.getPrecision(), rngName);
                    if (GENN_PREFERENCES::autoRefractory) {
                        os << "bool oldSpike= (" << thCode << ");" << std::endl;
                    }
                }

                // check for current sources and insert code if necessary
                StandardGeneratedSections::neuronCurrentInjection(os, n->second,
                                                 "dd_", localID, cudaFunctions,
                                                 model.getPrecision(), rngName);

                os << "// calculate membrane potential" << std::endl;
                string sCode = nm->getSimCode();
                substitute(sCode, "$(id)", localID);
                StandardSubstitutions::neuronSim(sCode, n->second,
                                                nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                cudaFunctions, model.getPrecision(), rngName);
                os << sCode << std::endl;

                // look for spike type events first.
                if (n->second.isSpikeEventRequired()) {
                // Generate spike event test
                    StandardGeneratedSections::neuronSpikeEventTest(os, n->second,
                                                                    nmVars, nmExtraGlobalParams, localID,
                                                                    cudaFunctions, model.getPrecision(), rngName);

                    os << "// register a spike-like event" << std::endl;
                    os << "if (spikeLikeEvent)";
                    {
                        CodeStream::Scope b(os);
                        os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << std::endl;
                        os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << std::endl;
                    }
                }

                // test for true spikes if condition is provided
                if (!thCode.empty()) {
                    os << "// test for and register a true spike" << std::endl;
                    if (GENN_PREFERENCES::autoRefractory) {
                        os << "if ((" << thCode << ") && !(oldSpike)) ";
                    }
                    else {
                        os << "if (" << thCode << ") ";
                    }
                    {
                        CodeStream::Scope b(os);
                        os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << std::endl;
                        os << "shSpk[spkIdx] = " << localID << ";" << std::endl;

                        // add after-spike reset if provided
                        if (!nm->getResetCode().empty()) {
                            string rCode = nm->getResetCode();
                            substitute(rCode, "$(id)", localID);
                            StandardSubstitutions::neuronReset(rCode, n->second,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            cudaFunctions, model.getPrecision(), rngName);
                            os << "// spike reset code" << std::endl;
                            os << rCode << std::endl;
                        }
                    }
                }

                // store the defined parts of the neuron state into the global state variables dd_V etc
                StandardGeneratedSections::neuronLocalVarWrite(os, n->second, nmVars, "dd_", localID);

                if (!n->second.getMergedInSyn().empty()) {
                    os << "// the post-synaptic dynamics" << std::endl;
                }
                for(const auto &m : n->second.getMergedInSyn()) {
                    const auto *sg = m.first;
                    const auto *psm = sg->getPSModel();

                    string pdCode = psm->getDecayCode();
                    substitute(pdCode, "$(id)", localID);
                    substitute(pdCode, "$(inSyn)", "linSyn" + sg->getPSModelTargetName());
                    StandardSubstitutions::postSynapseDecay(pdCode, sg, n->second,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            cudaFunctions, model.getPrecision(), rngName);
                    if (!psm->getSupportCode().empty()) {
                        os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                    }
                    os << pdCode << std::endl;
                    if (!psm->getSupportCode().empty()) {
                        os << CodeStream::CB(29) << " // namespace bracket closed" << endl;
                    }

                    os << "dd_inSyn"  << sg->getPSModelTargetName() << "[" << localID << "] = linSyn" << sg->getPSModelTargetName() << ";" << std::endl;
                    for(const auto &v : psm->getVars()) {
                        os << "dd_" <<  v.first << sg->getPSModelTargetName() << "[" << localID << "] = lps" << v.first << sg->getPSModelTargetName() << ";"<< std::endl;
                    }
                }

                os << CodeStream::CB(20);
                os << "__syncthreads();" << std::endl;

                if (n->second.isSpikeEventRequired()) {
                    os << "if (threadIdx.x == 1)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << n->first;
                        if (n->second.isDelayRequired()) {
                            os << "[dd_spkQuePtr" << n->first << "], spkEvntCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkEvntCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 0)
                    os << "__syncthreads();" << std::endl;
                }

                if (!nm->getThresholdConditionCode().empty()) {
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << n->first;
                        if (n->second.isDelayRequired() && n->second.isTrueSpikeRequired()) {
                            os << "[dd_spkQuePtr" << n->first << "], spkCount);" << std::endl;
                        }
                        else {
                            os << "[0], spkCount);" << std::endl;
                        }
                    } // end if (threadIdx.x == 1)
                    os << "__syncthreads();" << std::endl;
                }

                const string queueOffset = n->second.isDelayRequired() ? "writeDelayOffset + " : "";
                if (n->second.isSpikeEventRequired()) {
                    os << "if (threadIdx.x < spkEvntCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpkEvnt" << n->first << "[" << queueOffset << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
                    }   // end if (threadIdx.x < spkEvntCount)
                }

                if (!nm->getThresholdConditionCode().empty()) {
                    string queueOffsetTrueSpk = n->second.isTrueSpikeRequired() ? queueOffset : "";

                    os << "if (threadIdx.x < spkCount)";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_glbSpk" << n->first << "[" << queueOffsetTrueSpk << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << std::endl;
                        if (n->second.isSpikeTimeRequired()) {
                            os << "dd_sT" << n->first << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << std::endl;
                        }
                    }   // end if (threadIdx.x < spkCount)
                }
            }   // end if (id < model.padSumNeuronN[i] )
            os << std::endl;
        }
    }   // end of neuron kernel

    os << "#endif" << std::endl;
    fs.close();
}

//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are 
  synapse-related and the actual CUDA kernel for simulating one time step of 
  the synapses.
*/
//-------------------------------------------------------------------------

void genSynapseKernel(const NNmodel &model, //!< Model description
                      const string &path,   //!< Path for code generation
                      int localHostID)      //!< ID of local host
{
    string localID; //!< "id" if first synapse group, else "lid". lid =(thread index- last thread of the last synapse group)
    ofstream fs;
    string name = model.getGeneratedCodePath(path, "synapseKrnl.cc");
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // write header content
    writeHeader(os);
    os << std::endl;

    // compiler/include control (include once)
    os << "#ifndef _" << model.getName() << "_synapseKrnl_cc" << std::endl;
    os << "#define _" << model.getName() << "_synapseKrnl_cc" << std::endl;
    os << "#define BLOCKSZ_SYN " << synapseBlkSz << std::endl;
    os << std::endl;
 
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file synapseKrnl.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName();
    os << " containing the synapse kernel and learning kernel functions." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    // If a reset kernel is required to be run before the synapse kernel
    if(model.isPreSynapseResetRequired())
    {
        // pre synapse reset kernel header
        os << "extern \"C\" __global__ void preSynapseReset()";
        {
            CodeStream::Scope b(os);

            os << "unsigned int id = " << preSynapseResetBlkSize << " * blockIdx.x + threadIdx.x;" << std::endl;

            // Loop through neuron groups
            unsigned int groupID = 0;
            for(const auto &n : model.getLocalNeuronGroups()) {
                // Loop through incoming synaptic populations
                for(const auto &m : n.second.getMergedInSyn()) {
                    const auto *sg = m.first;

                     // If this kernel requires dendritic delay
                    if(sg->isDendriticDelayRequired()) {
                        if(groupID > 0) {
                            os << "else ";
                        }
                        os << "if(id == " << (groupID++) << ")";
                        {
                            CodeStream::Scope b(os);

                            os << "dd_denDelayPtr" << sg->getPSModelTargetName() << " = (dd_denDelayPtr" << sg->getPSModelTargetName() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;
                        }
                    }
                }
            }
        }
    }

    if (!model.getSynapseDynamicsGroups().empty()) {
        os << "#define BLOCKSZ_SYNDYN " << synDynBlkSz << endl;

        // SynapseDynamics kernel header
        os << "extern \"C\" __global__ void calcSynapseDynamics(";
        for(const auto &p : model.getSynapseDynamicsKernelParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header

        // synapse dynamics kernel code
        {
            CodeStream::Scope b(os);

            // common variables for all cases
            os << "unsigned int id = BLOCKSZ_SYNDYN * blockIdx.x + threadIdx.x;" << std::endl;
            os << model.getPrecision() << " addtoinSyn;" << std::endl;
            os << std::endl;

            os << "// execute internal synapse dynamics if any" << std::endl;
            for(auto s = model.getSynapseDynamicsGroups().cbegin(); s != model.getSynapseDynamicsGroups().cend(); ++s)
            {
                const SynapseGroup *sg = model.findSynapseGroup(s->first);
                const auto *wu = sg->getWUModel();

                // if there is some internal synapse dynamics
                if (!wu->getSynapseDynamicsCode().empty()) {
                    // Create iteration context to iterate over the variables and derived parameters
                    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                    VarNameIterCtx wuVars(wu->getVars());

                    os << "// synapse group " << s->first << std::endl;
                    if (s == model.getSynapseDynamicsGroups().cbegin()) {
                        os << "if (id < " << s->second.second << ")";
                        localID = "id";
                    }
                    else {
                        os << "if ((id >= " << s->second.first << ") && (id < " << s->second.second << "))";
                        localID = "lid";
                    }
                    {
                        CodeStream::Scope b(os);
                        if(s != model.getSynapseDynamicsGroups().cbegin()) {
                            os << "unsigned int lid = id - " << s->second.first << ";" << std::endl;
                        }

                        // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                        if(sg->getSrcNeuronGroup()->isDelayRequired()) {
                            os << "const unsigned int preReadDelayOffset = " << sg->getPresynapticAxonalDelaySlot("dd_") << " * " << sg->getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                        }

                        // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                        if(sg->getTrgNeuronGroup()->isDelayRequired()) {
                            os << "const unsigned int postReadDelayOffset = " << sg->getTrgNeuronGroup()->getCurrentQueueOffset("dd_") << ";" << std::endl;
                        }

                        string SDcode = wu->getSynapseDynamicsCode();
                        substitute(SDcode, "$(t)", "t");

                        if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                            if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                os << "if (" << localID << " < dd_indInG" << s->first << "[" << sg->getSrcNeuronGroup()->getNumNeurons() << "])";
                            }
                            else {
                                os << "if (" << localID << " < dd_synRemap" << s->first << "[0])";
                            }
                            {
                                CodeStream::Scope b(os);

                                // Determine synapse and presynaptic indices for this thread
                                std::string synIdx;
                                std::string preIdx;
                                if(sg->getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                                    os << "const unsigned int s = dd_synRemap" << s->first << "[1 + " << localID << "];" << std::endl;
                                    synIdx = "s";
                                    preIdx = "s / " + to_string(sg->getMaxConnections());

                                }
                                else {
                                    synIdx = localID;
                                    preIdx = "dd_preInd" + s->first +"[" + localID + "]";
                                }

                                // Determine postsynaptic index from ind array
                                const std::string postIdx = "dd_ind" + s->first + "[" + synIdx + "]";

                                os << "// all threads participate that can work on an existing synapse" << std::endl;
                                if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                                    os << " using namespace " << s->first << "_weightupdate_synapseDynamics;" << std::endl;
                                }
                                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                    // name substitute synapse var names in synapseDynamics code
                                    name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s->first + "[" + synIdx +"]");
                                }

                                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                                if(sg->isDendriticDelayRequired()) {
                                    functionSubstitute(SDcode, "addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg->getPSModelTargetName() + "[" + sg->getDendriticDelayOffset("dd_", "$(1)") + postIdx + "], $(0))");
                                }
                                // Otherwise
                                else {
                                    functionSubstitute(SDcode, "addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg->getPSModelTargetName() + "[" + postIdx + "], $(0))");

                                    // **DEPRECATED**
                                    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(model.getPrecision()) + "(&$(inSyn), $(addtoinSyn))");
                                    substitute(SDcode, "$(inSyn)", "dd_inSyn" + sg->getPSModelTargetName() + "[" + postIdx + "]");
                                }

                                StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                            preIdx, postIdx, "dd_", cudaFunctions, model.getPrecision());
                                os << SDcode << std::endl;
                            }
                        }
                        else { // DENSE
                            os << "if (" << localID << " < " << sg->getSrcNeuronGroup()->getNumNeurons() * sg->getTrgNeuronGroup()->getNumNeurons() << ")";
                            {
                                CodeStream::Scope b(os);
                                os << "// all threads participate that can work on an existing synapse" << std::endl;
                                if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                                        os << " using namespace " << s->first << "_weightupdate_synapseDynamics;" << std::endl;
                                }
                                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                    // name substitute synapse var names in synapseDynamics code
                                    name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s->first + "[" + localID + "]");
                                }

                                const std::string postIdx = localID +"%" + to_string(sg->getTrgNeuronGroup()->getNumNeurons());

                                // If dendritic delay is required, always use atomic operation to update dendritic delay buffer
                                if(sg->isDendriticDelayRequired()) {
                                    functionSubstitute(SDcode, "addToInSynDelay", 2, getFloatAtomicAdd(model.getPrecision()) + "(&dd_denDelay" + sg->getPSModelTargetName() + "[" + sg->getDendriticDelayOffset("dd_", "$(1)") + postIdx + "], $(0))");
                                }
                                // Otherwise
                                else {
                                    functionSubstitute(SDcode, "addToInSyn", 1, getFloatAtomicAdd(model.getPrecision()) + "(&dd_inSyn" + sg->getPSModelTargetName() + "[" + postIdx + "], $(0))");

                                    // **DEPRECATED**
                                    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(model.getPrecision()) + "(&$(inSyn), $(addtoinSyn))");
                                    substitute(SDcode, "$(inSyn)", "dd_inSyn" + sg->getPSModelTargetName() + "[" + postIdx + "]");
                                }

                                StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                            localID +"/" + to_string(sg->getTrgNeuronGroup()->getNumNeurons()),
                                                                            postIdx, "dd_", cudaFunctions, model.getPrecision());
                                os << SDcode << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    // count how many neuron blocks to use: one thread for each synapse target
    // targets of several input groups are counted multiply
    const unsigned int numSynapseBlocks = model.getSynapseKernelGridSize() / synapseBlkSz;

    // synapse kernel header
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getTimePrecision() << " t)" << std::endl; // end of synapse kernel header

    // synapse kernel code
    {
        CodeStream::Scope b(os);
        // common variables for all cases
        os << "unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;" << std::endl;
        os << "unsigned int lmax, j, r;" << std::endl;
        os << model.getPrecision() << " addtoinSyn;" << std::endl;

        // We need shLg if any synapse groups accumulate into shared memory
        if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
            [](const NNmodel::SynapseGroupValueType &s){ return shouldAccumulateInSharedMemory(s.second); }))
        {
            os << "__shared__ " << model.getPrecision() << " shLg[BLOCKSZ_SYN];" << std::endl;
        }

        // We need linsyn if any synapse groups accumulate directly into a register
        for(const auto &s : model.getLocalSynapseGroups()) {
            if (shouldAccumulateInLinSyn(s.second)) {
                os << model.getPrecision() << " linSyn;" << std::endl;
                break;
            }
        }
        // we need ipost in any case, and we need npost if there are any SPARSE connections
        os << "unsigned int ipost;" << std::endl;
        for(const auto &s : model.getLocalSynapseGroups()) {
            if (s.second.getMatrixType()  & SynapseMatrixConnectivity::SPARSE){
                os << "unsigned int prePos; " << std::endl;
                os << "unsigned int npost; " << std::endl;
                break;
            }
        }
        for(const auto &s : model.getLocalSynapseGroups()) {
            if (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first)) {
                os << "__shared__ unsigned int shSpk[BLOCKSZ_SYN];" << std::endl;
                //os << "__shared__ " << model.getPrecision() << " shSpkV[BLOCKSZ_SYN];" << std::endl;
                os << "unsigned int lscnt, numSpikeSubsets;" << std::endl;
                break;
            }
        }
        for(const auto &s : model.getLocalSynapseGroups()) {
            if (s.second.isSpikeEventRequired()) {
                os << "__shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];" << std::endl;
                os << "unsigned int lscntEvnt, numSpikeSubsetsEvnt;" << std::endl;
                break;
            }
        }
        os << std::endl;

        for(auto s = model.getLocalSynapseGroups().cbegin(); s != model.getLocalSynapseGroups().cend(); ++s) {
            os << "// synapse group " << s->first << std::endl;
            const auto &groupIDRange = s->second.getPaddedKernelIDRange();
            if (s == model.getLocalSynapseGroups().cbegin()) {
                os << "if (id < " << groupIDRange.second << ")";
                localID = "id";
            }
            else {
                os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))";
                localID = "lid";
            }
            {
                CodeStream::Scope b(os);
                if(s != model.getLocalSynapseGroups().cbegin()) {
                    os << "unsigned int lid = id - " << groupIDRange.first<< ";" << std::endl;
                }

                // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                if(s->second.getSrcNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int preReadDelaySlot = " << s->second.getPresynapticAxonalDelaySlot("dd_") << ";" << std::endl;
                    os << "const unsigned int preReadDelayOffset = preReadDelaySlot * " << s->second.getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                }

                // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                if(s->second.getTrgNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int postReadDelayOffset = " << s->second.getTrgNeuronGroup()->getCurrentQueueOffset("dd_") << ";" << std::endl;
                }

                // If we are going to accumulate postsynaptic input into a register, copy current value into register from global memory
                if (shouldAccumulateInLinSyn(s->second)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << "if (" << localID << " < " << s->second.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "linSyn = dd_inSyn" << s->second.getPSModelTargetName() << "[" << localID << "];" << std::endl;
                    }
                }
                // Otherwise, if we are going to accumulate into shared memory, copy current value into correct array index
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(s->second)) {
                    os << "if (threadIdx.x < " << s->second.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "shLg[threadIdx.x] = dd_inSyn" << s->second.getPSModelTargetName() << "[threadIdx.x];"<< std::endl;
                    }
                    os << "__syncthreads();" << std::endl;
                }

                if (s->second.isSpikeEventRequired()) {
                    os << "lscntEvnt = dd_glbSpkCntEvnt" << s->second.getSrcNeuronGroup()->getName();
                    if (s->second.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[preReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                    os << "numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << std::endl;
                }

                if (s->second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s->first)) {
                    os << "lscnt = dd_glbSpkCnt" << s->second.getSrcNeuronGroup()->getName();
                    if (s->second.getSrcNeuronGroup()->isDelayRequired()) {
                        os << "[preReadDelaySlot];" << std::endl;
                    }
                    else {
                        os << "[0];" << std::endl;
                    }
                    os << "numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << std::endl;
                }

                // generate the code for processing spike-like events
                if (s->second.isSpikeEventRequired()) {
                    generate_process_presynaptic_events_code(os, s->second, localID, "Evnt", model.getPrecision());
                }

                // generate the code for processing true spike events
                if (s->second.isTrueSpikeRequired()) {
                    generate_process_presynaptic_events_code(os, s->second, localID, "", model.getPrecision());
                }
                os << std::endl;

                // If we have been accumulating into a register, write value back to global memory
                if (shouldAccumulateInLinSyn(s->second)) {
                    os << "// only do this for existing neurons" << std::endl;
                    os << "if (" << localID << " < " << s->second.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << s->second.getPSModelTargetName() << "[" << localID << "] = linSyn;" << std::endl;
                    }
                }
                // Otherwise, if we have been accumulating into shared memory, write value back to global memory
                // **NOTE** is ok as number of target neurons <= synapseBlkSz
                else if(shouldAccumulateInSharedMemory(s->second)) {
                    os << "__syncthreads();" << std::endl;
                    os << "if (threadIdx.x < " << s->second.getTrgNeuronGroup()->getNumNeurons() << ")";
                    {
                        CodeStream::Scope b(os);
                        os << "dd_inSyn" << s->second.getPSModelTargetName() << "[threadIdx.x] = shLg[threadIdx.x];"<< std::endl;
                    }
                }

                // need to do reset operations in this kernel (no learning kernel)
                if (model.getResetKernel() == GENN_FLAGS::calcSynapses) {
                    os << "__syncthreads();" << std::endl;
                    os << "if (threadIdx.x == 0)";
                    {
                        CodeStream::Scope b(os);
                        os << "j = atomicAdd((unsigned int *) &d_done, 1);" << std::endl;
                        os << "if (j == " << numSynapseBlocks - 1 << ")";
                        {
                            // Update device delay slot pointers for remote neuron groups that require them
                            CodeStream::Scope b(os);
                            for(const auto &n : model.getRemoteNeuronGroups()) {
                                if(n.second.hasOutputToHost(localHostID) && n.second.isDelayRequired()) {
                                    os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
                                }
                            }
                            for(const auto &n : model.getLocalNeuronGroups()) {
                                StandardGeneratedSections::neuronOutputInit(os, n.second, "dd_");
                            }
                            os << "d_done = 0;" << std::endl;
                        }   // end "if (j == " << numOfBlocks - 1 << ")"
                    }   // end "if (threadIdx.x == 0)"
                }
            }
            os << std::endl;
        }
    }
    os << std::endl;


    ///////////////////////////////////////////////////////////////
    // Kernel for learning synapses, post-synaptic spikes

    if (!model.getSynapsePostLearnGroups().empty()) {

        // count how many learn blocks to use: one thread for each synapse source
        // sources of several output groups are counted multiply
        const unsigned int numPostLearnBlocks = model.getSynapsePostLearnGridSize() / learnBlkSz;
  
        // Kernel header
        os << "extern \"C\" __global__ void learnSynapsesPost(";
        for(const auto &p : model.getSimLearnPostKernelParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getTimePrecision() << " t)";
        os << std::endl;

        // kernel code
        {
            CodeStream::Scope b(os);
            os << "unsigned int id = " << learnBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;
            os << "__shared__ unsigned int shSpk[" << learnBlkSz << "];" << std::endl;
            os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << std::endl;
            os << std::endl;
            for(auto s = model.getSynapsePostLearnGroups().cbegin(); s != model.getSynapsePostLearnGroups().cend(); ++s)
            {
                const SynapseGroup *sg = model.findSynapseGroup(s->first);
                const auto *wu = sg->getWUModel();
                const bool sparse = sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE;

                // NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM
                os << "// synapse group " << s->first << std::endl;
                if (s == model.getSynapsePostLearnGroups().cbegin()) {
                    os << "if (id < " << s->second.second << ")";
                    localID = "id";
                }
                else {
                    os << "if ((id >= " << s->second.first << ") && (id < " << s->second.second << "))";
                    localID = "lid";
                }
                {
                    CodeStream::Scope b(os);

                    if(s != model.getSynapsePostLearnGroups().cbegin()) {
                        os << "unsigned int lid = id - " << s->second.first << ";" << std::endl;
                    }

                    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
                    if(sg->getSrcNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int preReadDelayOffset = " << sg->getPresynapticAxonalDelaySlot("dd_") << " * " << sg->getSrcNeuronGroup()->getNumNeurons() << ";" << std::endl;
                    }

                    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
                    if(sg->getTrgNeuronGroup()->isDelayRequired()) {
                        os << "const unsigned int postReadDelayOffset = " << sg->getTrgNeuronGroup()->getCurrentQueueOffset("dd_") << ";" << std::endl;
                    }

                    if (sg->getTrgNeuronGroup()->isDelayRequired() && sg->getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[dd_spkQuePtr" << sg->getTrgNeuronGroup()->getName() << "];" << std::endl;
                    }
                    else {
                        os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[0];" << std::endl;
                    }

                    os << "numSpikeSubsets = (lscnt+" << learnBlkSz-1 << ") / " << learnBlkSz << ";" << std::endl;
                    os << "for (r = 0; r < numSpikeSubsets; r++)";
                    {
                        CodeStream::Scope b(os);
                        os << "if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % " << learnBlkSz << ")+1;" << std::endl;
                        os << "else lmax = " << learnBlkSz << ";" << std::endl;

                        os << "if (threadIdx.x < lmax)";
                        {
                            CodeStream::Scope b(os);
                            const string offsetTrueSpkPost = (sg->getTrgNeuronGroup()->isTrueSpikeRequired() && sg->getTrgNeuronGroup()->isDelayRequired()) ? "postReadDelayOffset + " : "";
                            os << "shSpk[threadIdx.x] = dd_glbSpk" << sg->getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << learnBlkSz << ") + threadIdx.x];" << std::endl;
                        }

                        os << "__syncthreads();" << std::endl;
                        os << "// only work on existing neurons" << std::endl;
                        os << "if (" << localID << " < " << sg->getMaxSourceConnections() << ")";
                        {
                            CodeStream::Scope b(os);
                            os << "// loop through all incoming spikes for learning" << std::endl;
                            os << "for (j = 0; j < lmax; j++)";
                            {
                                CodeStream::Scope b(os);
                                if (sparse) {
                                    if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                        os << "unsigned int iprePos = dd_revIndInG" <<  s->first << "[shSpk[j]];" << std::endl;
                                        os << "unsigned int npre = dd_revIndInG" << s->first << "[shSpk[j] + 1] - iprePos;" << std::endl;
                                    }
                                    else {
                                        os << "unsigned int iprePos = shSpk[j] * " << to_string(sg->getMaxSourceConnections()) << ";" << std::endl;
                                        os << "unsigned int npre = dd_colLength" << s->first << "[shSpk[j]];" << std::endl;
                                    }
                                    os << "if (" << localID << " < npre)" << CodeStream::OB(1540);
                                    os << "iprePos += " << localID << ";" << std::endl;
                                    //Commenting out the next line as it is not used rather than deleting as I'm not sure if it may be used by different learning models
                                    //os << "unsigned int ipre = dd_revInd" << sgName << "[iprePos];" << std::endl;
                                }

                                if (!wu->getLearnPostSupportCode().empty()) {
                                    os << " using namespace " << s->first << "_weightupdate_simLearnPost;" << std::endl;
                                }

                                // Create iteration context to iterate over the variables; derived and extra global parameters
                                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                                VarNameIterCtx wuVars(wu->getVars());

                                string code = wu->getLearnPostCode();
                                substitute(code, "$(t)", "t");
                                // Code substitutions ----------------------------------------------------------------------------------
                                std::string preIndex;
                                if (sparse) { // SPARSE
                                    name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, s->first + "[dd_remap" + s->first + "[iprePos]]");
                                    
                                    if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                        preIndex = "dd_revInd" + s->first + "[iprePos]";
                                    }
                                    else {
                                        preIndex = "(dd_remap" + s->first + "[iprePos] / " + to_string(sg->getMaxConnections()) + ")";
                                    }
                                }
                                else { // DENSE
                                    name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, s->first + "[" + localID + " * " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + " + shSpk[j]]");
                                    preIndex = localID;
                                }
                                StandardSubstitutions::weightUpdatePostLearn(code, sg, wuDerivedParams, wuExtraGlobalParams,
                                                                            preIndex, "shSpk[j]", "dd_", cudaFunctions, model.getPrecision());
                                // end Code substitutions -------------------------------------------------------------------------
                                os << code << std::endl;
                                if (sparse) {
                                    os << CodeStream::CB(1540);
                                }
                            }
                        }
                    }
                    if (model.getResetKernel() == GENN_FLAGS::learnSynapsesPost) {
                        os << "__syncthreads();" << std::endl;
                        os << "if (threadIdx.x == 0)";
                        {
                            CodeStream::Scope b(os);
                            os << "j = atomicAdd((unsigned int *) &d_done, 1);" << std::endl;
                            os << "if (j == " << numPostLearnBlocks - 1 << ")";
                            {
                                CodeStream::Scope b(os);

                                // Update device delay slot pointers for remote neuorn groups that require them
                                for(const auto &n : model.getRemoteNeuronGroups()) {
                                    if(n.second.hasOutputToHost(localHostID) && n.second.isDelayRequired()) {
                                        os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
                                    }
                                }
                                for(const auto &n : model.getLocalNeuronGroups()) {
                                    StandardGeneratedSections::neuronOutputInit(os, n.second, "dd_");
                                }
                                os << "d_done = 0;" << std::endl;
                            }   // end "if (j == " << numOfBlocks - 1 << ")"
                        }   // end "if (threadIdx.x == 0)"
                    }
                }
            }
        }
    }
    os << std::endl;
    
    os << "#endif" << std::endl;
    fs.close();

//    cout << "exiting genSynapseKernel" << endl;
}

#endif
