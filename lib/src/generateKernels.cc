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
// parallelisation along pre-synaptic spikes, looped over post-synaptic neurons
void generatePreParallelisedSparseCode(
    CodeStream &os, //!< output stream for code
    const SynapseGroup &sg,
    const string &localID, //!< the variable name of the local ID of the thread within the synapse group
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    const bool evnt = (postfix == "Evnt");
    const int UIntSz = sizeof(unsigned int) * 8;
    const int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f);
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    //int maxConnections;
    if (sg.isPSAtomicAddRequired(synapseBlkSz)) {
        if (sg.getMaxConnections() < 1) {
            fprintf(stderr, "Model Generation warning: for every SPARSE synapse group used you must also supply (in your model)\
a max possible number of connections via the model.setMaxConn() function.\n");
        }
    }

    os << "if (" << localID << " < " ;
    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[delaySlot])" << CodeStream::OB(102);
    }
    else {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[0])" << CodeStream::OB(102);
    }

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
    }

    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[(delaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + " << localID << "];" << std::endl;
    }
    else {
        os << "int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[" << localID << "];" << std::endl;
    }
    os << "prePos = dd_indInG" << sg.getName() << "[preInd];" << std::endl;
    os << "npost = dd_indInG" << sg.getName() << "[preInd + 1] - prePos;" << std::endl;

    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "unsigned int gid = (dd_glbSpkCnt" << postfix << "[" << localID << "] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + i);" << std::endl;
    }

    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << "if ";
        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
            os << "((B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << ")) && ";
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
        os << "if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << CodeStream::OB(135);
    }

    os << "for (int i = 0; i < npost; ++i)" << CodeStream::OB(103);
    os << "        ipost = dd_ind" <<  sg.getName() << "[prePos];" << std::endl;

// Code substitutions ----------------------------------------------------------------------------------
    string wCode = evnt ? wu->getEventCode() : wu->getSimCode();
    substitute(wCode, "$(t)", "t");

    if (sg.isPSAtomicAddRequired(synapseBlkSz)) { // SPARSE using atomicAdd
        substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");
        substitute(wCode, "$(inSyn)", "dd_inSyn" + sg.getName() + "[ipost]");
    }
    else { // using shared memory
        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
        substitute(wCode, "$(inSyn)", "shLg[ipost]");
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
    os << CodeStream::CB(103);
    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CodeStream::CB(130);
    }
    else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << CodeStream::CB(135);
    }
    os << CodeStream::CB(102);
    //os << CodeStream::CB(101);
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
    const int UIntSz = sizeof(unsigned int) * 8;
    const int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f);
    const auto *wu = sg.getWUModel();

    // Create iteration context to iterate over the variables; derived and extra global parameters
    DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
    ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
    VarNameIterCtx wuVars(wu->getVars());

    os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << std::endl;
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)" << CodeStream::OB(90);
    os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = ((lscnt" << postfix << "-1) % BLOCKSZ_SYN) +1;" << std::endl;
    os << "else lmax = BLOCKSZ_SYN;" << std::endl;
    os << "__syncthreads();" << std::endl;
    os << "if (threadIdx.x < lmax)" << CodeStream::OB(100);
    os << "shSpk" << postfix << "[threadIdx.x] = dd_glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << sg.getOffsetPre() << "(r * BLOCKSZ_SYN) + threadIdx.x];" << std::endl;
    os << CodeStream::CB(100);

    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !sg.isPSAtomicAddRequired(synapseBlkSz)) {
        // set shLg to 0 for all postsynaptic neurons; is ok as model.neuronN[model.synapseTarget[i]] <= synapseBlkSz
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ") shLg[threadIdx.x] = 0;" << std::endl;
    }
    os << "__syncthreads();" << std::endl;

    int maxConnections;
    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.isPSAtomicAddRequired(synapseBlkSz)) {
        if (sg.getMaxConnections() < 1) {
            fprintf(stderr, "Model Generation warning: for every SPARSE synapse group used you must also supply (in your model)\
a max possible number of connections via the model.setMaxConn() function.\n");
            maxConnections = sg.getTrgNeuronGroup()->getNumNeurons();
        }
        else {
            maxConnections = sg.getMaxConnections();
        }
    }
    else {
        maxConnections = sg.getTrgNeuronGroup()->getNumNeurons();
    }
    os << "// loop through all incoming spikes" << std::endl;
    os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(110);
    os << "// only work on existing neurons" << std::endl;
    os << "if (" << localID << " < " << maxConnections << ")" << CodeStream::OB(120);
    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "unsigned int gid = (shSpk" << postfix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << localID << ");" << std::endl;
    }

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << std::endl;
    }
    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << "if ";
        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            // Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp will be coalesced - no worries
            os << "((B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << ")) && ";
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
        os << "if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << CodeStream::OB(135);
    }

    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
        os << "prePos = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j]];" << std::endl;
        os << "npost = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j] + 1] - prePos;" << std::endl;
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
    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
        if (sg.isPSAtomicAddRequired(synapseBlkSz)) { // SPARSE using atomicAdd
            substitute(wCode, "$(updatelinsyn)", getFloatAtomicAdd(ftype) + "(&$(inSyn), $(addtoinSyn))");
            substitute(wCode, "$(inSyn)", "dd_inSyn" + sg.getName() + "[ipost]");
        }
        else { // SPARSE using shared memory
            substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
            substitute(wCode, "$(inSyn)", "shLg[ipost]");
        }

        if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[prePos]");
        }
    }
    else { // DENSE
        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
        substitute(wCode, "$(inSyn)", "linSyn");
        if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            name_substitutions(wCode, "dd_", wuVars.nameBegin, wuVars.nameEnd, sg.getName() + "[shSpk"
                                + postfix + "[j] * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + "+ ipost]");
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
        os << CodeStream::CB(135); // end if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid
    }
    os << CodeStream::CB(120) << std::endl;

    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !sg.isPSAtomicAddRequired(synapseBlkSz)) {
        os << "__syncthreads();" << std::endl;
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(136); // need to write back results
        os << "linSyn += shLg[" << localID << "];" << std::endl;
        os << "shLg[" << localID << "] = 0;" << std::endl;
        os << CodeStream::CB(136) << std::endl;

        os << "__syncthreads();" << std::endl;
    }
    os << CodeStream::CB(110) << std::endl;
    os << CodeStream::CB(90) << std::endl;
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
        if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && sg.getSpanType() == SynapseGroup::SpanType::PRESYNAPTIC) {
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
    os << model.getPrecision() << " t)" << std::endl;
    os << CodeStream::OB(5);

    // kernel code
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
        os << "if (id == 0)" << CodeStream::OB(6);
        for(const auto &n : model.getLocalNeuronGroups()) {
            StandardGeneratedSections::neuronOutputInit(os, n.second, "dd_");
        }
        os << CodeStream::CB(6);
        os << "__threadfence();" << std::endl << std::endl;
    }

    // Initialise shared spike count vars
    for(const auto &n : model.getLocalNeuronGroups()) {
        if (!n.second.getNeuronModel()->getThresholdConditionCode().empty()) {
            os << "if (threadIdx.x == 0)" << CodeStream::OB(8);
            os << "spkCount = 0;" << std::endl;
            os << CodeStream::CB(8);
            break;
        }
    }
    for(const auto &n : model.getLocalNeuronGroups()) {
        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x == 1)" << CodeStream::OB(7);
            os << "spkEvntCount = 0;" << std::endl;
            os << CodeStream::CB(7);
            break;
        }
    }
    os << "__syncthreads();" << std::endl;
    os << std::endl;

    
    bool firstNeuronGroup = true;
    for(const auto &n : model.getLocalNeuronGroups()) {
        os << "// neuron group " << n.first << std::endl;
        const auto &groupIDRange = n.second.getPaddedIDRange();
        if (firstNeuronGroup) {
            os << "if (id < " << groupIDRange.second << ")" << CodeStream::OB(10);
            localID = "id";
            firstNeuronGroup = false;
        }
        else {
            os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))" << CodeStream::OB(10);
            os << "unsigned int lid = id - " << groupIDRange.first << ";" << std::endl;
            localID = "lid";
        }

        if (n.second.isVarQueueRequired() && n.second.isDelayRequired()) {
            os << "unsigned int delaySlot = (dd_spkQuePtr" << n.first;
            os << " + " << (n.second.getNumDelaySlots() - 1);
            os << ") % " << n.second.getNumDelaySlots() << ";" << std::endl;
        }
        os << std::endl;

        os << "// only do this for existing neurons" << std::endl;
        os << "if (" << localID << " < " << n.second.getNumNeurons() << ")" << CodeStream::OB(20);

        os << "// pull neuron variables in a coalesced access" << std::endl;

        const auto *nm = n.second.getNeuronModel();

        // Get name of rng to use for this neuron
        // **TODO** Phillox option
        const std::string rngName = "&dd_rng" + n.first + "[" + localID + "]";

        // Create iteration context to iterate over the variables; derived and extra global parameters
        VarNameIterCtx nmVars(nm->getVars());
        DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
        ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

        // Generate code to copy neuron state into local variables
        StandardGeneratedSections::neuronLocalVarInit(os, n.second, nmVars, "dd_", localID);

        if ((nm->getSimCode().find("$(sT)") != string::npos)
            || (nm->getThresholdConditionCode().find("$(sT)") != string::npos)
            || (nm->getResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
            os << model.getPrecision() << " lsT = dd_sT" <<  n.first << "[";
            if (n.second.isDelayRequired()) {
                os << "(delaySlot * " << n.second.getNumNeurons() << ") + ";
            }
            os << localID << "];" << std::endl;
        }
        os << std::endl;

        if (n.second.getInSyn().size() > 0 || (nm->getSimCode().find("Isyn") != string::npos)) {
            os << model.getPrecision() << " Isyn = 0;" << std::endl;
        }

        // Initialise any additional input variables supported by neuron model
        for(const auto &a : nm->getAdditionalInputVars()) {
            os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
        }

        for(const auto *sg : n.second.getInSyn()) {
            const auto *psm = sg->getPSModel();

            os << "// pull inSyn values in a coalesced access" << std::endl;
            os << model.getPrecision() << " linSyn" << sg->getName() << " = dd_inSyn" << sg->getName() << "[" << localID << "];" << std::endl;
            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : psm->getVars()) {
                    os << v.second << " lps" << v.first << sg->getName();
                    os << " = dd_" <<  v.first << sg->getName() << "[" << localID << "];" << std::endl;
                }
            }
            string psCode = psm->getApplyInputCode();
            substitute(psCode, "$(id)", localID);
            substitute(psCode, "$(inSyn)", "linSyn" + sg->getName());
            StandardSubstitutions::postSynapseApplyInput(psCode, sg, n.second,
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
            os << " using namespace " << n.first << "_neuron;" << std::endl;
        }
        string thCode = nm->getThresholdConditionCode();
        if (thCode.empty()) { // no condition provided
            cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << n.first << "\" was provided. There will be no spikes detected in this population!" << endl;
        }
        else {
            os << "// test whether spike condition was fulfilled previously" << std::endl;
            substitute(thCode, "$(id)", localID);
            StandardSubstitutions::neuronThresholdCondition(thCode, n.second,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            cudaFunctions, model.getPrecision(), rngName);
            if (GENN_PREFERENCES::autoRefractory) {
                os << "bool oldSpike= (" << thCode << ");" << std::endl;
            }
        }

        os << "// calculate membrane potential" << std::endl;
        string sCode = nm->getSimCode();
        substitute(sCode, "$(id)", localID);
        StandardSubstitutions::neuronSim(sCode, n.second,
                                         nmVars, nmDerivedParams, nmExtraGlobalParams,
                                         cudaFunctions, model.getPrecision(), rngName);
        os << sCode << std::endl;

        // look for spike type events first.
        if (n.second.isSpikeEventRequired()) {
           // Generate spike event test
            StandardGeneratedSections::neuronSpikeEventTest(os, n.second,
                                                            nmVars, nmExtraGlobalParams, localID,
                                                            cudaFunctions, model.getPrecision(), rngName);

            os << "// register a spike-like event" << std::endl;
            os << "if (spikeLikeEvent)" << CodeStream::OB(30);
            os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << std::endl;
            os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << std::endl;
            os << CodeStream::CB(30);
        }

        // test for true spikes if condition is provided
        if (!thCode.empty()) {
            os << "// test for and register a true spike" << std::endl;
            if (GENN_PREFERENCES::autoRefractory) {
                os << "if ((" << thCode << ") && !(oldSpike)) " << CodeStream::OB(40);
            }
            else {
                os << "if (" << thCode << ") " << CodeStream::OB(40);
            }
            os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << std::endl;
            os << "shSpk[spkIdx] = " << localID << ";" << std::endl;

            // add after-spike reset if provided
            if (!nm->getResetCode().empty()) {
                string rCode = nm->getResetCode();
                substitute(rCode, "$(id)", localID);
                StandardSubstitutions::neuronReset(rCode, n.second,
                                                   nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                   cudaFunctions, model.getPrecision(), rngName);
                os << "// spike reset code" << std::endl;
                os << rCode << std::endl;
            }
            os << CodeStream::CB(40);
        }

        // store the defined parts of the neuron state into the global state variables dd_V etc
        StandardGeneratedSections::neuronLocalVarWrite(os, n.second, nmVars, "dd_", localID);

        if (!n.second.getInSyn().empty()) {
            os << "// the post-synaptic dynamics" << std::endl;
        }
        for(const auto *sg : n.second.getInSyn()) {
            const auto *psm = sg->getPSModel();

            string pdCode = psm->getDecayCode();
            substitute(pdCode, "$(id)", localID);
            substitute(pdCode, "$(inSyn)", "linSyn" + sg->getName());
            StandardSubstitutions::postSynapseDecay(pdCode, sg, n.second,
                                                    nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                    cudaFunctions, model.getPrecision(), rngName);
            if (!psm->getSupportCode().empty()) {
                os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
            }
            os << pdCode << std::endl;
            if (!psm->getSupportCode().empty()) {
                os << CodeStream::CB(29) << " // namespace bracket closed" << endl;
            }

            os << "dd_inSyn"  << sg->getName() << "[" << localID << "] = linSyn" << sg->getName() << ";" << std::endl;
            for(const auto &v : psm->getVars()) {
                os << "dd_" <<  v.first << sg->getName() << "[" << localID << "] = lps" << v.first << sg->getName() << ";"<< std::endl;
            }
        }

        os << CodeStream::CB(20);
        os << "__syncthreads();" << std::endl;

        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x == 1)" << CodeStream::OB(50);
            os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << n.first;
            if (n.second.isDelayRequired()) {
                os << "[dd_spkQuePtr" << n.first << "], spkEvntCount);" << std::endl;
            }
            else {
                os << "[0], spkEvntCount);" << std::endl;
            }
            os << CodeStream::CB(50); // end if (threadIdx.x == 0)
            os << "__syncthreads();" << std::endl;
        }

        if (!nm->getThresholdConditionCode().empty()) {
            os << "if (threadIdx.x == 0)" << CodeStream::OB(51);
            os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << n.first;
            if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
                os << "[dd_spkQuePtr" << n.first << "], spkCount);" << std::endl;
            }
            else {
                os << "[0], spkCount);" << std::endl;
            }
            os << CodeStream::CB(51); // end if (threadIdx.x == 1)

            os << "__syncthreads();" << std::endl;
        }

        string queueOffset = n.second.getQueueOffset("dd_");
        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x < spkEvntCount)" << CodeStream::OB(60);
            os << "dd_glbSpkEvnt" << n.first << "[" << queueOffset << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << std::endl;
            os << CodeStream::CB(60); // end if (threadIdx.x < spkEvntCount)
        }

        if (!nm->getThresholdConditionCode().empty()) {
            string queueOffsetTrueSpk = n.second.isTrueSpikeRequired() ? queueOffset : "";

            os << "if (threadIdx.x < spkCount)" << CodeStream::OB(70);
            os << "dd_glbSpk" << n.first << "[" << queueOffsetTrueSpk << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << std::endl;
            if (n.second.isSpikeTimeRequired()) {
                os << "dd_sT" << n.first << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << std::endl;
            }
            os << CodeStream::CB(70); // end if (threadIdx.x < spkCount)
        }
        os << CodeStream::CB(10); // end if (id < model.padSumNeuronN[i] )
        os << std::endl;
    }
    os << CodeStream::CB(5) << std::endl; // end of neuron kernel

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
                      const string &path) //!< Path for code generation
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


    if (!model.getSynapseDynamicsGroups().empty()) {
        os << "#define BLOCKSZ_SYNDYN " << synDynBlkSz << endl;

        // SynapseDynamics kernel header
        os << "extern \"C\" __global__ void calcSynapseDynamics(";
        for(const auto &p : model.getSynapseDynamicsKernelParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.getPrecision() << " t)" << std::endl; // end of synapse kernel header

        // synapse dynamics kernel code
        os << CodeStream::OB(75);

        // common variables for all cases
        os << "unsigned int id = BLOCKSZ_SYNDYN * blockIdx.x + threadIdx.x;" << std::endl;
        os << model.getPrecision() << " addtoinSyn;" << std::endl;
        os << std::endl;

        os << "// execute internal synapse dynamics if any" << std::endl;

        bool firstSynapseDynamicsGroup = true;
        for(const auto &s : model.getSynapseDynamicsGroups())
        {
            const SynapseGroup *sg = model.findSynapseGroup(s.first);
            const auto *wu = sg->getWUModel();

            // if there is some internal synapse dynamics
            if (!wu->getSynapseDynamicsCode().empty()) {
                // Create iteration context to iterate over the variables and derived parameters
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                VarNameIterCtx wuVars(wu->getVars());

                os << "// synapse group " << s.first << std::endl;
                if (firstSynapseDynamicsGroup) {
                    os << "if (id < " << s.second.second << ")" << CodeStream::OB(77);
                    localID = "id";
                    firstSynapseDynamicsGroup = false;
                }
                else {
                    os << "if ((id >= " << s.second.first << ") && (id < " << s.second.second << "))" << CodeStream::OB(77);
                    os << "unsigned int lid = id - " << s.second.first << ";" << std::endl;
                    localID = "lid";
                }

                if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                    os << "unsigned int delaySlot = (dd_spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                    os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                    os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                }

                string SDcode = wu->getSynapseDynamicsCode();
                substitute(SDcode, "$(t)", "t");

                if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                    os << "if (" << localID << " < dd_indInG" << s.first << "[" << sg->getSrcNeuronGroup()->getNumNeurons() << "])" << CodeStream::OB(25);
                    os << "// all threads participate that can work on an existing synapse" << std::endl;
                    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                        os << " using namespace " << s.first << "_weightupdate_synapseDynamics;" << std::endl;
                    }
                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                        // name substitute synapse var names in synapseDynamics code
                        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[" + localID +"]");
                    }

                    const std::string postIdx = "dd_ind" + s.first + "[" + localID + "]";
                    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(model.getPrecision()) + "(&$(inSyn), $(addtoinSyn))");
                    substitute(SDcode, "$(inSyn)", "dd_inSyn" + s.first + "[" + postIdx + "]");

                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                "dd_preInd" + s.first +"[" + localID + "]",
                                                                postIdx, "dd_", cudaFunctions, model.getPrecision());
                    os << SDcode << std::endl;
                }
                else { // DENSE
                    os << "if (" << localID << " < " << sg->getSrcNeuronGroup()->getNumNeurons() * sg->getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(25);
                    os << "// all threads participate that can work on an existing synapse" << std::endl;
                    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                            os << " using namespace " << s.first << "_weightupdate_synapseDynamics;" << std::endl;
                    }
                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                        // name substitute synapse var names in synapseDynamics code
                        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[" + localID + "]");
                    }

                    const std::string postIdx = localID +"%" + to_string(sg->getTrgNeuronGroup()->getNumNeurons());
                    substitute(SDcode, "$(updatelinsyn)", getFloatAtomicAdd(model.getPrecision()) + "(&$(inSyn), $(addtoinSyn))");
                    substitute(SDcode, "$(inSyn)", "dd_inSyn" + s.first + "[" + postIdx + "]");

                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                localID +"/" + to_string(sg->getTrgNeuronGroup()->getNumNeurons()),
                                                                postIdx, "dd_", cudaFunctions, model.getPrecision());
                    os << SDcode << std::endl;
                }
                os << CodeStream::CB(25);
                os << CodeStream::CB(77);
            }
        }
        os << CodeStream::CB(75);
    }

    // count how many neuron blocks to use: one thread for each synapse target
    // targets of several input groups are counted multiply
    const unsigned int numSynapseBlocks = model.getSynapseKernelGridSize() / synapseBlkSz;

    // synapse kernel header
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.getPrecision() << " t)" << std::endl; // end of synapse kernel header

    // synapse kernel code
    os << CodeStream::OB(75);

    // common variables for all cases
    os << "unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;" << std::endl;
    os << "unsigned int lmax, j, r;" << std::endl;
    os << model.getPrecision() << " addtoinSyn;" << std::endl;
    os << "volatile __shared__ " << model.getPrecision() << " shLg[BLOCKSZ_SYN];" << std::endl;

    // case-dependent variables
    for(const auto &s : model.getLocalSynapseGroups()) {
        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)){
            os << model.getPrecision() << " linSyn;" << std::endl;
            break;
        }
    }
    // we need ipost in any case, and we need npost if there are any SPARSE connections
    os << "unsigned int ipost;" << std::endl;
    for(const auto &s : model.getLocalSynapseGroups()) {
        if (s.second.getMatrixType()  & SynapseMatrixConnectivity::SPARSE) {
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

    bool firstSynapseGroup = true;
    for(const auto &s : model.getLocalSynapseGroups()) {
        os << "// synapse group " << s.first << std::endl;
        const auto &groupIDRange = s.second.getPaddedKernelIDRange();
        if (firstSynapseGroup) {
            os << "if (id < " << groupIDRange.second << ")" << CodeStream::OB(77);
            localID = "id";
            firstSynapseGroup = false;
        }
        else {
            os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))" << CodeStream::OB(77);
            os << "unsigned int lid = id - " << groupIDRange.first<< ";" << std::endl;
            localID = "lid";
        }

        if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
            os << "unsigned int delaySlot = (dd_spkQuePtr" << s.second.getSrcNeuronGroup()->getName();
            os << " + " << (s.second.getSrcNeuronGroup()->getNumDelaySlots() - s.second.getDelaySteps());
            os << ") % " << s.second.getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
        }

        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)){
            os << "// only do this for existing neurons" << std::endl;
            os << "if (" << localID << " < " << s.second.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(80);
            os << "linSyn = dd_inSyn" << s.first << "[" << localID << "];" << std::endl;
            os << CodeStream::CB(80);
        }

        if (s.second.isSpikeEventRequired()) {
            os << "lscntEvnt = dd_glbSpkCntEvnt" << s.second.getSrcNeuronGroup()->getName();
            if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << std::endl;
            }
            else {
                os << "[0];" << std::endl;
            }
            os << "numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << std::endl;
        }
  
        if (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first)) {
            os << "lscnt = dd_glbSpkCnt" << s.second.getSrcNeuronGroup()->getName();
            if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << std::endl;
            }
            else {
                os << "[0];" << std::endl;
            }
            os << "numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << std::endl;
        }

        // generate the code for processing spike-like events
        if (s.second.isSpikeEventRequired()) {
            generate_process_presynaptic_events_code(os, s.second, localID, "Evnt", model.getPrecision());
        }

        // generate the code for processing true spike events
        if (s.second.isTrueSpikeRequired()) {
            generate_process_presynaptic_events_code(os, s.second, localID, "", model.getPrecision());
        }
        os << std::endl;

        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)) {
            os << "// only do this for existing neurons" << std::endl;
            os << "if (" << localID << " < " << s.second.getTrgNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(190);
            os << "dd_inSyn" << s.first << "[" << localID << "] = linSyn;" << std::endl;
            os << CodeStream::CB(190);
        }
        // need to do reset operations in this kernel (no learning kernel)
        if (model.getResetKernel() == GENN_FLAGS::calcSynapses) {
            os << "__syncthreads();" << std::endl;
            os << "if (threadIdx.x == 0)" << CodeStream::OB(200);
            os << "j = atomicAdd((unsigned int *) &d_done, 1);" << std::endl;
            os << "if (j == " << numSynapseBlocks - 1 << ")" << CodeStream::OB(210);

            for(const auto &n : model.getLocalNeuronGroups()) {
                if (n.second.isDelayRequired()) { // WITH DELAY
                    os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    if (n.second.isTrueSpikeRequired()) {
                        os << "dd_glbSpkCnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                    }
                    else {
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                    }
                }
                else { // NO DELAY
                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
                    }
                    os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                }
            }
            os << "d_done = 0;" << std::endl;

            os << CodeStream::CB(210); // end "if (j == " << numOfBlocks - 1 << ")"
            os << CodeStream::CB(200); // end "if (threadIdx.x == 0)"
        }

        os << CodeStream::CB(77);
        os << std::endl;
    }
    os << CodeStream::CB(75);
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
        os << model.getPrecision() << " t)";
        os << std::endl;

        // kernel code
        os << CodeStream::OB(215);
        os << "unsigned int id = " << learnBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;
        os << "__shared__ unsigned int shSpk[" << learnBlkSz << "];" << std::endl;
        os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << std::endl;
        os << std::endl;

        bool firstPostLearnGroup = true;
        for(const auto &s : model.getSynapsePostLearnGroups())
        {
            const SynapseGroup *sg = model.findSynapseGroup(s.first);
            const auto *wu = sg->getWUModel();
            const bool sparse = sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE;

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM
            os << "// synapse group " << s.first << std::endl;
            if (firstPostLearnGroup) {
                os << "if (id < " << s.second.second << ")" << CodeStream::OB(220);
                localID = "id";
                firstPostLearnGroup = false;
            }
            else {
                os << "if ((id >= " << s.second.first << ") && (id < " << s.second.second << "))" << CodeStream::OB(220);
                os << "unsigned int lid = id - " << s.second.first << ";" << std::endl;
                localID = "lid";
            }

            if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                os << "unsigned int delaySlot = (dd_spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
            }

            if (sg->getTrgNeuronGroup()->isDelayRequired() && sg->getTrgNeuronGroup()->isTrueSpikeRequired()) {
                os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[dd_spkQuePtr" << sg->getTrgNeuronGroup()->getName() << "];" << std::endl;
            }
            else {
                os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[0];" << std::endl;
            }

            os << "numSpikeSubsets = (lscnt+" << learnBlkSz-1 << ") / " << learnBlkSz << ";" << std::endl;
            os << "for (r = 0; r < numSpikeSubsets; r++)" << CodeStream::OB(230);
            os << "if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % " << learnBlkSz << ")+1;" << std::endl;
            os << "else lmax = " << learnBlkSz << ";" << std::endl;

            string offsetTrueSpkPost = sg->getTrgNeuronGroup()->isTrueSpikeRequired()
                ? sg->getOffsetPost("dd_")
                : "";
            os << "if (threadIdx.x < lmax)" << CodeStream::OB(240);
            os << "shSpk[threadIdx.x] = dd_glbSpk" << sg->getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << learnBlkSz << ") + threadIdx.x];" << std::endl;
            os << CodeStream::CB(240);

            os << "__syncthreads();" << std::endl;
            os << "// only work on existing neurons" << std::endl;
            os << "if (" << localID << " < " << sg->getSrcNeuronGroup()->getNumNeurons() << ")" << CodeStream::OB(250);
            os << "// loop through all incoming spikes for learning" << std::endl;
            os << "for (j = 0; j < lmax; j++)" << CodeStream::OB(260) << std::endl;

            if (sparse) {
                os << "unsigned int iprePos = dd_revIndInG" <<  s.first << "[shSpk[j]];" << std::endl;
                os << "unsigned int npre = dd_revIndInG" << s.first << "[shSpk[j] + 1] - iprePos;" << std::endl;
                os << "if (" << localID << " < npre)" << CodeStream::OB(1540);
                os << "iprePos += " << localID << ";" << std::endl;
                //Commenting out the next line as it is not used rather than deleting as I'm not sure if it may be used by different learning models 
                //os << "unsigned int ipre = dd_revInd" << sgName << "[iprePos];" << std::endl;
            }

            if (!wu->getLearnPostSupportCode().empty()) {
                os << " using namespace " << s.first << "_weightupdate_simLearnPost;" << std::endl;
            }
            
             // Create iteration context to iterate over the variables; derived and extra global parameters
            DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
            ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
            VarNameIterCtx wuVars(wu->getVars());

            string code = wu->getLearnPostCode();
            substitute(code, "$(t)", "t");
            // Code substitutions ----------------------------------------------------------------------------------
            if (sparse) { // SPARSE
                name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[dd_remap" + s.first + "[iprePos]]");
            }
            else { // DENSE
                name_substitutions(code, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[" + localID + " * " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + " + shSpk[j]]");
            }
            StandardSubstitutions::weightUpdatePostLearn(code, sg, wuDerivedParams, wuExtraGlobalParams,
                                                         sparse ?  "dd_revInd" + s.first + "[iprePos]" : localID,
                                                         "shSpk[j]", "dd_", cudaFunctions, model.getPrecision());
            // end Code substitutions -------------------------------------------------------------------------
            os << code << std::endl;
            if (sparse) {
                os << CodeStream::CB(1540);
            }
            os << CodeStream::CB(260);
            os << CodeStream::CB(250);
            os << CodeStream::CB(230);
            if (model.getResetKernel() == GENN_FLAGS::learnSynapsesPost) {
                os << "__syncthreads();" << std::endl;
                os << "if (threadIdx.x == 0)" << CodeStream::OB(320);
                os << "j = atomicAdd((unsigned int *) &d_done, 1);" << std::endl;
                os << "if (j == " << numPostLearnBlocks - 1 << ")" << CodeStream::OB(330);

                for(const auto &n : model.getLocalNeuronGroups()) {
                    if (n.second.isDelayRequired()) { // WITH DELAY
                        os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
                        if (n.second.isSpikeEventRequired()) {
                            os << "dd_glbSpkCntEvnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                        }
                        if (n.second.isTrueSpikeRequired()) {
                            os << "dd_glbSpkCnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << std::endl;
                        }
                        else {
                            os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                        }
                    }
                    else { // NO DELAY
                        if (n.second.isSpikeEventRequired()) {
                            os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
                        }
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
                    }
                }
                os << "d_done = 0;" << std::endl;

                os << CodeStream::CB(330); // end "if (j == " << numOfBlocks - 1 << ")"
                os << CodeStream::CB(320); // end "if (threadIdx.x == 0)"
            }
            os << CodeStream::CB(220);
        }

        os << CodeStream::CB(215);
    }
    os << std::endl;
    
    os << "#endif" << std::endl;
    fs.close();

//    cout << "exiting genSynapseKernel" << endl;
}

#endif
