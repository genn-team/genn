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
#include "standardSubstitutions.h"
#include "codeGenUtils.h"
#include "CodeHelper.h"

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
    ostream &os, //!< output stream for code
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
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[delaySlot])" << OB(102);
    }
    else {
        os << "dd_glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[0])" << OB(102);
    }

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << ENDL;
    }

    if (sg.getSrcNeuronGroup()->isDelayRequired()) {
        os << "int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[(delaySlot * " << sg.getSrcNeuronGroup()->getNumNeurons() << ") + " << localID << "];" << ENDL;
    }
    else {
        os << "int preInd = dd_glbSpk"  << postfix << sg.getSrcNeuronGroup()->getName();
        os << "[" << localID << "];" << ENDL;
    }
    os << "prePos = dd_indInG" << sg.getName() << "[preInd];" << ENDL;
    os << "npost = dd_indInG" << sg.getName() << "[preInd + 1] - prePos;" << ENDL;

    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "unsigned int gid = (dd_glbSpkCnt" << postfix << "[" << localID << "] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + i);" << ENDL;
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
                                                              "preInd", "i", "dd_", ftype);
        // end code substitutions ----
        os << "(" << eCode << ")";

        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << ")";
        }
        os << OB(130);
    }
    else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << OB(135);
    }

    os << "for (int i = 0; i < npost; ++i)" << OB(103);
    os << "        ipost = dd_ind" <<  sg.getName() << "[prePos];" << ENDL;

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
                                           "preInd", "ipost", "dd_", ftype);
    // end code substitutions -------------------------------------------------------------------------

    os << wCode << ENDL;

    os << "prePos += 1;" << ENDL;
    os << CB(103);
    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CB(130);
    }
    else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << CB(135);
    }
    os << CB(102);
    //os << CB(101);
}

// classical parallelisation of post-synaptic neurons in parallel and spikes in a loop
void generatePostParallelisedCode(
    ostream &os, //!< output stream for code
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

    os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << ENDL;
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)" << OB(90);
    os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = ((lscnt" << postfix << "-1) % BLOCKSZ_SYN) +1;" << ENDL;
    os << "else lmax = BLOCKSZ_SYN;" << ENDL;
    os << "__syncthreads();" << ENDL;
    os << "if (threadIdx.x < lmax)" << OB(100);
    os << "shSpk" << postfix << "[threadIdx.x] = dd_glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << sg.getOffsetPre() << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
    os << CB(100);

    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !sg.isPSAtomicAddRequired(synapseBlkSz)) {
        // set shLg to 0 for all postsynaptic neurons; is ok as model.neuronN[model.synapseTarget[i]] <= synapseBlkSz
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ") shLg[threadIdx.x] = 0;" << ENDL;
    }
    os << "__syncthreads();" << ENDL;

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
    os << "// loop through all incoming spikes" << ENDL;
    os << "for (j = 0; j < lmax; j++)" << OB(110);
    os << "// only work on existing neurons" << ENDL;
    os << "if (" << localID << " < " << maxConnections << ")" << OB(120);
    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "unsigned int gid = (shSpk" << postfix << "[j] * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + " << localID << ");" << ENDL;
    }

    if (!wu->getSimSupportCode().empty()) {
        os << "using namespace " << sg.getName() << "_weightupdate_simCode;" << ENDL;
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
                                                              "shSpkEvnt[j]", "ipost", "dd_", ftype);
        // end code substitutions ----
        os << "(" << eCode << ")";

        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << ")";
        }
        os << OB(130);
    }
    else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << "if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << OB(135);
    }

    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
        os << "prePos = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j]];" << ENDL;
        os << "npost = dd_indInG" << sg.getName() << "[shSpk" << postfix << "[j] + 1] - prePos;" << ENDL;
        os << "if (" << localID << " < npost)" << OB(140);
        os << "prePos += " << localID << ";" << ENDL;
        os << "ipost = dd_ind" << sg.getName() << "[prePos];" << ENDL;
    }
    else { // DENSE
    os << "ipost = " << localID << ";" << ENDL;
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
                                           "shSpk" + postfix + "[j]", "ipost", "dd_", ftype);
    // end Code substitutions -------------------------------------------------------------------------
    os << wCode << ENDL;

    if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        os << CB(140); // end if (id < npost)
    }

    if (evnt && sg.isEventThresholdReTestRequired()) {
        os << CB(130); // end if (eCode)
    }
    else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        os << CB(135); // end if (B(dd_gp" << sg.getName() << "[gid >> " << logUIntSz << "], gid
    }
    os << CB(120) << ENDL;

    if ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && !sg.isPSAtomicAddRequired(synapseBlkSz)) {
        os << "__syncthreads();" << ENDL;
        os << "if (threadIdx.x < " << sg.getTrgNeuronGroup()->getNumNeurons() << ")" << OB(136); // need to write back results
        os << "linSyn += shLg[" << localID << "];" << ENDL;
        os << "shLg[" << localID << "] = 0;" << ENDL;
        os << CB(136) << ENDL;

        os << "__syncthreads();" << ENDL;
    }
    os << CB(110) << ENDL;
    os << CB(90) << ENDL;
}
//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic
  spikes or spike type events

*/
//-------------------------------------------------------------------------
void generate_process_presynaptic_events_code(
    ostream &os, //!< output stream for code
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
    ofstream os;

    string name = path + "/" + model.name + "_CODE/neuronKrnl.cc";
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_neuronKrnl_cc" << ENDL;
    os << "#define _" << model.name << "_neuronKrnl_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the neuron kernel function." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    //os << "__device__ __host__ float exp(int i) { return exp((float) i); }" << endl;

    os << "// include the support codes provided by the user for neuron or synaptic models" << ENDL;
    os << "#include \"support_code.h\"" << ENDL << ENDL;

    // kernel header
    os << "extern \"C\" __global__ void calcNeurons(";
    for(const auto &p : model.getNeuronKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.ftype << " t)" << ENDL;
    os << OB(5);

    // kernel code
    unsigned int neuronGridSz = model.getNeuronGridSize();
    neuronGridSz = neuronGridSz / neuronBlkSz;
    if (neuronGridSz < (unsigned int)deviceProp[theDevice].maxGridSize[1]) {
        os << "unsigned int id = " << neuronBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
    }
    else {
        os << "unsigned int id = " << neuronBlkSz << " * (blockIdx.x * " << ceil(sqrt((float) neuronGridSz)) << " + blockIdx.y) + threadIdx.x;" << ENDL;
    }

    // these variables deal with high V "spike type events"
    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isSpikeEventRequired()) {
            os << "__shared__ volatile unsigned int posSpkEvnt;" << ENDL;
            os << "__shared__ unsigned int shSpkEvnt[" << neuronBlkSz << "];" << ENDL;
            os << "unsigned int spkEvntIdx;" << ENDL;
            os << "__shared__ volatile unsigned int spkEvntCount;" << ENDL;
            break;
        }
    }

    // these variables now deal only with true spikes, not high V "events"
    for(const auto &n : model.getNeuronGroups()) {
        if(!n.second.getNeuronModel()->getThresholdConditionCode().empty()) {
            os << "__shared__ unsigned int shSpk[" << neuronBlkSz << "];" << ENDL;
            os << "__shared__ volatile unsigned int posSpk;" << ENDL;
            os << "unsigned int spkIdx;" << ENDL;
            os << "__shared__ volatile unsigned int spkCount;" << ENDL;
            break;
        }
    }
    os << ENDL;

    // Reset global spike counting vars here if there are no synapses at all
    if (model.resetKernel == GENN_FLAGS::calcNeurons) {
        os << "if (id == 0)" << OB(6);
        for(const auto &n : model.getNeuronGroups()) {
            StandardGeneratedSections::neuronOutputInit(os, n.second, "dd_");
        }
        os << CB(6);
        os << "__threadfence();" << ENDL << ENDL;
    }

    // Initialise shared spike count vars
    for(const auto &n : model.getNeuronGroups()) {
        if (!n.second.getNeuronModel()->getThresholdConditionCode().empty()) {
            os << "if (threadIdx.x == 0)" << OB(8);
            os << "spkCount = 0;" << ENDL;
            os << CB(8);
            break;
        }
    }
    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x == 1)" << OB(7);
            os << "spkEvntCount = 0;" << ENDL;
            os << CB(7);
            break;
        }
    }
    os << "__syncthreads();" << ENDL;
    os << ENDL;

    
    bool firstNeuronGroup = true;
    for(const auto &n : model.getNeuronGroups()) {
        os << "// neuron group " << n.first << ENDL;
        const auto &groupIDRange = n.second.getPaddedCumSumNeurons();
        if (firstNeuronGroup) {
            os << "if (id < " << groupIDRange.second << ")" << OB(10);
            localID = "id";
            firstNeuronGroup = false;
        }
        else {
            os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))" << OB(10);
            os << "unsigned int lid = id - " << groupIDRange.first << ";" << ENDL;
            localID = "lid";
        }

        if (n.second.isVarQueueRequired() && n.second.isDelayRequired()) {
            os << "unsigned int delaySlot = (dd_spkQuePtr" << n.first;
            os << " + " << (n.second.getNumDelaySlots() - 1);
            os << ") % " << n.second.getNumDelaySlots() << ";" << ENDL;
        }
        os << ENDL;

        os << "// only do this for existing neurons" << ENDL;
        os << "if (" << localID << " < " << n.second.getNumNeurons() << ")" << OB(20);

        os << "// pull neuron variables in a coalesced access" << ENDL;

        const auto *nm = n.second.getNeuronModel();

        // Create iteration context to iterate over the variables; derived and extra global parameters
        VarNameIterCtx nmVars(nm->getVars());
        DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
        ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

        // Generate code to copy neuron state into local variables
        StandardGeneratedSections::neuronLocalVarInit(os, n.second, nmVars, "dd_", localID);

        if ((nm->getSimCode().find("$(sT)") != string::npos)
            || (nm->getThresholdConditionCode().find("$(sT)") != string::npos)
            || (nm->getResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
            os << model.ftype << " lsT = dd_sT" <<  n.first << "[";
            if (n.second.isDelayRequired()) {
                os << "(delaySlot * " << n.second.getNumNeurons() << ") + ";
            }
            os << localID << "];" << ENDL;
        }
        os << ENDL;

        if (n.second.getInSyn().size() > 0 || (nm->getSimCode().find("Isyn") != string::npos)) {
            os << model.ftype << " Isyn = 0;" << ENDL;
        }
        for(const auto *sg : n.second.getInSyn()) {
            const auto *psm = sg->getPSModel();

            os << "// pull inSyn values in a coalesced access" << ENDL;
            os << model.ftype << " linSyn" << sg->getName() << " = dd_inSyn" << sg->getName() << "[" << localID << "];" << ENDL;
            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : psm->getVars()) {
                    os << v.second << " lps" << v.first << sg->getName();
                    os << " = dd_" <<  v.first << sg->getName() << "[" << localID << "];" << ENDL;
                }
            }
            string psCode = psm->getCurrentConverterCode();
            substitute(psCode, "$(id)", localID);
            substitute(psCode, "$(inSyn)", "linSyn" + sg->getName());
            StandardSubstitutions::postSynapseCurrentConverter(psCode, sg, n.second,
                nmVars, nmDerivedParams, nmExtraGlobalParams, model.ftype);

            if (!psm->getSupportCode().empty()) {
                os << OB(29) << " using namespace " << sg->getName() << "_postsyn;" << ENDL;
            }
            os << "Isyn += " << psCode << ";" << ENDL;
            if (!psm->getSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << ENDL;
            }
        }

        if (!nm->getSupportCode().empty()) {
            os << " using namespace " << n.first << "_neuron;" << ENDL;
        }
        string thCode = nm->getThresholdConditionCode();
        if (thCode.empty()) { // no condition provided
            cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << n.first << "\" was provided. There will be no spikes detected in this population!" << endl;
        }
        else {
            os << "// test whether spike condition was fulfilled previously" << ENDL;
            substitute(thCode, "$(id)", localID);
            StandardSubstitutions::neuronThresholdCondition(thCode, n.second,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            model.ftype);
            if (GENN_PREFERENCES::autoRefractory) {
                os << "bool oldSpike= (" << thCode << ");" << ENDL;
            }
        }

        os << "// calculate membrane potential" << ENDL;
        string sCode = nm->getSimCode();
        substitute(sCode, "$(id)", localID);
        StandardSubstitutions::neuronSim(sCode, n.second,
                                         nmVars, nmDerivedParams, nmExtraGlobalParams,
                                         model.ftype);
        os << sCode << ENDL;

        // look for spike type events first.
        if (n.second.isSpikeEventRequired()) {
           // Generate spike event test
            StandardGeneratedSections::neuronSpikeEventTest(os, n.second,
                                                            nmVars, nmExtraGlobalParams,
                                                            localID, model.ftype);

            os << "// register a spike-like event" << ENDL;
            os << "if (spikeLikeEvent)" << OB(30);
            os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << ENDL;
            os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << ENDL;
            os << CB(30);
        }

        // test for true spikes if condition is provided
        if (!thCode.empty()) {
            os << "// test for and register a true spike" << ENDL;
            if (GENN_PREFERENCES::autoRefractory) {
                os << "if ((" << thCode << ") && !(oldSpike)) " << OB(40);
            }
            else {
                os << "if (" << thCode << ") " << OB(40);
            }
            os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << ENDL;
            os << "shSpk[spkIdx] = " << localID << ";" << ENDL;

            // add after-spike reset if provided
            if (!nm->getResetCode().empty()) {
                string rCode = nm->getResetCode();
                substitute(rCode, "$(id)", localID);
                StandardSubstitutions::neuronReset(rCode, n.second,
                                                   nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                   model.ftype);
                os << "// spike reset code" << ENDL;
                os << rCode << ENDL;
            }
            os << CB(40);
        }

        // store the defined parts of the neuron state into the global state variables dd_V etc
        StandardGeneratedSections::neuronLocalVarWrite(os, n.second, nmVars, "dd_", localID);

        if (!n.second.getInSyn().empty()) {
            os << "// the post-synaptic dynamics" << ENDL;
        }
        for(const auto *sg : n.second.getInSyn()) {
            const auto *psm = sg->getPSModel();

            string pdCode = psm->getDecayCode();
            substitute(pdCode, "$(id)", localID);
            substitute(pdCode, "$(inSyn)", "linSyn" + sg->getName());
            StandardSubstitutions::postSynapseDecay(pdCode, sg, n.second,
                                                    nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                    model.ftype);
            if (!psm->getSupportCode().empty()) {
                os << OB(29) << " using namespace " << sg->getName() << "_postsyn;" << ENDL;
            }
            os << pdCode << ENDL;
            if (!psm->getSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << endl;
            }

            os << "dd_inSyn"  << sg->getName() << "[" << localID << "] = linSyn" << sg->getName() << ";" << ENDL;
            for(const auto &v : psm->getVars()) {
                os << "dd_" <<  v.first << sg->getName() << "[" << localID << "] = lps" << v.first << sg->getName() << ";"<< ENDL;
            }
        }

        os << CB(20);
        os << "__syncthreads();" << ENDL;

        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x == 1)" << OB(50);
            os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvnt" << n.first;
            if (n.second.isDelayRequired()) {
                os << "[dd_spkQuePtr" << n.first << "], spkEvntCount);" << ENDL;
            }
            else {
                os << "[0], spkEvntCount);" << ENDL;
            }
            os << CB(50); // end if (threadIdx.x == 0)
            os << "__syncthreads();" << ENDL;
        }

        if (!nm->getThresholdConditionCode().empty()) {
            os << "if (threadIdx.x == 0)" << OB(51);
            os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCnt" << n.first;
            if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) {
                os << "[dd_spkQuePtr" << n.first << "], spkCount);" << ENDL;
            }
            else {
                os << "[0], spkCount);" << ENDL;
            }
            os << CB(51); // end if (threadIdx.x == 1)

            os << "__syncthreads();" << ENDL;
        }

        string queueOffset = n.second.getQueueOffset("dd_");
        if (n.second.isSpikeEventRequired()) {
            os << "if (threadIdx.x < spkEvntCount)" << OB(60);
            os << "dd_glbSpkEvnt" << n.first << "[" << queueOffset << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << ENDL;
            os << CB(60); // end if (threadIdx.x < spkEvntCount)
        }

        if (!nm->getThresholdConditionCode().empty()) {
            string queueOffsetTrueSpk = n.second.isTrueSpikeRequired() ? queueOffset : "";

            os << "if (threadIdx.x < spkCount)" << OB(70);
            os << "dd_glbSpk" << n.first << "[" << queueOffsetTrueSpk << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << ENDL;
            if (n.second.isSpikeTimeRequired()) {
                os << "dd_sT" << n.first << "[" << queueOffset << "shSpk[threadIdx.x]] = t;" << ENDL;
            }
            os << CB(70); // end if (threadIdx.x < spkCount)
        }
        os << CB(10); // end if (id < model.padSumNeuronN[i] )
        os << ENDL;
    }
    os << CB(5) << ENDL; // end of neuron kernel

    os << "#endif" << ENDL;
    os.close();
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
    ofstream os;

//    cout << "entering genSynapseKernel" << endl;
    string name = path + "/" + model.name + "_CODE/synapseKrnl.cc";
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_synapseKrnl_cc" << ENDL;
    os << "#define _" << model.name << "_synapseKrnl_cc" << ENDL;
    os << "#define BLOCKSZ_SYN " << synapseBlkSz << ENDL;
    os << ENDL;
 
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file synapseKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name;
    os << " containing the synapse kernel and learning kernel functions." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;


    if (!model.getSynapseDynamicsGroups().empty()) {
        os << "#define BLOCKSZ_SYNDYN " << synDynBlkSz << endl;
	
        // SynapseDynamics kernel header
        os << "extern \"C\" __global__ void calcSynapseDynamics(";
        for(const auto &p : model.getSynapseDynamicsKernelParameters()) {
            os << p.second << " " << p.first << ", ";
        }
        os << model.ftype << " t)" << ENDL; // end of synapse kernel header

        // synapse dynamics kernel code
        os << OB(75);

        // common variables for all cases
        os << "unsigned int id = BLOCKSZ_SYNDYN * blockIdx.x + threadIdx.x;" << ENDL;

        os << "// execute internal synapse dynamics if any" << ENDL;
        os << ENDL;

        bool firstSynapseDynamicsGroup = true;
        for(const auto &s : model.getSynapseDynamicsGroups())
        {
            const SynapseGroup *sg = model.findSynapseGroup(s.first);
            const auto *wu = sg->getWUModel();

            // if there is some internal synapse dynamics
            if (!wu->getSynapseDynamicsCode().empty()) {
                // Create iteration context to iterate over the variables and derived parameters
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                VarNameIterCtx wuVars(wu->getVars());

                os << "// synapse group " << s.first << ENDL;
                const auto &groupIDRange = sg->getPaddedKernelCumSum();
                if (firstSynapseDynamicsGroup) {
                    os << "if (id < " << groupIDRange.second << ")" << OB(77);
                    localID = "id";
                    firstSynapseDynamicsGroup = false;
                }
                else {
                    os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))" << OB(77);
                    os << "unsigned int lid = id - " << groupIDRange.first << ";" << ENDL;
                    localID = "lid";
                }

                if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                    os << "unsigned int delaySlot = (dd_spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                    os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                    os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
                }

                string SDcode = wu->getSynapseDynamicsCode();
                substitute(SDcode, "$(t)", "t");

                if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                    os << "if (" << localID << " < dd_indInG" << s.first << "[" << sg->getSrcNeuronGroup()->getNumNeurons() << "])" << OB(25);
                    os << "// all threads participate that can work on an existing synapse" << ENDL;
                    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                        os << " using namespace " << s.first << "_weightupdate_synapseDynamics;" << ENDL;
                    }
                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                        // name substitute synapse var names in synapseDynamics code
                        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[" + localID +"]");
                    }

                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams,
                                                                "dd_preInd" + s.first +"[" + localID + "]",
                                                                "dd_ind" + s.first + "[" + localID + "]",
                                                                "dd_", model.ftype);
                    os << SDcode << ENDL;
                }
                else { // DENSE
                    os << "if (" << localID << " < " << sg->getSrcNeuronGroup()->getNumNeurons() * sg->getTrgNeuronGroup()->getNumNeurons() << ")" << OB(25);
                    os << "// all threads participate that can work on an existing synapse" << ENDL;
                    if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                            os << " using namespace " << s.first << "_weightupdate_synapseDynamics;" << ENDL;
                    }
                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                        // name substitute synapse var names in synapseDynamics code
                        name_substitutions(SDcode, "dd_", wuVars.nameBegin, wuVars.nameEnd, s.first + "[" + localID + "]");
                    }
                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams,
                                                                localID +"/" + to_string(sg->getTrgNeuronGroup()->getNumNeurons()),
                                                                localID +"%" + to_string(sg->getTrgNeuronGroup()->getNumNeurons()),
                                                                "dd_", model.ftype);
                    os << SDcode << ENDL;
                }
                os << CB(25);
                os << CB(77);
            }
        }
        os << CB(75);
    }

    // count how many neuron blocks to use: one thread for each synapse target
    // targets of several input groups are counted multiply
    const unsigned int numSynapseBlocks = model.getSynapseKernelGridSize() / synapseBlkSz;

    // synapse kernel header
    os << "extern \"C\" __global__ void calcSynapses(";
    for (const auto &p : model.getSynapseKernelParameters()) {
        os << p.second << " " << p.first << ", ";
    }
    os << model.ftype << " t)" << ENDL; // end of synapse kernel header

    // synapse kernel code
    os << OB(75);

    // common variables for all cases
    os << "unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;" << ENDL;
    os << "unsigned int lmax, j, r;" << ENDL;
    os << model.ftype << " addtoinSyn;" << ENDL;  
    os << "volatile __shared__ " << model.ftype << " shLg[BLOCKSZ_SYN];" << ENDL;

    // case-dependent variables
    for(const auto &s : model.getSynapseGroups()) {
        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)){
            os << model.ftype << " linSyn;" << ENDL;
            break;
        }
    }
    // we need ipost in any case, and we need npost if there are any SPARSE connections
    os << "unsigned int ipost;" << ENDL;
    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType()  & SynapseMatrixConnectivity::SPARSE) {
            os << "unsigned int prePos; " << ENDL;
            os << "unsigned int npost; " << ENDL;
            break;
        }
    }
    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first)) {
            os << "__shared__ unsigned int shSpk[BLOCKSZ_SYN];" << ENDL;
            //os << "__shared__ " << model.ftype << " shSpkV[BLOCKSZ_SYN];" << ENDL;
            os << "unsigned int lscnt, numSpikeSubsets;" << ENDL;
            break;
        }
    }
    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.isSpikeEventRequired()) {
            os << "__shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];" << ENDL;
            os << "unsigned int lscntEvnt, numSpikeSubsetsEvnt;" << ENDL;
            break;
        }
    }
    os << ENDL;

    bool firstSynapseGroup = true;
    for(const auto &s : model.getSynapseGroups()) {
        os << "// synapse group " << s.first << ENDL;
        const auto &groupIDRange = s.second.getPaddedKernelCumSum();
        if (firstSynapseGroup) {
            os << "if (id < " << groupIDRange.second << ")" << OB(77);
            localID = "id";
            firstSynapseGroup = false;
        }
        else {
            os << "if ((id >= " << groupIDRange.first << ") && (id < " << groupIDRange.second << "))" << OB(77);
            os << "unsigned int lid = id - " << groupIDRange.first<< ";" << ENDL;
            localID = "lid";
        }

        if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
            os << "unsigned int delaySlot = (dd_spkQuePtr" << s.second.getSrcNeuronGroup()->getName();
            os << " + " << (s.second.getSrcNeuronGroup()->getNumDelaySlots() - s.second.getDelaySteps());
            os << ") % " << s.second.getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
        }

        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)){
            os << "// only do this for existing neurons" << ENDL;
            os << "if (" << localID << " < " << s.second.getTrgNeuronGroup()->getNumNeurons() << ")" << OB(80);
            os << "linSyn = dd_inSyn" << s.first << "[" << localID << "];" << ENDL;
            os << CB(80);
        }

        if (s.second.isSpikeEventRequired()) {
            os << "lscntEvnt = dd_glbSpkCntEvnt" << s.second.getSrcNeuronGroup()->getName();
            if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << ENDL;
            }
            else {
                os << "[0];" << ENDL;
            }
            os << "numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << ENDL;
        }
  
        if (s.second.isTrueSpikeRequired() || model.isSynapseGroupPostLearningRequired(s.first)) {
            os << "lscnt = dd_glbSpkCnt" << s.second.getSrcNeuronGroup()->getName();
            if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
                os << "[delaySlot];" << ENDL;
            }
            else {
                os << "[0];" << ENDL;
            }
            os << "numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;" << ENDL;
        }

        // generate the code for processing spike-like events
        if (s.second.isSpikeEventRequired()) {
            generate_process_presynaptic_events_code(os, s.second, localID, "Evnt", model.ftype);
        }

        // generate the code for processing true spike events
        if (s.second.isTrueSpikeRequired()) {
            generate_process_presynaptic_events_code(os, s.second, localID, "", model.ftype);
        }
        os << ENDL;

        if (!(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) || !s.second.isPSAtomicAddRequired(synapseBlkSz)) {
            os << "// only do this for existing neurons" << ENDL;
            os << "if (" << localID << " < " << s.second.getTrgNeuronGroup()->getNumNeurons() << ")" << OB(190);
            os << "dd_inSyn" << s.first << "[" << localID << "] = linSyn;" << ENDL;
            os << CB(190);
        }
        // need to do reset operations in this kernel (no learning kernel)
        if (model.resetKernel == GENN_FLAGS::calcSynapses) {
            os << "__syncthreads();" << ENDL;
            os << "if (threadIdx.x == 0)" << OB(200);
            os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
            os << "if (j == " << numSynapseBlocks - 1 << ")" << OB(210);

            for(const auto &n : model.getNeuronGroups()) {
                if (n.second.isDelayRequired()) { // WITH DELAY
                    os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << ENDL;
                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << ENDL;
                    }
                    if (n.second.isTrueSpikeRequired()) {
                        os << "dd_glbSpkCnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << ENDL;
                    }
                    else {
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
                    }
                }
                else { // NO DELAY
                    if (n.second.isSpikeEventRequired()) {
                        os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << ENDL;
                    }
                    os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
                }
            }
            os << "d_done = 0;" << ENDL;

            os << CB(210); // end "if (j == " << numOfBlocks - 1 << ")"
            os << CB(200); // end "if (threadIdx.x == 0)"
        }

        os << CB(77);
        os << ENDL;
    }
    os << CB(75);
    os << ENDL;


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
        os << model.ftype << " t)";
        os << ENDL;

        // kernel code
        os << OB(215);
        os << "unsigned int id = " << learnBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
        os << "__shared__ unsigned int shSpk[" << learnBlkSz << "];" << ENDL;
        os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << ENDL;
        os << ENDL;

        bool firstPostLearnGroup = true;
        for(const auto &s : model.getSynapsePostLearnGroups())
        {
            const SynapseGroup *sg = model.findSynapseGroup(s.first);
            const auto *wu = sg->getWUModel();
            const bool sparse = sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE;

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM
            os << "// synapse group " << s.first << ENDL;
            //const auto &groupIDRange = s.second.getPaddedCumSumNeurons();
            if (firstPostLearnGroup) {
                os << "if (id < " << s.second.second << ")" << OB(220);
                localID = "id";
                firstPostLearnGroup = false;
            }
            else {
                os << "if ((id >= " << s.second.first << ") && (id < " << s.second.second << "))" << OB(220);
                os << "unsigned int lid = id - " << s.second.first << ";" << ENDL;
                localID = "lid";
            }

            if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                os << "unsigned int delaySlot = (dd_spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
            }

            if (sg->getTrgNeuronGroup()->isDelayRequired() && sg->getTrgNeuronGroup()->isTrueSpikeRequired()) {
                os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[dd_spkQuePtr" << sg->getTrgNeuronGroup()->getName() << "];" << ENDL;
            }
            else {
                os << "lscnt = dd_glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[0];" << ENDL;
            }

            os << "numSpikeSubsets = (lscnt+" << learnBlkSz-1 << ") / " << learnBlkSz << ";" << ENDL;
            os << "for (r = 0; r < numSpikeSubsets; r++)" << OB(230);
            os << "if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % " << learnBlkSz << ")+1;" << ENDL;
            os << "else lmax = " << learnBlkSz << ";" << ENDL;

            string offsetTrueSpkPost = sg->getTrgNeuronGroup()->isTrueSpikeRequired()
                ? sg->getOffsetPost("dd_")
                : "";
            os << "if (threadIdx.x < lmax)" << OB(240);
            os << "shSpk[threadIdx.x] = dd_glbSpk" << sg->getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "(r * " << learnBlkSz << ") + threadIdx.x];" << ENDL;
            os << CB(240);

            os << "__syncthreads();" << ENDL;
            os << "// only work on existing neurons" << ENDL;
            os << "if (" << localID << " < " << sg->getSrcNeuronGroup()->getNumNeurons() << ")" << OB(250);
            os << "// loop through all incoming spikes for learning" << ENDL;
            os << "for (j = 0; j < lmax; j++)" << OB(260) << ENDL;

            if (sparse) {
                os << "unsigned int iprePos = dd_revIndInG" <<  s.first << "[shSpk[j]];" << ENDL;
                os << "unsigned int npre = dd_revIndInG" << s.first << "[shSpk[j] + 1] - iprePos;" << ENDL;
                os << "if (" << localID << " < npre)" << OB(1540);
                os << "iprePos += " << localID << ";" << ENDL;
                //Commenting out the next line as it is not used rather than deleting as I'm not sure if it may be used by different learning models 
                //os << "unsigned int ipre = dd_revInd" << sgName << "[iprePos];" << ENDL;
            }

            if (!wu->getLearnPostSupportCode().empty()) {
                os << " using namespace " << s.first << "_weightupdate_simLearnPost;" << ENDL;
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
                                                         "shSpk[j]", "dd_", model.ftype);
            // end Code substitutions -------------------------------------------------------------------------
            os << code << ENDL;
            if (sparse) {
                os << CB(1540);
            }
            os << CB(260);
            os << CB(250);
            os << CB(230);
            if (model.resetKernel == GENN_FLAGS::learnSynapsesPost) {
                os << "__syncthreads();" << ENDL;
                os << "if (threadIdx.x == 0)" << OB(320);
                os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
                os << "if (j == " << numPostLearnBlocks - 1 << ")" << OB(330);

                for(const auto &n : model.getNeuronGroups()) {
                    if (n.second.isDelayRequired()) { // WITH DELAY
                        os << "dd_spkQuePtr" << n.first << " = (dd_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << ENDL;
                        if (n.second.isSpikeEventRequired()) {
                            os << "dd_glbSpkCntEvnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << ENDL;
                        }
                        if (n.second.isTrueSpikeRequired()) {
                            os << "dd_glbSpkCnt" << n.first << "[dd_spkQuePtr" << n.first << "] = 0;" << ENDL;
                        }
                        else {
                            os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
                        }
                    }
                    else { // NO DELAY
                        if (n.second.isSpikeEventRequired()) {
                            os << "dd_glbSpkCntEvnt" << n.first << "[0] = 0;" << ENDL;
                        }
                        os << "dd_glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
                    }
                }
                os << "d_done = 0;" << ENDL;

                os << CB(330); // end "if (j == " << numOfBlocks - 1 << ")"
                os << CB(320); // end "if (threadIdx.x == 0)"
            }
            os << CB(220);
        }

        os << CB(215);
    }
    os << ENDL;
    
    os << "#endif" << ENDL;
    os.close();

//    cout << "exiting genSynapseKernel" << endl;
}

#endif
