#include "generateInit.h"

// Standard C++ includes
#include <fstream>

// Standard C includes
#include <cmath>

// GeNN includes
#include "codeStream.h"
#include "global.h"
#include "modelSpec.h"

// ------------------------------------------------------------------------
// Anonymous namespace
// ------------------------------------------------------------------------
namespace
{
unsigned int genInitNeuronKernel(CodeStream &os, const NNmodel &model)
{
    unsigned int startThread = 0;
#ifndef CPU_ONLY
     // init kernel header
    os << "extern \"C\" __global__ void init()" << std::endl;

    // initialization kernel code
    os << CodeStream::OB(10);

    // common variables for all cases
    os << "const unsigned int id = " << initBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;

    // Loop through neuron groups
    unsigned int sequence = 0;
    for(const auto &n : model.getNeuronGroups()) {
        // If this group requires an RNG
        if(n.second.isRNGRequired()) {
            // Get padded size of group and hence it's end thread
            const unsigned int paddedSize = (unsigned int)(ceil((double)n.second.getNumNeurons() / (double)initBlkSz) * (double)initBlkSz);
            const unsigned int endThread = startThread + paddedSize;

            // Write if block to determine if this thread should be used for this neuron group
            os << "// neuron group " << n.first << std::endl;
            if(startThread == 0) {
                os << "if (id < " << endThread << ")" << CodeStream::OB(20);
            }
            else {
                os << "if ((id >= " << startThread << ") && (id < " << endThread << "))" << CodeStream::OB(20);
            }
            os << "const unsigned int lid = id - " << startThread << ";" << std::endl;

            os << "// only do this for existing neurons" << std::endl;
            os << "if (lid < " << n.second.getNumNeurons() << ")" << CodeStream::OB(30);
            os << "curand_init(" << model.getSeed() << ", " << sequence << " + lid, 0, &dd_rng" << n.first << "[lid]);" << std::endl;
            os << CodeStream::CB(30);
            os << CodeStream::CB(20);

            // Increment sequence number
            sequence++;

            // Update start thread
            startThread = endThread;
        }
    }

    // initialization kernel code
    os << CodeStream::CB(10);
#endif  // CPU_ONLY

    return startThread;
}
}   // Anonymous namespace

void genInit(const NNmodel &model,          //!< Model description
             const std::string &path)       //!< Path for code generationn
{
    const std::string runnerName= path + "/" + model.getName() + "_CODE/init.cc";
    std::ofstream fs;
    fs.open(runnerName.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    writeHeader(os);
    os << std::endl;

    // Insert kernel to initialize neurons
    const unsigned int numInitThreads = genInitNeuronKernel(os, model);

    // ------------------------------------------------------------------------
    // initializing variables
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to (re)set all model variables to their compile-time, homogeneous initial values." << std::endl;
    os << " Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "void initialize()" << std::endl;
    os << "{" << std::endl;

    // Extra braces around Windows for loops to fix https://support.microsoft.com/en-us/kb/315481
#ifdef _WIN32
    std::string oB = "{", cB = "}";
#else
    std::string oB = "", cB = "";
#endif // _WIN32

    if (model.getSeed() == 0) {
        os << "    srand((unsigned int) time(NULL));" << std::endl;
        //os << std::endl;
        //os << "    std::random_device rd;" << std::endl;
        //os << "    rng.seed(rd);" << std::endl;
    }
    else {
        os << "    srand((unsigned int) " << model.getSeed() << ");" << std::endl;
    }
    os << std::endl;

    // INITIALISE NEURON VARIABLES
    os << "    // neuron variables" << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
        if (n.second.isDelayRequired()) {
            os << "    spkQuePtr" << n.first << " = 0;" << std::endl;
#ifndef CPU_ONLY
            os << "CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtr" << n.first;
            os << ", &spkQuePtr" << n.first;
            os << ", sizeof(unsigned int), 0, cudaMemcpyHostToDevice));" << std::endl;
#endif
        }

        if (n.second.isTrueSpikeRequired() && n.second.isDelayRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            os << "        glbSpkCnt" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            os << "        glbSpk" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
        }
        else {
            os << "    glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << std::endl;
            os << "        glbSpk" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
        }

        if (n.second.isSpikeEventRequired() && n.second.isDelayRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            os << "        glbSpkCntEvnt" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            os << "        glbSpkEvnt" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
        }
        else if (n.second.isSpikeEventRequired()) {
            os << "    glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << std::endl;
            os << "        glbSpkEvnt" << n.first << "[i] = 0;" << std::endl;
            os << "    }" << cB << std::endl;
        }

        if (n.second.isSpikeTimeRequired()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            os << "        sT" <<  n.first << "[i] = -SCALAR_MAX;" << std::endl;
            os << "    }" << cB << std::endl;
        }

        auto neuronModelVars = n.second.getNeuronModel()->getVars();
        for (size_t j = 0; j < neuronModelVars.size(); j++) {
            if (n.second.isVarQueueRequired(neuronModelVars[j].first)) {
                os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++) {" << std::endl;
            }
            else {
                os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << std::endl;
            }
            if (neuronModelVars[j].second == model.getPrecision()) {
                os << "        " << neuronModelVars[j].first << n.first << "[i] = " << model.scalarExpr(n.second.getInitVals()[j]) << ";" << std::endl;
            }
            else {
                os << "        " << neuronModelVars[j].first << n.first << "[i] = " << n.second.getInitVals()[j] << ";" << std::endl;
            }
            os << "    }" << cB << std::endl;
        }

        if (n.second.getNeuronModel()->isPoisson()) {
            os << "    " << oB << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++) {" << std::endl;
            os << "        seed" << n.first << "[i] = rand();" << std::endl;
            os << "    }" << cB << std::endl;
        }

        /*if ((model.neuronType[i] == IZHIKEVICH) && (model.getDT() != 1.0)) {
            os << "    fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); " << std::endl;
        }*/
    }
    os << std::endl;

    // INITIALISE SYNAPSE VARIABLES
    os << "    // synapse variables" << std::endl;
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
        const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

        os << "    " << oB << "for (int i = 0; i < " << numTrgNeurons << "; i++) {" << std::endl;
        os << "        inSyn" << s.first << "[i] = " << model.scalarExpr(0.0) << ";" << std::endl;
        os << "    }" << cB << std::endl;

        if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
            auto wuVars = wu->getVars();
            for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                os << "    " << oB << "for (int i = 0; i < " << numSrcNeurons * numTrgNeurons << "; i++) {" << std::endl;
                if     (wuVars[k].second == model.getPrecision()) {
                    os << "        " << wuVars[k].first << s.first << "[i] = " << model.scalarExpr(s.second.getWUInitVals()[k]) << ";" << std::endl;
                }
                else {
                    os << "        " << wuVars[k].first << s.first << "[i] = " << s.second.getWUInitVals()[k] << ";" << std::endl;
                }

                os << "    }" << cB << std::endl;
            }
        }

        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            auto psmVars = psm->getVars();
            for (size_t k= 0, l= psmVars.size(); k < l; k++) {
                os << "    " << oB << "for (int i = 0; i < " << numTrgNeurons << "; i++) {" << std::endl;
                if (psmVars[k].second == model.getPrecision()) {
                    os << "        " << psmVars[k].first << s.first << "[i] = " << model.scalarExpr(s.second.getPSInitVals()[k]) << ";" << std::endl;
                }
                else {
                    os << "        " << psmVars[k].first << s.first << "[i] = " << s.second.getPSInitVals()[k] << ";" << std::endl;
                }
                os << "    }" << cB << std::endl;
            }
        }
    }
    os << std::endl << std::endl;
#ifndef CPU_ONLY
    os << "    copyStateToDevice();" << std::endl << std::endl;

    // If any init threads were required, perform init kernel launch
    if(numInitThreads > 0) {
        os << "    // perform on-device init" << std::endl;
        os << "    dim3 iThreads(" << initBlkSz << ", 1);" << std::endl;
        os << "    dim3 iGrid(" << numInitThreads / initBlkSz << ", 1);" << std::endl;
        os << "    init <<<iGrid, iThreads>>>();" << std::endl;
    }
    os << "    //initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later" << std::endl;
#endif
    os << "}" << std::endl << std::endl;

}