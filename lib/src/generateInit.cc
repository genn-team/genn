#include "generateInit.h"

// Standard C++ includes
#include <fstream>

// Standard C includes
#include <cmath>

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
#ifndef CPU_ONLY
unsigned int genInitNeuronKernel(CodeStream &os, const NNmodel &model)
{
     // init kernel header
    os << "extern \"C\" __global__ void init()" << std::endl;

    // initialization kernel code
    os << CodeStream::OB(10);

    // common variables for all cases
    os << "const unsigned int id = " << initBlkSz << " * blockIdx.x + threadIdx.x;" << std::endl;

    // Loop through neuron groups
    unsigned int startThread = 0;
    unsigned int sequence = 0;
    for(const auto &n : model.getNeuronGroups()) {
        // If this group requires an RNG to simulate or intialisation code to be run on device
        if(n.second.isSimRNGRequired() || (model.getInitMode() == InitMode::DEVICE && n.second.isInitCodeRequired())) {
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
            if(n.second.isSimRNGRequired()) {
                os << "curand_init(" << model.getSeed() << ", " << sequence << " + lid, 0, &dd_rng" << n.first << "[lid]);" << std::endl;
            }

            // If initialisation should be performed on device
            /*if(model.getInitMode() == InitMode::DEVICE) {
                auto neuronModelVars = n.second.getNeuronModel()->getVars();
                for (size_t j = 0; j < neuronModelVars.size(); j++) {
                    const auto &varInit = n.second.getVarInitialisers()[j];
                    os << StandardSubstitutions::initVariable(varInit, "dd_" +  neuronModelVars[j].first + n.first + "[lid] = $(0)", model.getPrecision()) << std::endl;
                }
            }*/
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

    return startThread;
}
#endif  // CPU_ONLY
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
#ifndef CPU_ONLY
    const unsigned int numInitThreads = genInitNeuronKernel(os, model);
#endif  // CPU_ONLY

    // ------------------------------------------------------------------------
    // initializing variables
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\brief Function to (re)set all model variables to their compile-time, homogeneous initial values." << std::endl;
    os << " Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "void initialize()" << std::endl;
    os << CodeStream::OB(10) << std::endl;

    // Extra braces around Windows for loops to fix https://support.microsoft.com/en-us/kb/315481
#ifdef _WIN32
    std::string oB = "{", cB = "}";
#else
    std::string oB = "", cB = "";
#endif // _WIN32

    // Seed legacy RNG
    if (model.getSeed() == 0) {
        os << "srand((unsigned int) time(NULL));" << std::endl;
    }
    else {
        os << "srand((unsigned int) " << model.getSeed() << ");" << std::endl;
    }

    // If model requires an RNG
    if(model.isRNGRequired()) {
        // If no seed is specified, use system randomness to generate seed sequence
        os << CodeStream::OB(20);
        if (model.getSeed() == 0) {
            os << "uint32_t seedData[std::mt19937::state_size];" << std::endl;
            os << "std::random_device seedSource;" << std::endl;
            os << CodeStream::OB(30) << "for(int i = 0; i < std::mt19937::state_size; i++)" << CodeStream::OB(40);
            os << "seedData[i] = seedSource();" << std::endl;
            os << CodeStream::CB(40) << CodeStream::CB(30);
            os << "std::seed_seq seeds(std::begin(seedData), std::end(seedData));" << std::endl;
        }
        // Otherwise, create a seed sequence from model seed
        // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
        else {
            os << "std::seed_seq seeds{" << model.getSeed() << "};" << std::endl;
        }

        // Seed RNG from seed sequence
        os << "rng.seed(seeds);" << std::endl;
        os << CodeStream::CB(20);
    }
    os << std::endl;

    // INITIALISE NEURON VARIABLES
    os << "// neuron variables" << std::endl;
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
            os << CodeStream::OB(50) << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(60);
            os << "glbSpkCnt" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(60) << CodeStream::CB(50) << std::endl;

            os << CodeStream::OB(70) << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(80);
            os << "glbSpk" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(80) << CodeStream::CB(70) << std::endl;
        }
        else {
            os << "glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
            os << CodeStream::OB(90) << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)" << CodeStream::OB(100);
            os << "glbSpk" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(100) << CodeStream::CB(90) << std::endl;
        }

        if (n.second.isSpikeEventRequired() && n.second.isDelayRequired()) {
            os << CodeStream::OB(110) << "for (int i = 0; i < " << n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(120);
            os << "glbSpkCntEvnt" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(120) << CodeStream::CB(110) << std::endl;

            os << CodeStream::OB(130) << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(140);
            os << "glbSpkEvnt" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(140) << CodeStream::CB(130) << std::endl;
        }
        else if (n.second.isSpikeEventRequired()) {
            os << "glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
            os << CodeStream::OB(150) << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)" << CodeStream::OB(160);
            os << "glbSpkEvnt" << n.first << "[i] = 0;" << std::endl;
            os << CodeStream::CB(160) << CodeStream::OB(150) << std::endl;
        }

        if (n.second.isSpikeTimeRequired()) {
            os << CodeStream::OB(170) << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(180);
            os << "sT" <<  n.first << "[i] = -SCALAR_MAX;" << std::endl;
            os << CodeStream::CB(180) << CodeStream::CB(170) << std::endl;
        }

        // If intialisation mode should occur on the host
        if(model.getInitMode() == InitMode::HOST) {
            auto neuronModelVars = n.second.getNeuronModel()->getVars();
            for (size_t j = 0; j < neuronModelVars.size(); j++) {
                const auto &varInit = n.second.getVarInitialisers()[j];

                // If this variable has any initialisation code
                if(!varInit.getSnippet()->getCode().empty()) {
                    if (n.second.isVarQueueRequired(neuronModelVars[j].first)) {
                        os << CodeStream::OB(190) << "for (int i = 0; i < " << n.second.getNumNeurons() * n.second.getNumDelaySlots() << "; i++)" << CodeStream::OB(200);
                    }
                    else {
                        os << CodeStream::OB(190) << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)" << CodeStream::OB(200);
                    }

                    os << StandardSubstitutions::initVariable(varInit, neuronModelVars[j].first + n.first + "[i] = $(0)",
                                                              cpuFunctions, model.getPrecision(), "rng") << std::endl;

                    os << CodeStream::CB(200) << CodeStream::CB(190) << std::endl;
                }
            }
        }

        if (n.second.getNeuronModel()->isPoisson()) {
            os << CodeStream::OB(210) << "for (int i = 0; i < " << n.second.getNumNeurons() << "; i++)" << CodeStream::OB(220);
            os << "seed" << n.first << "[i] = rand();" << std::endl;
            os << CodeStream::CB(220) << CodeStream::CB(210) << std::endl;
        }

        /*if ((model.neuronType[i] == IZHIKEVICH) && (model.getDT() != 1.0)) {
            os << "    fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); " << std::endl;
        }*/
    }
    os << std::endl;

    // If intialisation mode should occur on the host
    if(model.getInitMode() == InitMode::HOST) {
        // INITIALISE SYNAPSE VARIABLES
        os << "// synapse variables" << std::endl;
        for(const auto &s : model.getSynapseGroups()) {
            const auto *wu = s.second.getWUModel();
            const auto *psm = s.second.getPSModel();

            const unsigned int numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const unsigned int numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();

            os << CodeStream::OB(230) << "for (int i = 0; i < " << numTrgNeurons << "; i++)" << CodeStream::OB(240);
            os << "inSyn" << s.first << "[i] = " << model.scalarExpr(0.0) << ";" << std::endl;
            os << CodeStream::CB(240) << CodeStream::CB(230) << std::endl;

            // If matrix is dense (i.e. can be initialised here) and each synapse has individual values (i.e. needs initialising at all)
            if ((s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE) && (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                auto wuVars = wu->getVars();
                for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                    os << CodeStream::OB(250) << "for (int i = 0; i < " << numSrcNeurons * numTrgNeurons << "; i++)" << CodeStream::OB(260);
                    if(wuVars[k].second == model.getPrecision()) {
                        os << wuVars[k].first << s.first << "[i] = " << model.scalarExpr(s.second.getWUInitVals()[k]) << ";" << std::endl;
                    }
                    else {
                        os << wuVars[k].first << s.first << "[i] = " << s.second.getWUInitVals()[k] << ";" << std::endl;
                    }

                    os << CodeStream::CB(260) << CodeStream::CB(250) << std::endl;
                }
            }

            // If matrix has individual state variables
            // **THINK** should this REALLY also apply to postsynaptic models
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                auto psmVars = psm->getVars();
                for (size_t k= 0, l= psmVars.size(); k < l; k++) {
                    const auto &varInit = s.second.getPSVarInitialisers()[k];

                    // If this variable has any initialisation code
                    if(!varInit.getSnippet()->getCode().empty()) {
                        // Loop through postsynaptic neurons and substitute in initialisation code
                        os << CodeStream::OB(270) << "for (int i = 0; i < " << numTrgNeurons << "; i++)" << CodeStream::OB(280);
                        os << StandardSubstitutions::initVariable(varInit, psmVars[k].first + s.first + "[i] = $(0)",
                                                                  cpuFunctions, model.getPrecision(), "rng") << std::endl;
                        os << CodeStream::CB(280) << CodeStream::CB(270) << std::endl;
                    }
                }
            }
        }
    }
    os << std::endl << std::endl;
#ifndef CPU_ONLY
    os << "copyStateToDevice();" << std::endl << std::endl;

    // If any init threads were required, perform init kernel launch
    if(numInitThreads > 0) {
        os << "// perform on-device init" << std::endl;
        os << "dim3 iThreads(" << initBlkSz << ", 1);" << std::endl;
        os << "dim3 iGrid(" << numInitThreads / initBlkSz << ", 1);" << std::endl;
        os << "init <<<iGrid, iThreads>>>();" << std::endl;
    }
    os << "//initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later" << std::endl;
#endif
    os << CodeStream::CB(10) << std::endl;

}