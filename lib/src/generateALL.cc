/*--------------------------------------------------------------------------
   Author: Thomas Nowotny

   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK

   email to:  T.Nowotny@sussex.ac.uk

   initial version: 2010-02-07

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateALL.cc

  \brief Main file combining the code for code generation. Part of the code generation section.

  The file includes separate files for generating kernels (generateKernels.cc), generating the CPU side code for running simulations on either the CPU or GPU (generateRunner.cc) and for CPU-only simulation code (generateCPU.cc).

*/
//--------------------------------------------------------------------------

#include "global.h"
#include MODEL
#include "generateALL.h"
#include "generateRunner.h"
#include "generateCPU.h"
#include "generateKernels.h"
#include "modelSpec.h"
#include "utils.h"
#include "stringUtils.h"
#include "CodeHelper.h"

#include <cmath>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h> // needed for mkdir
#endif

CodeHelper hlp;
//hlp.setVerbose(true);//this will show the generation of bracketing (brace) levels. Helps to debug a bracketing issue


//--------------------------------------------------------------------------
/*! \brief This function will call the necessary sub-functions to generate the code for simulating a model.
 */
//--------------------------------------------------------------------------

void generate_model_runner(const NNmodel &model,  //!< Model description
                           const string &path      //!< Path where the generated code will be deposited
                           )
{
#ifdef _WIN32
  _mkdir((path + "\\" + model.name + "_CODE").c_str());
#else // UNIX
  mkdir((path + "/" + model.name + "_CODE").c_str(), 0777);
#endif

  // general shared code for GPU and CPU versions
  genRunner(model, path);

#ifndef CPU_ONLY
  // GPU specific code generation
  genRunnerGPU(model, path);

  // generate neuron kernels
  genNeuronKernel(model, path);

  // generate synapse and learning kernels
  if (model.synapseGrpN > 0) genSynapseKernel(model, path);
#endif

  // Generate the equivalent of neuron kernel
  genNeuronFunction(model, path);

  // Generate the equivalent of synapse and learning kernel
  if (model.synapseGrpN > 0) genSynapseFunction(model, path);

  // Generate the Makefile for the generated code
  genMakefile(model, path);
}


//--------------------------------------------------------------------------
/*!
  \brief Helper function that prepares data structures and detects the hardware properties to enable the code generation code that follows.

  The main tasks in this function are the detection and characterization of the GPU device present (if any), choosing which GPU device to use, finding and appropriate block size, taking note of the major and minor version of the CUDA enabled device chosen for use, and populating the list of standard neuron models. The chosen device number is returned.
*/
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
void chooseDevice(NNmodel &model, //!< the nn model we are generating code for
                  const string &path     //!< path the generated code will be deposited
    )
{
    const int krnlNo = 4;
    const char *kernelName[krnlNo]= {"calcSynapses", "learnSynapsesPost", "calcSynapseDynamics", "calcNeurons"};
    size_t globalMem, mostGlobalMem = 0;
    int chosenDevice = 0;

    // IF OPTIMISATION IS ON: Choose the device which supports the highest warp occupancy.
    if (GENN_PREFERENCES::optimiseBlockSize) {
        cout << "optimizing block size..." << endl;
        int reqRegs, reqSmem, requiredBlocks;
        int warpSize= 32;

        unsigned int **bestBlkSz= new unsigned int*[krnlNo];
        int **smallModel= new int*[krnlNo];
        int **deviceOccupancy= new int*[krnlNo];
        vector<unsigned int> *groupSize= new vector<unsigned int>[krnlNo];
        float blockLimit, mainBlockLimit;
        // initialise the smallXXX flags and bestBlkSz
        for (int kernel= 0; kernel < krnlNo; kernel++) {
            bestBlkSz[kernel]= new unsigned int[deviceCount];
            smallModel[kernel]= new int[deviceCount];
            deviceOccupancy[kernel]= new int[deviceCount];
            for (theDevice = 0; theDevice < deviceCount; theDevice++) { // initialise all whether used or not
                bestBlkSz[kernel][theDevice]= 0;
                smallModel[kernel][theDevice]= 0;
                deviceOccupancy[kernel][theDevice]= 0;
            }
        }

        // Get the sizes of each synapse / learn group present on this host and device
        vector<unsigned int> synapseN, learnN;
        for (int group = 0; group < model.synapseGrpN; group++) {
            if ((model.synapseConnType[group] == SPARSE) && (model.maxConn[group]>0)) {
                groupSize[0].push_back(model.maxConn[group]);
            }
            else {
                groupSize[0].push_back(model.neuronN[model.synapseTarget[group]]);
            }
            if (model.synapseUsesPostLearning[group]) {     // TODO: this needs updating where learning is detected properly!
                groupSize[1].push_back(model.neuronN[model.synapseSource[group]]);
            }
            if (model.synapseUsesSynapseDynamics[group]) {
                if ((model.synapseConnType[group] == SPARSE) && (model.maxConn[group]>0)) {
                    groupSize[2].push_back(model.neuronN[model.synapseSource[group]]*model.maxConn[group]);
                }
                else {
                    groupSize[2].push_back(model.neuronN[model.synapseSource[group]]*model.neuronN[model.synapseTarget[group]]);
                }
            }
        }
        groupSize[3]= model.neuronN;
#ifdef BLOCKSZ_DEBUG
        for (int i= 0; i < krnlNo; i++) {
            cerr << "BLOCKSZ_DEBUG: ";
            for (int j= 0; j < groupSize[i].size(); j++) {
                cerr << "groupSize[" << i << "][" << j << "]=" << groupSize[i][j] << "; ";
            }
            cerr << endl;
        }
#endif

        for (theDevice = 0; theDevice < deviceCount; theDevice++) {
            if (!(GENN_PREFERENCES::autoChooseDevice) && (theDevice != GENN_PREFERENCES::defaultDevice)) continue;

            // This data is required for block size optimisation, but cannot be found in deviceProp.
            float warpAllocGran, regAllocGran, smemAllocGran, maxBlocksPerSM;
            if (deviceProp[theDevice].major == 1) {
                smemAllocGran = 512;
                warpAllocGran = 2;
                regAllocGran = (deviceProp[theDevice].minor < 2) ? 256 : 512;
                maxBlocksPerSM = 8;
            }
            else if (deviceProp[theDevice].major == 2) {
                smemAllocGran = 128;
                warpAllocGran = 2;
                regAllocGran = 64;
                maxBlocksPerSM = 8;
            }
            else if (deviceProp[theDevice].major == 3) {
                smemAllocGran = 256;
                warpAllocGran = 4;
                regAllocGran = 256;
                maxBlocksPerSM = 16;
            }
            else if (deviceProp[theDevice].major == 5) {
                smemAllocGran = 256;
                warpAllocGran = 4;
                regAllocGran = 256;
                maxBlocksPerSM = 32;
            }
            else {
                cerr << "Error: unsupported CUDA device major version: " << deviceProp[theDevice].major << endl;
                exit(EXIT_FAILURE);
            }

            // Signal error and exit if SM version < 1.3 and double precision floats are requested.
            if ((deviceProp[theDevice].major == 1) && (deviceProp[theDevice].minor < 3))
            {
                if (model.ftype != "float")
                {
                    cerr << "Error: This CUDA device does not support double precision floating-point." << endl;
                    cerr << "       Either change the ftype parameter to GENN_FLOAT or find a newer GPU" << endl;
                    exit(EXIT_FAILURE);
                }
            }

#ifdef BLOCKSZ_DEBUG
            cerr << "BLOCKSZ_DEBUG: smemAllocGran= " <<  smemAllocGran << endl;
            cerr << "BLOCKSZ_DEBUG: warpAllocGran= " <<  warpAllocGran << endl;
            cerr << "BLOCKSZ_DEBUG: regAllocGran= " <<  regAllocGran << endl;
            cerr << "BLOCKSZ_DEBUG: maxBlocksPerSM= " <<  maxBlocksPerSM << endl;
#endif

            // obtaining ptxas info.
            CUmodule module;
            CUdevice cuDevice;
            CUcontext cuContext;
            CHECK_CU_ERRORS(cuDeviceGet(&cuDevice, theDevice));
            CHECK_CU_ERRORS(cuCtxCreate(&cuContext, 0, cuDevice));
            CHECK_CU_ERRORS(cuCtxSetCurrent(cuContext));

            string nvccFlags = "-cubin -x cu -arch sm_";
            nvccFlags += tS(deviceProp[theDevice].major) + tS(deviceProp[theDevice].minor);
            nvccFlags += " " + GENN_PREFERENCES::userNvccFlags;
            if (GENN_PREFERENCES::optimizeCode) nvccFlags += " -O3 -use_fast_math";
            if (GENN_PREFERENCES::debugCode) nvccFlags += " -O0 -g -G";
            if (GENN_PREFERENCES::showPtxInfo) nvccFlags += " -Xptxas \"-v\"";

#ifdef _WIN32
            nvccFlags += " -I\"%GENN_PATH%\\lib\\include\"";
            string runnerPath = path + "\\" + model.name + "_CODE\\runner.cc";
            string cubinPath = path + "\\runner.cubin";
            string nvccCommand = "\"\"" + tS(NVCC) + "\" " + nvccFlags;
            nvccCommand += " -o \"" + cubinPath + "\" \"" + runnerPath + "\"\"";
#else
            nvccFlags += " -I\"$GENN_PATH/lib/include\"";
            string runnerPath = path + "/" + model.name + "_CODE/runner.cc";
            string cubinPath = path + "/runner.cubin";
            string nvccCommand = "\"" + tS(NVCC) + "\" " + nvccFlags;
            nvccCommand += " -o \"" + cubinPath + "\" \"" + runnerPath + "\"";
#endif

            cudaFuncAttributes krnlAttr[2][krnlNo];
            CUfunction kern;
            CUresult res;
            int KrnlExist[krnlNo];
            for (int rep= 0; rep < 2; rep++) {
                // do two repititions with different candidate kernel size
                synapseBlkSz = warpSize*(rep+1);
                learnBlkSz = warpSize*(rep+1);
                synDynBlkSz= warpSize*(rep+1);
                neuronBlkSz = warpSize*(rep+1);
                model.setPopulationSums();
                generate_model_runner(model, path);

                // Run NVCC
                cout << "dry-run compile for device " << theDevice << endl;
                cout << nvccCommand << endl;
                system(nvccCommand.c_str());

                CHECK_CU_ERRORS(cuModuleLoad(&module, cubinPath.c_str()));
                for (int i= 0; i < krnlNo; i++) {
#ifdef BLOCKSZ_DEBUG
                    cerr << "BLOCKSZ_DEBUG: ptxas info for " << kernelName[i] << " ..." << endl;
#endif
                    res= cuModuleGetFunction(&kern, module, kernelName[i]);
                    if (res == CUDA_SUCCESS) {
                        cudaFuncGetAttributesDriver(&krnlAttr[rep][i], kern);
                        KrnlExist[i]= 1;
                    }
                    else {
                        KrnlExist[i]= 0;
                    }
                }
                CHECK_CU_ERRORS(cuModuleUnload(module));

                if (remove(cubinPath.c_str())) {
                    cerr << "generateALL: Error deleting dry-run cubin file" << endl;
                    exit(EXIT_FAILURE);
                }
            }

            for (int kernel= 0; kernel < krnlNo; kernel++) {
                if (KrnlExist[kernel]) {
                    reqRegs= krnlAttr[0][kernel].numRegs;
                    // estimate shared memory requirement as function of kernel size (assume constant+linear in size)
                    double x1= warpSize;
                    double x2= 2*warpSize;
                    double y1= krnlAttr[0][kernel].sharedSizeBytes;
                    double y2= krnlAttr[1][kernel].sharedSizeBytes;
                    double reqSmemM= (y2-y1)/(x2-x1);
                    double reqSmemB= y1-reqSmemM*x1;
                    for (int blkSz = 1, mx= deviceProp[theDevice].maxThreadsPerBlock / warpSize; blkSz <= mx; blkSz++) {

#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel << ": Candidate block size: " << blkSz*warpSize << endl;
#endif
                        // BLOCK LIMIT DUE TO THREADS
                        blockLimit = floor((float) deviceProp[theDevice].maxThreadsPerMultiProcessor/warpSize/blkSz);
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Block limit due to maxThreadsPerMultiProcessor: " << blockLimit << endl;
#endif
                        if (blockLimit > maxBlocksPerSM) blockLimit = maxBlocksPerSM;
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Block limit corrected for maxBlocksPerSM: " << blockLimit << endl;
#endif
                        mainBlockLimit = blockLimit;

                        // BLOCK LIMIT DUE TO REGISTERS
                        if (deviceProp[theDevice].major == 1) { // if register allocation is per block
                            blockLimit = ceil(blkSz/warpAllocGran)*warpAllocGran;
                            blockLimit = ceil(blockLimit*reqRegs*warpSize/regAllocGran)*regAllocGran;
                            blockLimit = floor(deviceProp[theDevice].regsPerBlock/blockLimit);
                        }
                        else { // if register allocation is per warp
                            blockLimit = ceil(reqRegs*warpSize/regAllocGran)*regAllocGran;
                            blockLimit = floor(deviceProp[theDevice].regsPerBlock/blockLimit/warpAllocGran)*warpAllocGran;
                            blockLimit = floor(blockLimit/blkSz);
                        }
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Block limit due to registers: " << blockLimit << endl;
#endif
                        if (blockLimit < mainBlockLimit) mainBlockLimit= blockLimit;

                        // BLOCK LIMIT DUE TO SHARED MEMORY
                        reqSmem= (unsigned int) (reqSmemM*blkSz*warpSize+reqSmemB);
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Required shared memory for block size " << blkSz*warpSize << " is: " << reqSmem << endl;
#endif
                        blockLimit = ceil(reqSmem/smemAllocGran)*smemAllocGran;
                        blockLimit = floor(deviceProp[theDevice].sharedMemPerBlock/blockLimit);
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Block limit due to shared memory: " << blockLimit << endl;
#endif
                        if (blockLimit < mainBlockLimit) mainBlockLimit= blockLimit;

                        // The number of thread blocks required to simulate all groups
                        requiredBlocks = 0;
                        for (int group = 0; group < groupSize[kernel].size(); group++) {
                            requiredBlocks+= ceil(((float) groupSize[kernel][group])/(blkSz*warpSize));
                        }
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                        cerr << ": Required blocks (according to padded sum): " << requiredBlocks << endl;
#endif

                        // Use a small block size if it allows all groups to occupy the device concurrently
                        if (requiredBlocks <= (mainBlockLimit*deviceProp[theDevice].multiProcessorCount)) {
                            bestBlkSz[kernel][theDevice] = (unsigned int) blkSz*warpSize;
                            deviceOccupancy[kernel][theDevice]= blkSz*mainBlockLimit*deviceProp[theDevice].multiProcessorCount;
                            smallModel[kernel][theDevice] = 1;

#ifdef BLOCKSZ_DEBUG
                            cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                            cerr << ": Small model situation detected; bestBlkSz: " << bestBlkSz[kernel][theDevice] << endl;
                            cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                            cerr << ": Setting smallModel[" << kernel << "][" << theDevice << "] to 1" << endl;
#endif
                            break; // for small model the first (smallest) block size allowing it is chosen
                        }

                        // Update the best warp occupancy and the block size which enables it.
                        int newOccupancy= blkSz*mainBlockLimit*deviceProp[theDevice].multiProcessorCount;
                        if (newOccupancy > deviceOccupancy[kernel][theDevice]) {
                            bestBlkSz[kernel][theDevice] = (unsigned int) blkSz*warpSize;
                            deviceOccupancy[kernel][theDevice]= newOccupancy;

#ifdef BLOCKSZ_DEBUG
                            cerr << "BLOCKSZ_DEBUG: Kernel " << kernel;
                            cerr << ": Small model not enabled; device occupancy criterion; deviceOccupancy ";
                            cerr << deviceOccupancy[kernel][theDevice] << "; blocksize for " << kernelName[kernel];
                            cerr << ": " << (unsigned int) blkSz * warpSize << endl;
#endif
                        }
                    }
                }
            }
        }

        // Now choose the device
        if (GENN_PREFERENCES::autoChooseDevice) {

            int anySmall= 0;
            int *smallModelCnt= new int[deviceCount];
            int *sumOccupancy= new int[deviceCount];
            float smVersion, bestSmVersion = 0.0;
            int bestSmallModelCnt= 0;
            int bestDeviceOccupancy = 0;

            for (theDevice = 0; theDevice < deviceCount; theDevice++) {
                if (!(GENN_PREFERENCES::autoChooseDevice) && (theDevice != GENN_PREFERENCES::defaultDevice)) continue;

                smallModelCnt[theDevice]= 0;
                sumOccupancy[theDevice]= 0;
                for (int kernel= 0; kernel < krnlNo; kernel++) {
#ifdef BLOCKSZ_DEBUG
                    cerr << "BLOCKSZ_DEBUG: smallModel[" << kernel << "][" << theDevice << "]= ";
                    cerr << smallModel[kernel][theDevice] << endl;
#endif
                    if (smallModel[kernel][theDevice]) {
                        smallModelCnt[theDevice]++;
                    }
                    sumOccupancy[theDevice]+= deviceOccupancy[kernel][theDevice];
                }
                smVersion= deviceProp[theDevice].major+((float) deviceProp[theDevice].minor/10);
#ifdef BLOCKSZ_DEBUG
                cerr << "BLOCKSZ_DEBUG: Choosing device: First criterion: Small model count" << endl;
#endif
                if (smallModelCnt[theDevice] > bestSmallModelCnt) {
                    bestSmallModelCnt= smallModelCnt[theDevice];
                    bestDeviceOccupancy= sumOccupancy[theDevice];
                    bestSmVersion= smVersion;
                    chosenDevice= theDevice;
#ifdef BLOCKSZ_DEBUG
                    cerr << "BLOCKSZ_DEBUG: Choosing based on larger small model count;";
                    cerr << "device: " << chosenDevice << "; bestSmallModelCnt: " <<  bestSmallModelCnt << endl;
#endif
                }
                else {
                    if (smallModelCnt[theDevice] == bestSmallModelCnt) {
#ifdef BLOCKSZ_DEBUG
                        cerr << "BLOCKSZ_DEBUG: Equal small model count: Next criterion: Occupancy" << endl;
#endif
                        if (sumOccupancy[theDevice] > bestDeviceOccupancy) {
                            bestDeviceOccupancy = sumOccupancy[theDevice];
                            bestSmVersion= smVersion;
                            chosenDevice= theDevice;
#ifdef BLOCKSZ_DEBUG
                            cerr << "BLOCKSZ_DEBUG: Choose device based on occupancy;";
                            cerr << "device: " << chosenDevice << "; bestDeviceOccupancy (sum): " << bestDeviceOccupancy << endl;
#endif
                        }
                        else {
                            if (sumOccupancy[theDevice] == bestDeviceOccupancy) {
#ifdef BLOCKSZ_DEBUG
                                cerr << "BLOCKSZ_DEBUG: Equal device occupancy: Next criterion: smVersion" << endl;
#endif
                                if (smVersion > bestSmVersion) {
                                    bestSmVersion= smVersion;
                                    chosenDevice= theDevice;
#ifdef BLOCKSZ_DEBUG
                                    cerr << "BLOCKSZ_DEBUG: Choosing based on bestSmVersion;";
                                    cerr << "device:  " << chosenDevice <<  "; bestSmVersion: " << bestSmVersion << endl;
#endif
                                }
#ifdef BLOCKSZ_DEBUG
                                else {
                                    cerr << "BLOCKSZ_DEBUG: Devices are tied;";
                                    cerr << "chosen device remains: " << chosenDevice << endl;
                                }
#endif
                            }
#ifdef BLOCKSZ_DEBUG
                            else {
                                cerr << "BLOCKSZ_DEBUG: Device has inferior occupancy;";
                                cerr << "chosen device remains: " << chosenDevice << endl;
                            }
#endif
                        }
                    }
#ifdef BLOCKSZ_DEBUG
                    else {
                        cerr << "BLOCKSZ_DEBUG: Device has inferior small model count;";
                        cerr << "chosen device remains: " << chosenDevice << endl;
                    }
#endif
                }
            }
            cout << "Using device " << chosenDevice << " (" << deviceProp[chosenDevice].name << "), with up to ";
            cout << bestDeviceOccupancy << " warps of summed kernel occupancy." << endl;
            delete[] smallModelCnt;
            delete[] sumOccupancy;
        }
        else {
            chosenDevice= GENN_PREFERENCES::defaultDevice;
        }
        synapseBlkSz = bestBlkSz[0][chosenDevice];
        learnBlkSz = bestBlkSz[1][chosenDevice];
        synDynBlkSz= bestBlkSz[2][chosenDevice];
        neuronBlkSz = bestBlkSz[3][chosenDevice];
        for (int kernel= 0; kernel < krnlNo; kernel++) {
            delete[] bestBlkSz[kernel];
            delete[] smallModel[kernel];
            delete[] deviceOccupancy[kernel];
        }
        delete[] bestBlkSz;
        delete[] smallModel;
        delete[] deviceOccupancy;
        delete[] groupSize;
    }

    // IF OPTIMISATION IS OFF: Simply choose the device with the most global memory.
    else {
        cout << "skipping block size optimisation..." << endl;
        synapseBlkSz= GENN_PREFERENCES::synapseBlockSize;
        learnBlkSz= GENN_PREFERENCES::learningBlockSize;
        synDynBlkSz= GENN_PREFERENCES::synapseDynamicsBlockSize;
        neuronBlkSz= GENN_PREFERENCES::neuronBlockSize;
        if (GENN_PREFERENCES::autoChooseDevice) {
            for (theDevice = 0; theDevice < deviceCount; theDevice++) {
                CHECK_CUDA_ERRORS(cudaSetDevice(theDevice));
                CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[theDevice]), theDevice));
                globalMem = deviceProp[theDevice].totalGlobalMem;
                if (globalMem >= mostGlobalMem) {
                    mostGlobalMem = globalMem;
                    chosenDevice = theDevice;
                }
            }
            cout << "Using device " << chosenDevice << ", which has " << mostGlobalMem << " bytes of global memory." << endl;
        }
        else {
            chosenDevice= GENN_PREFERENCES::defaultDevice;
        }
    }

    theDevice = chosenDevice;
    model.setPopulationSums();

    ofstream sm_os((path + "/sm_version.mk").c_str());
#ifdef _WIN32
    sm_os << "NVCCFLAGS =$(NVCCFLAGS) -arch sm_";
#else // UNIX
    sm_os << "NVCCFLAGS += -arch sm_";
#endif
    sm_os << deviceProp[chosenDevice].major << deviceProp[chosenDevice].minor << endl;
    sm_os.close();

    cout << "synapse block size: " << synapseBlkSz << endl;
    cout << "learn block size: " << learnBlkSz << endl;
    cout << "synapseDynamics block size: " << synDynBlkSz << endl;
    cout << "neuron block size: " << neuronBlkSz << endl;
}
#endif


//--------------------------------------------------------------------------
/*! \brief Main entry point for the generateALL executable that generates
  the code for GPU and CPU.

  The main function is the entry point for the code generation engine. It
  prepares the system and then invokes generate_model_runner to inititate
  the different parts of actual code generation.
*/
//--------------------------------------------------------------------------

int main(int argc,     //!< number of arguments; expected to be 2
         char *argv[]  //!< Arguments; expected to contain the target directory for code generation.
    )
{
    if (argc != 2) {
        cerr << "usage: generateALL <target dir>" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "call was ";
    for (int i= 0; i < argc; i++) {
        cout << argv[i] << " ";
    }
    cout << endl;

#ifdef DEBUG
    GENN_PREFERENCES::optimizeCode = false;
    GENN_PREFERENCES::debugCode = true;
#endif // DEBUG

#ifndef CPU_ONLY
    CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
    deviceProp = new cudaDeviceProp[deviceCount];
    for (int device = 0; device < deviceCount; device++) {
        CHECK_CUDA_ERRORS(cudaSetDevice(device));
        CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));
    }
#endif // CPU_ONLY

    NNmodel *model = new NNmodel();
#ifdef DT
    model->setDT(DT);
    cout << "Setting integration step size from global DT macro: " << DT << endl;
#endif // DT
    modelDefinition(*model);
    if (!model->final) {
        gennError("Model was not finalized in modelDefinition(). Please call model.finalize().");
    }

    string path= toString(argv[1]);
#ifndef CPU_ONLY
    chooseDevice(*model, path);
#endif // CPU_ONLY
    generate_model_runner(*model, path);

    return EXIT_SUCCESS;
}
