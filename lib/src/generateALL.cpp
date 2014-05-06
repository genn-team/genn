/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

/*! \file generateALL.cpp
  \brief Main file combining the code for code generation. Part of the code generation section.

  The file includes separate files for generating kernels (generateKernels.cc),
  generating the CPU side code for running simulations on either the CPU or GPU (generateRunner.cc) and for CPU-only simulation code (generateCPU.cc).
*/


#include "global.h"
#include "utils.h"
#include "currentModel.cc"
#include "generateKernels.cc"
#include "generateRunner.cc"
#include "generateCPU.cc"

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else
#include <dirent.h> // for directory removal
#include <sys/stat.h> // needed for mkdir
#endif


/*! \brief This function will call the necessary sub-functions to generate the code for simulating a model. */
void generate_model_runner(int deviceID)
{

#ifdef _WIN32
  _mkdir((path + "/" + model->name + "_CODE_" + toString(deviceID)).c_str());
#else // UNIX
  mkdir((path + "/" + model->name + "_CODE_" + toString(deviceID)).c_str(), 0777); 
#endif

  // general shared code for GPU and CPU versions
  genRunner(deviceID, cerr);

  // GPU specific code generation
  genRunnerGPU(deviceID, cerr);
  
  // generate neuron kernels
  genNeuronKernel(deviceID, cerr);

  // generate synapse and learning kernels
  if (model->synapseGrpN > 0) genSynapseKernel(deviceID, cerr);

  // CPU specific code generation
  genRunnerCPU(deviceID, cerr);

  // Generate the equivalent of neuron kernel
  genNeuronFunction(deviceID, cerr);
  
  // Generate the equivalent of synapse and learning kernel
  if (model->synapseGrpN > 0) genSynapseFunction(deviceID, cerr);

  // Generate NVCC compiler flag file
  ofstream sm_os((path + "/" + model->name + "_CODE_" + toString(deviceID) + "/sm_version").c_str());
  sm_os << "-arch sm_" << deviceProp[deviceID].major << deviceProp[deviceID].minor << endl;
  sm_os.close();
}


//--------------------------------------------------------------------------
/*! 
  \brief Helper function that prepares data structures and detects the hardware properties to enable the code generation code that follows.

  The main tasks in this function are the detection and characterization of the GPU device present (if any), choosing which GPU device to use, finding and appropriate block size, taking note of the major and minor version of the CUDA enabled device chosen for use, and populating the list of standard neuron models. The chosen device number is returned.
*/
//--------------------------------------------------------------------------

int blockSizeOptimise(int deviceID)
{
  cerr << "optimizing block sizes for device " << deviceID << endl;

  int *blockSizePtr;
  char pipeBuffer[256];
  stringstream command, ptxInfo;
  string kernelName, junk;
  int reqRegs, reqSmem, deviceOccupancy, ptxInfoFound = 0;
  float warpAllocGran, regAllocGran, smemAllocGran;
  float blockLimit, mainBlockLimit, bestOccupancy;

  CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
  generate_model_runner(deviceID);

  // Run NVCC and pipe output to this process
  command << "nvcc -x cu -cubin -Xptxas=-v -arch=sm_" << deviceProp[deviceID].major;
  command << deviceProp[deviceID].minor << " -DDT -D\"CHECK_CUDA_ERRORS(call){call;}\" ";
  command << path << "/" << model->name << "_CODE_" << deviceID << "/runner.cc 2>&1 1>/dev/null";

#ifdef _WIN32
  FILE *nvccPipe = _popen(command.str().c_str(), "r");
#else // UNIX
  FILE *nvccPipe = popen(command.str().c_str(), "r");
#endif

  command.str("");
  cerr << "dry-run compile for device " << deviceID << endl;
  //cerr << command.str() << endl;
  if (!nvccPipe) {
    cerr << "ERROR: failed to open nvcc pipe" << endl;
    exit(EXIT_FAILURE);
  }

  // Read pipe until reg / smem usage is found, then calculate optimum block size for each kernel
  while (fgets(pipeBuffer, 256, nvccPipe) != NULL) {
    if (strstr(pipeBuffer, "error:") != NULL) {
      cout << pipeBuffer;
    }
    else if (strstr(pipeBuffer, "calcSynapses") != NULL) {
      blockSizePtr = &synapseBlkSz[deviceID];
      kernelName = "synapse";
    }
    else if (strstr(pipeBuffer, "learnSynapses") != NULL) {
      blockSizePtr = &learnBlkSz[deviceID];
      kernelName = "learn";
    }
    else if (strstr(pipeBuffer, "calcNeurons") != NULL) {
      blockSizePtr = &neuronBlkSz[deviceID];
      kernelName = "neuron";
    }
    if (strncmp(pipeBuffer, "ptxas info    : Used", 20) == 0) {
      ptxInfoFound = 1;
      bestOccupancy = 0;
      ptxInfo << pipeBuffer;
      ptxInfo >> junk >> junk >> junk >> junk >> reqRegs >> junk >> reqSmem;
      ptxInfo.str("");
      cerr << "kernel: " << kernelName << ", regs needed: " << reqRegs << ", smem needed: " << reqSmem << endl;

      // This data is required for block size optimisation, but cannot be found in deviceProp
      if (deviceProp[deviceID].major == 1) {
	smemAllocGran = 512;
	warpAllocGran = 2;
	regAllocGran = (deviceProp[deviceID].minor < 2) ? 256 : 512;
      }
      else if (deviceProp[deviceID].major == 2) {
	smemAllocGran = 128;
	warpAllocGran = 2;
	regAllocGran = 128;
      }
      else if (deviceProp[deviceID].major == 3) {
	smemAllocGran = 256;
	warpAllocGran = 4;
	regAllocGran = 256;
      }
      else {
	cerr << "Error: unsupported CUDA device major version: " << deviceProp[deviceID].major << endl;
	exit(EXIT_FAILURE);
      }

      // Test all block sizes (in warps) up to [max warps per block]
      for (int blockSize = 1; blockSize <= deviceProp[deviceID].maxThreadsPerBlock / 32; blockSize++) {

	// BLOCK LIMIT DUE TO THREADS
	blockLimit = floor(deviceProp[deviceID].maxThreadsPerMultiProcessor / 32 / blockSize);
	if (blockLimit > 8) blockLimit = 8; // mind the blocks per SM limit
	mainBlockLimit = blockLimit;

	// BLOCK LIMIT DUE TO REGISTERS
	if (deviceProp[deviceID].major == 1) { // if register allocation is per block
	  blockLimit = ceil(blockSize / warpAllocGran) * warpAllocGran;
	  blockLimit = ceil(blockLimit * reqRegs * 32 / regAllocGran) * regAllocGran;
	  blockLimit = floor(deviceProp[deviceID].regsPerBlock / blockLimit);
	}
	else { // if register allocation is per warp
	  blockLimit = ceil(reqRegs * 32 / regAllocGran) * regAllocGran;
	  blockLimit = floor(deviceProp[deviceID].regsPerBlock / blockLimit / warpAllocGran) * warpAllocGran;
	  blockLimit = floor(blockLimit / blockSize);
	}
	if (blockLimit < mainBlockLimit) mainBlockLimit = blockLimit;

	// BLOCK LIMIT DUE TO SHARED MEMORY
	blockLimit = ceil(reqSmem / smemAllocGran) * smemAllocGran;
	blockLimit = floor(deviceProp[deviceID].sharedMemPerBlock / blockLimit);
	if (blockLimit < mainBlockLimit) mainBlockLimit = blockLimit;

	// Update the best warp occupancy and the block size which enables it
	if ((blockSize * mainBlockLimit) > bestOccupancy) {
	  bestOccupancy = blockSize * mainBlockLimit;
	  *blockSizePtr = (int) blockSize * 32;

	  // Update the neuron kernel warp occupancy for this deviuce
	  if (blockSizePtr == &neuronBlkSz[deviceID]) {
	    deviceOccupancy = bestOccupancy * deviceProp[deviceID].multiProcessorCount;
	  }
	}
      }
    }
  }

  // Close the NVCC pipe
#ifdef _WIN32
  _pclose(nvccPipe);
#else // UNIX
  pclose(nvccPipe);
#endif

  // Delete dry-run code
  char filepath[256];
  struct dirent *file;
  const char *dirpath = (path + "/" + model->name + "_CODE_" + toString(deviceID) + "/").c_str();
  DIR *dir = opendir(dirpath);
  if (dir != NULL) {
    while ((file = readdir(dir)) != NULL) {
      if (strcmp(file->d_name, ".") && strcmp(file->d_name, "..")) {
	remove(strcat(strcpy(filepath, dirpath), file->d_name));
      }
    }
  }
  remove(dirpath);
  closedir(dir);

  if (!ptxInfoFound) {
    cerr << "ERROR: did not find any PTX info" << endl;
    cerr << "ensure nvcc is on your $PATH, and fix any NVCC errors listed above" << endl;
    exit(EXIT_FAILURE);
  }

  return deviceOccupancy;
}


//--------------------------------------------------------------------------
/*! 
  \brief Function that generates CUDA simulation code for the best device present on the host.

  Returns the device which supports the most simultaniously active neuron kernel warps, if block size
  optimisation is requested. Otherwise returns the device with the most global memory.
*/
//--------------------------------------------------------------------------

void bestDeviceGenerate()
{
  int best = 0, current, bestDevice;
  for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
    if (optCudaBlockSize) {
      current = blockSizeOptimise(deviceID); // find the device which supports the highest warp occupancy
    }
    else {
      current = deviceProp[deviceID].totalGlobalMem; // find the device with the most global memory
    }
    if (current > best) {
      best = current;
      bestDevice = deviceID;
    }
  }

  cerr << "using device " << bestDevice;
  if (optCudaBlockSize) cerr << " which supports " << best << " warps of occupancy" << endl;
  else cerr << " with " << best << " bytes of global memory" << endl;

  generate_model_runner(bestDevice);
}


//--------------------------------------------------------------------------
/*! 
  \brief Distributes the neuron and synapse groups of the simulation across 
  all available devices on the host.
*/
//--------------------------------------------------------------------------

void multiDeviceGenerate()
{

  // this needs to split the network evenly (fancy load balancing algorithm will go here, in future),
  // set the hostID:deviceID of each neuron and synapse group accordingly,
  // perform block size optimisation (if requested) for each hostID:deviceID included in the computation,
  // and finally update the padSum* variables of the NNmodel to reflect the new block sizes

  model->calcPaddedThreadSums();
}


/*! \brief Main entry point for the generateALL executable that generates
  the code for GPU and CPU.

  The main function is the entry point for the code generation engine. It 
  prepares the system and then invokes generate_model_runner to inititate
  the different parts of actual code generation.
*/

int main(int argc,     //!< number of arguments; expected to be 2
	 char *argv[]  //!< Arguments; expected to contain the target directory for code generation.
	 )
{
  if (argc != 2) {
    cerr << "usage: generateALL <target dir>" << endl;
    exit(EXIT_FAILURE);
  }
  cerr << "call was ";
  for (int i= 0; i < argc; i++) {
    cerr << argv[i] << " ";
  }
  cerr << endl;

  // initialise network model
  path = toString(argv[1]);
  model = new NNmodel();
  CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
  synapseBlkSz = new int[deviceCount];
  learnBlkSz = new int[deviceCount];
  neuronBlkSz = new int[deviceCount];
  deviceProp = new cudaDeviceProp[deviceCount];

  for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[deviceID]), deviceID));
    synapseBlkSz[deviceID] = 256;
    learnBlkSz[deviceID] = 256;
    neuronBlkSz[deviceID] = 256;
  }

  prepareStandardModels();
  preparePostSynModels();
  modelDefinition(*model);

  if (useAllCudaDevices) {
    multiDeviceGenerate();
  }
  else {
    bestDeviceGenerate();
  }

  delete[] synapseBlkSz;
  delete[] learnBlkSz;
  delete[] neuronBlkSz;
  delete[] deviceProp;
  delete model;

  return EXIT_SUCCESS;
}
