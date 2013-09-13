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
#include <sys/stat.h> //needed for mkdir
#endif


/*! \brief This function will call the necessary sub-functions to generate the code for simulating a model. */

void generate_model_runner(NNmodel &model,  //!< Model description
			   string path      //!< Path where the generated code will be deposited
			   )
{

#ifdef _WIN32
  _mkdir((path + "\\" + model.name + "_CODE").c_str());
#else // UNIX
  mkdir((path + "/" + model.name + "_CODE").c_str(), 0777); 
#endif

  // general shared code for GPU and CPU versions
  genRunner(model, path, cerr);

  // GPU specific code generation
  genRunnerGPU(model, path, cerr);
  
  // generate neuron kernels
  genNeuronKernel(model, path, cerr);

  // generate synapse and learning kernels
  if (model.synapseGrpN>0) genSynapseKernel(model, path, cerr);

  // CPU specific code generation
  genRunnerCPU(model, path, cerr);

  // Generate the equivalent of neuron kernel
  genNeuronFunction(model, path, cerr);
  
  // Generate the equivalent of synapse and learning kernel
  if (model.synapseGrpN>0) genSynapseFunction(model, path, cerr);
}


//--------------------------------------------------------------------------
/*! 
  \brief Helper function that prepares data structures and detects the hardware properties to enable the code generation code that follows.

  The main tasks in this function are the detection and characterization of the GPU device present (if any), choosing which GPU device to use, finding and appropriate block size, taking note of the major and minor version of the CUDA enabled device chosen for use, and populating the list of standard neuron models. The chosen device number is returned.
*/
//--------------------------------------------------------------------------

int chooseDevice(ostream &mos,   //!< output stream for messages
		 NNmodel &model, //!< the nn model we are generating code for
		 string path     //!< path the generated code will be deposited
		 )
{
  // Get the specifications of all available cuda devices, then work out which one we will use.
  int deviceCount, chosenDevice = 0;
  CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
  deviceProp = new cudaDeviceProp[deviceCount];

  if (optimiseBlockSize) { // IF OPTIMISATION IS ON: Choose the device which supports the highest warp occupancy.
    mos << "optimiting block size..." << endl;

    stringstream command;
    char pipeBuffer[128];
    stringstream ptxInfo;
    string kernelName, junk;
    int reqRegs, reqSmem;
    bool ptxInfoFound = false;

    unsigned int bestSynBlkSz[deviceCount];
    unsigned int bestLrnBlkSz[deviceCount];
    unsigned int bestNrnBlkSz[deviceCount];
    unsigned int (*blockSizePointer)[deviceCount];
    float warpAllocGran, regAllocGran, smemAllocGran;
    float blockLimit, mainBlockLimit, bestOccupancy;
    int deviceOccupancy, bestDeviceOccupancy = 0;

    for (int device = 0; device < deviceCount; device++) {
      CHECK_CUDA_ERRORS(cudaSetDevice(device));
      CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));      
      generate_model_runner(model, path);
      bestSynBlkSz[device] = 0;
      bestLrnBlkSz[device] = 0;
      bestNrnBlkSz[device] = 0;

      // Run NVCC and pipe output to this process.
      command << "nvcc -x cu -cubin -Xptxas=-v -arch=sm_" << deviceProp[device].major
	      << deviceProp[device].minor << " -DDT -D\"CHECK_CUDA_ERRORS(call){call;}\" "; 
#ifdef _WIN32
      command << path << "\\" << model.name << "_CODE\\runner.cc 2>&1";
      FILE *nvccPipe = _popen(command.str().c_str(), "r");
#else // UNIX
      command << path << "/" << model.name << "_CODE/runner.cc 2>&1";
      FILE *nvccPipe = popen(command.str().c_str(), "r");
#endif
      mos << "dry-run compile for device " << device << endl;
      //mos << command.str() << endl;
      command.str("");
      if (!nvccPipe) {
	mos << "ERROR: faied to open nvcc pipe" << endl;
	exit(EXIT_FAILURE);
      }

      // Read pipe until reg / smem usage is found, then calculate optimum block size for each kernel.
      while (fgets(pipeBuffer, 128, nvccPipe) != NULL) {
	if (strstr(pipeBuffer, "calcSynapses") != NULL) {
	  blockSizePointer = &bestSynBlkSz;
	  kernelName = "synapse";
	}
	else if (strstr(pipeBuffer, "learnSynapses") != NULL) {
	  blockSizePointer = &bestLrnBlkSz;
	  kernelName = "learn";
	}
	else if (strstr(pipeBuffer, "calcNeurons") != NULL) {
	  blockSizePointer = &bestNrnBlkSz;
	  kernelName = "neuron";
	}
	if (strncmp(pipeBuffer, "ptxas info    : Used", 20) == 0) {
	  ptxInfoFound = true;
	  bestOccupancy = 0;
	  ptxInfo << pipeBuffer;
	  ptxInfo >> junk >> junk >> junk >> junk >> reqRegs >> junk >> reqSmem;
	  ptxInfo.str("");
	  mos << "kernel: " << kernelName << ", regs needed: " << reqRegs << ", smem needed: " << reqSmem << endl;

	  // This data is required for block size optimisation, but cannot be found in deviceProp.
	  if (deviceProp[device].major == 1) {
	    smemAllocGran = 512;
	    warpAllocGran = 2;
	    regAllocGran = (deviceProp[device].minor < 2) ? 256 : 512;
	  }
	  else if (deviceProp[device].major == 2) {
	    smemAllocGran = 128;
	    warpAllocGran = 2;
	    regAllocGran = 64;
	  }
	  else { // major == 3
	    smemAllocGran = 256;
	    warpAllocGran = 4;
	    regAllocGran = 256;
	  }

	  // Test all block sizes (in warps) up to [max warps per block].
	  for (int blockSize = 1; blockSize <= deviceProp[device].maxThreadsPerBlock / 32; blockSize++) {

	    // BLOCK LIMIT DUE TO THREADS
	    blockLimit = floor(deviceProp[device].maxThreadsPerMultiProcessor / 32 / blockSize);
	    if (blockLimit > 8) blockLimit = 8; // mind the blocks per SM limit
	    mainBlockLimit = blockLimit;

	    // BLOCK LIMIT DUE TO REGISTERS
	    if (deviceProp[device].major == 1) { // if register allocation is per block
	      blockLimit = ceil(blockSize / warpAllocGran) * warpAllocGran;
	      blockLimit = ceil(blockLimit * reqRegs * 32 / regAllocGran) * regAllocGran;
	      blockLimit = floor(deviceProp[device].regsPerBlock / blockLimit);
	    }
	    else { // if register allocation is per warp
	      blockLimit = ceil(reqRegs * 32 / regAllocGran) * regAllocGran;
	      blockLimit = floor(deviceProp[device].regsPerBlock / blockLimit / warpAllocGran) * warpAllocGran;
	      blockLimit = floor(blockLimit / blockSize);
	    }
	    if (blockLimit < mainBlockLimit) mainBlockLimit = blockLimit;

	    // BLOCK LIMIT DUE TO SHARED MEMORY
	    blockLimit = ceil(reqSmem / smemAllocGran) * smemAllocGran;
	    blockLimit = floor(deviceProp[device].sharedMemPerBlock / blockLimit);
	    if (blockLimit < mainBlockLimit) mainBlockLimit = blockLimit;

	    // Update the best thread occupancy and the block size which enables it.
	    if ((blockSize * 32 * mainBlockLimit) > bestOccupancy) {
	      bestOccupancy = blockSize * 32 * mainBlockLimit;
	      (*blockSizePointer)[device] = (unsigned int)blockSize * 32;

	      // Choose this device and set optimal block sizes if it enables higher neuron kernel occupancy.
	      if (blockSizePointer == &bestNrnBlkSz) {
		deviceOccupancy = bestOccupancy * deviceProp[device].multiProcessorCount;
		if (deviceOccupancy >= bestDeviceOccupancy) {
		  bestDeviceOccupancy = deviceOccupancy;
		  chosenDevice = device;
		}
	      }
	    }
	  }
	}
      }

      // Close the NVCC pipe after each invocation.
#ifdef _WIN32
      _pclose(nvccPipe);
#else // UNIX
      pclose(nvccPipe);
#endif

    }
    if (!ptxInfoFound) {
      mos << "ERROR: did not find any PTX info (is nvcc on your $PATH ?)" << endl;
      exit(EXIT_FAILURE);
    }
    synapseBlkSz = bestSynBlkSz[chosenDevice];
    learnBlkSz = bestLrnBlkSz[chosenDevice];
    neuronBlkSz = bestNrnBlkSz[chosenDevice];
    mos << "Using device " << chosenDevice << ", with a neuron kernel occupancy of "
	<< bestDeviceOccupancy << " threads." << endl;
  }

  else { // IF OPTIMISATION IS OFF: Simply choose the device with the most global memory.
    mos << "skipping block size optimisation..." << endl;
    size_t globalMem, mostGlobalMem = 0;
    for (int device = 0; device < deviceCount; device++) {
      CHECK_CUDA_ERRORS(cudaSetDevice(device));
      CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[device]), device));
      globalMem = deviceProp[device].totalGlobalMem;
      if (globalMem >= mostGlobalMem) {
	mostGlobalMem = globalMem;
	chosenDevice = device;
      }
    }
    mos << "Using device " << chosenDevice << ", which has "
	<< mostGlobalMem << " bytes of global memory." << endl;
  }

  ofstream sm_os("sm_Version.mk");
  sm_os << "NVCCFLAGS += -arch sm_" << deviceProp[chosenDevice].major << deviceProp[chosenDevice].minor << endl;
  sm_os.close();

  mos << "synapse block size: " << synapseBlkSz << endl;
  mos << "learn block size: " << learnBlkSz << endl;
  mos << "neuron block size: " << neuronBlkSz << endl;
  UIntSz= sizeof(unsigned int)*8; // in bits
  logUIntSz= (int) (logf((float) UIntSz)/logf(2.0f)+1e-5f);
  mos << "UIntSz: " << UIntSz << endl;
  mos << "logUIntSz: " << logUIntSz << endl;

  return chosenDevice;
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

  synapseBlkSz = 256;
  learnBlkSz = 256;
  neuronBlkSz = 256;

  NNmodel model;
  prepareStandardModels();
  modelDefinition(model);
  cerr << "model allocated" << endl;

  string path= toString(argv[1]);
  theDev = chooseDevice(cerr, model, path);	
  generate_model_runner(model, path);
  
  return EXIT_SUCCESS;
}

