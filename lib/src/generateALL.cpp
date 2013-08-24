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
			   string path //!< Path where the generated code will be deposited
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

int chooseDevice(ostream &mos, //!< output stream for messages
		 NNmodel &model, //!< the nn model we are generating code for
		 string path //!< path the generated code will be deposited
		 )
{
  // Get the specifications of all available cuda devices, then work out which one we will use.
  int deviceCount;
  int chosenDevice = 0;
  CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
  deviceProp = new cudaDeviceProp[deviceCount];
  int warpOccupancy;
  int bestWarpOccupancy = 0;
  int globalMem;
  int mostGlobalMem = 0;
  for (int dev = 0; dev < deviceCount; dev++) {
    CHECK_CUDA_ERRORS(cudaSetDevice(dev));
    CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&(deviceProp[dev]), dev));

    if (optimiseBlockSize) { // IF OPTIMISATION IS ON
      // Choose the device which supports the highest warp occupancy.
      generate_model_runner(model, path);
      stringstream command;
      command << "nvcc -x cu -cubin -Xptxas=-v -arch=sm_" << deviceProp[dev].major
	      << deviceProp[dev].minor << " -DDT -D\"CHECK_CUDA_ERRORS(call){call;}\" "; 

      // Run NVCC and pipe stderr to this process to get kernel resource requirements.
#ifdef _WIN32
      command << path << "\\" << model.name << "_CODE\\runner.cc 2>&1";
      FILE *nvccPipe = _popen(command.str().c_str(), "r");
#else // UNIX
      command << path << "/" << model.name << "_CODE/runner.cc 2>&1";
      FILE *nvccPipe = popen(command.str().c_str(), "r");
#endif
      if (!nvccPipe) {
	mos << "ERROR: faied to open nvcc pipe" << endl;
	exit(EXIT_FAILURE);
      }

      // Store each kernel's resource usage in separate stringstreams.
      char pipeBuffer[128];
      stringstream ptxInfo[3];
      int ptxInfoPtr; // 0 = calcSynapses, 1 = learnSynapses, 2 = calcNeurons
      while (fgets(pipeBuffer, 128, nvccPipe) != NULL) {
	if (strstr(pipeBuffer, "calcSynapses") != NULL) {
	  ptxInfoPtr = 0;
	}
	else if (strstr(pipeBuffer, "learnSynapses") != NULL) {
	  ptxInfoPtr = 1;
	}
	else if (strstr(pipeBuffer, "calcNeurons") != NULL) {
	  ptxInfoPtr = 2;
	}
	// If ptxas info is found, store it in its own stringstream.
	if (strncmp(pipeBuffer, "ptxas info    : Used", 20) == 0) {
	  ptxInfo[ptxInfoPtr] << pipeBuffer;
	}
      }

      // Close the NVCC pipe.
#ifdef _WIN32
      _pclose(nvccPipe);
#else // UNIX
      pclose(nvccPipe);
#endif

      // Parsing the PTX info string retrieved from the NVCC pipe.
      int registers, sharedMem;
      string junk;
      for (int i = 0; i < 3; i++) {
	ptxInfo[i] >> junk >> junk >> junk >> junk;  // ptxas info    : Used 
	ptxInfo[i] >> registers >> junk;             // [registers] registers, 
	ptxInfo[i] >> sharedMem;                     // [sharedMem] bytes smem, 
	mos << "kernel " << i << " " << ptxInfo[i].str();
	mos << "registers needed: " << registers << endl;
	mos << "shared mem needed: " << sharedMem << endl;






      }

      mos << "device " << dev << " supports a max warp occupancy of " << warpOccupancy << endl;
      if (warpOccupancy >= bestWarpOccupancy) {
	bestWarpOccupancy = warpOccupancy;
	chosenDevice = dev;
      }
    }

    else { // IF OPTIMISATION IS OFF
      // Simply choose the device with the most global memory.
      globalMem = deviceProp[dev].totalGlobalMem;
      mos << "device " << dev << " has " << globalMem << " bytes of global memory" << endl;
      if (globalMem >= mostGlobalMem) {
	mostGlobalMem = globalMem;
	chosenDevice = dev;
      }
    }
  }

  ofstream sm_os("sm_Version.mk");
  sm_os << "NVCCFLAGS += -arch sm_" << deviceProp[chosenDevice].major << deviceProp[chosenDevice].minor << endl;
  sm_os.close();

  mos << "We are using CUDA device " << chosenDevice << endl;
  mos << "global memory: " << deviceProp[chosenDevice].totalGlobalMem << " bytes" << endl;
  mos << "neuron block size: " << neuronBlkSz << endl;
  mos << "synapse block size: " << synapseBlkSz << endl;
  mos << "learn block size: " << learnBlkSz << endl;
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

int main(int argc, //!< number of arguments; expected to be 2
	 char *argv[]  //!< Arguments; expected to contain the name of the  
	               //!< target directory for code generation.
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

  NNmodel model;
  prepareStandardModels();
  modelDefinition(model);
  cerr << "model allocated" << endl;

  string path= toString(argv[1]);
  theDev = chooseDevice(cerr, model, path);	
  generate_model_runner(model, path);
  
  return EXIT_SUCCESS;
}

