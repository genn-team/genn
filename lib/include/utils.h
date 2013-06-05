/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */
/** Standard neuron model definitions are at the end of the code. */

#ifndef _UTILS_H_
#define _UTILS_H_

#include "toString.h"

// Shared Utilities (QA Testing)

// std::system includes
#include <memory>
#include <iostream>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>

//GeNN-related
#include <vector>
#include "modelSpec.h"

int *pArgc = NULL;
char **pArgv = NULL;

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

// Function to output an error code and exit

void error(const char *msg) 
{
  cerr << msg << endl;
  exit(1);
}

void writeHeader(ostream &os) 
{
  string s;
  ifstream is("header.src");
  getline(is, s);
  while (is.good()) {
    os << s << endl;
    getline(is, s);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Function to check devices and capabilities
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv)
int checkDevices(ostream &mos) {
    //D pArgc = &argc;
    //D pArgv = argv;

    //D printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
		mos << "There is no device supporting CUDA.\n";
        //D printf("There are no available device(s) that support CUDA\n");
		return 0;
    }
    else
    {
		mos << "Detected " << deviceCount << "CUDA Capable device(s)\n";
        //D printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

	deviceProp= new cudaDeviceProp[deviceCount];
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        //D cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&(deviceProp[dev]), dev);
		mos << "\nDevice" << dev << ": \""  << deviceProp[dev].name << "\"\n";
        //D printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

		mos << "  CUDA Driver Version: " << driverVersion/1000 << ".";
		mos << driverVersion%100 << endl;

		mos << "  CUDA Runtime Version: " << runtimeVersion/1000 << ".";
        mos << runtimeVersion%100 << endl;

		mos << "  CUDA Capability Major/Minor version number: ";
		mos << deviceProp[dev].major << "." << deviceProp[dev].minor << endl;


        //D printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        //D printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        //

		//
		mos << "  Total amount of global memory: ";
		mos << (unsigned long long) deviceProp[dev].totalGlobalMem/1048576.0f << " Mbytes\n";

		//
        printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp[dev].multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor),
               _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor) * deviceProp[dev].multiProcessorCount);
		//
		mos << "  Multiprocessors x Cores/MP = Cores: ";
		mos << deviceProp[dev].multiProcessorCount << "(MP) x ";
		mos << _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor);
		mos << " (Cores/MP) = ";
		mos << _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor) * 

		printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp[dev].clockRate * 1e-3f, deviceProp[dev].clockRate * 1e-6f);

		// these may not work -- they don't exist in v5.0 of the code
		mos << "  Total amount of constant memory: " << deviceProp[dev].totalConstMem; 
	    mos << " bytes\n";
		mos << "  Total amount of shared memory per block: ";
		mos << deviceProp[dev].sharedMemPerBlock << " bytes\n";
		mos << "  Total number of registers available per block: ";
		mos << deviceProp[dev].regsPerBlock << endl;
		mos << "  Warp size: " << deviceProp[dev].warpSize << endl;
		mos << "  Maximum number of threads per block: ";
		mos << deviceProp[dev].maxThreadsPerBlock << endl;
		mos << "  Maximum sizes of each dimension of a block: ";
		mos << deviceProp[dev].maxThreadsDim[0] << " x ";
		mos << deviceProp[dev].maxThreadsDim[1] << " x ";
		mos << deviceProp[dev].maxThreadsDim[2] << endl;
		mos << "  Maximum sizes of each dimension of a grid: ";
		mos << deviceProp[dev].maxGridSize[0] << " x ";
		mos << deviceProp[dev].maxGridSize[1] << " x ";
		mos << deviceProp[dev].maxGridSize[2] << endl;
		mos << "  Maximum memory pitch: " << deviceProp[dev].memPitch;
		mos << " bytes\n";
		mos << "  Texture alignment: " << deviceProp[dev].textureAlignment;
		mos << " bytes\n";
		mos << "  Clock rate: " << deviceProp[dev].clockRate * 1e-6f;
		mos << " GHz\n";


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp[dev].memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp[dev].memoryBusWidth);

		mos << "Memory Clock rate: " << deviceProp[dev].memoryClockRate * 1e-3f << "Mhz \n" ;
        mos << "Memory Bus Width: "<< deviceProp[dev].memoryBusWidth <<"\n";

        if (deviceProp[dev].l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp[dev].l2CacheSize);
			mos << "L2 Cache Size:" << deviceProp[dev].l2CacheSize << " bytes\n";
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        mos << "  Memory Clock rate:" << memoryClock * 1e-3f <<" Mhz\n";
        
		int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        mos << "  Memory Bus Width:" << memBusWidth << " -bit\n";
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
			mos << "  L2 Cache Size: " << L2CacheSize << " bytes\n";
        }
#endif
		//HERE
        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
               deviceProp[dev].maxTexture1D   , deviceProp[dev].maxTexture2D[0], deviceProp[dev].maxTexture2D[1],
			   deviceProp[dev].maxTexture3D[0], deviceProp[dev].maxTexture3D[1], deviceProp[dev].maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
               deviceProp[dev].maxTexture1DLayered[0], deviceProp[dev].maxTexture1DLayered[1],
               deviceProp[dev].maxTexture2DLayered[0], deviceProp[dev].maxTexture2DLayered[1], deviceProp[dev].maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp[dev].totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp[dev].sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp[dev].regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp[dev].warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp[dev].maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp[dev].maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp[dev].maxThreadsDim[0],
               deviceProp[dev].maxThreadsDim[1],
               deviceProp[dev].maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp[dev].maxGridSize[0],
               deviceProp[dev].maxGridSize[1],
               deviceProp[dev].maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp[dev].memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp[dev].textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp[dev].deviceOverlap ? "Yes" : "No"), deviceProp[dev].asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp[dev].kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp[dev].integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp[dev].canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp[dev].surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp[dev].ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp[dev].tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp[dev].unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp[dev].pciBusID, deviceProp[dev].pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp[dev].computeMode]);
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#ifdef _WIN32
    sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
    sprintf(cTemp, ", Device%d = ", dev);
#endif
	    
	//    deviceProp= new cudaDeviceProp[deviceCount]; 
        //cudaDeviceProp deviceProp;

	//		cudaGetDeviceProperties(&deviceProp[dev], dev);
			sProfileString += cTemp;
			sProfileString += deviceProp[dev].name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

	return deviceCount;
    // finish
    //exit(EXIT_SUCCESS);
}

//TO HERE

////////////////////////////////////////////////////////////////////////////////
// Tools for templae matching/cod expansion 
////////////////////////////////////////////////////////////////////////////////

void substitute(string &s, const string trg, const string rep)
{
  size_t found= s.find(trg);
  while (found != string::npos) {
    s.replace(found,trg.length(),rep);
    found= s.find(trg);
  }
}

unsigned int theSize(string type) 
{
  unsigned int sz= sizeof(int);
  if (type == tS("float")) sz= sizeof(float);
  if (type == tS("usigned int")) sz= sizeof(unsigned int);
  if (type == tS("int")) sz= sizeof(int);
  return sz;
}



////////////////////////////////////////////////////////////////////////////////
// Standard neuron model definitions 
////////////////////////////////////////////////////////////////////////////////

vector<neuronModel> nModels;

void prepareStandardModels()
{
  neuronModel n;

  //Rulkov neurons
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("preV"));
  n.varTypes.push_back(tS("float"));
  n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("alpha"));
  n.pNames.push_back(tS("y"));
  n.pNames.push_back(tS("beta"));
  n.dpNames.push_back(tS("ip0"));
  n.dpNames.push_back(tS("ip1"));
  n.dpNames.push_back(tS("ip2"));
  n.simCode= tS("    if ($(V) <= 0.0) {\n\
      $(preV)= $(V);\n\
      $(V)= $(ip0)/(($(Vspike)) - $(V) - ($(beta))*$(Isyn)) +($(ip1));\n\
    }\n\
    else {\n\
      if (($(V) < $(ip2)) && ($(preV) <= 0.0)) {\n\
        $(preV)= $(V);\n\
        $(V)= $(ip2);\n\
      }\n\
      else {\n\
        $(preV)= $(V);\n\
        $(V)= -($(Vspike));\n\
      }\n\
    }\n");
  nModels.push_back(n);

  // Poisson neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("seed"));
  n.varTypes.push_back(tS("unsigned int"));
  n.varNames.push_back(tS("spikeTime"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.pNames.push_back(tS("therate"));
  n.pNames.push_back(tS("trefract"));
  n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("Vrest"));
  n.dpNames.clear();
  n.simCode= tS("    unsigned int theRnd;\n\
    if ($(V) > $(Vrest)) {\n\
      $(V)= $(Vrest);\n\
    }\n\
    else {\n\
      if (t - $(spikeTime) > ($(trefract))) {\n\
        RAND($(seed),theRnd);\n\
        if (theRnd < lrate) {\n\
          $(V)= $(Vspike);\n\
          $(spikeTime)= t;\n\
        }\n\
      }\n\
    }\n");
  nModels.push_back(n);

// Traub and Miles HH neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("m"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("h"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("n"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.pNames.push_back(tS("gNa"));
  n.pNames.push_back(tS("ENa"));
  n.pNames.push_back(tS("gK"));
  n.pNames.push_back(tS("EK"));
  n.pNames.push_back(tS("gl"));
  n.pNames.push_back(tS("El"));
  n.pNames.push_back(tS("C"));
  n.dpNames.clear();
  n.simCode= tS("    float Imem;\n\
    unsigned int mt;\n\
    float mdt= DT/25.0f;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-Isyn);\n\
      float _a= 0.32f*(-52.0f-$(V)) / (exp((-52.0f-$(V))/4.0f)-1.0f);\n\
      float _b= 0.28f*($(V)+25.0f)/(exp(($(V)+25.0f)/5.0f)-1.0f);\n\
      $(m)+= (_a*(1.0f-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*expf((-48.0f-$(V))/18.0f);\n\
      _b= 4.0f / (expf((-25.0f-$(V))/5.0f)+1.0f);\n\
      $(h)+= (_a*(1.0f-$(h))-_b*$(h))*mdt;\n\
      _a= .032f*(-50.0f-$(V)) / (expf((-50.0f-$(V))/5.0f)-1.0f); \n\
      _b= 0.5f*expf((-55.0f-$(V))/40.0f);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n");
  nModels.push_back(n);
  
 //Izhikevich neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));  
  n.varNames.push_back(tS("U"));
  n.varTypes.push_back(tS("float"));
  n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("a")); // time scale of U
  n.pNames.push_back(tS("b")); // sensitivity of U
  n.pNames.push_back(tS("c")); // after-spike reset value of V
  n.pNames.push_back(tS("d")); // after-spike reset value of U
  n.dpNames.clear(); 
  n.simCode= tS(" $(V)+=0.5f*(0.04*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn)); //at two times for numerical stability \n\
  $(V)+=0.5f*(0.04*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn));\n\
  $(U)+=$(a)*($(b)*$(V)-$(U));\n\\n\
  if ($(V) > 30){\n\
		$(V)=$(c);\n\
		$(U)+=$(d);  \n\	
  	} \n\
}\n");
  nModels.push_back(n);}
// bit tool macros
#include "simpleBit.h"

#endif // _UTILS_H_
