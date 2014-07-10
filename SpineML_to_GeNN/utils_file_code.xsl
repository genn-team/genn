<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:SMLLOWNL="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer" xmlns:SMLNL="http://www.shef.ac.uk/SpineMLNetworkLayer" xmlns:SMLCL="http://www.shef.ac.uk/SpineMLComponentLayer" xmlns:fn="http://www.w3.org/2005/xpath-functions">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes"/>

<!-- TEMPLATE FOR INSERTING THE START OF THE GENN UTILS FILE -->
<xsl:template name="insert_utils_file_start_code">
/*
 * IN PART COPYRIGHTED AS BELOW BECAUSE IT'S ADAPTED FROM NVIDIA CUDA SOFTWARE DEVELOPMENT TOOLKIT:
 *
 *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*--------------------------------------------------------------------------
   Author/Modifier: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef _UTILS_H_
#define _UTILS_H_ //!&lt; macro for avoiding multiple inclusion during compilation

//--------------------------------------------------------------------------
/*! \file utils.h

\brief This file contains standard utility functions provide within the NVIDIA CUDA software development toolkit (SDK). The remainder of the file contains a function that defines the standard neuron models.
*/
//--------------------------------------------------------------------------

#include "toString.h"

// Shared Utilities (QA Testing)

// std::system includes
#include &lt;memory&gt;
#include &lt;iostream&gt;

// CUDA-C includes
#include &lt;cuda.h&gt;
#include &lt;cuda_runtime.h&gt;

#include &lt;helper_cuda.h&gt;

//GeNN-related
#include &lt;vector&gt;
#include "modelSpec.h"

int *pArgc = NULL;
char **pArgv = NULL;

//--------------------------------------------------------------------------
//! \brief CUDA SDK: This function wraps the CUDA Driver API into a template function
//--------------------------------------------------------------------------

template &lt;class T&gt;
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file &lt;%s&gt;, line %i.\n",
                error, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

//--------------------------------------------------------------------------
//! \brief CUDA SDK: Function to output an error code and exit
//--------------------------------------------------------------------------

void error(const char *msg) 
{
  cerr &lt;&lt; msg &lt;&lt; endl;
  exit(1);
}


//--------------------------------------------------------------------------
/* \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &amp;os) 
{
  string s;
  ifstream is("header.src");
  getline(is, s);
  while (is.good()) {
    os &lt;&lt; s &lt;&lt; endl;
    getline(is, s);
  }
}

//--------------------------------------------------------------------------
//! \bief Function to check devices and capabilities (modified from CUDA SDK)
//--------------------------------------------------------------------------

int checkDevices(ostream &amp;mos) {
    //D pArgc = &amp;argc;
    //D pArgv = argv;

    //D printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&amp;deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-&gt; %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
		mos &lt;&lt; "There is no device supporting CUDA.\n";
        //D printf("There are no available device(s) that support CUDA\n");
		return 0;
    }
    else
    {
		mos &lt;&lt; "Detected " &lt;&lt; deviceCount &lt;&lt; "CUDA Capable device(s)\n";
        //D printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

	deviceProp= new cudaDeviceProp[deviceCount];
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev &lt; deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        //D cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&amp;(deviceProp[dev]), dev);
		mos &lt;&lt; "\nDevice" &lt;&lt; dev &lt;&lt; ": \""  &lt;&lt; deviceProp[dev].name &lt;&lt; "\"\n";
        //D printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&amp;driverVersion);
        cudaRuntimeGetVersion(&amp;runtimeVersion);

		mos &lt;&lt; "  CUDA Driver Version: " &lt;&lt; driverVersion/1000 &lt;&lt; ".";
		mos &lt;&lt; driverVersion%100 &lt;&lt; endl;

		mos &lt;&lt; "  CUDA Runtime Version: " &lt;&lt; runtimeVersion/1000 &lt;&lt; ".";
        mos &lt;&lt; runtimeVersion%100 &lt;&lt; endl;

		mos &lt;&lt; "  CUDA Capability Major/Minor version number: ";
		mos &lt;&lt; deviceProp[dev].major &lt;&lt; "." &lt;&lt; deviceProp[dev].minor &lt;&lt; endl;


        //D printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        //D printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        //

		//
		mos &lt;&lt; "  Total amount of global memory: ";
		mos &lt;&lt; (unsigned long long) deviceProp[dev].totalGlobalMem/1048576.0f &lt;&lt; " Mbytes\n";

		//
        printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp[dev].multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor),
               _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor) * deviceProp[dev].multiProcessorCount);
		//
		mos &lt;&lt; "  Multiprocessors x Cores/MP = Cores: ";
		mos &lt;&lt; deviceProp[dev].multiProcessorCount &lt;&lt; "(MP) x ";
		mos &lt;&lt; _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor);
		mos &lt;&lt; " (Cores/MP) = ";
		mos &lt;&lt; _ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor) * 

		printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp[dev].clockRate * 1e-3f, deviceProp[dev].clockRate * 1e-6f);

		// these may not work -- they don't exist in v5.0 of the code
		mos &lt;&lt; "  Total amount of constant memory: " &lt;&lt; deviceProp[dev].totalConstMem; 
	    mos &lt;&lt; " bytes\n";
		mos &lt;&lt; "  Total amount of shared memory per block: ";
		mos &lt;&lt; deviceProp[dev].sharedMemPerBlock &lt;&lt; " bytes\n";
		mos &lt;&lt; "  Total number of registers available per block: ";
		mos &lt;&lt; deviceProp[dev].regsPerBlock &lt;&lt; endl;
		mos &lt;&lt; "  Warp size: " &lt;&lt; deviceProp[dev].warpSize &lt;&lt; endl;
		mos &lt;&lt; "  Maximum number of threads per block: ";
		mos &lt;&lt; deviceProp[dev].maxThreadsPerBlock &lt;&lt; endl;
		mos &lt;&lt; "  Maximum sizes of each dimension of a block: ";
		mos &lt;&lt; deviceProp[dev].maxThreadsDim[0] &lt;&lt; " x ";
		mos &lt;&lt; deviceProp[dev].maxThreadsDim[1] &lt;&lt; " x ";
		mos &lt;&lt; deviceProp[dev].maxThreadsDim[2] &lt;&lt; endl;
		mos &lt;&lt; "  Maximum sizes of each dimension of a grid: ";
		mos &lt;&lt; deviceProp[dev].maxGridSize[0] &lt;&lt; " x ";
		mos &lt;&lt; deviceProp[dev].maxGridSize[1] &lt;&lt; " x ";
		mos &lt;&lt; deviceProp[dev].maxGridSize[2] &lt;&lt; endl;
		mos &lt;&lt; "  Maximum memory pitch: " &lt;&lt; deviceProp[dev].memPitch;
		mos &lt;&lt; " bytes\n";
		mos &lt;&lt; "  Texture alignment: " &lt;&lt; deviceProp[dev].textureAlignment;
		mos &lt;&lt; " bytes\n";
		mos &lt;&lt; "  Clock rate: " &lt;&lt; deviceProp[dev].clockRate * 1e-6f;
		mos &lt;&lt; " GHz\n";


#if CUDART_VERSION &gt;= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp[dev].memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp[dev].memoryBusWidth);

		mos &lt;&lt; "Memory Clock rate: " &lt;&lt; deviceProp[dev].memoryClockRate * 1e-3f &lt;&lt; "Mhz \n" ;
        mos &lt;&lt; "Memory Bus Width: "&lt;&lt; deviceProp[dev].memoryBusWidth &lt;&lt;"\n";

        if (deviceProp[dev].l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp[dev].l2CacheSize);
			mos &lt;&lt; "L2 Cache Size:" &lt;&lt; deviceProp[dev].l2CacheSize &lt;&lt; " bytes\n";
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute&lt;int&gt;(&amp;memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        mos &lt;&lt; "  Memory Clock rate:" &lt;&lt; memoryClock * 1e-3f &lt;&lt;" Mhz\n";
        
		int memBusWidth;
        getCudaAttribute&lt;int&gt;(&amp;memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        mos &lt;&lt; "  Memory Bus Width:" &lt;&lt; memBusWidth &lt;&lt; " -bit\n";
        int L2CacheSize;
        getCudaAttribute&lt;int&gt;(&amp;L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
			mos &lt;&lt; "  L2 Cache Size: " &lt;&lt; L2CacheSize &lt;&lt; " bytes\n";
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
        printf("     &lt; %s &gt;\n", sComputeMode[deviceProp[dev].computeMode]);
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
    for (dev = 0; dev &lt; deviceCount; ++dev)
    {
#ifdef _WIN32
    sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
    sprintf(cTemp, ", Device%d = ", dev);
#endif
	    
	//    deviceProp= new cudaDeviceProp[deviceCount]; 
        //cudaDeviceProp deviceProp;

	//		cudaGetDeviceProperties(&amp;deviceProp[dev], dev);
			sProfileString += cTemp;
			sProfileString += deviceProp[dev].name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

	return deviceCount;
    // finish
    //exit(EXIT_SUCCESS);
}


//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &amp;s, const string trg, const string rep)
{
  size_t found= s.find(trg);
  while (found != string::npos) {
    s.replace(found,trg.length(),rep);
    found= s.find(trg);
  }
}

//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(string type) 
{
  unsigned int sz= sizeof(int);
  if (type == tS("float")) sz= sizeof(float);
  if (type == tS("unsigned int")) sz= sizeof(unsigned int);
  if (type == tS("int")) sz= sizeof(int);
  return sz;
}


vector&lt;neuronModel&gt; nModels; //!&lt; Global c++ vector containing all neuron model descriptions

//--------------------------------------------------------------------------
/*! \brief FUnction that defines standard neuron models

The neuron models are defined and added to the C++ vector nModels that is holding all neuron model descriptions. User defined neuron models can be appended to this vector later in (a) separate function(s).
*/
//--------------------------------------------------------------------------

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
  n.simCode= tS("    if ($(V) &lt;= 0.0) {\n\
      $(preV)= $(V);\n\
      $(V)= $(ip0)/(($(Vspike)) - $(V) - ($(beta))*$(Isyn)) +($(ip1));\n\
    }\n\
    else {\n\
      if (($(V) &lt; $(ip2)) &amp;&amp; ($(preV) &lt;= 0.0)) {\n\
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
    if ($(V) &gt; $(Vrest)) {\n\
      $(V)= $(Vrest);\n\
    }\n\
    else {\n\
      if (t - $(spikeTime) &gt; ($(trefract))) {\n\
        RAND($(seed),theRnd);\n\
        if (theRnd &lt; lrate) {\n\
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
  n.simCode= tS("   float Imem;\n\
    unsigned int mt;\n\
    float mdt= DT/25.0f;\n\
    for (mt=0; mt &lt; 25; mt++) {\n\
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
  n.pNames.clear();
  //n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("a")); // time scale of U
  n.pNames.push_back(tS("b")); // sensitivity of U
  n.pNames.push_back(tS("c")); // after-spike reset value of V
  n.pNames.push_back(tS("d")); // after-spike reset value of U
  n.dpNames.clear(); 
  n.simCode= tS(" $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn)); //at two times for numerical stability \n\
  $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn));\n\
  $(U)+=$(a)*($(b)*$(V)-$(U));\n\
  if ($(V) &gt; 30){\n\
		$(V)=$(c);\n\
		$(U)+=$(d);\n\
  }\n");
  nModels.push_back(n);
  
</xsl:template>

<!-- TEMPLATE FOR INSERTING THE END OF THE GENN UTILS FILE -->
<xsl:template name="insert_utils_file_end_code">

cout &lt;&lt; "AT END OF UTILS.H\n\n\n";
}

// bit tool macros
#include "simpleBit.h"

#endif // _UTILS_H_
</xsl:template>

</xsl:stylesheet>
