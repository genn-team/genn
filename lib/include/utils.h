/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

/* parts of the source code obtained and modified from NVIDIA developer 
 * kit under the condtions below:
 *
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

#ifndef _UTILS_H_
#define _UTILS_H_

#include "toString.h"

// utilities and system includes
#include <shrUtils.h>

// CUDA-C includes
#include <cuda_runtime_api.h>

#include <vector>
#include "modelSpec.h"

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
int
checkDevices(ostream &mos) 
{
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    mos << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n";
    return -1;
  }
  
  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    mos << "There is no device supporting CUDA\n";
    return 0;
  }

  deviceProp= new cudaDeviceProp[deviceCount];
  int dev;
  int driverVersion = 0, runtimeVersion = 0;     
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaGetDeviceProperties(&(deviceProp[dev]), dev);
    
    if (dev == 0) {
      // This function call returns 9999 for both major & minor fields, 
      // if no CUDA capable devices are present
      if (deviceProp[dev].major == 9999 && deviceProp[dev].minor == 9999) {
	mos << "There is no device supporting CUDA.\n";
	return 0;
      }
      else if (deviceCount == 1)
	mos << "There is 1 device supporting CUDA\n";
      else
	mos << "There are " << deviceCount << " devices supporting CUDA\n";
    }
    mos << "\nDevice " << dev << ": \"" << deviceProp[dev].name << "\"\n";
    
#if CUDART_VERSION >= 2020
    // Console log
    cudaDriverGetVersion(&driverVersion);
    mos << "  CUDA Driver Version: " << driverVersion/1000 << ".";
    mos << driverVersion%100 << endl;
    cudaRuntimeGetVersion(&runtimeVersion);
    mos << "  CUDA Runtime Version: " << runtimeVersion/1000 << ".";
    mos << runtimeVersion%100 << endl;
#endif
    mos << "  CUDA Capability Major/Minor version number: ";
    mos << deviceProp[dev].major << "." << deviceProp[dev].minor << endl;

    mos << "  Total amount of global memory: ";
    mos << (unsigned long long) deviceProp[dev].totalGlobalMem << " bytes\n";
		
#if CUDART_VERSION >= 2000
    mos << "  Multiprocessors x Cores/MP = Cores: ";
    mos << deviceProp[dev].multiProcessorCount << "(MP) x ";
    mos << ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor);
    mos << " (Cores/MP) = ";
    mos << ConvertSMVer2Cores(deviceProp[dev].major, deviceProp[dev].minor) * deviceProp[dev].multiProcessorCount << " (Cores)\n"; 
#endif
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
#if CUDART_VERSION >= 2000
    mos << "  Concurrent copy and execution: ";
    if( deviceProp[dev].deviceOverlap) 
      mos <<  "Yes\n";
    else mos << "No\n";
#endif
#if CUDART_VERSION >= 2020
    mos << "  Run time limit on kernels: ";
    if ( deviceProp[dev].kernelExecTimeoutEnabled)
      mos << "Yes\n";
    else mos << "No\n";
    mos << "  Integrated: ";
    if (deviceProp[dev].integrated) 
      mos << "Yes\n";
    else mos << "No\n";
    mos << "  Support host page-locked memory mapping: ";
    if (deviceProp[dev].canMapHostMemory)
      mos << "Yes\n";
    else mos << "No\n";
    mos << "  Compute mode: ";
    if (deviceProp[dev].computeMode == cudaComputeModeDefault)
      mos << "Default (multiple host threads can use this device simultaneously)\n";
    else if (deviceProp[dev].computeMode == cudaComputeModeExclusive) 
      mos << "Exclusive (only one host thread at a time can use this device)\n";
    else if (deviceProp[dev].computeMode == cudaComputeModeProhibited)
      mos << "Prohibited (no host thread can use this device)\n";
    else mos << "Unknown\n";
#endif
#if CUDART_VERSION >= 3000
    mos << "  Concurrent kernel execution: ";
    if ( deviceProp[dev].concurrentKernels)
      mos << "Yes\n";
    else mos << "No\n";
#endif
#if CUDART_VERSION >= 3010
    mos << "  Device has ECC support enabled: ";
    if (deviceProp[dev].ECCEnabled)
      mos << "Yes\n" ;
    else mos << "No\n";
#endif
#if CUDART_VERSION >= 3020
    mos << "  Device is using TCC driver mode: ";
    if (deviceProp[dev].tccDriver) 
      mos << "Yes\n";
    else mos << "No\n" << endl;
#endif
  }
  
  // csv masterlog info
  // *****************************
  // exe and CUDA driver name 
  shrLog("\n");    
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";        
  char cTemp[10];
    
  // driver version
  sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
  sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, driverVersion%100);    
#else
  sprintf(cTemp, "%d.%d", driverVersion/1000, driverVersion%100);	
#endif
  sProfileString +=  cTemp;
	    
  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
  sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
#else
  sprintf(cTemp, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
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
  
  // First 2 device names, if any
  for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev)  {
    sProfileString += ", Device = ";
    sProfileString += deviceProp[dev].name;
  }
  sProfileString += "\n";
  mos << sProfileString.c_str();
  
  return deviceCount;
  }


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
}



// bit tool macros
#include "simpleBit.h"

#endif // _UTILS_H_
