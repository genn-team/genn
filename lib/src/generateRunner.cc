/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "utils.h"
#include <string>
////////////////////////////////////////////////////////////////////////////////
//!
//!
//!
//!
////////////////////////////////////////////////////////////////////////////////
void prepare(ostream &mos)
{
	cout << "prepare starts" << endl;
  devN= checkDevices(mos);
  cout << "checkdevices ends" << endl;
  if (devN > 0) {
    neuronBlkSz= 0;
    theDev= 0;
    // choose the device with largest block size for the time being
    // among equals we use the last device
    for (int i= 0; i < devN; i++) {
      if (deviceProp[i].maxThreadsPerBlock >= neuronBlkSz) {
	neuronBlkSz= deviceProp[i].maxThreadsPerBlock;
	logNeuronBlkSz= (int) (logf((float)neuronBlkSz)/logf(2.0f)+1e-5f);
	logNeuronBlkSz-= 1; // for HH neurons (need a better solution for this!)
	neuronBlkSz= 1 << logNeuronBlkSz;
	synapseBlkSz= deviceProp[i].maxThreadsPerBlock;
	logSynapseBlkSz= (int) (logf((float)synapseBlkSz)/logf(2.0f)+1e-5f);
	logSynapseBlkSz-= 1; // for learning syns (need a better solution for this!) 
	synapseBlkSz= 1 << logSynapseBlkSz;
	learnBlkSz= deviceProp[i].maxThreadsPerBlock;
	logLearnBlkSz= (int) (logf((float)learnBlkSz)/logf(2.0f)+1e-5f);
	learnBlkSz= 1 << logLearnBlkSz;
	theDev= i;
      }
    }
	mos << "I am in generate_run" << endl;
    mos << "We are using CUDA device " << theDev << endl;
    mos << "max logNeuronBlkSz: " << logNeuronBlkSz << endl;
    mos << "max neuronBlkSz: " << neuronBlkSz << endl;
    mos << "max logSynapseBlkSz: " << logSynapseBlkSz << endl;
    mos << "max synapseBlkSz: " << synapseBlkSz << endl;
    mos << "max logLearnBlkSz: " << logLearnBlkSz << endl;
    mos << "max learnBlkSz: " << learnBlkSz << endl;
    UIntSz= sizeof(unsigned int)*8;   // in bit!
    logUIntSz= (int) (logf((float) UIntSz)/logf(2.0f)+1e-5f);
    mos << "logUIntSz: " << logUIntSz << endl;
    mos << "UIntSz: " << UIntSz << endl;
    if (optimiseBlockSize) {
      mos << "optimising block size ..." << endl;
    }
  }
}

void genRunner(NNmodel &model, ostream &os, ostream &mos)
{
  unsigned int nt;
  unsigned int mem= 0;

  writeHeader(os);
  os << endl;
  //os << "#include <cutil_inline.h>" << endl;
  os << "#include <helper_cuda.h>" << endl;

  // ------------------------------------------------------------------------
  // gloabl host variables (matching some of the device ones)
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    os << "unsigned int glbscnt" << model.neuronName[i] << ";" << endl;
    os << "unsigned int *glbSpk" << model.neuronName[i] << ";" << endl;
    for (int j= 0; j < model.inSyn[i].size(); j++) {
      os << "float *inSyn" << model.neuronName[i] << j << ";" << endl;
    } 
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << nModels[nt].varTypes[k] << " *";
      os << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
    }
    if (model.neuronNeedSt[i]) {
      os << "float *sT" << model.neuronName[i] << ";" << endl;
    }
  }
  for (int i= 0; i < model.synapseGrpN; i++) {
    if (model.synapseGType[i] == INDIVIDUALG) {
      os << "float *gp" << model.synapseName[i] << ";" << endl;
      if (model.synapseType[i] == LEARN1SYNAPSE) {
	os << "float *grawp" << model.synapseName[i] << ";" << endl;
      }
    }
    if (model.synapseGType[i] == INDIVIDUALID) {
      os << "unsigned int *gp" << model.synapseName[i] << ";" << endl;
    }
  }
  os << endl;
  
  // ------------------------------------------------------------------------
  // Code for setting the CUDA device and
  // setting up the host's global variables.
  // Also estimating memory usage on device ...
  os << "void allocateMem()" << endl;
  os << "{" << endl;
  //if cuda v <5 os << "  cutilSafeCall(cudaSetDevice(theDev));" << endl;
  os << "  checkCudaErrors(cudaSetDevice(theDev));" << endl;
  //if cuda v <5 os << "  cutilCheckMsg(\"CUDA setDevice failed.\\n\");" << endl;
  os << "  getLastCudaError(\"CUDA setDevice failed.\\n\");" << endl;
  cerr << "model.neuronGroupN " << model.neuronGrpN << endl;
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    os << "  glbSpk" << model.neuronName[i] << "= new unsigned int[";
    os << model.neuronN[i] << "];" << endl;
    mem+= model.neuronN[i]*sizeof(unsigned int);
    for (int j= 0; j < model.inSyn[i].size(); j++) {
      os << "  inSyn" << model.neuronName[i] << j << "= new float[";
      os << model.neuronN[i] << "];" << endl;
      mem+= model.neuronN[i]*sizeof(float);
    } 
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "  " << nModels[nt].varNames[k] << model.neuronName[i] << "= new ";
      os << nModels[nt].varTypes[k] << "[";
      os << model.neuronN[i] << "];" << endl;
      unsigned int sz= theSize(nModels[nt].varTypes[k]);
      mem+= model.neuronN[i]*sz;
    }
    if (model.neuronNeedSt[i]) {
      os << "  sT" << model.neuronName[i] << "= new float[";
      os << model.neuronN[i] << "];" << endl;
    }   
  }
  for (int i= 0; i < model.synapseGrpN; i++) {
    if (model.synapseGType[i] == INDIVIDUALG) {
      os << "  gp" << model.synapseName[i] << "= new float[";
      unsigned int size= model.neuronN[model.synapseSource[i]]
	*model.neuronN[model.synapseTarget[i]];
      os << size;
      os << "];      // synaptic conductances of group " << model.neuronName[i];
      os << endl;
      mem+= size*sizeof(float);
      if (model.synapseType[i] == LEARN1SYNAPSE) {
	os << "  grawp" << model.synapseName[i] << "= new float[";
	unsigned int size= model.neuronN[model.synapseSource[i]]
	  *model.neuronN[model.synapseTarget[i]];
	os << size;
	os << "];      // raw synaptic conductances of group " << model.neuronName[i];
	os << endl;
	mem+= size*sizeof(float);
      }
    }
    if (model.synapseGType[i] == INDIVIDUALID) {
     os << "  gp" << model.synapseName[i] << "= new unsigned int[";
      unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
      unsigned int size= tmp >> logUIntSz;
      if (tmp > (size << logUIntSz)) size++;
      os << size;
      os << "];     // synaptic connectivity of group " << model.neuronName[i];
      os << endl;
      mem+= size*sizeof(unsigned int);
    }
  }
  os << "}" << endl;
  os << endl;
  
  // ------------------------------------------------------------------------
  // freeing global memory structures
  os << "void freeMem()" << endl;
  os << "{" << endl;
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    for (int j= 0; j < model.inSyn[i].size(); j++) {
      os << "  delete[] inSyn" << model.neuronName[i] << j << ";" << endl;
    } 
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "  delete[] " << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
    }
  }
  for (int i= 0; i < model.synapseGrpN; i++) {
    if ((model.synapseGType[i] == INDIVIDUALG) ||
	(model.synapseGType[i] == INDIVIDUALID)) {
      os << "  delete[] gp" << model.synapseName[i] << ";" << endl;
    }
  }
  os << "}" << endl;
  os << endl;
  
  // ------------------------------------------------------------------------
  // initializing variables
  os << "void initialize()" << endl;
  os << "{" << endl;
  os << "//  srand((unsigned int) time(NULL));" << endl;
  os << "srand(101);" << endl;
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    os << "  glbscnt" << model.neuronName[i] << "= 0;" << endl;
    os << "  for (int i= 0; i < " << model.neuronN[i] << "; i++) {" << endl;
    os << "    glbSpk" << model.neuronName[i] << "[i]= 0;" << endl;
    for (int j= 0; j < model.inSyn[i].size(); j++) {
      os << "    inSyn" << model.neuronName[i] << j << "[i]= 0.0f;" << endl;
    } 
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "    " << nModels[nt].varNames[k] << model.neuronName[i];
      os << "[i]= " << model.neuronIni[i][k] << ";" << endl;
    }
    if (model.neuronType[i] == POISSONNEURON) {
      os << "    seed" << model.neuronName[i] << "[i]= rand();" << endl;
    }
    if (model.neuronNeedSt[i]) {
      os << "    sT" <<  model.neuronName[i] << "[i]= -10000.0;" << endl;
    }
    os << "  }" << endl;
  }
  os << "}" << endl;

  os << endl;
  if (model.lrnGroups > 0) {
    for (int i= 0; i < model.synapseGrpN; i++) {
      if (model.synapseType[i] == LEARN1SYNAPSE) {
	os << "__host__ __device__ float gFunc" << model.synapseName[i] << "(float graw)" << endl;
	os << "{" << endl;
	os << "  return " << SAVEP(model.synapsePara[i][8]/2.0) << "*(tanh(";
	os << SAVEP(model.synapsePara[i][10]) << "*(graw - ";
	os << SAVEP(model.synapsePara[i][9]) << "))+1.0);" << endl;
	os << "}" << endl;
	os << endl;
	os << "__host__ __device__ float invGFunc" << model.synapseName[i] << "(float g)" << endl;
	os << "{" << endl;
	os << "float tmp= g/" << SAVEP(model.synapsePara[i][8]*2.0) << "- 1.0;" << endl;
	os << "return 0.5*log((1.0+tmp)/(1.0-tmp))/" << SAVEP(model.synapsePara[i][10]);
	os << " + " << SAVEP(model.synapsePara[i][9]) << ";" << endl;
	os << "}" << endl;
	os << endl;
      }
    }
    os << "void initGRaw()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
      if (model.synapseType[i] == LEARN1SYNAPSE) {
	os << "  for (int i= 0; i < ";
	os << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	os << "; i++) {" << endl;
	os << "    grawp"  << model.synapseName[i] << "[i]= invGFunc" << model.synapseName[i];
	os << "(gp" << model.synapseName[i] << "[i]);" << endl;
	os << "  }" << endl;
      }
    }
    os << "}" << endl;
    os << endl;
  }

  os << "#include \"runnerGPU.cc\"" << endl;
  os << "#include \"runnerCPU.cc\"" << endl;
  os << endl;

  mos << "Global memory required for core model: " << mem/1e6;
  mos << " MB" << endl;
cerr << deviceProp[theDev].totalGlobalMem << " theDev " << theDev << endl;  
  if (0.5*deviceProp[theDev].totalGlobalMem < mem) {
    mos << "memory required for core model (" << mem/1e6;
    mos << "MB) is more than 50% of global memory on the chosen device";
    mos << "(" << deviceProp[theDev].totalGlobalMem/1e6 << "MB)." << endl;
    mos << "Experience shows that this is UNLIKELY TO WORK ... " << endl;
  }
}

void genRunnerGPU(NNmodel &model, ostream &os, ostream &mos)
{
  unsigned int nt;

  if (devN > 0) {
    writeHeader(os);
    //os << "#include <cutil.h>" << endl;
	os << "#include <cuda_runtime.h>" << endl;//EY
    os << "#include <helper_cuda.h>" << endl;//EY
    os << "#include <helper_timer.h>" << endl;//EY
    os << "#define RAND(Y,X) Y = Y * 1103515245 +12345;";
    os << "X= (unsigned int)(Y >> 16) & 32767" << endl;
    os << endl;


    os << "#include \"neuronKrnl.cc\"" << endl;
    os << "#include \"synapseKrnl.cc\"" << endl;
    os << endl;

    unsigned int size;     

    // ------------------------------------------------------------------------
    // copying conductances to device
    os << "void copyGToDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
      if (model.synapseGType[i] == INDIVIDUALG) {
    /**
	/	from CUDA 5.0 release notes:
	/	The use of a character string to indicate a device symbol, which was possible with
	/	certain API functions, is no longer supported. Instead, the symbol should be used
	/	directly.
	/	Therefore, call of cudaGetSymboAddress inthis file is modified wrt GeNN 0.3.
	/	
	*/
	os << "  cudaGetSymbolAddress(&devPtr, d_gp" << model.synapseName[i] << ");" << endl;
	//if cuda version <5 os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, gp" << model.synapseName[i];
	os << "  checkCudaErrors(cudaMemcpy(devPtr, gp" << model.synapseName[i];
	os << ", ";
	size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*sizeof(float);
	os << size << ", cudaMemcpyHostToDevice));" << endl;
	if (model.synapseType[i] == LEARN1SYNAPSE) {
	  os << "  cudaGetSymbolAddress(&devPtr, d_grawp" << model.synapseName[i] << ");" << endl;
	  //if cuda version <5 os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, grawp" << model.synapseName[i];
	  os << "  checkCudaErrors(cudaMemcpy(devPtr, grawp" << model.synapseName[i];
	  os << ", ";
	  size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*sizeof(float);
	  os << size << ", cudaMemcpyHostToDevice));" << endl;
	} 
      }
      if (model.synapseGType[i] == INDIVIDUALID) {
	os << "  cudaGetSymbolAddress(&devPtr, d_gp" << model.synapseName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, gp" << model.synapseName[i];
	os << "  checkCudaErrors(cudaMemcpy(devPtr, gp" << model.synapseName[i];
	os << ", ";
	unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	size= (tmp >> logUIntSz);
	if (tmp > (size << logUIntSz)) size++;
	size= size*sizeof(unsigned int);
	os << size << ", cudaMemcpyHostToDevice));" << endl;
      }
    }  
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // copying conductances from device
    os << "void copyGFromDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
      if (model.synapseGType[i] == INDIVIDUALG) {
	os << "  cudaGetSymbolAddress(&devPtr, d_gp" << model.synapseName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(gp" << model.synapseName[i] << ", devPtr,";
	os << "  checkCudaErrors(cudaMemcpy(gp" << model.synapseName[i] << ", devPtr,";
	size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*sizeof(float);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
	if (model.synapseType[i] == LEARN1SYNAPSE) {
	  os << "  cudaGetSymbolAddress(&devPtr, d_grawp" << model.synapseName[i] << ");" << endl;
	  //os << "  CUDA_SAFE_CALL(cudaMemcpy(grawp" << model.synapseName[i] << ", devPtr,";
	  os << "  checkCudaErrors(cudaMemcpy(grawp" << model.synapseName[i] << ", devPtr,";
	  size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*sizeof(float);
	  os << size << ", cudaMemcpyDeviceToHost));" << endl;
	}
      }
      if (model.synapseGType[i] == INDIVIDUALID) {
	os << "  cudaGetSymbolAddress(&devPtr, d_gp" << model.synapseName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(gp" << model.synapseName[i] << ", devPtr,";
	os << "  checkCudaErrors(cudaMemcpy(gp" << model.synapseName[i] << ", devPtr,";
	unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	size= (tmp >> logUIntSz);
	if (tmp > (size << logUIntSz)) size++;
	size= size*sizeof(unsigned int);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
      }
    }            
    os << "}" << endl;
    os << endl;
  
    // ------------------------------------------------------------------------
    // copying particular conductances group from device
  
    

    // ------------------------------------------------------------------------
    // copying values to device
    os << "void copyStateToDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    os << "  unsigned int tmp= 0;" << endl;
    os << "  cudaMemcpyToSymbol(\"d_done\",";
    os << " &tmp, sizeof(unsigned int), 0,";
    os << " cudaMemcpyHostToDevice);" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
      nt= model.neuronType[i];
      os << "  cudaMemcpyToSymbol(\"d_glbscnt" << model.neuronName[i] << "\",";
      os << " &glbscnt" << model.neuronName[i] << ", sizeof(unsigned int), 0,";
      os << " cudaMemcpyHostToDevice);" << endl;
      os << "  cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << ");" << endl;
      //os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, " << "glbSpk" << model.neuronName[i] << ",";
      os << "  checkCudaErrors(cudaMemcpy(devPtr, " << "glbSpk" << model.neuronName[i] << ",";
      size= model.neuronN[i]*sizeof(unsigned int);
      os << size << ", cudaMemcpyHostToDevice));" << endl;
      for (int j= 0; j < model.inSyn[i].size(); j++) {
	os << "  cudaGetSymbolAddress(&devPtr, d_inSyn";
	os << model.neuronName[i] << j << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, ";
	os << "  checkCudaErrors(cudaMemcpy(devPtr, ";
	os << "inSyn" << model.neuronName[i] << j << ",";
	size= model.neuronN[i]*sizeof(float);
	os << size << ", cudaMemcpyHostToDevice));" << endl;
      }
      for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	os << "  cudaGetSymbolAddress(&devPtr, d_";
	os << nModels[nt].varNames[k] << model.neuronName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, ";
	os << "  checkCudaErrors(cudaMemcpy(devPtr, ";
	os << nModels[nt].varNames[k] << model.neuronName[i] << ",";
	os << model.neuronN[i] << "*sizeof(" << nModels[nt].varTypes[k];
	os << "), cudaMemcpyHostToDevice));" << endl;
      }
      if (model.neuronNeedSt[i]) {
	os << "  cudaGetSymbolAddress(&devPtr, d_sT" << model.neuronName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(devPtr, " << "sT" << model.neuronName[i] << ",";
	os << "  checkCudaErrors(cudaMemcpy(devPtr, " << "sT" << model.neuronName[i] << ",";
	size= model.neuronN[i]*sizeof(float);
	os << size << ", cudaMemcpyHostToDevice));" << endl;
      }
    }            
    os << "}" << endl;
    os << endl;
    
    // ------------------------------------------------------------------------
    // copying values from device
    os << "void copyStateFromDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
      nt= model.neuronType[i];
      os << "  cudaMemcpyFromSymbol(&glbscnt" << model.neuronName[i];
      os << ", \"d_glbscnt" << model.neuronName[i] << "\", sizeof(unsigned int), 0, ";
      os << " cudaMemcpyDeviceToHost);" << endl;
      os << "  cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << ");" << endl;
      //os << "  CUDA_SAFE_CALL(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
      os << "  checkCudaErrors(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
      size= model.neuronN[i]*sizeof(unsigned int);
      os << size << ", cudaMemcpyDeviceToHost));" << endl;

      for (int j= 0; j < model.inSyn[i].size(); j++) {
	os << "  cudaGetSymbolAddress(&devPtr, d_inSyn";
	os << model.neuronName[i] << j << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(";
	os << "  checkCudaErrors(cudaMemcpy(";
	os << "inSyn" << model.neuronName[i] << j << ", devPtr,";
	size= model.neuronN[i]*sizeof(float);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
      }
      for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	os << "  cudaGetSymbolAddress(&devPtr, d_";
	os << nModels[nt].varNames[k] << model.neuronName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(";
	os << "  checkCudaErrors(cudaMemcpy(";
	os << nModels[nt].varNames[k] << model.neuronName[i] << ", devPtr, ";
	os << model.neuronN[i] << "*sizeof(" << nModels[nt].varTypes[k];
	os << "), cudaMemcpyDeviceToHost));" << endl;
      }
      if (model.neuronNeedSt[i]) {
	os << "  cudaGetSymbolAddress(&devPtr, d_sT" << model.neuronName[i] << ");" << endl;
	//os << "  CUDA_SAFE_CALL(cudaMemcpy(sT" << model.neuronName[i] << ", devPtr, ";
	os << "  checkCudaErrors(cudaMemcpy(sT" << model.neuronName[i] << ", devPtr, ";
	size= model.neuronN[i]*sizeof(float);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
      }
    }            
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // copying spikes from device                                               
    os << "void copySpikesFromDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
      os << "  cudaMemcpyFromSymbol(&glbscnt" << model.neuronName[i];
      os << ", \"d_glbscnt" << model.neuronName[i] << "\", sizeof(unsigned int), 0, ";
      os << " cudaMemcpyDeviceToHost);" << endl;
      os << "  cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << ");";
      os << endl;
      //os << "  CUDA_SAFE_CALL(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
      os << "  checkCudaErrors(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
      os << "glbscnt" << model.neuronName[i] << "*sizeof(unsigned int), ";
      os << "cudaMemcpyDeviceToHost));" << endl;
    }
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // copying spike numbers from device
    os << "void copySpikeNFromDevice()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
      os << "  cudaMemcpyFromSymbol(&glbscnt" << model.neuronName[i];
      os << ", \"d_glbscnt" << model.neuronName[i] << "\", sizeof(unsigned int), 0, ";
      os << " cudaMemcpyDeviceToHost);" << endl;
    }
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // the actual time stepping procedure
    os << "void stepTimeGPU(";
    for (int i= 0; i < model.neuronGrpN; i++) {
      if (model.neuronType[i] == POISSONNEURON) {
	os << "unsigned int *rates" << model.neuronName[i];
	os << ",   // pointer to the rates of the Poisson neurons in grp ";
	os << model.neuronName[i] << endl;
	os << "unsigned int offset" << model.neuronName[i];
	os << ",   // offset on pointer to the rates in grp ";
	os << model.neuronName[i] << endl;
      }
    }
    os << "float t)" << endl;
    os << "{" << endl;
    unsigned int gridSz= model.padSumNeuronN[model.neuronGrpN-1];
    gridSz= gridSz >> logNeuronBlkSz;
    unsigned int sgridSz= model.padSumSynapseTrgN[model.synapseGrpN-1];
    sgridSz= sgridSz >> logSynapseBlkSz;
    unsigned int lgridSz= model.padSumLearnN[model.lrnGroups-1];
    lgridSz= lgridSz >> logLearnBlkSz;
    unsigned int blkSz= neuronBlkSz;
    unsigned int sblkSz= synapseBlkSz;
    unsigned int lblkSz= learnBlkSz;

    os << "  dim3 sThreads(" << sblkSz << ", 1);" << endl;
    os << "  dim3 sGrid(" << sgridSz << ", 1);" << endl;
    os << endl;
    os << "  dim3 nThreads(" << blkSz << ", 1);" << endl;
    os << "  dim3 nGrid(" << gridSz << ", 1);" << endl;
    os << endl;
    if (model.lrnGroups > 0) {
      os << "  dim3 lThreads(" << lblkSz << ", 1);" << endl;
      os << "  dim3 lGrid(" << lgridSz << ", 1);" << endl;
      os << endl;
    }
    os << "  if (t > 0.0) {" << endl; 
    os << "    calcSynapses <<< sGrid, sThreads >>> (";
    if (model.needSt) {
      os << "t";
    }
    os << ");" << endl;
    if (model.lrnGroups > 0) {
      os << "    learnSynapsesPost <<< lGrid, lThreads >>> (t);";
    }
    os << "  }" << endl;
    os << "  calcNeurons <<< nGrid, nThreads >>> (";
    for (int i= 0; i < model.neuronGrpN; i++) {
      if (model.neuronType[i] == POISSONNEURON) {
	os << "rates" << model.neuronName[i] << ", ";
	os << "offset" << model.neuronName[i] << ",";
      }
    }
    os << "t);" << endl;
    os << "}" << endl;
  }
  else {
    mos << "No CUDA enabled device deteced - cowardly refusing to";
    mos << " generate GPU code." << endl;
    os << endl;
  }
}


void genRunnerCPU(NNmodel &model, ostream &os, ostream &mos)
{
  writeHeader(os);
  //os << "#include <cutil.h>" << endl;
  os << "#include <cuda_runtime.h>" << endl;//EY
  os << "#include <helper_cuda.h>" << endl;//EY
  os << "#ifndef RAND" << endl;
  os << "#define RAND(Y,X) Y = Y * 1103515245 +12345;";
  os << "X= (unsigned int)(Y >> 16) & 32767" << endl;
  os << "#endif" << endl;
  os << endl;
  
  os << "#include \"neuronFnct.cc\"" << endl;
  os << "#include \"synapseFnct.cc\"" << endl;
  os << endl;

  // ------------------------------------------------------------------------
  // the actual time stepping procedure
  os << "void stepTimeCPU(";
  for (int i= 0; i < model.neuronGrpN; i++) {
    if (model.neuronType[i] == POISSONNEURON) {
      os << "unsigned int *rates" << model.neuronName[i];
      os << ",   // pointer to the rates of the Poisson neurons in grp ";
      os << model.neuronName[i] << endl;
      os << "unsigned int offset" << model.neuronName[i];
      os << ",   // offset on pointer to the rates in grp ";
      os << model.neuronName[i] << endl;
    }
  }
  os << "float t)" << endl;
  os << "{" << endl;
  os << "  if (t > 0.0) {" << endl; 
  os << "    calcSynapsesCPU();";
  os << endl;
  if (model.lrnGroups > 0) {
    os << "learnSynapsesPostHost(t);" << endl;
  }
  os << "  }" << endl;
  os << "  calcNeuronsCPU(";
  for (int i= 0; i < model.neuronGrpN; i++) {
    if (model.neuronType[i] == POISSONNEURON) {
      os << "rates" << model.neuronName[i] << ", ";
      os << "offset" << model.neuronName[i] << ",";
    }
  }
  os << "t);" << endl;
  os << "}" << endl;
}
