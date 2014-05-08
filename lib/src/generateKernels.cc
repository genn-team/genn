/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//------------------------------------------------------------------------
/*! \file generateKernels.cc

  \brief Contains functions that generate code for CUDA kernels. Part of the code generation section.
*/
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA kernel that simulates all neurons in the model->

  The code generated upon execution of this function is for defining GPU side global variables that will hold model state in the GPU global memory and for the actual kernel function for simulating the neurons for one time step.
*/
//-------------------------------------------------------------------------

unsigned int nt;
short *isGrpVarNeeded;

void genCudaNeuron(int deviceID, ostream &mos)
{
  // write header content
  string name, s, localID;
  ofstream os;
  isGrpVarNeeded = new short[model->neuronGrpN];
  name = path + toString("/") + model->name + toString("_CODE_CUDA_") + toString(deviceID) + toString("/neuron.cu");
  os.open(name.c_str());
  
  writeHeader(os);
  // compiler/include control (include once)
  os << "#ifndef _" << model->name << "_neuronKrnl_cc" << endl;
  os << "#define _" << model->name << "_neuronKrnl_cc" << endl;
  
  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file neuronKrnl.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model->name << " containing the neuron kernel function." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;

  // global device variables
  os << "// relevant neuron variables" << endl;
  os << "__device__ volatile unsigned int d_done;" << endl;
  for (int i= 0; i < model->neuronGrpN; i++) {
    nt= model->neuronType[i];
    isGrpVarNeeded[i] = 0;

    //these now hold just true spikes for benefit of the user (raster plots etc)
    os << "__device__ volatile unsigned int d_glbscnt" << model->neuronName[i] << ";" << endl;
    os << "__device__ volatile unsigned int d_glbSpk" << model->neuronName[i] << "[" << model->neuronN[i] << "];" << endl;

    if (model->neuronDelaySlots[i] == 1) {//i.e. no delay
      os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model->neuronName[i] << ";" << endl;
      os << "__device__ volatile unsigned int d_glbSpkEvnt" << model->neuronName[i] << "[" << model->neuronN[i] << "];" << endl;
    }
    else {
      os << "__device__ volatile unsigned int d_spkEvntQuePtr" << model->neuronName[i] << ";" << endl;
      os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model->neuronName[i] << "[";
      os << model->neuronDelaySlots[i] << "];" << endl;
      os << "__device__ volatile unsigned int d_glbSpkEvnt" << model->neuronName[i] << "[";
      os << model->neuronN[i] * model->neuronDelaySlots[i] << "];" << endl;
    }
    if (model->neuronType[i] != POISSONNEURON) {
      for (int j= 0; j < model->inSyn[i].size(); j++) {
	os << "__device__ " << model->ftype << " d_inSyn" << model->neuronName[i] << j << "[" << model->neuronN[i] << "];";
	os << "    // summed input for neurons in grp" << model->neuronName[i] << endl;
      }
    }
    if (model->neuronNeedSt[i]) { 
      os << "__device__ volatile " << model->ftype << " d_sT" << model->neuronName[i] << "[" << model->neuronN[i] << "];" << endl;
    }
    os << endl;
  }

  //! Binary spike flag for sparse synapse sources. This is necessary for parallelisation of the synapse kernel for postsynaptic spike queue.  
  for (int i= 0; i < model->synapseGrpN; i++) {
    if ((model->synapseConnType[i] == SPARSE)&& (model->neuronN[model->synapseTarget[i]] > neuronBlkSz[deviceID]) && (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << "__device__ short d_spikeFlag" << model->neuronName[model->synapseTarget[i]] << "[" << model->neuronN[model->synapseTarget[i]] << "];" << endl;
      isGrpVarNeeded[model->synapseTarget[i]] = 1;
    }		
  }
  
  // kernel header
  os << "__global__ void calcNeurons(" << endl;

  for (int i= 0; i < model->neuronGrpN; i++) {
    nt = model->neuronType[i];
    if (nt == POISSONNEURON) {
      // Note: Poisson neurons only used as input neurons; they do not receive any inputs
      os << "  unsigned int *d_rates" << model->neuronName[i];
      os << ", // poisson \"rates\" of grp " << model->neuronName[i] << endl;
      os << "  unsigned int offset" << model->neuronName[i];
      os << ", // poisson \"rates\" offset of grp " << model->neuronName[i] << endl;
    }
    if (model->receivesInputCurrent[i] > 1) {
      os << "  float *d_inputI" << model->neuronName[i];
      os << ", // explicit input current to grp " << model->neuronName[i] << endl;    	
    }
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "  " << nModels[nt].varTypes[k] << " *d_" << nModels[nt].varNames[k];
      os << model->neuronName[i]<< ", " << endl;
    }
  }
  
  for (int i=0; i< model->synapseName.size(); i++){
    int pst= model->postSynapseType[i];
    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
      os << "  " << postSynModels[pst].varTypes[k] << " *d_" << postSynModels[pst].varNames[k];
      os << model->synapseName[i]<< ", " << endl;
    }
  }
  
  os << model->ftype << " t // absolute time" << endl; 
  os << ")" << endl;

  // kernel code
  os << "{" << endl;
  unsigned int neuronGridSz = model->padSumNeuronN[model->neuronGrpN - 1];
  neuronGridSz = neuronGridSz / neuronBlkSz[deviceID];
  if (neuronGridSz < deviceProp[deviceID].maxGridSize[1]){
    os << "  unsigned int id = " << neuronBlkSz[deviceID] << " * blockIdx.x + threadIdx.x;" << endl;
  }
  else {
    os << "  unsigned int id = " << neuronBlkSz[deviceID] << " * (blockIdx.x * " << ceil(sqrt(neuronGridSz));
    os << " + blockIdx.y) + threadIdx.x;" << endl;  	
  }
  //these variables deal with high V "spike type events"
  os << "  __shared__ volatile unsigned int posSpkEvnt;" << endl;
  os << "  __shared__ unsigned int shSpkEvnt[" << neuronBlkSz[deviceID] << "];" << endl;
  os << "  unsigned int spkEvntIdx;" << endl;
  os << "  __shared__ volatile unsigned int spkEvntCount;" << endl; //was scnt

  //these variables now deal only with true spikes , not high V "events"
  os << "  __shared__ unsigned int shSpk[" << neuronBlkSz[deviceID] << "];" << endl;
  os << "  __shared__ volatile unsigned int posSpk;" << endl;
  os << "  unsigned int spkIdx;" << endl; //was sidx
  os << "  __shared__ volatile unsigned int spkCount;" << endl; //was scnt

  os << endl;
  os << "  if (threadIdx.x == 0) { spkEvntCount = 0; spkCount = 0;}" << endl;
  os << "  __syncthreads();" << endl;
  
  for (int i= 0; i < model->neuronGrpN; i++) {
    nt= model->neuronType[i];
    if (i == 0) {
      os << "  if (id < " << model->padSumNeuronN[i] << ") {" << endl;
      localID = string("id");
    }
    else {
      os << "  if ((id >= " << model->padSumNeuronN[i-1] << ") && ";
      os << "(id < " << model->padSumNeuronN[i] << ")) {" << endl;
      os << "    unsigned int lid;" << endl;
      os << "    lid = id - " << model->padSumNeuronN[i-1] << ";" << endl;
      localID = string("lid");
    }
    os << "    // only do this for existing neurons" << endl;
    os << "    if (" << localID << " < " << model->neuronN[i] << ") {" << endl;
    os << "      // pull V values in a coalesced access" << endl;
    if (nt == POISSONNEURON) {
      os << "      unsigned int lrate = d_rates" << model->neuronName[i];
      os << "[offset" << model->neuronName[i] << " + " << localID << "];" << endl;
    }
    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      os << "      " << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k];
      os << " = d_" <<  nModels[nt].varNames[k] << model->neuronName[i] << "[";
      if ((nModels[nt].varNames[k] == "V") && (model->neuronDelaySlots[i] != 1)) {
	os << "(((d_spkEvntQuePtr" << model->neuronName[i] << " + " << (model->neuronDelaySlots[i] - 1) << ") % ";
	os << model->neuronDelaySlots[i] << ") * " << model->neuronN[i] << ") + ";
      }
      os << localID << "];" << endl;
    }
    if (nt != POISSONNEURON) {
      os << "      // pull inSyn values in a coalesced access" << endl;
      for (int j = 0; j < model->inSyn[i].size(); j++) {
      	os << "      " << model->ftype << " linSyn" << j << " = d_inSyn" << model->neuronName[i] << j << "[" << localID << "];" << endl;
      }
      os << "    " << model->ftype << " Isyn = 0;" << endl;
      if (isGrpVarNeeded[i]==1) {
      	os << "    d_spikeFlag" << model->neuronName[i] << "[" << localID << "]=0;" << endl;
      }
      if (model->inSyn[i].size() > 0) {
	for (int j = 0; j < model->inSyn[i].size(); j++) {
	  for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	    os << "      " << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varTypes[k] << " lps" << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << j;
	    os << " = d_" <<  postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << model->synapseName[model->inSyn[i][j]] << "[";
	    os << localID << "];" << endl;
	  }

	  os << "      Isyn += ";
	  string psCode = postSynModels[model->postSynapseType[model->inSyn[i][j]]].postSyntoCurrent;

	  substitute(psCode, tS("$(inSyn)"), tS("linSyn")+tS(j));

	  for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    substitute(psCode, tS("$(") + nModels[nt].varNames[k] + tS(")"),
		       tS("l") + nModels[nt].varNames[k]);
	  }
	
	  for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	    substitute(psCode, tS("$(") + nModels[nt].pNames[k] + tS(")"),
		       tS("l") + nModels[nt].pNames[k]);
	  }
    
	  for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].pNames.size(); k < l; k++) {
	    substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].pNames[k] + tS(")"),
		       tS(model->postSynapsePara[model->inSyn[i][j]][k]));
	  }
		   
	  for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	    substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] + tS(")"), 
		       tS("lps") +tS(postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k])+tS(j));
	  }  
		 
	  for (int k = 0; k < postSynModels[model->postSynapseType[model->inSyn[i][j]]].dpNames.size(); ++k)
	    substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].dpNames[k] + tS(")"), tS(model->dpsp[model->inSyn[i][j]][k]));
		
	  os << psCode;
	  os << ";" << endl;
	}
      }
    }
    if (model->receivesInputCurrent[i] == 1) { // receives constant  input
      os << "      Isyn += " << model->globalInp[i] << ";" << endl;
    }    	
    if (model->receivesInputCurrent[i] >= 2) { // receives explicit input from file
      os << "    Isyn += (" << model->ftype<< ") d_inputI" << model->neuronName[i] << "[" << localID << "];" << endl;
    }
    os << "      // calculate membrane potential" << endl;
    //new way of doing it
    string code = nModels[nt].simCode;
    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
    }
    substitute(code, tS("$(Isyn)"), tS("Isyn"));
    for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model->neuronPara[i][k]));
    }
    for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model->dnp[i][k]));
    }
    os << code;

    //insert condition code provided that tests for a true spike
    if (nModels[nt].thresholdConditionCode  == tS("")) { //no condition provided
      cerr << "Generation Error: You must provide thresholdConditionCode for neuron type :  " << model->neuronType[i]  << " used for " << model->name[i];
      exit(1);
    }
    else {
      code= nModels[nt].thresholdConditionCode;
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
      }
      substitute(code, tS("$(Isyn)"), tS("Isyn"));
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model->neuronPara[i][k]));
      }
      for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model->dnp[i][k]));
      }
      os << "      if (" << code << ") {" << endl;
      os << "        // register a true spike" << endl;
      os << "        spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << endl;
      os << "        shSpk[spkIdx] = " << localID << ";" << endl;
    }

    //add optional reset code after a true spike, if provided
    if (nModels[nt].resetCode != tS("")) {
      code = nModels[nt].resetCode;
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
      }
      substitute(code, tS("$(Isyn)"), tS("Isyn"));
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model->neuronPara[i][k]));
      }
      for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model->dnp[i][k]));
      }
      os << "        // spike reset code" << endl;
      os << "        " << code << endl;
    }
    os << "      }" << endl;

    //test if a spike type event occurred
    os << "      if (lV >= " << model->nSpkEvntThreshold[i] << ") {" << endl;
    os << "        // register a spike type event" << endl;
    os << "        spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << endl;
    os << "        shSpkEvnt[spkEvntIdx] = " << localID << ";" << endl;
    os << "      }" << endl;

    //store the defined parts of the neuron state into the global state variables d_V.. etc
    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      os << "      d_" << nModels[nt].varNames[k] <<  model->neuronName[i] << "[";
      if ((nModels[nt].varNames[k] == "V") && (model->neuronDelaySlots[i] != 1)) {
	os << "(d_spkEvntQuePtr" << model->neuronName[i] << " * " << model->neuronN[i] << ") + ";
      }
      os << localID << "] = l" << nModels[nt].varNames[k] << ";" << endl;
    }
    for (int j = 0; j < model->inSyn[i].size(); j++) {
    
      string psCode = postSynModels[model->postSynapseType[model->inSyn[i][j]]].postSynDecay;

      substitute(psCode, tS("$(inSyn)"), tS("linSyn") + tS(j));
      for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].pNames.size(); k < l; k++) {
	substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].pNames[k] + tS(")"), 
		   tS(model->postSynapsePara[model->inSyn[i][j]][k]));
      }  
		 
      for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] + tS(")"), 
		   tS("lps") +tS(postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k])+tS(j));
      }  
		       
      for (int k = 0; k < postSynModels[model->postSynapseType[model->inSyn[i][j]]].dpNames.size(); ++k)
	substitute(psCode, tS("$(") + postSynModels[model->postSynapseType[model->inSyn[i][j]]].dpNames[k] + tS(")"), tS(model->dpsp[model->inSyn[i][j]][k]));
			
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(psCode, tS("$(") + nModels[nt].varNames[k] + tS(")"),
		   tS("l") + nModels[nt].varNames[k]);
      }
	
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(psCode, tS("$(") + nModels[nt].pNames[k] + tS(")"),
		   tS("l") + nModels[nt].pNames[k]);
      }
    
      os << psCode;
      os << "      d_inSyn"  << model->neuronName[i] << j << "[" << localID << "] = linSyn"<< j << ";" << endl;
    
      for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	os << "      " << "d_" <<  postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << model->synapseName[model->inSyn[i][j]] << "[";
	os << localID << "] = lps" << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << j << ";"<< endl;
      }
    }
    os << "    }" << endl;
    os << "    __syncthreads();" << endl;
    
    os << "    if (threadIdx.x == 0) {" << endl;
    os << "      posSpkEvnt = atomicAdd((unsigned int *) &d_glbSpkEvntCnt" << model->neuronName[i];
    if (model->neuronDelaySlots[i] != 1) {
      os << "[d_spkEvntQuePtr" << model->neuronName[i] << "]";
    }
    os << ", spkEvntCount);" << endl;

    os << "      posSpk = atomicAdd((unsigned int *) &d_glbscnt" << model->neuronName[i] << ", spkCount);" << endl;

    os << "    }" << endl;  //end if (threadIdx.x == 0)

    os << "    __syncthreads();" << endl;

    os << "    if (threadIdx.x < spkEvntCount) {" << endl;
    os << "      d_glbSpkEvnt" << model->neuronName[i] << "[";

    if (model->neuronDelaySlots[i] != 1) {
      os << "(d_spkEvntQuePtr" << model->neuronName[i] << " * " << model->neuronN[i] << ") + ";
    }
    os << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << endl;
    os << "    }" << endl;

    os << "    if (threadIdx.x < spkCount) {" << endl;
    os << "      d_glbSpk" << model->neuronName[i] << "[";

    os << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << endl;
    if (model->neuronNeedSt[i]) {
      os << "      d_sT" << model->neuronName[i] << "[shSpk[threadIdx.x]] = t;" << endl;
    }
    os << "    }" << endl;
    os << "  }" << endl;
    os << endl;
  }
  os << "}" << endl;
  os << endl;
  os << "#endif" << endl;
  os.close();
}

//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are synapse-related and the actual CUDA kernel for simulating one time step of the synapses.
*/
//-------------------------------------------------------------------------

void genCudaSynapse(int deviceID, ostream &mos)
{
  string name, s, localID, theLG;
  unsigned int numOfBlocks,trgN;
  ofstream os;

  // count how many neuron blocks to use: one thread for each synapse target
  // targets of several input groups are counted multiply
  numOfBlocks = model->padSumSynapseKrnl[model->synapseGrpN - 1] / synapseBlkSz[deviceID];
  name = path + toString("/") + model->name + toString("_CODE_CUDA_") + toString(deviceID) + toString("/synapse.cu");
  os.open(name.c_str());
  writeHeader(os);  // write header content
  // compiler/include control (include once)
  os << "#ifndef _" << model->name << "_synapseKrnl_cc" << endl;
  os << "#define _" << model->name << "_synapseKrnl_cc" << endl;
  os << endl;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file synapseKrnl.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model->name;
  os << " containing the synapse kernel and learning kernel functions." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;
  os << endl;
  
  // Kernel header
  unsigned int src;
  os << "__global__ void calcSynapses(" << endl;
  for (int i = 0; i < model->synapseGrpN; i++) {    
    if (model->synapseGType[i] == INDIVIDUALG) {
      os << " " << model->ftype << " * d_gp" << model->synapseName[i] << "," << endl;	
    }
    if (model->synapseConnType[i] == SPARSE){
      os << " unsigned int * d_gp" << model->synapseName[i] << "_ind," << endl;
      os << " unsigned int * d_gp" << model->synapseName[i] << "_indInG," << endl;
      trgN = model->neuronN[model->synapseTarget[i]];
    }
    if (model->synapseGType[i] == (INDIVIDUALID )) {
      os << "  unsigned int *d_gp" << model->synapseName[i] << "," << endl;	
    }
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << model->ftype << " * d_grawp" << model->synapseName[i] << "," << endl;
    }   	
  }
  for (int i = 0; i < model->neuronGrpN; i++) {
    nt = model->neuronType[i];
    os << "  " << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model->neuronName[i]; // Vm
    if (i < (model->neuronGrpN - 1) || model->needSt) os << "," << endl;
  }
  if (model->needSt) {
    os << "  " << model->ftype << " t";
  }
  os << endl << ")";
  os << endl;

  // kernel code
  os << "{" << endl;
  os << "  unsigned int id = " << synapseBlkSz[deviceID] << " * blockIdx.x + threadIdx.x;" << endl;
  os << "  __shared__ unsigned int shSpkEvnt[" << synapseBlkSz[deviceID] << "];" << endl;
  os << "  __shared__ " << model->ftype << " shSpkEvntV[" << synapseBlkSz[deviceID] << "];" << endl;
  os << "  volatile __shared__ " << model->ftype << " shLg[" << neuronBlkSz[deviceID] << "];" << endl;
  os << "  unsigned int lscnt, numSpikeSubsets, lmax, j, p, r, ipost, npost;" << endl;
  for (int i = 0; i < model->synapseGrpN; i++) {
    if (model->synapseConnType[i] != SPARSE){
      os << "  " << model->ftype << " linSyn, lg;" << endl;
      break;
    }
  }
  if (model->needSynapseDelay == 1) {
    os << "  int delaySlot;" << endl;
  }
  os << endl;
  os << "  ipost = 0;" << endl;
  for (int i = 0; i < model->synapseGrpN; i++) {
    if (i == 0) {
      os << "  if (id < " << model->padSumSynapseKrnl[i] << ") {  //synapse group " << model->synapseName[i] << endl;
      localID = string("id");
    }
    else {
      os << "  if ((id >= " << model->padSumSynapseKrnl[i - 1] << ") && ";
      os << "(id < " << model->padSumSynapseKrnl[i] << ")) {  //synapse group " << model->synapseName[i] << endl;
      os << "    unsigned int lid;" << endl;
      os << "    lid = id - " << model->padSumSynapseKrnl[i - 1] << ";" << endl;
      localID = string("lid");
    }
    unsigned int trg = model->synapseTarget[i];
    unsigned int nN = model->neuronN[trg];
    src = model->synapseSource[i];
    float Epre = model->synapsePara[i][1];
    float Vslope;
    if (model->synapseType[i] == NGRADSYNAPSE) {
      Vslope = model->synapsePara[i][3]; 
    }
    unsigned int inSynNo = model->synapseInSynNo[i];
    if (model->neuronDelaySlots[src] != 1) {
      os << "    delaySlot = (d_spkEvntQuePtr" << model->neuronName[src] << " + ";
      os << (int) (model->neuronDelaySlots[src] - model->synapseDelay[i] + 1);
      os << ") % " << model->neuronDelaySlots[src] << ";" << endl;
    }
    if (model->synapseConnType[i] != SPARSE) {
      os << "    // only do this for existing neurons" << endl;
      os << "    if (" << localID << " < " << nN <<") {" << endl;
      os << "      linSyn = d_inSyn" << model->neuronName[trg] << inSynNo << "[" << localID << "];" << endl;

      os << "    }" << endl;
      os << "    __threadfence();" << endl;
    }
    os << "    lscnt = d_glbSpkEvntCnt" << model->neuronName[src];
    if (model->neuronDelaySlots[src] != 1) {
      os << "[delaySlot]";
    }
    os << ";" << endl;
    os << "    numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << synapseBlkSz[deviceID] << ".0f));" << endl;
    os << "    for (r = 0; r < numSpikeSubsets; r++) {" << endl;
    os << "      if (r == numSpikeSubsets - 1) lmax = lscnt % " << synapseBlkSz[deviceID] << ";" << endl;
    os << "      else lmax = " << synapseBlkSz[deviceID] << ";" << endl;
    os << "      if (threadIdx.x < lmax) {" << endl;
    os << "        shSpkEvnt[threadIdx.x] = d_glbSpkEvnt" << model->neuronName[src] << "[";
    if (model->neuronDelaySlots[src] != 1) {
      os << "(delaySlot * " << model->neuronN[src] << ") + ";
    }
    os << "(r * " << synapseBlkSz[deviceID] << ") + threadIdx.x];" << endl;
    if (model->neuronType[src] != POISSONNEURON) {
      os << "        shSpkEvntV[threadIdx.x] = d_V" << model->neuronName[src] << "[";
      if (model->neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model->neuronN[src] << ") + ";
      }
      os << "shSpkEvnt[threadIdx.x]];" << endl;
    }
    os << "      }" << endl;
    if ((model->synapseConnType[i] == SPARSE) && (model->neuronN[trg] <= neuronBlkSz[deviceID])) {
      os << "        if (threadIdx.x < " << neuronBlkSz[deviceID] << ") shLg[threadIdx.x] = 0;" << endl;
    }
    os << "      __syncthreads();" << endl;
    os << "      // only work on existing neurons" << endl;
    if (model->synapseConnType[i] == SPARSE) {
      if(model->maxConn.size()==0) {
	fprintf(stderr,"Model Generation error: for every SPARSE synapse group used you must also supply (in your model) a max possible number of connections via the model->setMaxConn() function.");
	exit(1);
      }
      int maxConnections  = model->maxConn[i];
      os << "      if (" << localID << " < " << maxConnections << ") {" << endl;
    }
    else {
      os << "      if (" << localID << " < " << model->neuronN[trg] << ") {" << endl;
    }

    os << "        // loop through all incoming spikes" << endl;
    os << "        for (j = 0; j < lmax; j++) {" << endl;
    if (model->synapseGType[i] == INDIVIDUALID) {
      os << "          unsigned int gid = (shSpkEvnt[j] * " << model->neuronN[trg];
      os << " + " << localID << ");" << endl;
    }
    if (model->neuronType[src] != POISSONNEURON) {
      os << "          if ";
      if (model->synapseGType[i] == INDIVIDUALID) {
	// Note: we will just access global mem. For compute >= 1.2
	// simultaneous access to same global mem in the (half-)warp
	// will be coalesced - no worries
	os << "((B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz - 1 << ")) && ";
      } 
      os << "(shSpkEvntV[j] > " << Epre << ")";
      if (model->synapseGType[i] == INDIVIDUALID) {
	os << ")";
      }
      os << " {" << endl;
    }
    else {
      if (model->synapseGType[i] == INDIVIDUALID) {
	os << "          if (B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz - 1 << ")) {" << endl;
      }
    }

    if (model->synapseConnType[i] == SPARSE) {
      os << "          npost = d_gp" << model->synapseName[i] << "_indInG[shSpkEvnt[j] + 1] - d_gp";
      os << model->synapseName[i] << "_indInG[shSpkEvnt[j]];" << endl;
      os << "          if ("<< localID <<" < npost) {" << endl;
      os << "            ipost = d_gp" << model->synapseName[i] << "_ind[d_gp";
      os << model->synapseName[i] << "_indInG[shSpkEvnt[j]] + "<< localID <<"];" << endl;
      if (isGrpVarNeeded[model->synapseTarget[i]] == 0) {
	theLG = toString("shLg[" + localID + "]");
	if (model->synapseGType[i] != INDIVIDUALG) {
	  os << "            shLg[ipost] += " << model->g0[i] <<"];";
	}
	else {
	  os << "            shLg[ipost] += d_gp" << model->synapseName[i] << "[d_gp" << model->synapseName[i] << "_indInG[shSpkEvnt[j]] + "<< localID <<"];";
	}
      }
      else {
	if (model->synapseGType[i] != INDIVIDUALG) {
	  os << "            atomicAdd(&d_inSyn" << model->neuronName[trg] << inSynNo << "[ipost]," << model->g0[i] <<");";
	}
	else{
	  os << "            atomicAdd(&d_inSyn" << model->neuronName[trg] << inSynNo << "[ipost],d_gp" << model->synapseName[i] << "[d_gp" << model->synapseName[i] << "_indInG[shSpkEvnt[j]] + "<< localID <<"]);" << endl;
	}
      }

      os << "          }" << endl; // end if (id < npost)
      //os << "        }" << endl; // end if (shSpkEvntV[j]>postthreshold)
      os << "        __syncthreads();" << endl;
    }
    else {
      if (model->synapseGType[i] == INDIVIDUALG){
	os << "            lg = d_gp" << model->synapseName[i] << "[shSpkEvnt[j]*" << model->neuronN[trg] << " + " << localID << "];";
	theLG = toString("lg");
      }
    }
    os << endl;
    //if (model->synapseConnType[i] == SPARSE) { // need to close the parenthesis to synchronize threads
    //os << "          }" << endl; // end if (id < npost)
    //os << "        }" << endl; // end if (shSpkEvntV[j]>postthreshold)
    //os << "        __syncthreads();" << endl;
    //}
    if ((model->synapseGType[i] == GLOBALG) || (model->synapseGType[i] == INDIVIDUALID)) {
      theLG = toString(model->g0[i]);
    }

    if (model->synapseConnType[i] != SPARSE) {
      if ((model->synapseType[i] == NSYNAPSE) || (model->synapseType[i] == LEARN1SYNAPSE)) {
	os << "          linSyn = linSyn + " << theLG << "; " << endl;
      }
      if (model->synapseType[i] == NGRADSYNAPSE) {
      	if (model->neuronType[src] == POISSONNEURON) {
	  os << "          linSyn = linSyn + " << theLG << " * tanh((";
	  os << SAVEP(model->neuronPara[src][2]) << " - " << SAVEP(Epre);
      	}
      	else {
	  os << "          linSyn = linSyn + " << theLG << " * tanh((shSpkEvntV[j] - " << SAVEP(Epre);
      	}
      	os << ") / " << Vslope << ");" << endl;
      }
    }
    // if needed, do some learning (this is for pre-synaptic spikes)
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      // simply assume INDIVIDUALG for now
      os << "            lg = d_grawp" << model->synapseName[i] << "[shSpkEvnt[j] * " << model->neuronN[trg] << " + " << localID << "];" << endl;
      os << "            " << model->ftype << " dt = d_sT" << model->neuronName[trg] << "[" << localID << "] - t - ";
      os << SAVEP(model->synapsePara[i][11]) << ";" << endl;
      os << "            if (dt > " << model->dsp[i][1] << ") {" << endl;
      os << "              dt = - " << SAVEP(model->dsp[i][5]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > 0.0) {" << endl;
      os << "              dt = " << SAVEP(model->dsp[i][3]) << " * dt + " << SAVEP(model->dsp[i][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > " << model->dsp[i][2] << ") {" << endl;
      os << "              dt = " << SAVEP(model->dsp[i][4]) << " * dt + " << SAVEP(model->dsp[i][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else {" << endl;
      os << "              dt = - " << SAVEP(model->dsp[i][7]) << ";" << endl;
      os << "            }" << endl;
      os << "            lg = lg + dt;" << endl;
      os << "            d_grawp" << model->synapseName[i] << "[shSpkEvnt[j] * " << model->neuronN[trg] << " + " << localID << "] = lg;" << endl;
      os << "            d_gp" << model->synapseName[i] << "[shSpkEvnt[j] * " << model->neuronN[trg] << " + " << localID << "] = ";
      os << "gFunc" << model->synapseName[i] << "(lg);" << endl; 
    }
    if (model->synapseConnType[i] != SPARSE) {
      if ((model->neuronType[src] != POISSONNEURON) || (model->synapseGType[i] == INDIVIDUALID)) {
        os << "          }" << endl;////1 end if (shSpkEvntV[j]>postthreshold) !!!!!! is INDIVIDUALID part correct?
      }
    }

    os << "        }" << endl; ////2 for (j = 0; j < lmax; j++)
    os << "      }" << endl; ////3 if (id < Npre)
    os << "    }" << endl; ////4 for (r = 0; r < numSpikeSubsets; r++)

    if (model->synapseConnType[i] != SPARSE) {
      os << "    // only do this for existing neurons" << endl;
      os << "    if (" << localID << " < " << model->neuronN[trg] <<") {" << endl;
      os << "      d_inSyn" << model->neuronName[trg] << inSynNo << "[" << localID << "] = linSyn;" << endl;
      os << "    }" << endl;
    } 
    if (model->lrnGroups == 0) {
      os << "    if (threadIdx.x == 0) {" << endl;
      os << "      j = atomicAdd((unsigned int *) &d_done, 1);" << endl;
      os << "      if (j == " << numOfBlocks - 1 << ") {" << endl;
      for (int j = 0; j < model->neuronGrpN; j++) {
	os << "        d_glbscnt" << model->neuronName[j] << " = 0;" << endl;
	if (model->neuronDelaySlots[j] != 1) {
	  os << "        d_spkEvntQuePtr" << model->neuronName[j] << " = (d_spkEvntQuePtr";
	  os << model->neuronName[j] << " + 1) % " << model->neuronDelaySlots[j] << ";" << endl;
	  os << "        d_glbSpkEvntCnt" << model->neuronName[j] << "[d_spkEvntQuePtr";
	  os << model->neuronName[j] << "] = 0;" << endl;
	}
	else {
	  os << "        d_glbSpkEvntCnt" << model->neuronName[j] << " = 0;" << endl;
	}
      }
      os << "        d_done = 0;" << endl;
      os << "      }" << endl;
      os << "    }" << endl;
    }
    os << "  }" << endl;
    os << endl;
  }
  os << "}" << endl;
  os << endl;


  ///////////////////////////////////////////////////////////////
  // Kernel for learning synapses, post-synaptic spikes

  if (model->lrnGroups > 0) {

    // count how many learn blocks to use: one thread for each synapse source
    // sources of several output groups are counted multiply
    numOfBlocks = model->padSumLearnN[model->lrnGroups - 1] / learnBlkSz[deviceID];

    // Kernel header
    os << "__global__ void learnSynapsesPost(" << endl;
    for (int i = 0; i < model->synapseGrpN; i++) {
      if (model->synapseGType[i] == (INDIVIDUALG )) {
	os << "  " << model->ftype << " *d_gp" << model->synapseName[i] << "," << endl;	
      }
      if (model->synapseGType[i] == (INDIVIDUALID )) {
	os << "  unsigned int *d_gp" << model->synapseName[i] << "," << endl;	
      }
      if (model->synapseType[i] == LEARN1SYNAPSE) {
	os << "  " << model->ftype << " *d_grawp"  << model->synapseName[i] << "," << endl;
      }
    }
    for (int i = 0; i < model->neuronGrpN; i++) {
      nt = model->neuronType[i];
      os << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model->neuronName[i] << ","; // Vm
    }
    os << "  " << model->ftype << " t" << endl;
    os << ")" << endl;

    // kernel code
    os << "{" << endl;
    os << "  unsigned int id = " << learnBlkSz[deviceID] << " * blockIdx.x + threadIdx.x;" << endl;
    os << "  __shared__ unsigned int shSpkEvnt[" << learnBlkSz[deviceID] << "];" << endl;
    os << "  __shared__ " << model->ftype << " shSpkEvntV[" << learnBlkSz[deviceID] << "];" << endl;
    os << "  unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << endl;
    os << "  " << model->ftype << " lg;" << endl;
    os << endl;

    for (int i = 0; i < model->lrnGroups; i++) {
      if (i == 0) {
	os << "  if (id < " << model->padSumLearnN[i] << ") {" << endl;
	localID = string("id");
      }
      else {
	os << "  if ((id >= " << model->padSumLearnN[i - 1] << ") && ";
	os << "(id < " << model->padSumLearnN[i] << ")) {" << endl;
	os << "    unsigned int lid;" << endl;
	os << "    lid = id - " << model->padSumLearnN[i - 1] << ";" << endl;
	localID = string("lid");
      }
      unsigned int k = model->lrnSynGrp[i];
      unsigned int src = model->synapseSource[k];
      unsigned int nN = model->neuronN[src];
      unsigned int trg = model->synapseTarget[k];
      float Epre = model->synapsePara[k][1];

      os << "    lscnt = d_glbSpkEvntCnt" << model->neuronName[trg];
      if (model->neuronDelaySlots[trg] != 1) os << "[d_spkEvntQuePtr" << model->neuronName[trg] << "]";
      os << ";" << endl;
      os << "    numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << learnBlkSz[deviceID] << ".0f));" << endl;
      os << "    for (r = 0; r < numSpikeSubsets; r++) {" << endl;
      os << "      if (r == numSpikeSubsets - 1) lmax = lscnt % " << learnBlkSz[deviceID] << ";" << endl;
      os << "      else lmax = " << learnBlkSz[deviceID] << ";" << endl;
      os << "      if (threadIdx.x < lmax) {" << endl;
      os << "        shSpkEvnt[threadIdx.x] = d_glbSpkEvnt" << model->neuronName[trg] << "[";
      if (model->neuronDelaySlots[trg] != 1) {
	os << "(d_spkEvntQuePtr" << model->neuronName[trg] << " * " << model->neuronN[trg] << ") + ";
      }
      os << "(r * " << learnBlkSz[deviceID] << ") + threadIdx.x];" << endl;
      os << "        shSpkEvntV[threadIdx.x] = d_V" << model->neuronName[trg] << "[";
      if (model->neuronDelaySlots[trg] != 1) {
	os << "(d_spkEvntQuePtr" << model->neuronName[trg] << " * " << model->neuronN[trg] << ") + ";
      }
      os << "shSpkEvnt[threadIdx.x]];" << endl;
      os << "      }" << endl;
      os << "      __syncthreads();" << endl;
      os << "      // only work on existing neurons" << endl;
      os << "      if (" << localID << " < " << model->neuronN[src] << ") {" << endl;
      os << "        // loop through all incoming spikes for learning" << endl;
      os << "        for (j = 0; j < lmax; j++) {" << endl;
      os << "          if (shSpkEvntV[j] > " << Epre << ") {" << endl;
      os << "            lg = d_grawp" << model->synapseName[k] << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpkEvnt[j]];" << endl;
      os << "            " << model->ftype << " dt = t - d_sT" << model->neuronName[src] << "[" << localID << "]";
      if (model->neuronDelaySlots[src] != 1) {
	os << " + " << (DT * model->synapseDelay[k]);
      }
      os << " - " << SAVEP(model->synapsePara[k][11]) << ";" << endl;
      os << "            if (dt > " << model->dsp[k][1] << ") {" << endl;
      os << "              dt = - " << SAVEP(model->dsp[k][5]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > 0.0) {" << endl;
      os << "              dt = " << SAVEP(model->dsp[k][3]) << " * dt + " << SAVEP(model->dsp[k][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > " << model->dsp[k][2] << ") {" << endl;
      os << "              dt = " << SAVEP(model->dsp[k][4]) << " * dt + " << SAVEP(model->dsp[k][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else {" << endl;
      os << "              dt = - " << SAVEP(model->dsp[k][7]) << ";" << endl;
      os << "            }" << endl;
      os << "            lg = lg + dt;" << endl;
      os << "            d_grawp" << model->synapseName[k] << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpkEvnt[j]] = lg;" << endl;
      os << "            d_gp" << model->synapseName[k] << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpkEvnt[j]] = gFunc" << model->synapseName[k] << "(lg);" << endl;
      os << "          }" << endl;
      os << "        }" << endl;
      os << "      }" << endl;
      os << "    }" << endl;
      //os << "    __syncthreads();" << endl;
      os << "          __threadfence();" << endl;
      os << "    if (threadIdx.x == 0) {" << endl;
      os << "      j = atomicAdd((unsigned int *) &d_done, 1);" << endl;
      os << "      if (j == " << numOfBlocks - 1 << ") {" << endl;
      for (int j = 0; j < model->neuronGrpN; j++) {
	os << "        d_glbscnt" << model->neuronName[j] << " = 0;" << endl;
	if (model->neuronDelaySlots[j] != 1) {
	  os << "        d_spkEvntQuePtr" << model->neuronName[j] << " = (d_spkEvntQuePtr";
	  os << model->neuronName[j] << " + 1) % " << model->neuronDelaySlots[j] << ";" << endl;
	  os << "        d_glbSpkEvntCnt" << model->neuronName[j] << "[d_spkEvntQuePtr";
	  os << model->neuronName[j] << "] = 0;" << endl;
	}
	else {
	  os << "        d_glbSpkEvntCnt" << model->neuronName[j] << " = 0;" << endl;
	}
      }
      os << "        d_done = 0;" << endl;
      os << "      }" << endl;
      os << "    }" << endl;
      os << "  }" << endl;
    }
    os << "}" << endl;
  }
  os << endl;

  os << "#endif" << endl;
  os.close();
}
