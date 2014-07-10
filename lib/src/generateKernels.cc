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

short *isGrpVarNeeded;

void genCudaNeuron(unsigned int deviceID, //!< device number to generate code for
		   ostream &mos //!< output stream for messages
		   )
{
  unsigned int nt;
  string name, localID;
  ofstream os;
  isGrpVarNeeded = new short[model->neuronGrpN];
  name = path + toString("/") + model->name + toString("_CODE_CUDA") + toString(deviceID) + toString("/neuron.cu");
  os.open(name.c_str());
  
  // write header content
  writeHeader(os);

  // compiler/include control (include once)
  os << "#ifndef _" << model->name << "_neuronKrnl_cc" << ENDL;
  os << "#define _" << model->name << "_neuronKrnl_cc" << ENDL;
  os << ENDL;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << ENDL;
  os << "/*! \\file neuronKrnl.cc" << ENDL << ENDL;
  os << "\\brief File generated from GeNN for the model " << model->name << " containing the neuron kernel function." << ENDL;
  os << "*/" << ENDL;
  os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

  // global device variables
  os << "// relevant neuron variables" << ENDL;
  os << "__device__ volatile unsigned int d_done" << deviceID << ";" << ENDL;

  for (int i = 0; i < model->neuronGrpN; i++) {
    // conditional skip if neuron group is not on this device
    if ((model->nrnHostID[i] != hostID) || (model->nrnDevID[i] != deviceID)) {
      // ... but not if this host and device recieves this group's spikes
      if ((!model->hostRecvSpkFrom[hostID][i]) || (!model->deviceRecvSpkFrom[deviceID][i])) {
	continue;
      }
    }
    nt = model->neuronType[i];
    isGrpVarNeeded[i] = 0;

    // these now hold just true spikes for benefit of the user (raster plots etc)
    os << "__device__ volatile unsigned int d_glbscnt" << model->neuronName[i] << deviceID << ";" << ENDL;
    os << "__device__ volatile unsigned int d_glbSpk" << model->neuronName[i] << deviceID << "[" << model->neuronN[i] << "];" << ENDL;
    if (model->neuronDelaySlots[i] == 1) { // i.e. no delay
      os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model->neuronName[i] << deviceID << ";" << ENDL;
      os << "__device__ volatile unsigned int d_glbSpkEvnt" << model->neuronName[i] << deviceID << "[" << model->neuronN[i] << "];" << ENDL;
    }
    else {
      os << "__device__ volatile unsigned int d_spkEvntQuePtr" << model->neuronName[i] << deviceID << ";" << ENDL;
      os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model->neuronName[i] << deviceID << "[";
      os << model->neuronDelaySlots[i] << "];" << ENDL;
      os << "__device__ volatile unsigned int d_glbSpkEvnt" << model->neuronName[i] << deviceID << "[";
      os << model->neuronN[i] * model->neuronDelaySlots[i] << "];" << ENDL;
    }
    if (model->neuronType[i] != POISSONNEURON) {
      for (int j = 0; j < model->inSyn[i].size(); j++) {
	os << "__device__ " << model->ftype << " d_inSyn" << model->neuronName[i] << j << deviceID << "[" << model->neuronN[i];
	os << "]; // summed input for neurons in grp " << model->neuronName[i] << ENDL;
      }
    }
    if (model->neuronNeedSt[i]) {
      os << "__device__ volatile " << model->ftype << " d_sT" << model->neuronName[i] << deviceID << "[" << model->neuronN[i] << "];" << ENDL;
    }
    os << ENDL;
  }

  for (int i = 0; i < model->synapseGrpN; i++) {
    // conditional skip if synapse group is not on this device
    if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
      continue;
    }
    if ((model->synapseConnType[i] == SPARSE) && (model->neuronN[model->synapseTarget[i]] > synapseBlkSz[deviceID])) {
      isGrpVarNeeded[model->synapseTarget[i]] = 1; //! Binary flag for the sparse synapses to use atomic operations when the number of connections is bigger than the block size, and shared variables otherwise
    }		
  }

  // kernel header
  os << "__global__ void calcNeuronsCuda" << deviceID << "(" << ENDL;
  for (int i = 0; i < model->neuronGrpN; i++) {
    // conditional skip if neuron group is not on this device
    if ((model->nrnHostID[i] != hostID) || (model->nrnDevID[i] != deviceID)) {
      continue;
    }
    nt = model->neuronType[i];
    if (nt == POISSONNEURON) {
      // Note: Poisson neurons only used as input neurons; they do not receive any inputs
      os << "unsigned int *d_rates" << model->neuronName[i] << deviceID;
      os << ", // poisson \"rates\" of grp " << model->neuronName[i] << ENDL;
      os << "unsigned int offset" << model->neuronName[i];
      os << ", // poisson \"rates\" offset of grp " << model->neuronName[i] << ENDL;
    }
    if (model->receivesInputCurrent[i] > 1) {
      os << "float *d_inputI" << model->neuronName[i] << deviceID;
      os << ", // explicit input current to grp " << model->neuronName[i] << ENDL;    	
    }
    for (int j = 0; j < nModels[nt].varNames.size(); j++) {
      os << nModels[nt].varTypes[j] << " *d_" << nModels[nt].varNames[j];
      os << model->neuronName[i] << deviceID << ", " << ENDL;
    }
    for (int j = 0; j < nModels[nt].extraGlobalNeuronKernelParameters.size(); j++) {
      os << nModels[nt].extraGlobalNeuronKernelParameterTypes[j];
      os << " " << nModels[nt].extraGlobalNeuronKernelParameters[j];
      os << model->neuronName[i] << ", " << ENDL;
    }
  }
  for (int i = 0; i < model->synapseName.size(); i++) {
    // conditional skip if synapse group is not on this device
    if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
      continue;
    }
    int pst = model->postSynapseType[i];
    for (int j = 0; j < postSynModels[pst].varNames.size(); j++) {
      os << postSynModels[pst].varTypes[j] << " *d_" << postSynModels[pst].varNames[j];
      os << model->synapseName[i] << deviceID << ", " << ENDL;
    }
  }
  os << model->ftype << " t // absolute time" << ENDL;
  os << ")" << ENDL;

  // kernel code
  os << OB(5);
  unsigned int neuronGridSz = model->padSumNeuronN[deviceID].back();
  neuronGridSz = neuronGridSz / neuronBlkSz[deviceID];
  if (neuronGridSz < deviceProp[deviceID].maxGridSize[1]) {
    os << "unsigned int id = " << neuronBlkSz[deviceID] << " * blockIdx.x + threadIdx.x;" << ENDL;
  }
  else {
    os << "unsigned int id = " << neuronBlkSz[deviceID] << " * (blockIdx.x * " << ceil(sqrt(neuronGridSz));
    os << " + blockIdx.y) + threadIdx.x;" << ENDL;  	
  }
  //these variables deal with high V "spike type events"
  os << "__shared__ volatile unsigned int posSpkEvnt;" << ENDL;
  os << "__shared__ unsigned int shSpkEvnt[" << neuronBlkSz[deviceID] << "];" << ENDL;
  os << "unsigned int spkEvntIdx;" << ENDL;
  os << "__shared__ volatile unsigned int spkEvntCount;" << ENDL; //was scnt

  //these variables now deal only with true spikes , not high V "events"
  os << "__shared__ unsigned int shSpk[" << neuronBlkSz[deviceID] << "];" << ENDL;
  os << "__shared__ volatile unsigned int posSpk;" << ENDL;
  os << "unsigned int spkIdx;" << ENDL; //was sidx
  os << "__shared__ volatile unsigned int spkCount;" << ENDL; //was scnt
  os << ENDL;

  // reset spike counts and increment spike queue pointers
  os << "if (threadIdx.x == 0) " << OB(7);
  os << "spkEvntCount = 0;" << ENDL;
  os << CB(7);
  os << "if (threadIdx.x == 1) " << OB(8);
  os << "spkCount = 0;" << ENDL;
  os << CB(8);







  /*           MOVE THIS BACK!!!!!!!!!!!!!
  for (int i = 0; i < model->neuronGrpN; i++) {
    if (model->neuronDelaySlots[i] != 1) {
      os << "d_spkEvntQuePtr" << model->neuronName[i] << deviceID << " = (d_spkEvntQuePtr";
      os << model->neuronName[i] << deviceID << " + 1) % " << model->neuronDelaySlots[i] << ";" << ENDL;
    }
  }
  os << CB(7);
  */









  os << "__syncthreads();" << ENDL;
  for (int i = 0; i < model->neuronGrpN; i++) {
    // conditional skip if neuron group is not on this device
    if ((model->nrnHostID[i] != hostID) || (model->nrnDevID[i] != deviceID)) {
      continue;
    }
    nt = model->neuronType[i];
    if (model->localNeuronID[i] == 0) {
      os << "if (id < " << model->padSumNeuronN[deviceID][model->localNeuronID[i]] << ") " << OB(10);
      localID = string("id");
    }
    else {
      os << "if ((id >= " << model->padSumNeuronN[deviceID][model->localNeuronID[i - 1]] << ") && ";
      os << "(id < " << model->padSumNeuronN[deviceID][model->localNeuronID[i]] << ")) " << OB(10);
      os << "unsigned int lid;" << ENDL;
      os << "lid = id - " << model->padSumNeuronN[deviceID][model->localNeuronID[i - 1]] << ";" << ENDL;
      localID = string("lid");
    }
    os << "// only do this for existing neurons" << ENDL;
    os << "if (" << localID << " < " << model->neuronN[i] << ") " << OB(20);
    os << "// pull V values in a coalesced access" << ENDL;
    if (nt == POISSONNEURON) {
      os << "unsigned int lrate = d_rates" << model->neuronName[i] << deviceID;
      os << "[offset" << model->neuronName[i] << " + " << localID << "]";
      if (DT != 0.5) {
	os << " * " << DT / 0.5;
      }
      os << ";" << ENDL;
    }
    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k];
      os << " = d_" <<  nModels[nt].varNames[k] << model->neuronName[i] << deviceID << "[";
      if ((nModels[nt].varNames[k] == "V") && (model->neuronDelaySlots[i] != 1)) {
	os << "(((d_spkEvntQuePtr" << model->neuronName[i] << deviceID << " + " << (model->neuronDelaySlots[i] - 1) << ") % ";
	os << model->neuronDelaySlots[i] << ") * " << model->neuronN[i] << ") + ";
      }
      os << localID << "];" << ENDL;
    }
    if (nt != POISSONNEURON) {
      os << "// pull inSyn values in a coalesced access" << ENDL;
      for (int j = 0; j < model->inSyn[i].size(); j++) {
      	os << model->ftype << " linSyn" << j << " = d_inSyn" << model->neuronName[i] << j << deviceID << "[" << localID << "];" << ENDL;
      }
      os << model->ftype << " Isyn = 0;" << ENDL;
      if (model->inSyn[i].size() > 0) {
	for (int j = 0; j < model->inSyn[i].size(); j++) {
	  os << "// Synapse " << j << " of Population " << i << ENDL;
	  for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	    os << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varTypes[k] << " lps" << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << j;
	    os << " = d_" <<  postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << model->synapseName[model->inSyn[i][j]] << deviceID << "[";
	    os << localID << "];" << ENDL;
	  }
	  os << "Isyn += ";
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
	  os << ";" << ENDL;
	}
      }
    }
    if (model->receivesInputCurrent[i] == 1) { // receives constant input
      os << "Isyn += " << model->globalInp[i] << ";" << ENDL;
    }    	
    if (model->receivesInputCurrent[i] >= 2) { // receives explicit input from file
      os << "Isyn += (" << model->ftype<< ") d_inputI" << model->neuronName[i] << deviceID << "[" << localID << "];" << ENDL;
    }

    // test whether spike condition was fulfilled previously
    string thcode = nModels[nt].thresholdConditionCode;
    if (thcode == tS("")) { // no condition provided
      cerr << "Warning: No thresholdConditionCode for neuron type: " << model->neuronType[i] << " used for ";
      cerr << model->neuronName[i] << " was provided. There will be no spikes detected in this population!" << ENDL;
    }
    else {
      for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	substitute(thcode, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
      }
      for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	substitute(thcode, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model->neuronPara[i][k]));
      }
      for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	substitute(thcode, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model->dnp[i][k]));
      }
      os << "bool oldSpike = (" << thcode << ");" << ENDL;
    }

    // insert the provided neuron model code
    os << "// calculate membrane potential" << ENDL;
    string code = nModels[nt].simCode;
    substitute(code, tS("$(DT)"), tS(model->dt));
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
    for (int k = 0, l = nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
      substitute(code, tS("$(") + nModels[nt].extraGlobalNeuronKernelParameters[k] + tS(")"),
		 nModels[nt].extraGlobalNeuronKernelParameters[k] + model->neuronName[i]);
    }
    os << code;
    os << ENDL;

    // insert condition code provided that tests for a true spike
    if (thcode != tS("")) {
      os << "if ((" << thcode << ") && !(oldSpike)) " << OB(30);
      os << "// register a true spike" << ENDL;
      os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << ENDL;
      os << "shSpk[spkIdx] = " << localID << ";" << ENDL;

    // add optional reset code after a true spike, if provided
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
	os << "// spike reset code" << ENDL;
	os << code << ENDL;
      }
      os << CB(30);
    }

    // test if a spike type event occurred
    os << "if (lV >= " << model->nSpkEvntThreshold[i] << ") " << OB(40);
    os << "// register a spike type event" << ENDL;
    os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << ENDL;
    os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << ENDL;
    os << CB(40);

    // store the defined parts of the neuron state into the global state variables d_V.. etc
    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
      os << "d_" << nModels[nt].varNames[k] <<  model->neuronName[i] << deviceID << "[";
      if ((nModels[nt].varNames[k] == "V") && (model->neuronDelaySlots[i] != 1)) {
	os << "(d_spkEvntQuePtr" << model->neuronName[i] << deviceID << " * " << model->neuronN[i] << ") + ";
      }
      os << localID << "] = l" << nModels[nt].varNames[k] << ";" << ENDL;
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
      os << "d_inSyn"  << model->neuronName[i] << j << deviceID << "[" << localID << "] = linSyn"<< j << ";" << ENDL;
      for (int k = 0, l = postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames.size(); k < l; k++) {
	os << "d_" <<  postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << model->synapseName[model->inSyn[i][j]] << deviceID << "[";
	os << localID << "] = lps" << postSynModels[model->postSynapseType[model->inSyn[i][j]]].varNames[k] << j << ";"<< ENDL;
      }
    }
    os << CB(20);

    os << "__syncthreads();" << ENDL;
    os << "if (threadIdx.x == 0) " << OB(50);
    os << "if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &d_glbSpkEvntCnt" << model->neuronName[i] << deviceID;
    if (model->neuronDelaySlots[i] != 1) {
      os << "[d_spkEvntQuePtr" << model->neuronName[i] << deviceID << "]";
    }
    os << ", spkEvntCount);" << ENDL;
    os << "if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &d_glbscnt" << model->neuronName[i] << deviceID << ", spkCount);" << ENDL;
    os << CB(50); // end if (threadIdx.x == 0)

    os << "__syncthreads();" << ENDL;
    os << "if (threadIdx.x < spkEvntCount) " << OB(60);
    os << "d_glbSpkEvnt" << model->neuronName[i] << deviceID << "[";
    if (model->neuronDelaySlots[i] != 1) {
      os << "(d_spkEvntQuePtr" << model->neuronName[i] << deviceID << " * " << model->neuronN[i] << ") + ";
    }
    os << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << ENDL;
    os << CB(60);

    os << "if (threadIdx.x < spkCount) " << OB(70);
    os << "d_glbSpk" << model->neuronName[i] << deviceID << "[";
    os << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << ENDL;
    if (model->neuronNeedSt[i]) {
      os << "d_sT" << model->neuronName[i] << deviceID << "[shSpk[threadIdx.x]] = t;" << ENDL;
    }
    os << CB(70);

    os << CB(10);
    os << ENDL;
  }
  os << CB(5);
  os << ENDL;
  os << "#endif" << ENDL;
  os.close();
}


//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are synapse-related and the actual CUDA kernel for simulating one time step of the synapses.
*/
//-------------------------------------------------------------------------

void genCudaSynapse(unsigned int deviceID, //!< device number to generate code for
		    ostream &mos //!< output stream for messages
		    )
{
  // abort if no synapse groups are on this device
  if (model->padSumSynapseKrnl[deviceID].empty()) return;

  string name, localID, theLG;
  unsigned int numOfBlocks, trgN, nt;
  ofstream os;

  name = path + toString("/") + model->name + toString("_CODE_CUDA") + toString(deviceID) + toString("/synapse.cu");
  os.open(name.c_str());

  // write header content
  writeHeader(os);

  // count how many neuron blocks to use: one thread for each synapse target
  // targets of several input groups are counted multiply
  numOfBlocks = model->padSumSynapseKrnl[deviceID].back() / synapseBlkSz[deviceID];

  // compiler/include control (include once)
  os << "#ifndef _" << model->name << "_synapseKrnl_cc" << ENDL;
  os << "#define _" << model->name << "_synapseKrnl_cc" << ENDL;
  os << "#define BLOCKSZ_SYN " << synapseBlkSz[deviceID] << ENDL;
  os << ENDL;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << ENDL;
  os << "/*! \\file synapseKrnl.cc" << ENDL << ENDL;
  os << "\\brief File generated from GeNN for the model " << model->name;
  os << " containing the synapse kernel and learning kernel functions." << ENDL;
  os << "*/" << ENDL;
  os << "//-------------------------------------------------------------------------" << ENDL;
  os << ENDL;

  // Kernel header
  unsigned int src;
  os << "__global__ void calcSynapsesCuda" << deviceID << "(" << ENDL;
  for (int i = 0; i < model->synapseGrpN; i++) {
    // conditional skip if synapse group is not on this device
    if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
      continue;
    }
    if (model->synapseGType[i] == INDIVIDUALG) {
      os << model->ftype << " * d_gp" << model->synapseName[i] << deviceID << ", " << ENDL;	
    }
    if (model->synapseConnType[i] == SPARSE) {
      os << "unsigned int * d_gp" << model->synapseName[i] << "_ind" << deviceID << ", " << ENDL;
      os << "unsigned int * d_gp" << model->synapseName[i] << "_indInG" << deviceID << ", " << ENDL;
      trgN = model->neuronN[model->synapseTarget[i]];
    }
    if (model->synapseGType[i] == (INDIVIDUALID )) {
      os << "unsigned int *d_gp" << model->synapseName[i] << deviceID << ", " << ENDL;	
    }
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << model->ftype << " * d_grawp" << model->synapseName[i] << deviceID << ", " << ENDL;
    }   	
  }
  for (int i = 0; i < model->neuronGrpN; i++) {
    // conditional skip if neuron group is not on this device
    if ((model->nrnHostID[i] != hostID) || (model->nrnDevID[i] != deviceID)) {
      // ... but not if this host and device recieves this group's spikes
      if ((!model->hostRecvSpkFrom[hostID][i]) || (!model->deviceRecvSpkFrom[deviceID][i])) {
	continue;
      }
    }
    nt = model->neuronType[i];
    os << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model->neuronName[i] << deviceID; // Vm
    if (i < (model->neuronGrpN - 1) || model->needSt) os << "," << ENDL;
  }
  if (model->needSt) {
    os << model->ftype << " t";
  }
  os << ENDL << ")";
  os << ENDL;

  // kernel code
  os << OB(75);
  os << "unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;" << ENDL;
  os << "__shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];" << ENDL;
  os << "__shared__ " << model->ftype << " shSpkEvntV[BLOCKSZ_SYN];" << ENDL;
  os << "volatile __shared__ " << model->ftype << " shLg[" << neuronBlkSz[deviceID] << "];" << ENDL;
  os << "unsigned int lscntEvnt, numSpikeEvntSubsets, lmax, j, p, r, ipost, npost;" << ENDL;
  for (int i = 0; i < model->synapseGrpN; i++) {
    // conditional skip if synapse group is not on this device
    if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
      continue;
    }
    if ((model->synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << model->ftype << " linSyn, lg;" << ENDL;
      break;
    }
  }
  for (int i = 0; i < model->synapseGrpN; i++) {
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << "__shared__ unsigned int shSpk[BLOCKSZ_SYN];" << ENDL;
      os << "__shared__ " << model->ftype << " shSpkV[BLOCKSZ_SYN];" << ENDL;
      os << "unsigned int lscnt, numSpikeSubsets;" << ENDL;
      break;
    }
  }
  if (model->needSynapseDelay == 1) {
    os << "int delaySlot;" << ENDL;
  }
  os << ENDL;
  os << "ipost = 0;" << ENDL;

  for (int i = 0; i < model->synapseGrpN; i++) {
    // conditional skip if synapse group is not on this device
    if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
      continue;
    }
    if (model->localSynapseID[i] == 0) {
      os << "if (id < " << model->padSumSynapseKrnl[deviceID][model->localSynapseID[i]] << ") " << OB(77);
      os << " // synapse group " << model->synapseName[i] << ENDL;
      localID = string("id");
    }
    else {
      os << "if ((id >= " << model->padSumSynapseKrnl[deviceID][model->localSynapseID[i - 1]] << ") && ";
      os << "(id < " << model->padSumSynapseKrnl[deviceID][model->localSynapseID[i]] << ")) " << OB(77);
      os << " // synapse group " << model->synapseName[i] << ENDL;
      os << "unsigned int lid;" << ENDL;
      os << "lid = id - " << model->padSumSynapseKrnl[deviceID][model->localSynapseID[i - 1]] << ";" << ENDL;
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
      os << "delaySlot = (d_spkEvntQuePtr" << model->neuronName[src] << deviceID << " + ";
      os << (int) (model->neuronDelaySlots[src] - model->synapseDelay[i] + 1);
      os << ") % " << model->neuronDelaySlots[src] << ";" << ENDL;
    }
    if ((model->synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << "// only do this for existing neurons" << ENDL;
      os << "if (" << localID << " < " << nN << ") " << OB(80);
      os << "linSyn = d_inSyn" << model->neuronName[trg] << inSynNo << deviceID << "[" << localID << "];" << ENDL;
      os << CB(80);
    }
    os << "lscntEvnt = d_glbSpkEvntCnt" << model->neuronName[src] << deviceID;
    if (model->neuronDelaySlots[src] != 1) {
      os << "[delaySlot]";
    }
    os << ";" << ENDL;
    os << "numSpikeEvntSubsets = (unsigned int) (ceilf((float) lscntEvnt / (float) BLOCKSZ_SYN));" << ENDL;
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << "lscnt = d_glbscnt" << model->neuronName[src];
      if (model->neuronDelaySlots[src] != 1) {
	os << "[delaySlot]";
      }
      os << ";" << ENDL;
      os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / (float) BLOCKSZ_SYN));" << ENDL;
    }
    os << "for (r = 0; r < numSpikeEvntSubsets; r++) " << OB(90);
    os << "if (r == numSpikeEvntSubsets - 1) lmax = lscntEvnt % BLOCKSZ_SYN;" << ENDL;
    os << "else lmax = BLOCKSZ_SYN;" << ENDL;
    os << "if (threadIdx.x < lmax) " << OB(100);
    os << "shSpkEvnt[threadIdx.x] = d_glbSpkEvnt" << model->neuronName[src] << deviceID << "[";
    if (model->neuronDelaySlots[src] != 1) {
      os << "(delaySlot * " << model->neuronN[src] << ") + ";
    }
    os << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << "shSpk[threadIdx.x] = d_glbSpk" << model->neuronName[src] << "[";
      if (model->neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model->neuronN[src] << ") + ";
      }
      os << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
    }
    if (model->neuronType[src] != POISSONNEURON) {
      os << "shSpkEvntV[threadIdx.x] = d_V" << model->neuronName[src] << deviceID << "[";
      if (model->neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model->neuronN[src] << ") + ";
      }
      os << "shSpkEvnt[threadIdx.x]];" << ENDL;
      if (model->synapseType[i] == LEARN1SYNAPSE) {
	os << "shSpkV[threadIdx.x] = d_V" << model->neuronName[src] << "[";
	if (model->neuronDelaySlots[src] != 1) {
	  os << "(delaySlot * " << model->neuronN[src] << ") + ";
	}
	os << "shSpk[threadIdx.x]];" << ENDL;
      }
    }
    os << CB(100);
    if ((model->synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << "if (threadIdx.x < " << neuronBlkSz[deviceID] << ") shLg[threadIdx.x] = 0;" << ENDL;
    }
    os << "__syncthreads();" << ENDL;
    os << "// only work on existing neurons" << ENDL;
    if ((model->synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model->synapseTarget[i]] == 1)) {
      if (model->maxConn.size() == 0) {
	fprintf(stderr,"Model Generation error: for every SPARSE synapse group used you must also supply (in your model) a max possible number of connections via the model->setMaxConn() function.");
	exit(1);
      }
      int maxConnections = model->maxConn[i];
      os << "if (" << localID << " < " << maxConnections << ") " << OB(110);
    }
    else {
      os << "if (" << localID << " < " << model->neuronN[trg] << ") " << OB(110);
    }
    os << "// loop through all incoming spikes" << ENDL;
    os << "for (j = 0; j < lmax; j++) " << OB(120);
    if (model->synapseGType[i] == INDIVIDUALID) {
      os << "unsigned int gid = (shSpkEvnt[j] * " << model->neuronN[trg];
      os << " + " << localID << ");" << ENDL;
    }
    if (model->neuronType[src] != POISSONNEURON) {
      os << "if ";
      if (model->synapseGType[i] == INDIVIDUALID) {
	// Note: we will just access global mem. For compute >= 1.2
	// simultaneous access to same global mem in the (half-)warp
	// will be coalesced - no worries
	os << "((B(d_gp" << model->synapseName[i] << deviceID << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz - 1 << ")) && ";
      } 
      os << "(shSpkEvntV[j] > " << Epre << ")";
      if (model->synapseGType[i] == INDIVIDUALID) {
	os << ")";
      }
      os << " " << OB(130);
    }
    else {
      if (model->synapseGType[i] == INDIVIDUALID) {
	os << "if (B(d_gp" << model->synapseName[i] << deviceID << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz - 1 << ")) " << OB(135);
      }
    }
    if (model->synapseConnType[i] == SPARSE) {
      os << "npost = d_gp" << model->synapseName[i] << "_indInG" << deviceID << "[shSpkEvnt[j] + 1] - d_gp";
      os << model->synapseName[i] << "_indInG" << deviceID << "[shSpkEvnt[j]];" << ENDL;
      os << "if (" << localID << " < npost) " << OB(140);
      os << "ipost = d_gp" << model->synapseName[i] << "_ind" << deviceID << "[d_gp";
      os << model->synapseName[i] << "_indInG" << deviceID << "[shSpkEvnt[j]] + " << localID << "];" << ENDL;
      if (isGrpVarNeeded[model->synapseTarget[i]] == 0) {
	theLG = toString("shLg[" + localID + "]");
	if (model->synapseGType[i] != INDIVIDUALG) {
	  os << "shLg[ipost] += " << model->g0[i] << "];";
	}
	else {
	  os << "shLg[ipost] += d_gp" << model->synapseName[i] << deviceID << "[d_gp" << model->synapseName[i] << "_indInG" << deviceID << "[shSpkEvnt[j]] + " << localID << "];";
	}
      }
      else {
	if (model->synapseGType[i] != INDIVIDUALG) {
	  os << "atomicAdd(&d_inSyn" << model->neuronName[trg] << inSynNo << deviceID << "[ipost], " << model->g0[i] << ");";
	}
	else{
	  os << "atomicAdd(&d_inSyn" << model->neuronName[trg] << inSynNo << deviceID << "[ipost], d_gp";
	  os << model->synapseName[i] << deviceID << "[d_gp" << model->synapseName[i] << "_indInG" << deviceID << "[shSpkEvnt[j]] + " << localID << "]);" << ENDL;
	}
      }
      os << CB(140); // end if (id < npost)
      if (model->neuronType[src] != POISSONNEURON) {
        os << CB(130) << ENDL; // end if (shSpkEvntV[j] > postthreshold)
      }
      else {
	if (model->synapseGType[i] == INDIVIDUALID) {
	  os << CB(135) << ENDL; // end if (B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	}
      }
      if (isGrpVarNeeded[model->synapseTarget[i]] == 0) {	
        os << "__syncthreads();" << ENDL;
      }
    }
    else {
      if (model->synapseGType[i] == INDIVIDUALG) {
	os << "lg = d_gp" << model->synapseName[i] << deviceID << "[shSpkEvnt[j]*" << model->neuronN[trg] << " + " << localID << "];";
	theLG = toString("lg");
      }
    }
    os << ENDL;

    if ((model->synapseGType[i] == GLOBALG) || (model->synapseGType[i] == INDIVIDUALID)) {
      theLG = toString(model->g0[i]);
    }
    if ((model->synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      if ((model->synapseType[i] == NSYNAPSE) || (model->synapseType[i] == LEARN1SYNAPSE)) {
	os << "linSyn = linSyn + " << theLG << "; " << ENDL;
      }
      if (model->synapseType[i] == NGRADSYNAPSE) {
      	if (model->neuronType[src] == POISSONNEURON) {
	  os << "linSyn = linSyn + " << theLG << " * tanh((";
	  os << SAVEP(model->neuronPara[src][2]) << " - " << SAVEP(Epre);
      	}
      	else {
	  os << "linSyn = linSyn + " << theLG << " * tanh((shSpkEvntV[j] - " << SAVEP(Epre);
      	}
      	os << ") / " << Vslope << ");" << ENDL;
      }
    }
    if ((model->synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << theLG << " = 0;" << ENDL; 
      os << "__syncthreads();" << ENDL;
    }
    if (model->synapseConnType[i] != SPARSE) {
      if (model->neuronType[src] != POISSONNEURON) {
	os << CB(130) << ENDL; // end if (shSpkEvntV[j] > postthreshold)
      }
      else {
	if (model->synapseGType[i] == INDIVIDUALID) {
	  os << CB(135) << ENDL; // end if (B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid
	}
      }
    }
    os << CB(120);
    os << CB(110);
    os << CB(90);

    // if needed, do some learning (this is for pre-synaptic spikes)
    if (model->synapseType[i] == LEARN1SYNAPSE) {
      os << "for (r = 0; r < numSpikeSubsets; r++) " << OB(2090);
      os << "if (r == numSpikeSubsets - 1) lmax = lscnt % BLOCKSZ_SYN;" << ENDL;
      os << "else lmax = BLOCKSZ_SYN;" << ENDL;
      os << "if (threadIdx.x < lmax) " << OB(2100);
      os << "shSpk[threadIdx.x] = d_glbSpk" << model->neuronName[src] << "[";
      if (model->neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model->neuronN[src] << ") + ";
      }
      os << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
      os << "shSpk[threadIdx.x] = d_glbSpk" << model->neuronName[src] << "[";
      if (model->neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model->neuronN[src] << ") + ";
      }
      os << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
      if (model->neuronType[src] != POISSONNEURON) {
	os << "shSpkV[threadIdx.x] = d_V" << model->neuronName[src] << "[";
	if (model->neuronDelaySlots[src] != 1) {
	  os << "(delaySlot * " << model->neuronN[src] << ") + ";
	}
	os << "shSpk[threadIdx.x]];" << ENDL;
	os << "shSpkV[threadIdx.x] = d_V" << model->neuronName[src] << "[";
	if (model->neuronDelaySlots[src] != 1) {
	  os << "(delaySlot * " << model->neuronN[src] << ") + ";
	}
	os << "shSpk[threadIdx.x]];" << ENDL;
      }
      os << CB(2100);
      if ((model->synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
	os << "if (threadIdx.x < " << neuronBlkSz << ") shLg[threadIdx.x] = 0;" << ENDL;
      }
      os << "__syncthreads();" << ENDL;
      os << "// only work on existing neurons" << ENDL;
      if ((model->synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model->synapseTarget[i]] == 1)) {
	if(model->maxConn.size() == 0) {
	  fprintf(stderr, "Model Generation error: for every SPARSE synapse group used you must also supply (in your model) a max possible number of connections via the model->setMaxConn() function.");
	  exit(1);
	}
	int maxConnections  = model->maxConn[i];
	os << "if (" << localID << " < " << maxConnections << ") " << OB(2110);
      }
      else {
	os << "if (" << localID << " < " << model->neuronN[trg] << ") " << OB(2110);
      }
      os << "// loop through all incoming spikes" << ENDL;
      os << "for (j = 0; j < lmax; j++) " << OB(2120);
      if (model->synapseGType[i] == INDIVIDUALID) {
	os << "unsigned int gid = (shSpk[j] * " << model->neuronN[trg];
	os << " + " << localID << ");" << ENDL;
      }
      if (model->neuronType[src] != POISSONNEURON) {
	os << "if ";
	if (model->synapseGType[i] == INDIVIDUALID) {
	  // Note: we will just access global mem. For compute >= 1.2
	  // simultaneous access to same global mem in the (half-)warp
	  // will be coalesced - no worries
	  os << "((B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	  os << UIntSz - 1 << ")) && ";
	}
	os << "(shSpkV[j] > " << Epre << ")";
	if (model->synapseGType[i] == INDIVIDUALID) {
	  os << ")";
	}
	os << " " << OB(1130);
      }
      else {
	if (model->synapseGType[i] == INDIVIDUALID) {
	  os << "if (B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	  os << UIntSz - 1 << ")) " << OB(1135);
	}
      }
      //we may need something like the following, if sparse:
      /*
	if (model->synapseConnType[i] == SPARSE) {
	os << "npost = d_gp" << model->synapseName[i] << "_indInG[shSpkEvnt[j] + 1] - d_gp";
	os << model->synapseName[i] << "_indInG[shSpkEvnt[j]];" << ENDL;
	os << "if (" << localID << " < npost) " << OB(140);
	os << "ipost = d_gp" << model->synapseName[i] << "_ind[d_gp";
	os << model->synapseName[i] << "_indInG[shSpkEvnt[j]] + " << localID << "];" << ENDL;
	if (isGrpVarNeeded[model->synapseTarget[i]] == 0) {
	theLG = tS("shLg[" + localID + "]");
      */

      // simply assume INDIVIDUALG for now
      os << "lg = d_grawp" << model->synapseName[i] << deviceID << "[shSpk[j] * " << model->neuronN[trg] << " + " << localID << "];" << ENDL;
      os << model->ftype << " dt = d_sT" << model->neuronName[trg] << deviceID << "[" << localID << "] - t - ";
      os << SAVEP(model->synapsePara[i][11]) << ";" << ENDL;
      os << "if (dt > " << model->dsp[i][1] << ") " << OB(150);
      os << "dt = - " << SAVEP(model->dsp[i][5]) << ";" << ENDL;
      os << CB(150);
      os << "else if (dt > 0.0) " << OB(160);
      os << "dt = " << SAVEP(model->dsp[i][3]) << " * dt + " << SAVEP(model->dsp[i][6]) << ";" << ENDL;
      os << CB(160);
      os << "else if (dt > " << model->dsp[i][2] << ") " << OB(170);
      os << "dt = " << SAVEP(model->dsp[i][4]) << " * dt + " << SAVEP(model->dsp[i][6]) << ";" << ENDL;
      os << CB(170);
      os << "else " << OB(180);
      os << "dt = - " << SAVEP(model->dsp[i][7]) << ";" << ENDL;
      os << CB(180);
      os << "lg = lg + dt;" << ENDL;
      os << "d_grawp" << model->synapseName[i] << deviceID << "[shSpk[j] * " << model->neuronN[trg] << " + " << localID << "] = lg;" << ENDL;
      os << "d_gp" << model->synapseName[i] << deviceID << "[shSpk[j] * " << model->neuronN[trg] << " + " << localID << "] = ";
      os << "gFunc" << model->synapseName[i] << "Cuda" << deviceID << "(lg);" << ENDL; 
      if (model->synapseConnType[i] != SPARSE) {
	if (model->neuronType[src] != POISSONNEURON) {
	  os << CB(1130) << ENDL; // end if (shSpkEvntV[j] > postthreshold)
	}
	else {
	  if (model->synapseGType[i] == INDIVIDUALID) {
	    os << CB(1135) << ENDL; // end if (B(d_gp" << model->synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	  }
	}
      }
      os << CB(2120); //// 2 for (j = 0; j < lmax; j++)
      os << CB(2110); //// 3 if (id < Npre)
      os << CB(2090); //// 4 for (r = 0; r < numSpikeEvntSubsets; r++)
    }

    if ((model->synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model->synapseTarget[i]] == 0)) {
      os << "// only do this for existing neurons" << ENDL;
      os << "if (" << localID << " < " << model->neuronN[trg] << ") " << OB(190);
      os << "d_inSyn" << model->neuronName[trg] << inSynNo << deviceID << "[" << localID << "] = linSyn;" << ENDL;
      os << CB(190);
    }
    if (model->lrnGroups == 0) {
      os << "if (threadIdx.x == 0) " << OB(200);
      os << "j = atomicAdd((unsigned int *) &d_done" << deviceID << ", 1);" << ENDL;
      os << "if (j == " << numOfBlocks - 1 << ") " << OB(210);
      for (int j = 0; j < model->neuronGrpN; j++) {
	os << "d_glbscnt" << model->neuronName[j] << deviceID << " = 0;" << ENDL;
	if (model->neuronDelaySlots[j] != 1) {
	  os << "d_glbSpkEvntCnt" << model->neuronName[j] << deviceID << "[d_spkEvntQuePtr";
	  os << model->neuronName[j] << deviceID << "] = 0;" << ENDL;
	}
	else {
	  os << "d_glbSpkEvntCnt" << model->neuronName[j] << deviceID << " = 0;" << ENDL;
	}
      }
      os << "d_done" << deviceID << " = 0;" << ENDL;
      os << CB(210);
      os << CB(200);
    }
    os << CB(77);
    os << ENDL;
  }
  os << CB(75);
  os << ENDL;



  ///////////////////////////////////////////////////////////////
  // Kernel for learning synapses, post-synaptic spikes

  if ((model->lrnGroups > 0) && (!model->padSumLearnN[deviceID].empty())) {

    // count how many learn blocks to use: one thread for each synapse source
    // sources of several output groups are counted multiply
    numOfBlocks = model->padSumLearnN[deviceID].back() / learnBlkSz[deviceID];

    // Kernel header
    os << "__global__ void learnSynapsesPostCuda" << deviceID << "(" << ENDL;
    for (int i = 0; i < model->synapseGrpN; i++) {
      // conditional skip if synapse group is not on this device
      if ((model->synHostID[i] != hostID) || (model->synDevID[i] != deviceID)) {
	continue;
      }
      if (model->synapseGType[i] == (INDIVIDUALG )) {
	os << model->ftype << " *d_gp" << model->synapseName[i] << deviceID << ", " << ENDL;	
      }
      if (model->synapseGType[i] == (INDIVIDUALID )) {
	os << "unsigned int *d_gp" << model->synapseName[i] << deviceID << ", " << ENDL;	
      }
      if (model->synapseType[i] == LEARN1SYNAPSE) {
	os << model->ftype << " *d_grawp"  << model->synapseName[i] << deviceID << ", " << ENDL;
      }
    }
    for (int i = 0; i < model->neuronGrpN; i++) {
      // conditional skip if neuron group is not on this device
      if ((model->nrnHostID[i] != hostID) || (model->nrnDevID[i] != deviceID)) {
	continue;
      }
      nt = model->neuronType[i];
      os << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model->neuronName[i] << deviceID << ", "; // Vm
    }
    os << model->ftype << " t" << ENDL;
    os << ")" << ENDL;

    // kernel code
    os << OB(215);
    os << "unsigned int id = " << learnBlkSz[deviceID] << " * blockIdx.x + threadIdx.x;" << ENDL;
    os << "__shared__ unsigned int shSpk[" << learnBlkSz[deviceID] << "];" << ENDL;
    os << "__shared__ " << model->ftype << " shSpkV[" << learnBlkSz[deviceID] << "];" << ENDL;
    os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << ENDL;
    os << model->ftype << " lg;" << ENDL;
    os << ENDL;

    for (int i = 0; i < model->lrnGroups; i++) {
      // conditional skip if plastic synapse group is not on this device
      if ((model->synHostID[model->lrnSynGrp[i]] != hostID) || (model->synDevID[model->lrnSynGrp[i]] != deviceID)) {
	continue;
      }
      if (model->localLearnID[i] == 0) {
	os << "if (id < " << model->padSumLearnN[deviceID][model->localLearnID[i]] << ") " << OB(220);
	localID = string("id");
      }
      else {
	os << "if ((id >= " << model->padSumLearnN[deviceID][model->localLearnID[i - 1]] << ") && ";
	os << "(id < " << model->padSumLearnN[deviceID][model->localLearnID[i]] << ")) " << OB(220);
	os << "unsigned int lid;" << ENDL;
	os << "lid = id - " << model->padSumLearnN[deviceID][model->localLearnID[i - 1]] << ";" << ENDL;
	localID = string("lid");
      }
      unsigned int k = model->lrnSynGrp[i];
      unsigned int src = model->synapseSource[k];
      unsigned int nN = model->neuronN[src];
      unsigned int trg = model->synapseTarget[k];
      float Epre = model->synapsePara[k][1];
      os << "lscnt = d_glbscnt" << model->neuronName[trg] << deviceID;
      if (model->neuronDelaySlots[trg] != 1) os << "[d_spkQuePtr" << model->neuronName[trg] << deviceID << "]";
      os << ";" << ENDL;
      os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << learnBlkSz[deviceID] << ".0f));" << ENDL;
      os << "for (r = 0; r < numSpikeSubsets; r++) " << OB(230);
      os << "if (r == numSpikeSubsets - 1) lmax = lscnt % " << learnBlkSz[deviceID] << ";" << ENDL;
      os << "else lmax = " << learnBlkSz[deviceID] << ";" << ENDL;
      os << "if (threadIdx.x < lmax) " << OB(240);
      os << "shSpk[threadIdx.x] = d_glbSpk" << model->neuronName[trg] << deviceID << "[";
      if (model->neuronDelaySlots[trg] != 1) {
	os << "(d_spkQuePtr" << model->neuronName[trg] << deviceID << " * " << model->neuronN[trg] << ") + ";
      }
      os << "(r * " << learnBlkSz[deviceID] << ") + threadIdx.x];" << ENDL;
      os << "shSpkV[threadIdx.x] = d_V" << model->neuronName[trg] << deviceID << "[";
      if (model->neuronDelaySlots[trg] != 1) {
	os << "(d_spkQuePtr" << model->neuronName[trg] << deviceID << " * " << model->neuronN[trg] << ") + ";
      }
      os << "shSpk[threadIdx.x]];" << ENDL;
      os << CB(240);
      os << "__syncthreads();" << ENDL;
      os << "// only work on existing neurons" << ENDL;
      os << "if (" << localID << " < " << model->neuronN[src] << ") " << OB(250);
      os << "// loop through all incoming spikes for learning" << ENDL;
      os << "for (j = 0; j < lmax; j++) " << OB(260);
      os << "if (shSpkV[j] > " << Epre << ")" << OB(270); //!TODO: shouldn't that be something equivalent to Epost???
      os << "lg = d_grawp" << model->synapseName[k] << deviceID << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpk[j]];" << ENDL;
      os << model->ftype << " dt = t - d_sT" << model->neuronName[src] << deviceID << "[" << localID << "]";
      if (model->neuronDelaySlots[src] != 1) {
	os << " + " << (model->dt * model->synapseDelay[k]);
      }
      os << " - " << SAVEP(model->synapsePara[k][11]) << ";" << ENDL;
      os << "if (dt > " << model->dsp[k][1] << ") " << OB(280);
      os << "dt = - " << SAVEP(model->dsp[k][5]) << ";" << ENDL;
      os << CB(280);
      os << "else if (dt > 0.0) " << OB(290);
      os << "dt = " << SAVEP(model->dsp[k][3]) << " * dt + " << SAVEP(model->dsp[k][6]) << ";" << ENDL;
      os << CB(290);
      os << "else if (dt > " << model->dsp[k][2] << ") " << OB(300);
      os << "dt = " << SAVEP(model->dsp[k][4]) << " * dt + " << SAVEP(model->dsp[k][6]) << ";" << ENDL;
      os << CB(300);
      os << "else " << OB(310);
      os << "dt = - " << SAVEP(model->dsp[k][7]) << ";" << ENDL;
      os << CB(310);
      os << "lg = lg + dt;" << ENDL;
      os << "d_grawp" << model->synapseName[k] << deviceID << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpk[j]] = lg;" << ENDL;
      os << "d_gp" << model->synapseName[k] << deviceID << "[" << localID << " * ";
      os << model->neuronN[trg] << " + shSpk[j]] = gFunc" << model->synapseName[k] << "Cuda" << deviceID << "(lg);" << ENDL;
      os << CB(270);
      os << CB(260);
      os << CB(250);
      os << CB(230);
      os << "__threadfence();" << ENDL;
      os << "if (threadIdx.x == 0) " << OB(320);
      os << "j = atomicAdd((unsigned int *) &d_done" << deviceID << ", 1);" << ENDL;
      os << "if (j == " << numOfBlocks - 1 << ") " << OB(330);
      for (int j = 0; j < model->neuronGrpN; j++) {
	os << "d_glbscnt" << model->neuronName[j] << deviceID << " = 0;" << ENDL;
	if (model->neuronDelaySlots[j] != 1) {
	  os << "d_spkQuePtr" << model->neuronName[j] << deviceID << " = (d_spkQuePtr";
	  os << model->neuronName[j] << deviceID << " + 1) % " << model->neuronDelaySlots[j] << ";" << ENDL;
	  os << "d_glbSpkEvntCnt" << model->neuronName[j] << deviceID << "[d_spkQuePtr";
	  os << model->neuronName[j] << deviceID << "] = 0;" << ENDL;
	}
	else {
	  os << "d_glbSpkEvntCnt" << model->neuronName[j] << deviceID << " = 0;" << ENDL;
	}
      }
      os << "d_done" << deviceID << " = 0;" << ENDL;
      os << CB(330);
      os << CB(320);
      os << CB(220);
    }
    os << CB(215);
  }
  os << ENDL;

  os << "#endif" << ENDL;
  os.close();
}
