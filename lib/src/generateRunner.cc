/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//-----------------------------------------------------------------------
/*!  \file generateRunner.cc 
  
  \brief Contains functions to generate code for running the
  simulation on the GPU, and for I/O convenience functions between GPU
  and CPU space. Part of the code generation section.
*/
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
/*!
  \brief A function that generates predominantly host-side code.

  In this function host-side functions and other code are generated,
  including: Global host variables, "allocatedMem()" function for
  allocating memories, "freeMem" function for freeing the allocated
  memories, "initialize" for initializing host variables, "gFunc" and
  "initGRaw()" for use with plastic synapses if such synapses exist in
  the model.  
*/
//--------------------------------------------------------------------------

void genRunner(NNmodel &model, //!< Model description
	       string path, //!< path for code generation
	       ostream &mos //!< output stream for messages
    )
{
    string name;
    unsigned int nt;
    unsigned int mem = 0;
    float memremsparse= 0;
    int trgN;
    ofstream os;
    
    //initializing learning parameters to start
    model.initLearnGrps();  //Putting this here for the moment. Makes more sense to call it at the end of ModelDefinition, but this leaves the initialization to the user.
    
    cout << "entering genRunner" << endl;
    name= path + toString("/") + model.name + toString("_CODE/runner.cc");
    os.open(name.c_str());  
    writeHeader(os);
    os << endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << endl;
    os << "/*! \\file runner.cc" << endl << endl;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing general control code used for both GPU amd CPU versions." << endl;
    os << "*/" << endl;
    os << "//-------------------------------------------------------------------------" << endl << endl;

  os << "#include <cstdio>" << endl << endl;
  os << "#include <cassert>" << endl << endl;
  os << "#include <stdint.h>" << endl << endl;

  // write CUDA error handler macro
  os << "/*" << endl;
  os << "  CUDA error handling macro" << endl;
  os << "  -------------------------" << endl;
  os << "*/" << endl;
  os << "#ifndef CHECK_CUDA_ERRORS" << endl;
  os << "#define CHECK_CUDA_ERRORS(call) {\\" << endl;
  os << "  cudaError_t error = call;\\" << endl;
  os << "  if (error != cudaSuccess) {\\" << endl;
  os << "    fprintf(stderr, \"%s: %i: cuda error %i: %s\\n\", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\\" << endl;
  os << "    exit(EXIT_FAILURE);\\" << endl;
  os << "  }\\" << endl;
  os << "}" << endl;
  os << "#endif" << endl << endl;

  os << "template<class T>" << endl;
  os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)" << endl;
  os << "{" << endl;
  os << "  void *devptr;" << endl;
  os << "  CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << endl;
  os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << endl;
  os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << endl;
  os << "}" << endl << endl;

  os << "void convertProbabilityToRandomNumberThreshold(float *p_pattern, " << model.RNtype << " *pattern, int N)" << endl;
os << "{" << endl;
os << "    double fac= pow(2.0, (int) sizeof(" << model.RNtype << ")*8-16)*DT;" << endl;
os << "    for (int i= 0; i < N; i++) {" << endl;
//os << "	assert(p_pattern[i] <= 1.0);" << endl;
os << "	pattern[i]= (" << model.RNtype << ") (p_pattern[i]*fac);" << endl;
os << "    }" << endl;
os << "}" << endl;

    // global host variables (matching some of the device ones)  
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	os << "unsigned int glbscnt" << model.neuronName[i] << ";" << endl;
	os << "unsigned int *glbSpk" << model.neuronName[i] << ";" << endl;

	if (model.neuronDelaySlots[i] == 1) {
	    os << "unsigned int glbSpkEvntCnt" << model.neuronName[i] << ";" << endl;
	    os << "unsigned int *glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	}
	else {
	    os << "unsigned int spkEvntQuePtr" << model.neuronName[i] << ";" << endl;
	    os << "unsigned int *glbSpkEvntCnt" << model.neuronName[i] << ";" << endl;
	    os << "unsigned int *glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	}
	for (int j= 0; j < model.inSyn[i].size(); j++) {
	    os << model.ftype << " *inSyn" << model.neuronName[i] << j << ";" << endl;
	} 
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " *";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	}
	// write global variables for the extra global neuron kernel parameters. These are assumed not to be pointers, if they are the user needs to take care of allocation etc
	for (int k= 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    os << nModels[nt].extraGlobalNeuronKernelParameterTypes[k] << " ";
	    os << nModels[nt].extraGlobalNeuronKernelParameters[k] << model.neuronName[i] << ";" << endl;
	}
    
	if (model.neuronNeedSt[i]) {
	    os << model.ftype << " *sT" << model.neuronName[i] << ";" << endl;
	}
    }
  
    for (int i=0; i< model.synapseName.size(); i++){
    	int st= model.synapseType[i];
    	if (st >= MAXSYN){
    		for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
    			os << weightUpdateModels[st-MAXSYN].varTypes[k] << " *" << weightUpdateModels[st-MAXSYN].varNames[k];
    			os << model.synapseName[i]<< ";" << ENDL;
    		}
    		for (int k= 0, l= weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
    			os << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameterTypes[k] << " ";
    			os << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters[k] << model.synapseName[i] << ";" << endl;
    		}
    		
    		vector<float> tmpP;
    		for (int j=0; j < weightUpdateModels[st-MAXSYN].dpNames.size(); ++j) {
    			float retVal = weightUpdateModels[st-MAXSYN].dps->calculateDerivedParameter(j, model.synapsePara[i], DT);
    			tmpP.push_back(retVal);
    		}
    		//dsp[i].clear();
    		model.dsp[i].swap(tmpP);	
    	}
    }
  
    for (int i=0; i< model.postSynapseType.size(); i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	    os << postSynModels[pst].varTypes[k] << " *";
	    os << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
	}
    }
  
    os << endl;
    os << "struct Conductance{" << endl;
    os << "  " << model.ftype << " *gp;" << endl; // only if !globalg
    os << "  unsigned int *gIndInG;" << endl;
    os << "  unsigned int *gInd;" << endl;
    os << "  unsigned int connN;" << endl; 
    os << "};" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseConnType[i] == SPARSE){
	    os << "Conductance g" << model.synapseName[i] << ";" << endl;
	}
	else {
	    if (model.synapseGType[i] == INDIVIDUALG) {
		os << model.ftype << " *gp" << model.synapseName[i] << ";" << endl;
	    }
       
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << "unsigned int *gp" << model.synapseName[i] << ";" << endl;
	    }	 
	}
	if (model.synapseType[i] == LEARN1SYNAPSE) {
	    os << model.ftype << " *grawp" << model.synapseName[i] << ";" << endl;
	}
    }
    os << endl;
  
    //device memory
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " *d_" << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	    os << "__device__ " << nModels[nt].varTypes[k] << " *dd_" << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	}
    }
  
    for (int i=0; i< model.postSynapseType.size(); i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	    os << postSynModels[pst].varTypes[k] << " *" << "d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl; //should work at the moment but if we make postsynapse vectors independent of synapses this may change
	    os << "__device__ " << postSynModels[pst].varTypes[k] << " *dd_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl; //should work at the moment but if we make postsynapse vectors independent of synapses this may change
	}
    }
  
    for (int i = 0; i < model.synapseGrpN; i++) {
	if (model.synapseGType[i] == INDIVIDUALG) {
	    os << model.ftype << " *d_gp" << model.synapseName[i] << ";" << endl;
	    os << "__device__ " << model.ftype << " *dd_gp" << model.synapseName[i] << ";" << endl;
	    if (model.synapseType[i] == LEARN1SYNAPSE) { 
		os << model.ftype << " *d_grawp" << model.synapseName[i] << ";" << endl;
		os << "__device__ " << model.ftype << " *dd_grawp" << model.synapseName[i] << ";" << endl;
	    }
	}
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "unsigned int *d_gp" << model.synapseName[i] << ";" << endl;
	    os << "__device__ " << "unsigned int *dd_gp" << model.synapseName[i] << ";" << endl;
	}
	if (model.synapseConnType[i] == SPARSE) {
	    os << "unsigned int *d_gp" << model.synapseName[i] << "_indInG;" << endl;
	    os << "__device__ " << "unsigned int *dd_gp" << model.synapseName[i] << "_indInG;" << endl;
	    os << "unsigned int *d_gp" << model.synapseName[i] << "_ind;" << endl;
	    os << "__device__ " << "unsigned int *dd_gp" << model.synapseName[i] << "_ind;" << endl;
	    trgN = model.neuronN[model.synapseTarget[i]];

   	} 
   	
	int st= model.synapseType[i];
	if (st >= MAXSYN){
	    for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		os << weightUpdateModels[st-MAXSYN].varTypes[k] << " *d_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ";" << endl; 
		os << "__device__ " << weightUpdateModels[st-MAXSYN].varTypes[k] << " *dd_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ";" << endl; 
	    }
	}  	
    }
    os << endl;
    
    // ------------------------------------------------------------------------
    // Code for setting the CUDA device and
    // setting up the host's global variables.
    // Also estimating memory usage on device ...
  
    os << "void allocateMem()" << endl;
    os << "{" << endl;
    //os << "  " << model.ftype << " free_m,total_m;" << endl;
    //os << "  cudaMemGetInfo((size_t*)&free_m,(size_t*)&total_m);" << endl; //
    os << "  CHECK_CUDA_ERRORS(cudaSetDevice(" << theDev << "));" << endl;
    cout << "model.neuronGroupN " << model.neuronGrpN << endl;
    os << "  size_t size;" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	os << "  glbSpk" << model.neuronName[i] << " = new unsigned int[" << model.neuronN[i] << "];" << endl;

	if (model.neuronDelaySlots[i] == 1) {
	    os << "  glbSpkEvnt" << model.neuronName[i] << " = new unsigned int[" << model.neuronN[i] << "];" << endl;
	    mem += model.neuronN[i] * sizeof(unsigned int);
	}
	else {
	    os << "  glbSpkEvntCnt" << model.neuronName[i] << " = new unsigned int[" << model.neuronDelaySlots[i] << "];" << endl;
	    os << "  glbSpkEvnt" << model.neuronName[i] << " = new unsigned int[" << model.neuronN[i] * model.neuronDelaySlots[i] << "];" << endl;
	    mem += model.neuronN[i] * model.neuronDelaySlots[i] * sizeof(unsigned int);
	}
	for (int j= 0; j < model.inSyn[i].size(); j++) {
	    os << "  inSyn" << model.neuronName[i] << j << " = new " << model.ftype << "[";
	    os << model.neuronN[i] << "];" << endl;
	    mem += model.neuronN[i] * theSize(model.ftype);
	} 
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "  " << nModels[nt].varNames[k] << model.neuronName[i] << " = new " << nModels[nt].varTypes[k] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << (model.neuronDelaySlots[i] * model.neuronN[i]);
		mem += (model.neuronDelaySlots[i] * model.neuronN[i] * sizeof(nModels[nt].varTypes[k]));
	    }
	    else {
		os << (model.neuronN[i]);
		mem += (model.neuronN[i] * sizeof(nModels[nt].varTypes[k]));
	    }
	    os << "];" << endl;
	}
    
  
	if (model.neuronNeedSt[i]) {
	    os << "  sT" << model.neuronName[i] << " = new " << model.ftype << "[";
	    os << model.neuronN[i] << "];" << endl;
	    mem += model.neuronN[i] * theSize(model.ftype);
	}   
    
	//allocate device neuron variables
	void *devptr;
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "  size = sizeof(" << nModels[nt].varTypes[k] << ") * ";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << model.neuronN[i] * model.neuronDelaySlots[i] << ";" << endl;
	    }
	    else {
		os << model.neuronN[i] << ";" << endl;
	    }
	    os << "deviceMemAllocate(&d_" << nModels[nt].varNames[k] << model.neuronName[i] << ", dd_" << nModels[nt].varNames[k] << model.neuronName[i] << ", size);" << endl;
	}
	os << endl; 
    }
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseGType[i] == INDIVIDUALG) {
	    if (model.synapseConnType[i] != SPARSE) { 
		os << "  gp" << model.synapseName[i] << " = new " << model.ftype << "[";
		os << model.neuronN[model.synapseSource[i]] << " * " << model.neuronN[model.synapseTarget[i]];
		os << "];      // synaptic conductances of group " << model.synapseName[i];
		os << endl;
		mem += model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] * theSize(model.ftype);
	    }
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  grawp" << model.synapseName[i] << " = new " << model.ftype << "[";
		os << model.neuronN[model.synapseSource[i]] << " * " << model.neuronN[model.synapseTarget[i]];
		os << "];      // raw synaptic conductances of group " << model.synapseName[i];
		os << endl;
		mem += model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] * theSize(model.ftype);
	    }

	}
	// note, if GLOBALG we put the value at compile time
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "  gp" << model.synapseName[i] << "= new unsigned int[";
	    unsigned long int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	    unsigned long int size= tmp >> logUIntSz;
	    if (tmp > (size << logUIntSz)) size++;
	    os << size;
	    os << "];     // synaptic connectivity of group " << model.synapseName[i];
	    os << endl;
	    mem+= size*sizeof(unsigned int);
	}
    
	int st= model.synapseType[i];

	if ((st >= MAXSYN) && (model.synapseConnType[i] != SPARSE)){
	  os << "size = " << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]] << ";" << ENDL;
	    
	  for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		  os  << weightUpdateModels[st-MAXSYN].varNames[k];
		  os << model.synapseName[i]<< "= new " << weightUpdateModels[st-MAXSYN].varTypes[k] << "[size];" << ENDL;
    } 
  }
	  
	if (model.synapseGType[i] == INDIVIDUALG) {
	    if (model.synapseConnType[i]!=SPARSE){
		os << "  size = sizeof(" << model.ftype << ") * " << model.neuronN[model.synapseSource[i]] << " * " << model.neuronN[model.synapseTarget[i]] << "; " << endl;
		os << "deviceMemAllocate( &d_gp" << model.synapseName[i] << ", dd_gp" << model.synapseName[i] << ", size);" << endl;
		mem+= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*theSize(model.ftype);
      
	    }
	    if (model.synapseType[i] == LEARN1SYNAPSE) { //TODO: what if sparse && learning?
		os << "  size = sizeof(" << model.ftype << ") * " << model.neuronN[model.synapseSource[i]] << " * " << model.neuronN[model.synapseTarget[i]] << "; " << endl; //not sure if correct				
		os << "deviceMemAllocate( &d_grawp" << model.synapseName[i] << ", dd_grawp" << model.synapseName[i] << ", size);" << endl; // raw synaptic conductances of group " << model.synapseName[i];
		os << endl;
		mem+= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]]*theSize(model.ftype);
	    }
	}
	// note, if GLOBALG we put the value at compile time
	if (model.synapseGType[i] == INDIVIDUALID) {
	    unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	    os << "  size = sizeof(unsigned int)*" << tmp << "; " << endl;
	    os << " deviceMemAllocate( &d_gp" << model.synapseName[i] << ", dd_gp" << model.synapseName[i] << ", "; 
	    unsigned int size= tmp >> logUIntSz;
	    if (tmp > (size << logUIntSz)) size++;
	    os << size;
	    os << ");     // synaptic connectivity of group " << model.synapseName[i];
	    os << endl;
	    mem += size;
	}
    
	//allocate user-defined weight model variables

	st= model.synapseType[i];	
	if ((st >= MAXSYN) && (model.synapseConnType[i] != SPARSE)) { //if they are sparse, allocate later in the allocatesparsearrays function when we know the size of the network
	  for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
	      os << "  size = " << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]] << "*sizeof(" << weightUpdateModels[st-MAXSYN].varTypes[k] << ");" << ENDL;
	      os << "  deviceMemAllocate(&d_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", dd_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", size);" << endl; 
	  }
	}
	os << endl;
  }  
  
    //allocate postsynapse variables 
    for (int i=0; i< model.postSynapseType.size(); i++){
	    int pst= model.postSynapseType[i];
	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	      os << "  " << postSynModels[pst].varNames[k] << model.synapseName[i] << " = new " << postSynModels[pst].varTypes[k] << "[" << (model.neuronN[model.synapseTarget[i]]) <<  "];" << endl;
	      //allocate device variables
	      os << "  size = sizeof(" << postSynModels[pst].varTypes[k] << ") * "<< model.neuronN[model.synapseTarget[i]] << ";" << endl;
	      os << "  deviceMemAllocate(&d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ", dd_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ", size);" << endl;      
	    }
    }
    os << endl; 	
    os << "}" << endl;
    os << endl;
  
    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    os << "void allocateSparseArray(Conductance *C, unsigned int preN, bool isGlobalG)" << "{" << endl;
    os << "  if (isGlobalG == false) C->gp= new " << model.ftype << "[C->connN];" << endl;      // synaptic conductances of group " << model.synapseName[i];

    os << "  C->gIndInG= new unsigned int[preN + 1];";      // model.neuronN[model.synapseSource[i]] index where the next neuron starts in the synaptic conductances of group " << model.synapseName[i];
    os << endl;
    os << "  C->gInd= new unsigned int[C->connN];" << endl;      // postsynaptic neuron index in the synaptic conductances of group " << model.synapseName[i];

    //		}
  
    os << "}" << endl; 
 
    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    os << "void allocateAllHostSparseArrays() {" << endl;
    for (int i = 0; i < model.synapseGrpN; i++) {
		  if ((model.synapseConnType[i] == SPARSE) && (model.synapseType[i] >= MAXSYN)) {
        os << "size_t size;" << endl;
        break;
      }
    }
    	
    for (int i = 0; i < model.synapseGrpN; i++) {
	    if (model.synapseConnType[i] == SPARSE) {
	      os << "  allocateSparseArray(&g" << model.synapseName[i] << ", ";
	      os << model.neuronN[model.synapseSource[i]] << ",";
	      if (model.synapseGType[i] == GLOBALG) {
		      os << " true);	//globalG" << endl; 
	      }
	      else{
		      os << " false);	//individual G" << endl;				
	      }
	      int st= model.synapseType[i];
			  if (st >= MAXSYN){
			    os << "size = g" << model.synapseName[i] << ".connN;" << ENDL;
			    for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		        os  << weightUpdateModels[st-MAXSYN].varNames[k];
		        os << model.synapseName[i]<< "= new " << weightUpdateModels[st-MAXSYN].varTypes[k] << "[size];" << ENDL;
	        }
			  }
      }
    }
    os << "}" << endl;

  os << "void allocateAllDeviceSparseArrays() {" << endl;
  for (int i = 0; i < model.synapseGrpN; i++) {
		if (model.synapseConnType[i] == SPARSE) {
      os << "size_t size;" << endl;
      break;
    }
  }	
	for (int i = 0; i < model.synapseGrpN; i++) {
		if (model.synapseConnType[i] == SPARSE) {
		    os << "size = g" << model.synapseName[i] << ".connN;" << ENDL;
		    if (model.synapseGType[i] != GLOBALG) {
			os << "  deviceMemAllocate( &d_gp" << model.synapseName[i] << ", dd_gp" << model.synapseName[i] << ", sizeof(" << model.ftype << ") * g" << model.synapseName[i] << ".connN);" << endl;
		    }
		    os << "  deviceMemAllocate( &d_gp" << model.synapseName[i] << "_ind, dd_gp" << model.synapseName[i] << "_ind, sizeof(unsigned int) * g" << model.synapseName[i] << ".connN);" << endl;
		    os << "  deviceMemAllocate( &d_gp" << model.synapseName[i] << "_indInG, dd_gp" << model.synapseName[i] << "_indInG, sizeof(unsigned int) * ("<< model.neuronN[model.synapseSource[i]] <<" + 1));" << endl;
		    mem += model.neuronN[model.synapseSource[i]]*sizeof(unsigned int);     
		    memremsparse = deviceProp[theDev].totalGlobalMem - float(mem);
			
			int st= model.synapseType[i];
			if (st >= MAXSYN){
			  
			  //weight update variables
        for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {     
	    os << "deviceMemAllocate(&d_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", dd_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << "sizeof("  << weightUpdateModels[st-MAXSYN].varTypes[k] << ")*size);" << endl;       
	      }
			  //post-to-pre remapped arrays
			  if (model.usesPostLearning[i]==TRUE) {
				  string learncode = weightUpdateModels[model.synapseType[i]-MAXSYN].simLearnPost;
				  cout << endl << "learn code is: " << endl << learncode << endl;			
//				size_t found = learncode.find
		    }
			}
	}
  }
    os << "}" << endl; 

    os << "void allocateAllSparseArrays() {" << endl;
    os << "\t allocateAllHostSparseArrays();" << endl;
    os << "\t allocateAllDeviceSparseArrays();" << endl;
    os << "}" << endl;

    // ------------------------------------------------------------------------
    // freeing global memory structures

    os << "void freeMem()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];

	os << "  delete[] glbSpk" << model.neuronName[i] << ";" << endl;

	if (model.neuronDelaySlots[i] == 1) {
	    os << "  delete[] glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	}
	else {
	    os << "  delete[] glbSpkEvntCnt" << model.neuronName[i] << ";" << endl;
	    os << "  delete[] glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	}
	for (int j= 0; j < model.inSyn[i].size(); j++) {
	    os << "  delete[] inSyn" << model.neuronName[i] << j << ";" << endl;
	} 
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "  delete[] " << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	}
	if (model.neuronNeedSt[i]) {
	    os << "  delete[] sT" << model.neuronName[i] << ";" << endl;
	}
    }
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseConnType[i] == SPARSE){
	    if (model.synapseGType[i] != GLOBALG) os << "  delete[] g" << model.synapseName[i] << ".gp;" << endl;
	    os << "  delete[] g" << model.synapseName[i] << ".gIndInG;" << endl;
	    os << "  delete[] g" << model.synapseName[i] << ".gInd;" << endl;  
	}
	else {
	    if (model.synapseGType[i] != GLOBALG) os << "  delete[] gp" << model.synapseName[i] << ";" << endl;
	}
    }
  
    for (int i=0;i<model.postSynapseType.size();i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	    os << "  delete[] " << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
	}
    }
  
    os << "}" << endl;
    os << endl;
  
 
    // ------------------------------------------------------------------------
    // initializing variables

    os << "void initialize()" << endl;
    os << "{" << endl;
    os << "size_t size;" << endl;
    if (model.seed != 0) {
	os << "  srand((unsigned int) time(NULL));" << endl;
    }
    else {
	os << "  srand((insigned int) " << model.seed() << ");" << endl;
    }
    os << endl;
    os << "  //neuron variables" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	os << "  glbscnt" << model.neuronName[i] << " = 0;" << endl;
	if (model.neuronDelaySlots[i] == 1) {
	    os << "  glbSpkEvntCnt" << model.neuronName[i] << " = 0;" << endl;
	}
	else {
	    os << "  spkEvntQuePtr" << model.neuronName[i] << " = 0;" << endl;
	    os << "  for (int i = 0; i < " << model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "    glbSpkEvntCnt" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "  }" << endl;
	}
	os << "  for (int i = 0; i < " << model.neuronN[i] << "; i++) {" << endl;
	os << "    glbSpk" << model.neuronName[i] << "[i] = 0;" << endl;
	if (model.neuronDelaySlots[i] == 1) {
	    os << "    glbSpkEvnt" << model.neuronName[i] << "[i] = 0;" << endl;
	}
	else {
	    os << "    for (int j = 0; j < " << model.neuronDelaySlots[i] << "; j++) {" << endl;
	    os << "      glbSpkEvnt" << model.neuronName[i] << "[(j * " << model.neuronN[i] <<  ") + i] = 0;" << endl;
	    os << "    }" << endl;
	}
	for (int k = 0; k < model.inSyn[i].size(); k++) {
	    os << "    inSyn" << model.neuronName[i] << k << "[i] = 0.0f;" << endl;
	} 
	for (int k = 0; k < nModels[nt].varNames.size(); k++) {
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "    for (int j = 0; j < " << model.neuronDelaySlots[i] << "; j++) {" << endl;
		os << "      " << nModels[nt].varNames[k] << model.neuronName[i] << "[(j * ";
		os << model.neuronN[i] << ") + i] = " << model.neuronIni[i][k] << ";" << endl;
		os << "    }" << endl;
	    }
	    else {
		os << "    " << nModels[nt].varNames[k] << model.neuronName[i];
		os << "[i] = " << model.neuronIni[i][k] << ";" << endl;
	    }
	}
	if (model.neuronType[i] == POISSONNEURON) {
	    os << "    seed" << model.neuronName[i] << "[i]= rand();" << endl;
	} 
	if (model.neuronNeedSt[i]) {
	    os << "    sT" <<  model.neuronName[i] << "[i]= -10000.0;" << endl;
	}
	os << "  }" << endl;
	if ((model.neuronType[i] == IZHIKEVICH) && (DT!=1)){
	    os << "  fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); "<< endl;
	}

	//copy host to device mem
	//neuron variables
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "  size = sizeof(" << nModels[nt].varTypes[k] << ") * ";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << model.neuronN[i] * model.neuronDelaySlots[i] << ";" << endl;
	    }
	    else {
		os << model.neuronN[i] << ";" << endl;
	    }
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_" << nModels[nt].varNames[k] << model.neuronName[i] << ", ";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ", size, cudaMemcpyHostToDevice));" << endl;
	}	
    }
    os << "  //synapse variables" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
	unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	if (model.synapseGType[i] == INDIVIDUALG) {
	    if (model.synapseConnType[i]!=SPARSE){      
		os << "  size = sizeof("<< model.ftype<<") * " << model.neuronN[model.synapseSource[i]]<< " * " << model.neuronN[model.synapseTarget[i]] << "; " << endl;
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", gp" << model.synapseName[i] << ", size, cudaMemcpyHostToDevice));" << endl;
	    }
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_grawp" << model.synapseName[i];
		os << ", grawp" << model.synapseName[i] << ", size, cudaMemcpyHostToDevice));" << endl;
		os << endl;
	    }
	}
	// note, if GLOBALG we put the value at compile time
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "  size = sizeof(unsigned int)*" << tmp << "; " << endl;
	    if (model.synapseConnType[i] == SPARSE){
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", g" << model.synapseName[i] << ".gp,";      
	    }
	    else {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", gp" << model.synapseName[i] << ",";           
	    }      
	    unsigned int size = tmp >> logUIntSz;
	    if (tmp > (size << logUIntSz)) size++;
	    os << size;
	    os << ", cudaMemcpyHostToDevice));     // synaptic connectivity of group " << model.synapseName[i];
	    os << endl;
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_grawp" << model.synapseName[i];
		    os << ", grawp" << model.synapseName[i] << "," << size << ", cudaMemcpyHostToDevice));" << endl;
		    os << endl;
	    }
	  }
    
	  int st= model.synapseType[i];
	  if ((st >= MAXSYN) && (model.synapseConnType[i] != SPARSE)){
	    os << "size = " << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]] << ";" << ENDL;
	    for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", "  << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", sizeof(" << weightUpdateModels[st-MAXSYN].varTypes[k] << ") * size , cudaMemcpyHostToDevice));" << endl; 
	    }
	  }
    
    
    }
  
    os << "  //postsynapse variables" << endl;
    for (int i=0; i< model.postSynapseType.size(); i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {

	    //copy host to device mem
	    //postsynapse variables

	    os << "  for (int i = 0; i < " << model.neuronN[model.synapseTarget[i]] << "; i++) {" << endl;
	    os << "    " << postSynModels[pst].varNames[k] << model.synapseName[i];
	    os << "[i] = " << model.postSynIni[i][k] << ";" << endl;
	    os << "	}" << endl;
	    os << "  size = sizeof(" << postSynModels[pst].varTypes[k] << ") * " << model.neuronN[model.synapseTarget[i]]<< ";" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ", ";
	    os << postSynModels[pst].varNames[k] << model.synapseName[i] << ", size, cudaMemcpyHostToDevice));" << endl;
      
	}
    }
  
    os << "}" << endl;
    os << endl;

    if (model.lrnGroups > 0) {
	for (int i = 0; i < model.synapseGrpN; i++) {
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "__host__ __device__ " << model.ftype << " gFunc" << model.synapseName[i] << "(" << model.ftype << " graw)" << endl;
		os << "{" << endl;
		os << "  return " << SAVEP(model.synapsePara[i][8]/2.0) << " * (tanh(";
		os << SAVEP(model.synapsePara[i][10]) << " * (graw - ";
		os << SAVEP(model.synapsePara[i][9]) << ")) + 1.0);" << endl;
		os << "}" << endl;
		os << endl;
		os << "__host__ __device__ " << model.ftype << " invGFunc" << model.synapseName[i] << "(" << model.ftype << " g)" << endl;
		os << "{" << endl;
		os << model.ftype << " tmp = g / " << SAVEP(model.synapsePara[i][8]*2.0) << "- 1.0;" << endl;
		os << "return 0.5 * log((1.0 + tmp) / (1.0 - tmp)) /" << SAVEP(model.synapsePara[i][10]);
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
	
		if (model.synapseGType[i] != GLOBALG) {
		    if (model.synapseConnType[i]==SPARSE){
			os << "(g" << model.synapseName[i] << ".gp[i]);" << endl;
		    }
		    else {
			os << "(gp" << model.synapseName[i] << "[i]);" << endl;
		    }
		    os << "  }" << endl;
		}
		else{ // can be optimised: no need to create an array, a constant would be enough (TODO)
		    os << "(" << model.g0[i] << ");" << endl;
		}
	    }
	}
	os << "}" << endl;
	os << endl;
    }

    // ------------------------------------------------------------------------
    // initializing conductance arrays for sparse matrices

    os << "void initializeSparseArray(Conductance C, " << model.ftype << " * dg, unsigned int * dgInd, unsigned int * dgIndInG, unsigned int preN)" << "{" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dg, C.gp, C.connN*sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;  
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dgInd, C.gInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dgIndInG, C.gIndInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "}" << endl; 
 	
    os << "void initializeSparseArrayGlobalG(Conductance C, unsigned int * dgInd, unsigned int * dgIndInG, unsigned int preN)" << "{" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dgInd, C.gInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dgIndInG, C.gIndInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "}" << endl; 
    // ------------------------------------------------------------------------

    os << "void initializeAllSparseArrays()" << "{" << endl;
    for (int i = 0; i < model.synapseGrpN; i++) {
		  if ((model.synapseConnType[i] == SPARSE) && (model.synapseType[i] >= MAXSYN)) {
        os << "size_t size;" << endl;
        break;
      }
    }
    for (int i= 0; i < model.synapseGrpN; i++) {
	    if (model.synapseConnType[i]==SPARSE){
	      if (model.synapseGType[i] == GLOBALG) {
		      os << "  initializeSparseArrayGlobalG(g" << model.synapseName[i] << ",";
	      }
	      else{
		      os << "  initializeSparseArray(g" << model.synapseName[i] << ",";
		      os << "  d_gp" << model.synapseName[i] << ",";
	      }
	      os << "  d_gp" << model.synapseName[i] << "_ind,";
	      os << "  d_gp" << model.synapseName[i] << "_indInG,";
	      os << model.neuronN[model.synapseSource[i]] <<");" << endl;
	      int st= model.synapseType[i];
    	  if ((st >= MAXSYN) && (model.synapseConnType[i] == SPARSE)){
	        os << "size = g" << model.synapseName[i] << ".connN;" << ENDL;
	        for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		        os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", "  << weightUpdateModels[st-MAXSYN].varNames[k] << model.synapseName[i] << ", sizeof(" << weightUpdateModels[st-MAXSYN].varTypes[k] << ") * size , cudaMemcpyHostToDevice));" << endl; 
	        }
	      }
	    }
    }
    os << "}" << endl; 
    // ------------------------------------------------------------------------

    os << "#include \"runnerGPU.cc\"" << endl;
    os << "#include \"runnerCPU.cc\"" << endl;
    os << endl;

    mos << "Global memory required for core model: " << mem/1e6 << " MB. " << endl;
    mos << deviceProp[theDev].totalGlobalMem << " for the device " << theDev << endl;  
  
  
    if  (memremsparse !=0){
	int connEstim = int((memremsparse)/(theSize(model.ftype)+sizeof(unsigned int)));
	mos << "Remaining mem is " << memremsparse/1e6 << " MB." << endl;
	mos << "You may run into memory problems on device" << theDev << " if the total number of synapses is bigger than " << connEstim << ", which roughly stands for " << int(connEstim/model.sumNeuronN[model.neuronGrpN - 1])<< " connections per neuron, without considering any other dynamic memory load." << endl;
    }
    else{
	if (0.5*deviceProp[theDev].totalGlobalMem < mem) {
	    mos << "memory required for core model (" << mem/1e6;
	    mos << "MB) is more than 50% of global memory on the chosen device";
	    mos << "(" << deviceProp[theDev].totalGlobalMem/1e6 << "MB)." << endl;
	    mos << "Experience shows that this is UNLIKELY TO WORK ... " << endl;
	}
    }
    os.close();
}

//----------------------------------------------------------------------------
/*!
  \brief A function to generate the code that simulates the model on the GPU

  The function generates functions that will spawn kernel grids onto the GPU (but not the actual kernel code which is generated in "genNeuronKernel()" and "genSynpaseKernel()"). Generated functions include "copyGToDevice()", "copyGFromDevice()", "copyStateToDevice()", "copyStateFromDevice()", "copySpikesFromDevice()", "copySpikeNFromDevice()" and "stepTimeGPU()". The last mentioned function is the function that will initialize the execution on the GPU in the generated simulation engine. All other generated functions are "convenience functions" to handle data transfer from and to the GPU.
*/
//----------------------------------------------------------------------------

void genRunnerGPU(NNmodel &model, //!< Model description 
		  string &path, //!< path for code generation
		  ostream &mos //!< output stream for messages
    )
{
    string name;
    unsigned int nt;
    ofstream os;

    mos << "entering GenRunnerGPU" << endl;
    name= path + toString("/") + model.name + toString("_CODE/runnerGPU.cc");
    os.open(name.c_str());

    writeHeader(os);
    os << "#include <cuda_runtime.h>" << endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << endl;
    os << "/*! \\file runnerGPU.cc" << endl << endl;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the host side code for a GPU simulator version." << endl;
    os << "*/" << endl;
    os << "//-------------------------------------------------------------------------" << endl << endl;

    os << "#ifndef MYRAND" << endl;
    os << "#define MYRAND(Y,X) Y = Y * 1103515245 +12345; ";
    os << "X= (Y >> 16);" << endl;
    os << "#endif" << endl;
    os << endl;

    os << "#include \"neuronKrnl.cc\"" << endl;
    if (model.synapseGrpN>0) os << "#include \"synapseKrnl.cc\"" << endl;
  
    for (int i= 0; i < model.neuronGrpN; i++) {
	if (model.receivesInputCurrent[i]>=2) {
	    os << "#include <string>" << endl;
	    break;
	}
    }
    os << endl;

    unsigned int size;

    // ------------------------------------------------------------------------
    // copying conductances to device
    
    os << "void copyGToDevice()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseGType[i] == INDIVIDUALG) {
	    if (model.synapseConnType[i]==SPARSE){          
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", g" << model.synapseName[i];
		os << ".gp, g" << model.synapseName[i] << ".connN*sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;
	    }
	    else {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", gp" << model.synapseName[i];
		size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
		os << ","<< model.neuronN[model.synapseSource[i]] << " * " << model.neuronN[model.synapseTarget[i]]<< "*sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;
	    }      
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_grawp" << model.synapseName[i]<< ", grawp" << model.synapseName[i];   
		if (model.synapseConnType[i]==SPARSE) {
		    os << "," << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;}
		else {
		    os << "," << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;
		}
	    } 
	}
	if (model.synapseGType[i] == INDIVIDUALID) {
	    if (model.synapseConnType[i] == SPARSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", g" << model.synapseName[i];
		os << ".gp, ";
	    }
	    else {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i] << ", gp" << model.synapseName[i];
		os << ", ";
	    }
	    unsigned int tmp = model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	    size = (tmp >> logUIntSz);
	    if (tmp > (size << logUIntSz)) {
		size++;
	    }
	    size = size * sizeof(unsigned int);
	    os << "," << size << ", cudaMemcpyHostToDevice));" << endl;
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_grawp" << model.synapseName[i]<< ", grawp" << model.synapseName[i];   
		if (model.synapseConnType[i]==SPARSE) {
		    os << "," << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;}
		else {
		    os << "," << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;
		}
	    }     
	}
    }  
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // copying conductances from device
    
    os << "void copyGFromDevice()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseGType[i] == INDIVIDUALG) {
	    if (model.synapseConnType[i]==SPARSE){
		//size= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(g" << model.synapseName[i] << ".gp, d_gp" << model.synapseName[i];      
		os << ", g" << model.synapseName[i] << ".connN * sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << endl;
	    }
	    else {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(gp" << model.synapseName[i] << ", d_gp" << model.synapseName[i];      
		size = model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] * theSize(model.ftype);
		os << ","<< model.neuronN[model.synapseSource[i]] << "*" << model.neuronN[model.synapseTarget[i]]<< "*sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << endl;
	    }  
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(grawp" << model.synapseName[i]<< ", d_grawp" << model.synapseName[i];	        
		size = model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] * theSize(model.ftype);
		os << "," << size << ", cudaMemcpyDeviceToHost));" << endl;
	    }
	}
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(gp" << model.synapseName[i] << ", d_gp" << model.synapseName[i];            
	    unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	    size = (tmp >> logUIntSz);
	    if (tmp > (size << logUIntSz)) size++;
	    size = size*sizeof(unsigned int);
	    os << ", " << size << ", cudaMemcpyDeviceToHost));" << endl;
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(grawp" << model.synapseName[i]<< ", d_grawp" << model.synapseName[i];	        
		size = model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] * theSize(model.ftype);
		os << "," << size << ", cudaMemcpyDeviceToHost));" << endl;
	    }
	}
    }
    os << "}" << endl;
    os << endl;
    
    // ------------------------------------------------------------------------
    // copying particular conductances group from device
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // copying values to device
    
    os << "void copyStateToDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    os << "  unsigned int tmp= 0;" << endl;
    for (int i= 0; i < model.synapseGrpN; i++) {
 	    if (model.synapseType[i]>=MAXSYN){
        if (weightUpdateModels[model.synapseType[i]-MAXSYN].varNames.size() > 0) {
          os << "  size_t size;" << endl;
          break;
        }
      }
    }
    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_done));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, &tmp, sizeof(int), cudaMemcpyHostToDevice));" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	if (model.neuronDelaySlots[i] != 1) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_spkEvntQuePtr" << model.neuronName[i] << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, &spkEvntQuePtr" << model.neuronName[i] << ", ";
	    size = sizeof(unsigned int);
	    os << size << ", cudaMemcpyHostToDevice));" << endl;
	}
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbscnt" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, &glbscnt" << model.neuronName[i] << ", ";
	size = sizeof(unsigned int);
	os << size << ", cudaMemcpyHostToDevice));" << endl;

	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpkEvntCnt" << model.neuronName[i] << "));" << endl;
	if (model.neuronDelaySlots[i] == 1) {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, &glbSpkEvntCnt" << model.neuronName[i] << ", ";
	    size = sizeof(unsigned int);
	}
	else {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, glbSpkEvntCnt" << model.neuronName[i] << ", ";
	    size = model.neuronDelaySlots[i] * sizeof(unsigned int);
	}
	os << size << ", cudaMemcpyHostToDevice));" << endl;

	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, glbSpk" << model.neuronName[i] << ", ";
	size = model.neuronN[i] * sizeof(unsigned int);
	os << size << ", cudaMemcpyHostToDevice));" << endl;      
	

	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpkEvnt" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, glbSpkEvnt" << model.neuronName[i] << ", ";
	size = model.neuronN[i] * sizeof(unsigned int);
	if (model.neuronDelaySlots[i] != 1) {
	    size *= model.neuronDelaySlots[i];
	}
	os << size << ", cudaMemcpyHostToDevice));" << endl;      

	for (int j= 0; j < model.inSyn[i].size(); j++) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_inSyn" << model.neuronName[i] << j << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, inSyn" << model.neuronName[i] << j << ", ";
	    os << model.neuronN[i] << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl;
	}
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {   
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_" << nModels[nt].varNames[k] << model.neuronName[i]<< ", ";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ", ";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		size = model.neuronN[i] * model.neuronDelaySlots[i];
	    }
	    else {
		size = model.neuronN[i];
	    }
	    os << size << " * sizeof(" << nModels[nt].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl;
	}    
	
	if (model.neuronNeedSt[i]) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_sT" << model.neuronName[i] << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(devPtr, " << "sT" << model.neuronName[i] << ", ";
	    size = model.neuronN[i] * theSize(model.ftype);
	    os << size << ", cudaMemcpyHostToDevice));" << endl;
	    cout << "model.receivesInputCurrent[i]: " << model.receivesInputCurrent[i] << endl;
	}
    }
    
    for (int i= 0; i < model.synapseGrpN; i++) {
 	if (model.synapseType[i]>=MAXSYN){
	    unsigned int st= model.synapseType[i] - MAXSYN;
	    unsigned int src = model.synapseSource[i];
	    unsigned int trg = model.synapseTarget[i];
	    for (int k = 0, l = weightUpdateModels[st].varNames.size(); k < l; k++) {
	  if (model.synapseConnType[i] != SPARSE){
		  os << "size = " << model.neuronN[src] * model.neuronN[trg] << ";" <<endl; //! TODO This is for alltoall only 
		}
		else{
		  os << "size = g" << model.synapseName[i] << ".connN;" << ENDL;
		}
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i]<< ", ";
		os << weightUpdateModels[st].varNames[k] << model.synapseName[i] << ", ";
		os << "size * sizeof(" << weightUpdateModels[st].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl;
	    }
	    /*if ((model.usesPostLearning[i] == TRUE) && (model.synapseConnType[i] == SPARSE)) {
	    //!TODO create post-to-pre matrices here
	    }*/
	}  
    } 
    
    for (int i=0; i< model.postSynapseType.size(); i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	    
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(d_" << postSynModels[pst].varNames[k] << model.synapseName[i]<< ", ";
	    os << postSynModels[pst].varNames[k] << model.synapseName[i] << ", ";
	    size = model.neuronN[model.synapseTarget[i]];
	    os << size << " * sizeof(" << postSynModels[pst].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl;
	}
    }
    
    
    os << "}" << endl;
    os << endl;
    
    // ------------------------------------------------------------------------
    // copying values from device

    os << "void copyStateFromDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;    
    for (int i= 0; i < model.synapseGrpN; i++) {
 	    if (model.synapseType[i]>=MAXSYN){
        if (weightUpdateModels[model.synapseType[i]-MAXSYN].varNames.size() > 0) {
          os << "  size_t size;" << endl;
          break;
        }
      }
    }
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	if (model.neuronDelaySlots[i] != 1) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_spkEvntQuePtr" << model.neuronName[i] << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(&spkEvntQuePtr" << model.neuronName[i] << ", devPtr, ";
	    size = sizeof(unsigned int);
	    os << size << ", cudaMemcpyDeviceToHost));" << endl;
	}
	
	//glbscnt
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbscnt" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(&glbscnt" << model.neuronName[i] << ", devPtr, ";
	size = sizeof(unsigned int);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
	
	//glbSpkEvntCnt
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpkEvntCnt" << model.neuronName[i] << "));" << endl;
	if (model.neuronDelaySlots[i] == 1) {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(&glbSpkEvntCnt" << model.neuronName[i] << ", devPtr, ";
	    size = sizeof(unsigned int);
	}
	else {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntCnt" << model.neuronName[i] << ", devPtr, ";
	    size = model.neuronDelaySlots[i] * sizeof(unsigned int);
	}
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
	
	//glbSpk
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
	size = model.neuronN[i] * sizeof(unsigned int);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
	
	
	//glbSpkEvnt
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpkEvnt" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << model.neuronName[i] << ", devPtr, ";
	size = model.neuronN[i] * sizeof(unsigned int);
	if (model.neuronDelaySlots[i] != 1) size *= model.neuronDelaySlots[i];
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
	
	
	for (int j= 0; j < model.inSyn[i].size(); j++) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_inSyn" << model.neuronName[i] << j << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(inSyn" << model.neuronName[i] << j << ", devPtr, ";
	    os << model.neuronN[i] << " * sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << endl;
	}

	
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(" << nModels[nt].varNames[k] << model.neuronName[i] << ", ";
	    os << "d_" << nModels[nt].varNames[k] << model.neuronName[i] << ", ";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		size = model.neuronN[i] * model.neuronDelaySlots[i];
	    }
	    else {
		size = model.neuronN[i];
	    }
	    os << size << " * sizeof(" << nModels[nt].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl;
	}
    
    
	if (model.neuronNeedSt[i]) {
	    os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_sT" << model.neuronName[i] << "));" << endl;
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(sT" << model.neuronName[i] << ", devPtr, ";
	    os << model.neuronN[i] << " * sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << endl;
	}
    }
    
    for (int i= 0; i < model.synapseGrpN; i++) {
 	if (model.synapseType[i]>=MAXSYN){
	    unsigned int st= model.synapseType[i] - MAXSYN;
	    unsigned int src = model.synapseSource[i];
	    unsigned int trg = model.synapseTarget[i];
	    for (int k = 0, l = weightUpdateModels[st].varNames.size(); k < l; k++) {
			if (model.synapseConnType[i] != SPARSE){
		    os << "size = " << model.neuronN[src] * model.neuronN[trg] << ";" <<endl; //! TODO This is for alltoall only 
		  }
		  else{
		    os << "size = g" << model.synapseName[i] << ".connN;" << ENDL;
	  	}
		os << "  CHECK_CUDA_ERRORS(cudaMemcpy(" <<  weightUpdateModels[st].varNames[k] << model.synapseName[i] <<", ";
		os << "d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i]<< ", ";
			os <<  "size * sizeof(" << weightUpdateModels[st].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl;
	    }
	    /*if ((model.usesPostLearning[i] == TRUE) && (model.synapseConnType[i] == SPARSE)) {
	    //!TODO create post-to-pre matrices here
	    }*/
	}  
    }
 
 
    for (int i=0; i< model.postSynapseType.size(); i++){
	  int pst= model.postSynapseType[i];
	  for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {      
	    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(" << postSynModels[pst].varNames[k] << model.synapseName[i] << ", ";
	    os << "d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ", ";
	    size = model.neuronN[model.synapseTarget[i]];
	    os << size << " * sizeof(" << postSynModels[pst].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl;
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
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbSpk" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << model.neuronName[i] << ", devPtr, ";
	os << "glbscnt" << model.neuronName[i] << " * " << sizeof(unsigned int);
	os << ", cudaMemcpyDeviceToHost));" << endl;
    }
    os << "}" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // copying spike numbers from device

    os << "void copySpikeNFromDevice()" << endl;
    os << "{" << endl;
    os << "  void *devPtr;" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	os << "  CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_glbscnt" << model.neuronName[i] << "));" << endl;
	os << "  CHECK_CUDA_ERRORS(cudaMemcpy(&glbscnt" << model.neuronName[i] << ", devPtr, ";
	size = sizeof(unsigned int);
	os << size << ", cudaMemcpyDeviceToHost));" << endl;
    }
    os << "}" << endl;
    os << endl;
 
    // ------------------------------------------------------------------------
    // clean device memory that was allocated from the host

    os << "void freeDeviceMem()" << endl;
    os << "{" << endl;
    for (int i= 0; i < model.neuronGrpN; i++) {
	    nt= model.neuronType[i];
	    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	      os << "  CHECK_CUDA_ERRORS(cudaFree(d_" << nModels[nt].varNames[k] << model.neuronName[i] << "));" << endl;
	    }
    }
    for (int i= 0; i < model.synapseGrpN; i++) {
	    if ((model.synapseGType[i] == (INDIVIDUALG)) || (model.synapseGType[i] == INDIVIDUALID)) {
	      os << "  CHECK_CUDA_ERRORS(cudaFree(d_gp" << model.synapseName[i] << "));" <<endl;  	
	    }
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
	      os << " CHECK_CUDA_ERRORS(cudaFree(d_grawp"  << model.synapseName[i] << "));" <<endl;	
	    }
    }
    
    //postsynaptic variavles  
    for (int i=0; i< model.postSynapseType.size(); i++){
	    int pst= model.postSynapseType[i];
	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	      os << "  CHECK_CUDA_ERRORS(cudaFree(d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << "));" << endl;
	    }
    }
    
    //weight update variables
    for (int i= 0; i < model.synapseGrpN; i++) {
      if (model.synapseType[i] >= MAXSYN){    
        for (int k= 0, l= weightUpdateModels[model.synapseType[i]-MAXSYN].varNames.size(); k < l; k++) {
          os << "  CHECK_CUDA_ERRORS(cudaFree(d_" << weightUpdateModels[model.synapseType[i]-MAXSYN].varNames[k] << model.synapseName[i] << "));" << endl;
	      }
	    }
    }
    os << "}" << endl;
    os << endl;
    
    // ------------------------------------------------------------------------
    // the actual time stepping procedure
    
    os << "void stepTimeGPU(";
    for (int i= 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    os << model.RNtype << " *rates" << model.neuronName[i];
	    os << ",   // pointer to the rates of the Poisson neurons in grp ";
	    os << model.neuronName[i] << endl;
	    os << "unsigned int offset" << model.neuronName[i];
	    os << ",   // offset on pointer to the rates in grp ";
	    os << model.neuronName[i] << endl;
	}
	if (model.receivesInputCurrent[i]>=2) {
	    os << model.ftype << " *d_inputI" << model.neuronName[i];
	    os << ",   // Explicit input to the neurons in grp ";
	    os << model.neuronName[i] << endl;
	}
    }
    os << model.ftype << " t)" << endl;
    os << "{" << endl;

    if (model.synapseGrpN > 0) { 
	unsigned int synapseGridSz = model.padSumSynapseKrnl[model.synapseGrpN - 1];   
	os << "//model.padSumSynapseTrgN[model.synapseGrpN - 1] is " << model.padSumSynapseKrnl[model.synapseGrpN - 1] << endl; 
	synapseGridSz = synapseGridSz / synapseBlkSz;
	os << "  dim3 sThreads(" << synapseBlkSz << ", 1);" << endl;
	os << "  dim3 sGrid(" << synapseGridSz << ", 1);" << endl;
	
	os << endl;
    }
    if (model.lrnGroups > 0) {
	unsigned int learnGridSz = model.padSumLearnN[model.lrnGroups - 1];
	learnGridSz = ceil((float) learnGridSz / learnBlkSz);
	os << "  dim3 lThreads(" << learnBlkSz << ", 1);" << endl;
	os << "  dim3 lGrid(" << learnGridSz << ", 1);" << endl;
	os << endl;
    }


    unsigned int neuronGridSz = model.padSumNeuronN[model.neuronGrpN - 1];
    neuronGridSz = ceil((float) neuronGridSz / neuronBlkSz);
    os << "  dim3 nThreads(" << neuronBlkSz << ", 1);" << endl;
    if (neuronGridSz < deviceProp[theDev].maxGridSize[1]) {
	os << "  dim3 nGrid(" << neuronGridSz << ", 1);" << endl;
    }
    else {
      int sqGridSize = ceil((float) sqrt((float) neuronGridSz));
	os << "  dim3 nGrid(" << sqGridSize << ","<< sqGridSize <<");" << endl;
    }
    os << endl;
    int trgN;
    if (model.synapseGrpN > 0) {
	os << "  if (t > 0.0) {" << endl; 
	os << "    calcSynapses <<< sGrid, sThreads >>> (";
	for (int i= 0; i < model.synapseGrpN; i++) {
	    
	    for (int i=0; i< model.synapseName.size(); i++){
		int st= model.synapseType[i];
		if (st >= MAXSYN){
		    for (int k= 0, l= weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
			os << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters[k] << ", " ;
			os << model.synapseName[i];
		    }
		}
	    }
	}
	os << "t);"<< endl;

	if (model.lrnGroups > 0) {
	    os << "    learnSynapsesPost <<< lGrid, lThreads >>> (";         
	    for (int i=0; i< model.synapseName.size(); i++){
		int st= model.synapseType[i];
		if (st >= MAXSYN){
		    for (int k= 0, l= weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
			os << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters[k] << ", ";
			os << model.synapseName[i];
		    }
		}
	    }
	    
	    os << "t);" << endl;
	}
	os << "  }" << endl;
	
    }
    os << "  calcNeurons <<< nGrid, nThreads >>> (";
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	if (nt == POISSONNEURON) {
	    os << "rates" << model.neuronName[i] << ", ";
	    os << "offset" << model.neuronName[i] << ",";
	}
	if (model.receivesInputCurrent[i]>=2) {
	    os << "d_inputI" << model.neuronName[i] << ", ";
	}
	for (int k= 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    os << nModels[nt].extraGlobalNeuronKernelParameters[k] << model.neuronName[i]<< ", ";
	}    
    }
    os << "t);" << endl;
    os << "}" << endl;
    os.close();
    cerr << "done with generateGPU" << endl;
}


//----------------------------------------------------------------------------
/*!
  \brief A function to generate code for an equivalent CPU-only simulation engine

  The generated code provides the same functionality as the code generated for the GPU but only utilizing the CPU. That being so, no convenience functions for data transfer are necessary in this scenario and the only generated function is "stepTimeCPU()" for simulating a time step of the model on the CPU (using the CPU functions for simuations, equivalent to the kernels in the GPU case, that are generated by generateNeuronFunction() and generateSynapseFunction().
*/
//----------------------------------------------------------------------------

void genRunnerCPU(NNmodel &model, //!< Neuronal network model description 
		  string &path, //!< Path for code generation
		  ostream &mos) //!< Output stream for messages
{
    string name;
    ofstream os;
  
    name= path + toString("/") + model.name + toString("_CODE/runnerCPU.cc");
    os.open(name.c_str());
  
    writeHeader(os);
    os << "#include <cuda_runtime.h>" << endl;//EY
  
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << endl;
    os << "/*! \\file runnerCPU.cc" << endl << endl;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the control code for running a CPU only simulator version." << endl;
    os << "*/" << endl;
    os << "//-------------------------------------------------------------------------" << endl << endl;
  
    os << "#ifndef MYRAND" << endl;
    os << "#define MYRAND(Y,X) Y = Y * 1103515245 +12345; ";
    os << "X= (Y >> 16);" << endl;
    os << "#endif" << endl;
    os << endl;

    os << "#include \"neuronFnct.cc\"" << endl;
    if (model.synapseGrpN>0) os << "#include \"synapseFnct.cc\"" << endl;
    os << endl;

    // ------------------------------------------------------------------------
    // the actual time stepping procedure

    os << "void stepTimeCPU(";
    for (int i= 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    os << model.RNtype << " *rates" << model.neuronName[i];
	    os << ",   // pointer to the rates of the Poisson neurons in grp ";
	    os << model.neuronName[i] << endl;
	    os << "unsigned int offset" << model.neuronName[i];
	    os << ",   // offset on pointer to the rates in grp ";
	    os << model.neuronName[i] << endl;
	}
	if (model.receivesInputCurrent[i] >= 2) {
	    os << model.ftype << " *inputI" << model.neuronName[i];
	    os << ",   // pointer to the explicit input to neurons in grp ";
	    os << model.neuronName[i] << "," << endl;
	}
    }
    os << model.ftype << " t)" << endl;
    os << "{" << endl;
    if (model.synapseGrpN>0) {
	os << "  if (t > 0.0) {" << endl; 
	os << "    calcSynapsesCPU(t);" << endl;
	if (model.lrnGroups > 0) {
	    os << "learnSynapsesPostHost(t);" << endl;
	}
	os << "  }" << endl;
    }
    os << "  calcNeuronsCPU(";
    for (int i= 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    os << "rates" << model.neuronName[i] << ", ";
	    os << "offset" << model.neuronName[i] << ",";
	}
	if (model.receivesInputCurrent[i]>=2) {
	    os << "inputI" << model.neuronName[i] << ", ";
	}
    }
    os << "t);" << endl;
    os << "}" << endl;
    os.close();
}
