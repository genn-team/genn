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

#include <cfloat>

void genRunner(NNmodel &model, //!< Model description
	       string path, //!< path for code generation
	       ostream &mos //!< output stream for messages
    )
{
    string name;
    size_t size, tmp;
    unsigned int nt, st, pst;
    unsigned int mem = 0;
    float memremsparse= 0;
    ofstream os;
    
    //initializing learning parameters to start
    model.initLearnGrps();  //Putting this here for the moment. Makes more sense to call it at the end of ModelDefinition, but this leaves the initialization to the user.
    
    string SCLR_MIN;
    string SCLR_MAX;
    if (model.ftype == tS("float")) {
	SCLR_MIN= tS(FLT_MIN)+tS("f");
	SCLR_MAX= tS(FLT_MAX)+tS("f");
    }

    if (model.ftype == tS("double")) {
	SCLR_MIN= tS(DBL_MIN);
	SCLR_MAX= tS(DBL_MAX);
    }

    for (int i= 0; i < nModels.size(); i++) {
	for (int k= 0; k < nModels[i].varTypes.size(); k++) {
	    substitute(nModels[i].varTypes[k], "scalar", model.ftype);
	}
	substitute(nModels[i].simCode, "SCALAR_MIN", SCLR_MIN);
	substitute(nModels[i].resetCode, "SCALAR_MIN", SCLR_MIN);
	substitute(nModels[i].simCode, "SCALAR_MAX", SCLR_MAX);
	substitute(nModels[i].resetCode, "SCALAR_MAX", SCLR_MAX);
	substitute(nModels[i].simCode, "scalar", model.ftype);
	substitute(nModels[i].resetCode, "scalar", model.ftype);
    }
    for (int i= 0; i < weightUpdateModels.size(); i++) {
	for (int k= 0; k < weightUpdateModels[i].varTypes.size(); k++) {
	    substitute(weightUpdateModels[i].varTypes[k], "scalar", model.ftype);
	}
	substitute(weightUpdateModels[i].simCode, "SCALAR_MIN", SCLR_MIN);
	substitute(weightUpdateModels[i].simCodeEvnt, "SCALAR_MIN", SCLR_MIN);
	substitute(weightUpdateModels[i].simLearnPost, "SCALAR_MIN", SCLR_MIN);
	substitute(weightUpdateModels[i].synapseDynamics, "SCALAR_MIN", SCLR_MIN);
	substitute(weightUpdateModels[i].simCode, "SCALAR_MAX", SCLR_MAX);
	substitute(weightUpdateModels[i].simCodeEvnt, "SCALAR_MAX", SCLR_MAX);
	substitute(weightUpdateModels[i].simLearnPost, "SCALAR_MAX", SCLR_MAX);
	substitute(weightUpdateModels[i].synapseDynamics, "SCALAR_MAX", SCLR_MAX);
	substitute(weightUpdateModels[i].simCode, "scalar", model.ftype);
	substitute(weightUpdateModels[i].simCodeEvnt, "scalar", model.ftype);
	substitute(weightUpdateModels[i].simLearnPost, "scalar", model.ftype);
	substitute(weightUpdateModels[i].synapseDynamics, "scalar", model.ftype);
    }
    for (int i= 0; i < postSynModels.size(); i++) {
	for (int k= 0; k < postSynModels[i].varTypes.size(); k++) {
	    substitute(postSynModels[i].varTypes[k], "scalar", model.ftype);
	}
	substitute(postSynModels[i].postSyntoCurrent, "SCALAR_MIN", SCLR_MIN);
	substitute(postSynModels[i].postSynDecay, "SCALAR_MIN", SCLR_MIN);
	substitute(postSynModels[i].postSyntoCurrent, "SCALAR_MAX", SCLR_MAX);
	substitute(postSynModels[i].postSynDecay, "SCALAR_MAX", SCLR_MAX);
	substitute(postSynModels[i].postSyntoCurrent, "scalar", model.ftype);
	substitute(postSynModels[i].postSynDecay, "scalar", model.ftype);
    }
    
//    cout << "entering genRunner" << endl;
    name= path + toString("/") + model.name + toString("_CODE/runner.cc");
    os.open(name.c_str());  
    writeHeader(os);
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file runner.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing general control code used for both GPU amd CPU versions." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "#include <cstdio>" << ENDL;
    os << "#include <cassert>" << ENDL;
    os << "#include <stdint.h>" << ENDL;
    os << "#include \"numlib/simpleBit.h\"" << ENDL << ENDL;
    if (model.timing) os << "#include \"hr_time.cpp\"" << ENDL;
    os << ENDL;

    os << "#ifndef scalar" << ENDL;
    os << "typedef " << model.ftype << " scalar;" << ENDL;
    os << "#endif" << ENDL;
  
    os << "#ifndef SCALAR_MIN" << ENDL;
    os << "#define SCALAR_MIN " << SCLR_MIN << ENDL;
    os << "#endif" << ENDL;
  
    os << "#ifndef SCALAR_MAX" << ENDL;
    os << "#define SCALAR_MAX " << SCLR_MAX << ENDL;
    os << "#endif" << ENDL;
  os << "#define Conductance SparseProjection" << ENDL;
  os << "/*struct Conductance is deprecated. \n\
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. \n\
  Please consider updating your user code by renaming Conductance as SparseProjection \n\
  and making g member a synapse variable.*/" << ENDL;

    // write MYRAND macro
    os << "#ifndef MYRAND" << ENDL;
    os << "#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);" << ENDL;
    os << "#endif" << ENDL << ENDL;
  if (model.timing) {
      os << "cudaEvent_t neuronStart, neuronStop;" << endl;
      os << "double neuron_tme;" << endl;
      os << "CStopWatch neuron_timer;" << endl;
      if (model.synapseGrpN > 0) {
	  os << "cudaEvent_t synapseStart, synapseStop;" << endl;
	  os << "double synapse_tme;" << endl;
	  os << "CStopWatch synapse_timer;" << endl;
      }
      if (model.lrnGroups > 0) {
	  os << "cudaEvent_t learningStart, learningStop;" << endl;
	  os << "double learning_tme;" << endl;
	  os << "CStopWatch learning_timer;" << endl;
      }
  } 

    // write CUDA error handler macro
    os << "#ifndef CHECK_CUDA_ERRORS" << ENDL;
    os << "#define CHECK_CUDA_ERRORS(call) {\\" << ENDL;
    os << "    cudaError_t error = call;\\" << ENDL;
    os << "    if (error != cudaSuccess) {\\" << ENDL;
    os << "        fprintf(stderr, \"%s: %i: cuda error %i: %s\\n\", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\\" << ENDL;
    os << "        exit(EXIT_FAILURE);\\" << ENDL;
    os << "    }\\" << ENDL;
    os << "}" << ENDL;
    os << "#endif" << ENDL << ENDL;

    os << "template<class T>" << ENDL;
    os << "void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)" << ENDL;
    os << "{" << ENDL;
    os << "    void *devptr;" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));" << ENDL;
    os << "    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));" << ENDL;
    os << "}" << ENDL << ENDL;

    os << "void convertProbabilityToRandomNumberThreshold(" << model.ftype << " *p_pattern, " << model.RNtype << " *pattern, int N)" << ENDL;
    os << "{" << ENDL;
    os << "    " << model.ftype << " fac= pow(2.0, (int) sizeof(" << model.RNtype << ")*8-16)*DT;" << ENDL;
    os << "    for (int i= 0; i < N; i++) {" << ENDL;
    //os << "        assert(p_pattern[i] <= 1.0);" << ENDL;
    os << "        pattern[i]= (" << model.RNtype << ") (p_pattern[i]*fac);" << ENDL;
    os << "    }" << ENDL;
    os << "}" << ENDL << ENDL;

    //---------------------------------
    // HOST AND DEVICE NEURON VARIABLES

    os << "// neuron variables" << endl;
    os << "__device__ volatile unsigned int d_done;" << endl;
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	os << "unsigned int *glbSpkCnt" << model.neuronName[i] << ";" << endl;
	os << "unsigned int *d_glbSpkCnt" << model.neuronName[i] << ";" << endl;
	os << "__device__ unsigned int *dd_glbSpkCnt" << model.neuronName[i] << ";" << endl;

	os << "unsigned int *glbSpk" << model.neuronName[i] << ";" << endl;
	os << "unsigned int *d_glbSpk" << model.neuronName[i] << ";" << endl;
	os << "__device__ unsigned int *dd_glbSpk" << model.neuronName[i] << ";" << endl;

	if (model.neuronNeedSpkEvnt[i]) {
	    os << "unsigned int *glbSpkCntEvnt" << model.neuronName[i] << ";" << endl;
	    os << "unsigned int *d_glbSpkCntEvnt" << model.neuronName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_glbSpkCntEvnt" << model.neuronName[i] << ";" << endl;

	    os << "unsigned int *glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	    os << "unsigned int *d_glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	}
	if (model.neuronDelaySlots[i] > 1) {
	    os << "unsigned int spkQuePtr" << model.neuronName[i] << ";" << endl;
	    os << "__device__ volatile unsigned int dd_spkQuePtr" << model.neuronName[i] << ";" << endl;
	}
	if (model.neuronNeedSt[i]) {
	    os << model.ftype << " *sT" << model.neuronName[i] << ";" << endl;
	    os << model.ftype << " *d_sT" << model.neuronName[i] << ";" << endl;
	    os << "__device__ " << model.ftype << " *dd_sT" << model.neuronName[i] << ";" << endl;
	}
	for (int k = 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " *";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	    os << nModels[nt].varTypes[k] << " *d_";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	    os << "__device__ " << nModels[nt].varTypes[k] << " *dd_";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	}

	// write global variables for the extra global neuron kernel parameters.
	// These are assumed not to be pointers, if they are the user needs to take care of allocation etc
	for (int k = 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    os << nModels[nt].extraGlobalNeuronKernelParameterTypes[k] << " ";
	    os << nModels[nt].extraGlobalNeuronKernelParameters[k] << model.neuronName[i] << ";" << endl;
	}
    }
    os << endl;


    //----------------------------------
    // HOST AND DEVICE SYNAPSE VARIABLES

    os << "// synapse variables" << endl;
    os << "struct SparseProjection{" << endl;
    os << "    unsigned int *indInG;" << endl;
    os << "    unsigned int *ind;" << endl;
    os << "    unsigned int *revIndInG;" << endl;
    os << "    unsigned int *revInd;" << endl;
    os << "    unsigned int *remap;" << endl;
    os << "    unsigned int connN;" << endl; 
    os << "};" << endl;
    

    for (int i = 0; i < model.synapseGrpN; i++) {
    	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	os << model.ftype << " *inSyn" << model.synapseName[i] << ";" << endl;
	os << model.ftype << " *d_inSyn" << model.synapseName[i] << ";" << endl; 	
	os << "__device__ " << model.ftype << " *dd_inSyn" << model.synapseName[i] << ";" << endl; 	
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "uint32_t *gp" << model.synapseName[i] << ";" << endl;
	    os << "uint32_t *d_gp" << model.synapseName[i] << ";" << endl;
	    os << "__device__ uint32_t *dd_gp" << model.synapseName[i] << ";" << endl;
	}
	if (model.synapseConnType[i] == SPARSE) {
	    os << "SparseProjection C" << model.synapseName[i] << ";" << endl;
	    os << "unsigned int *d_indInG" << model.synapseName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_indInG" << model.synapseName[i] << ";" << endl;
	    os << "unsigned int *d_ind" << model.synapseName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_ind" << model.synapseName[i] << ";" << endl;
	    // TODO: make conditional on post-spike driven learning actually taking place
	    os << "unsigned int *d_revIndInG" << model.synapseName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_revIndInG" << model.synapseName[i] << ";" << endl;
	    os << "unsigned int *d_revInd" << model.synapseName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_revInd" << model.synapseName[i] << ";" << endl;
	    os << "unsigned int *d_remap" << model.synapseName[i] << ";" << endl;
	    os << "__device__ unsigned int *dd_remap" << model.synapseName[i] << ";" << endl;
	}
	if (model.synapseGType[i] == INDIVIDUALG) { // not needed for GLOBALG, INDIVIDUALID
	    for (int k = 0, l = weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << weightUpdateModels[st].varTypes[k] << " *";
		os << weightUpdateModels[st].varNames[k] << model.synapseName[i]<< ";" << ENDL;
		os << weightUpdateModels[st].varTypes[k] << " *d_";
		os << weightUpdateModels[st].varNames[k] << model.synapseName[i] << ";" << endl; 
		os << "__device__ " << weightUpdateModels[st].varTypes[k] << " *dd_";
		os << weightUpdateModels[st].varNames[k] << model.synapseName[i] << ";" << endl; 
	    }
	    for (int k = 0, l = postSynModels[pst].varNames.size(); k < l; k++) {
		os << postSynModels[pst].varTypes[k] << " *" << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
		//should work at the moment but if we make postsynapse vectors independent of synapses this may change
		os << postSynModels[pst].varTypes[k] << " *d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
		//should work at the moment but if we make postsynapse vectors independent of synapses this may change
		os << "__device__ " << postSynModels[pst].varTypes[k] << " *dd_" << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
	    }
	}
	for (int k = 0, l = weightUpdateModels[st].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
	    os << weightUpdateModels[st].extraGlobalSynapseKernelParameterTypes[k] << " ";
	    os << weightUpdateModels[st].extraGlobalSynapseKernelParameters[k] << model.synapseName[i] << ";" << endl;
	}
    }
    os << endl << endl;


    // include simulation kernels
    os << "#include \"runnerGPU.cc\"" << endl << endl;
    os << "#include \"neuronFnct.cc\"" << endl;
    if (model.synapseGrpN > 0) {
	os << "#include \"synapseFnct.cc\"" << endl;
    }


    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
  
    os << "void allocateMem()" << endl;
    os << "{" << endl;
    os << "    CHECK_CUDA_ERRORS(cudaSetDevice(" << theDev << "));" << endl;

    //cout << "model.neuronGroupN " << model.neuronGrpN << endl;
    //os << "    " << model.ftype << " free_m, total_m;" << endl;
    //os << "    cudaMemGetInfo((size_t*) &free_m, (size_t*) &total_m);" << endl;

    if (model.timing) {
	os << "    cudaEventCreate(&neuronStart);" << endl;
	os << "    cudaEventCreate(&neuronStop);" << endl;
	os << "    neuron_tme= 0.0;" << endl;
	if (model.synapseGrpN > 0) {
	    os << "    cudaEventCreate(&synapseStart);" << endl;
	    os << "    cudaEventCreate(&synapseStop);" << endl;
	    os << "    synapse_tme= 0.0;" << endl;
	}
	if (model.lrnGroups > 0) {
	    os << "    cudaEventCreate(&learningStart);" << endl;
	    os << "    cudaEventCreate(&learningStop);" << endl;
	    os << "    learning_tme= 0.0;" << endl;
	}
    }

    // ALLOCATE NEURON VARIABLES
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronDelaySlots[i];
	}
	else {
	    size = 1;
	}
	os << "    glbSpkCnt" << model.neuronName[i] << " = new unsigned int[" << size << "];" << endl;
	os << "    deviceMemAllocate(&d_glbSpkCnt" << model.neuronName[i];
	os << ", dd_glbSpkCnt" << model.neuronName[i] << ", ";
	os << size << " * sizeof(unsigned int));" << endl;
	mem += size * sizeof(unsigned int);

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	}
	else {
	    size = model.neuronN[i];
	}
	os << "    glbSpk" << model.neuronName[i] << " = new unsigned int[" << size << "];" << endl;
	os << "    deviceMemAllocate(&d_glbSpk" << model.neuronName[i];
	os << ", dd_glbSpk" << model.neuronName[i] << ", ";
	os << size << " * sizeof(unsigned int));" << endl;
	mem += size * sizeof(unsigned int);

	if (model.neuronNeedSpkEvnt[i]) {
	    size = model.neuronDelaySlots[i];
	    os << "    glbSpkCntEvnt" << model.neuronName[i] << " = new unsigned int[" << size << "];" << endl;
	    os << "    deviceMemAllocate(&d_glbSpkCntEvnt" << model.neuronName[i];
	    os << ", dd_glbSpkCntEvnt" << model.neuronName[i] << ", ";
	    os << size << " * sizeof(unsigned int));" << endl;
	    mem += size * sizeof(unsigned int);

	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "    glbSpkEvnt" << model.neuronName[i] << " = new unsigned int[" << size << "];" << endl;
	    os << "    deviceMemAllocate(&d_glbSpkEvnt" << model.neuronName[i];
	    os << ", dd_glbSpkEvnt" << model.neuronName[i] << ", ";
	    os << size << " * sizeof(unsigned int));" << endl;
	    mem += size * sizeof(unsigned int);
	}

	if (model.neuronNeedSt[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "    sT" << model.neuronName[i] << " = new " << model.ftype << "[" << size << "];" << endl;
	    os << "    deviceMemAllocate(&d_sT" << model.neuronName[i];
	    os << ", dd_sT" << model.neuronName[i] << ", ";
	    os << size << " * sizeof(" << model.ftype << "));" << endl;
	    mem += size * theSize(model.ftype);
	}

	// Variable are queued only if they are referenced in forward synapse code.
	for (int j = 0; j < nModels[nt].varNames.size(); j++) {
	    if (model.neuronVarNeedQueue[i][j]) {
		size = model.neuronN[i] * model.neuronDelaySlots[i];
	    }
	    else {
		size = model.neuronN[i];
	    }
	    os << "    " << nModels[nt].varNames[j] << model.neuronName[i];
	    os << " = new " << nModels[nt].varTypes[j] << "[" << size << "];" << endl;
	    os << "    deviceMemAllocate(&d_" << nModels[nt].varNames[j] << model.neuronName[i];
	    os << ", dd_" << nModels[nt].varNames[j] << model.neuronName[i] << ", ";
	    os << size << " * sizeof(" << nModels[nt].varTypes[j] << "));" << endl;
	    mem += size * theSize(nModels[nt].varTypes[j]);
	}
	os << endl; 
    }

    // ALLOCATE SYNAPSE VARIABLES
    for (int i = 0; i < model.synapseGrpN; i++) {
	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	size = model.neuronN[model.synapseTarget[i]];
	os << "    inSyn" << model.synapseName[i] << " = new " << model.ftype << "[" << size << "];" << endl;
	os << "    deviceMemAllocate(&d_inSyn" << model.synapseName[i];
	os << ", dd_inSyn" << model.synapseName[i];
	os << ", " << size << " * sizeof(" << model.ftype << "));" << endl; 
	mem += size * theSize(model.ftype);

	// note, if GLOBALG we put the value at compile time
	if (model.synapseGType[i] == INDIVIDUALID) {
	    size = (model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]]) / 32 + 1;
	    os << "    gp" << model.synapseName[i] << " = new uint32_t[" << size << "];" << endl;
	    os << "    deviceMemAllocate(&d_gp" << model.synapseName[i];
	    os << ", dd_gp" << model.synapseName[i];
	    os << ", " << size << " * sizeof(uint32_t));" << endl;
	    mem += size * sizeof(uint32_t);
	}

	// allocate user-defined weight model variables
	// if they are sparse, allocate later in the allocatesparsearrays function when we know the size of the network
	if ((model.synapseConnType[i] != SPARSE) && (model.synapseGType[i] == INDIVIDUALG)) {
	    size = model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]];
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "    " << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << " = new " << weightUpdateModels[st].varTypes[k] << "[" << size << "];" << ENDL;
		os << "    deviceMemAllocate(&d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", dd_" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", " << size << " * sizeof(" << weightUpdateModels[st].varTypes[k] << "));" << endl; 
		mem += size * theSize(weightUpdateModels[st].varTypes[k]);
	    } 
	}

	if (model.synapseGType[i] == INDIVIDUALG) { // not needed for GLOBALG, INDIVIDUALID
	    size = model.neuronN[model.synapseTarget[i]];
	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
		os << "    " << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << " = new " << postSynModels[pst].varTypes[k] << "[" << size << "];" << endl;
		os << "    deviceMemAllocate(&d_" << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", dd_" << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", " << size << " * sizeof(" << postSynModels[pst].varTypes[k] << "));" << endl;      
		mem += size * theSize(postSynModels[pst].varTypes[k]);
	    }
	}
	os << endl;
    }  
    os << "}" << endl << endl;


    // ------------------------------------------------------------------------
    // initializing variables

    os << "void initialize()" << endl;
    os << "{" << endl;

    if (model.seed == 0) {
	os << "    srand((unsigned int) time(NULL));" << endl;
    }
    else {
	os << "    srand((unsigned int) " << model.seed << ");" << endl;
    }
    os << endl;

    // INITIALISE NEURON VARIABLES
    os << "    // neuron variables" << endl;
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	if (model.neuronDelaySlots[i] > 1) {
	    os << "    spkQuePtr" << model.neuronName[i] << " = 0;" << endl;
	}

	if ((model.neuronNeedTrueSpk[i]) && (model.neuronDelaySlots[i] > 1)) {
	    os << "    for (int i = 0; i < " << model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "        glbSpkCnt" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	    os << "    for (int i = 0; i < " << model.neuronN[i] * model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "        glbSpk" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	}
	else {
	    os << "    glbSpkCnt" << model.neuronName[i] << "[0] = 0;" << endl;
	    os << "    for (int i = 0; i < " << model.neuronN[i] << "; i++) {" << endl;
	    os << "        glbSpk" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	}

	if ((model.neuronNeedSpkEvnt[i]) && (model.neuronDelaySlots[i] > 1)) {
	    os << "    for (int i = 0; i < " << model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "        glbSpkCntEvnt" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	    os << "    for (int i = 0; i < " << model.neuronN[i] * model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "        glbSpkEvnt" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	}
	else if (model.neuronNeedSpkEvnt[i]) {
	    os << "    glbSpkCntEvnt" << model.neuronName[i] << "[0] = 0;" << endl;
	    os << "    for (int i = 0; i < " << model.neuronN[i] << "; i++) {" << endl;
	    os << "        glbSpkEvnt" << model.neuronName[i] << "[i] = 0;" << endl;
	    os << "    }" << endl;
	}

	if (model.neuronNeedSt[i]) {
	    os << "    for (int i = 0; i < " << model.neuronN[i] * model.neuronDelaySlots[i] << "; i++) {" << endl;
	    os << "        sT" <<  model.neuronName[i] << "[i] = -10000.0;" << endl;
	    os << "    }" << endl;
	}

	for (int j = 0; j < nModels[nt].varNames.size(); j++) {
	    if (model.neuronVarNeedQueue[i][j]) {
		os << "    for (int i = 0; i < " << model.neuronN[i] * model.neuronDelaySlots[i] << "; i++) {" << endl;
	    }
	    else {
		os << "    for (int i = 0; i < " << model.neuronN[i] << "; i++) {" << endl;
	    }
	    os << "        " << nModels[nt].varNames[j] << model.neuronName[i] << "[i] = " << model.neuronIni[i][j] << ";" << endl;
	    os << "    }" << endl;
	}

	if (model.neuronType[i] == POISSONNEURON) {
	    os << "    for (int i = 0; i < " << model.neuronN[i] << "; i++) {" << endl;
	    os << "        seed" << model.neuronName[i] << "[i] = rand();" << endl;
	    os << "    }" << endl;
	}

	if ((model.neuronType[i] == IZHIKEVICH) && (DT != 1.0)) {
	    os << "    fprintf(stderr,\"WARNING: You use a time step different than 1 ms. Izhikevich model behaviour may not be robust.\\n\"); " << endl;
	}
    }
    os << endl;

    // INITIALISE SYNAPSE VARIABLES
    os << "    // synapse variables" << endl;
    for (int i = 0; i < model.synapseGrpN; i++) {
	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	os << "    for (int i = 0; i < " << model.neuronN[model.synapseTarget[i]] << "; i++) {" << endl;
	os << "        inSyn" << model.synapseName[i] << "[i] = 0.0f;" << endl;
	os << "    }" << endl;

	if ((model.synapseConnType[i] != SPARSE) && (model.synapseGType[i] == INDIVIDUALG)) {
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "    for (int i = 0; i < " << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << "; i++) {" << endl;
		os << "        " << weightUpdateModels[st].varNames[k] << model.synapseName[i] << "[i] = " << model.synapseIni[i][k] << ";" << endl;
		os << "    }" << endl;
	    }
	}

	if (model.synapseGType[i] == INDIVIDUALG) {
	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
		os << "    for (int i = 0; i < " << model.neuronN[model.synapseTarget[i]] << "; i++) {" << endl;
		os << "        " << postSynModels[pst].varNames[k] << model.synapseName[i] << "[i] = " << model.postSynIni[i][k] << ";" << endl;
		os << "    }" << endl;
	    }
	}
    }
    os << endl << endl;

    os << "    copyStateToDevice();" << endl << endl;

    os << "    //initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later" << endl;
    os << "}" << endl << endl;


    // ------------------------------------------------------------------------
    // allocating conductance arrays for sparse matrices

    for (int i = 0; i < model.synapseGrpN; i++) {	
	if (model.synapseConnType[i] == SPARSE) {
	    os << "void allocate" << model.synapseName[i] << "(unsigned int connN)" << "{" << endl;
	    os << "// Allocate host side variables" << endl;
	    os << "  C" << model.synapseName[i] << ".connN= connN;" << endl;
	    os << "  C" << model.synapseName[i] << ".indInG= new unsigned int[" << model.neuronN[model.synapseSource[i]] + 1 << "];" << endl;
	    os << "  C" << model.synapseName[i] << ".ind= new unsigned int[connN];" << endl;   
	    if (model.synapseUsesPostLearning[i]) {
		os << "  C" << model.synapseName[i] << ".revIndInG= new unsigned int[" << model.neuronN[model.synapseSource[i]] + 1 << "];" << endl;
		os << "  C" << model.synapseName[i] << ".revInd= new unsigned int[connN];" << endl;       
		os << "  C" << model.synapseName[i] << ".remap= new unsigned int[connN];" << endl;       
	    }
	    int st= model.synapseType[i];
	    string size = "C" + model.synapseName[i]+".connN";
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os  << weightUpdateModels[st].varNames[k];
		os << model.synapseName[i]<< "= new " << weightUpdateModels[st].varTypes[k] << "[" << size << "];" << ENDL;
	    }
	    os << "// Allocate device side variables" << endl;
	    os << "  deviceMemAllocate( &d_indInG" << model.synapseName[i] << ", dd_indInG" << model.synapseName[i];
	    os << ", sizeof(unsigned int) * ("<< model.neuronN[model.synapseSource[i]] + 1 <<"));" << endl;
	    os << "  deviceMemAllocate( &d_ind" << model.synapseName[i] << ", dd_ind" << model.synapseName[i];
	    os << ", sizeof(unsigned int) * (" << size << "));" << endl;
	    if (model.synapseUsesPostLearning[i]) {
		os << "  deviceMemAllocate( &d_revIndInG" << model.synapseName[i] << ", dd_revIndInG" << model.synapseName[i];
		os << ", sizeof(unsigned int) * ("<< model.neuronN[model.synapseTarget[i]] + 1 <<"));" << endl;
		os << "  deviceMemAllocate( &d_revInd" << model.synapseName[i] << ", dd_revInd" << model.synapseName[i];
		os << ", sizeof(unsigned int) * (" << size <<"));" << endl;
		os << "  deviceMemAllocate( &d_remap" << model.synapseName[i] << ", dd_remap" << model.synapseName[i];
		os << ", sizeof(unsigned int) * ("<< size << "));" << endl;
	    }
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {     
		os << "deviceMemAllocate(&d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", dd_" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", sizeof("  << weightUpdateModels[st].varTypes[k] << ")*(" << size << "));" << endl;       
	    }
	    os << "}" << endl; 
	    os << endl;
	}
    }


    // ------------------------------------------------------------------------
    // initializing conductance arrays for sparse matrices

    os << "void initializeSparseArray(SparseProjection C,  unsigned int * dInd, unsigned int * dIndInG, unsigned int preN)" << "{" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "}" << endl; 
 	
    os << "void initializeSparseArrayRev(SparseProjection C,  unsigned int * dRevInd, unsigned int * dRevIndInG, unsigned int * dRemap, unsigned int postN)" << "{" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dRevInd, C.revInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dRevIndInG, C.revIndInG, (postN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "  CHECK_CUDA_ERRORS(cudaMemcpy(dRemap, C.remap, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
    os << "}" << endl; 


    // ------------------------------------------------------------------------
    // initializing sparse arrays

    os << "void initializeAllSparseArrays() {" << endl;
    for (int i = 0; i < model.synapseGrpN; i++) {
	if (model.synapseConnType[i] == SPARSE) {
	    os << "size_t size;" << endl;
	    break;
	}
    }
    for (int i= 0; i < model.synapseGrpN; i++) {
	if (model.synapseConnType[i]==SPARSE){
	    os << "size = C" << model.synapseName[i] << ".connN;" << ENDL;
	    os << "  initializeSparseArray(C" << model.synapseName[i] << ",";
	    os << "  d_ind" << model.synapseName[i] << ",";
	    os << "  d_indInG" << model.synapseName[i] << ",";
	    os << model.neuronN[model.synapseSource[i]] <<");" << endl;
	    if (model.synapseUsesPostLearning[i]) {
		os << "  initializeSparseArrayRev(C" << model.synapseName[i] << ",";
		os << "  d_revInd" << model.synapseName[i] << ",";
		os << "  d_revIndInG" << model.synapseName[i] << ",";
		os << "  d_remap" << model.synapseName[i] << ",";
		os << model.neuronN[model.synapseTarget[i]] <<");" << endl;
	    }
	    int st= model.synapseType[i];
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i] << ", "  << weightUpdateModels[st].varNames[k] << model.synapseName[i] << ", sizeof(" << weightUpdateModels[st].varTypes[k] << ") * size , cudaMemcpyHostToDevice));" << endl; 
	    }
	}
    }
    os << "}" << endl; 
    os << endl;


    // ------------------------------------------------------------------------
    // freeing global memory structures

    os << "void freeMem()" << endl;
    os << "{" << endl;

    // FREE NEURON VARIABLES
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	os << "    delete[] glbSpkCnt" << model.neuronName[i] << ";" << endl;
	os << "    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCnt" << model.neuronName[i] << "));" << endl;
	os << "    delete[] glbSpk" << model.neuronName[i] << ";" << endl;
	os << "    CHECK_CUDA_ERRORS(cudaFree(d_glbSpk" << model.neuronName[i] << "));" << endl;
	if (model.neuronNeedSpkEvnt[i]) {
	    os << "    delete[] glbSpkCntEvnt" << model.neuronName[i] << ";" << endl;
	    os << "    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvnt" << model.neuronName[i] << "));" << endl;
	    os << "    delete[] glbSpkEvnt" << model.neuronName[i] << ";" << endl;
	    os << "    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvnt" << model.neuronName[i] << "));" << endl;
	}
	if (model.neuronNeedSt[i]) {
	    os << "    delete[] sT" << model.neuronName[i] << ";" << endl;
	    os << "    CHECK_CUDA_ERRORS(cudaFree(d_sT" << model.neuronName[i] << "));" << endl;
	}
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << "    delete[] " << nModels[nt].varNames[k] << model.neuronName[i] << ";" << endl;
	    os << "    CHECK_CUDA_ERRORS(cudaFree(d_" << nModels[nt].varNames[k] << model.neuronName[i] << "));" << endl;
	}
    }

    // FREE SYNAPSE VARIABLES
    for (int i = 0; i < model.synapseGrpN; i++) {
	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	os << "    delete[] inSyn" << model.synapseName[i] << ";" << endl;
	os << "    CHECK_CUDA_ERRORS(cudaFree(d_inSyn" << model.synapseName[i] << "));" << endl;
	if (model.synapseConnType[i] == SPARSE) {
	    os << "    delete[] C" << model.synapseName[i] << ".indInG;" << endl;
	    os << "    delete[] C" << model.synapseName[i] << ".ind;" << endl;  
	    if (model.synapseUsesPostLearning[i]) {
		os << "    delete[] C" << model.synapseName[i] << ".revIndInG;" << endl;
		os << "    delete[] C" << model.synapseName[i] << ".revInd;" << endl;  
		os << "    delete[] C" << model.synapseName[i] << ".remap;" << endl;
	    }
	}
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "    delete[] gp" << model.synapseName[i] << ";" << endl;
	    os << "    CHECK_CUDA_ERRORS(cudaFree(d_gp" << model.synapseName[i] << "));" <<endl;  	
	}
	if (model.synapseGType[i] == INDIVIDUALG) {
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "    CHECK_CUDA_ERRORS(cudaFree(d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i] << "));" << endl;
	    }
	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
		os << "    delete[] " << postSynModels[pst].varNames[k] << model.synapseName[i] << ";" << endl;
		os << "    CHECK_CUDA_ERRORS(cudaFree(d_" << postSynModels[pst].varNames[k] << model.synapseName[i] << "));" << endl;
	    }
	}
    }
    os << "}" << endl << endl;


// ------------------------------------------------------------------------
//! \brief Method for cleaning up and resetting device while quitting GeNN

  os << "void exitGeNN(){" << endl;  
  os << "  freeMem();" << endl;
  os << "  cudaDeviceReset();" << endl;
  os << "}" << endl;


    // ------------------------------------------------------------------------
    // the actual time stepping procedure

    os << "void stepTimeCPU(";
    for (int i = 0; i < model.neuronGrpN; i++) {
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

    if (model.synapseGrpN > 0) {
	os << "    if (t > 0.0) {" << endl;
	if (model.timing) os << "        synapse_timer.startTimer();" << endl;
	os << "        calcSynapsesCPU(t);" << endl;
	if (model.timing) {
	    os << "        synapse_timer.stopTimer();" << endl;
	    os << "        synapse_tme+= synapse_timer.getElapsedTime();"<< endl;
	}
	if (model.lrnGroups > 0) {
	    if (model.timing) os << "        learning_timer.startTimer();" << endl;
	    os << "        learnSynapsesPostHost(t);" << endl;
	    if (model.timing) {
		os << "        learning_timer.stopTimer();" << endl;
		os << "        learning_tme+= learning_timer.getElapsedTime();" << endl;
	    }
	}
	os << "    }" << endl;
    }
    if (model.timing) os << "    neuron_timer.startTimer();" << endl;
    os << "    calcNeuronsCPU(";
    for (int i = 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    os << "rates" << model.neuronName[i] << ", ";
	    os << "offset" << model.neuronName[i] << ", ";
	}
	if (model.receivesInputCurrent[i] >= 2) {
	    os << "inputI" << model.neuronName[i] << ", ";
	}
    }
    os << "t);" << endl;
    if (model.timing) {
	os << "    neuron_timer.stopTimer();" << endl;
	os << "    neuron_tme+= neuron_timer.getElapsedTime();" << endl;
    }
    os << "}" << endl;
    os.close();


    // ------------------------------------------------------------------------
    // finish up

    mos << "Global memory required for core model: " << mem/1e6 << " MB. " << endl;
    mos << deviceProp[theDev].totalGlobalMem << " for device " << theDev << endl;  
  
    if (memremsparse != 0) {
	int connEstim = int(memremsparse / (theSize(model.ftype) + sizeof(unsigned int)));
	mos << "Remaining mem is " << memremsparse/1e6 << " MB." << endl;
	mos << "You may run into memory problems on device" << theDev;
	mos << " if the total number of synapses is bigger than " << connEstim;
	mos << ", which roughly stands for " << int(connEstim/model.sumNeuronN[model.neuronGrpN - 1]);
	mos << " connections per neuron, without considering any other dynamic memory load." << endl;
    }
    else {
	if (0.5 * deviceProp[theDev].totalGlobalMem < mem) {
	    mos << "memory required for core model (" << mem/1e6;
	    mos << "MB) is more than 50% of global memory on the chosen device";
	    mos << "(" << deviceProp[theDev].totalGlobalMem/1e6 << "MB)." << endl;
	    mos << "Experience shows that this is UNLIKELY TO WORK ... " << endl;
	}
    }
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
    size_t size;
    unsigned int nt, st, pst;
    ofstream os;

    mos << "entering GenRunnerGPU" << endl;
    name= path + toString("/") + model.name + toString("_CODE/runnerGPU.cc");
    os.open(name.c_str());
    writeHeader(os);

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << endl;
    os << "/*! \\file runnerGPU.cc" << endl << endl;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the host side code for a GPU simulator version." << endl;
    os << "*/" << endl;
    os << "//-------------------------------------------------------------------------" << endl << endl;

  
    for (int i = 0; i < model.neuronGrpN; i++) {
	if (model.receivesInputCurrent[i] >= 2) {
	    os << "#include <string>" << endl;
	    break;
	}
    }
    os << endl;

    if ((deviceProp[theDev].major >= 2) || (deviceProp[theDev].minor >= 3)) {
	os << "__device__ double atomicAdd(double* address, double val)" << endl;
	os << "{" << endl;
	os << "    unsigned long long int* address_as_ull =" << endl;
	os << "                                          (unsigned long long int*)address;" << endl;
	os << "    unsigned long long int old = *address_as_ull, assumed;" << endl;
	os << "    do {" << endl;
	os << "        assumed = old;" << endl;
	os << "        old = atomicCAS(address_as_ull, assumed, " << endl;
	os << "                        __double_as_longlong(val + " << endl;
	os << "                        __longlong_as_double(assumed)));" << endl;
	os << "    } while (assumed != old);" << endl;
	os << "    return __longlong_as_double(old);" << endl;
	os << "}" << endl << endl;
    }

    if (deviceProp[theDev].major < 2) {
	os << "__device__ float atomicAddoldGPU(float* address, float val)" << endl;
	os << "{" << endl;
	os << "    int* address_as_ull =" << endl;
	os << "                                          (int*)address;" << endl;
	os << "    int old = *address_as_ull, assumed;" << endl;
	os << "    do {" << endl;
	os << "        assumed = old;" << endl;
	os << "        old = atomicCAS(address_as_ull, assumed, " << endl;
	os << "                        __float_as_int(val + " << endl;
	os << "                        __int_as_float(assumed)));" << endl;
	os << "    } while (assumed != old);" << endl;
	os << "    return __int_as_float(old);" << endl;
	os << "}" << endl << endl;
    }	

    os << "#include \"neuronKrnl.cc\"" << endl;
    if (model.synapseGrpN > 0) {
	os << "#include \"synapseKrnl.cc\"" << endl;
    }

    os << "// ------------------------------------------------------------------------" << endl;
    os << "// copying things to device" << endl << endl;

    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	// neuron state variables
	os << "void push" << model.neuronName[i] << "StateToDevice()" << ENDL;
	os << OB(1050);

	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    if (model.neuronVarNeedQueue[i][k]) {
		size = model.neuronN[i] * model.neuronDelaySlots[i];
	    }
	    else {
		size = model.neuronN[i];	
	    }
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << nModels[nt].varNames[k] << model.neuronName[i];
	    os << ", " << nModels[nt].varNames[k] << model.neuronName[i];
	    os << ", " << size << " * sizeof(" << nModels[nt].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl;
	}

	os << CB(1050);
	os << ENDL;	

	// neuron spike variables
	os << "void push" << model.neuronName[i] << "SpikesToDevice()" << ENDL;
	os << OB(1060);

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronDelaySlots[i];
	}
	else {
	    size = 1;
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCnt" << model.neuronName[i];
	os << ", glbSpkCnt" << model.neuronName[i];
	os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	}
	else {
	    size = model.neuronN[i];
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpk" << model.neuronName[i];
	os << ", glbSpk" << model.neuronName[i];
	os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;

	if (model.neuronNeedSpkEvnt[i]) {
	    size = model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvnt" << model.neuronName[i];
	    os << ", glbSpkCntEvnt" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;

	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvnt" << model.neuronName[i];
	    os << ", glbSpkEvnt" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
	}

	if (model.neuronNeedSt[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_sT" << model.neuronName[i];
	    os << ", sT" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyHostToDevice));" << endl;
	}

	os << CB(1060);
	os << ENDL;	
    }

    // synapse variables
    for (int i = 0; i < model.synapseGrpN; i++) {
	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	os << "void push" << model.synapseName[i] << "ToDevice()" << ENDL;
	os << OB(1100);

	if (model.synapseGType[i] == INDIVIDUALG) { // INDIVIDUALG
	    if (model.synapseConnType[i] != SPARSE) {
		os << "size_t size = " << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << ";" << ENDL;
	    }
	    else {
		os << "size_t size = C" << model.synapseName[i] << ".connN;" << ENDL;
	    }
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", " << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", size * sizeof(" << weightUpdateModels[st].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl; 
	    }

	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
		size = model.neuronN[model.synapseTarget[i]];
		os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_" << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", " << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", " << size << " * sizeof(" << postSynModels[pst].varTypes[k] << "), cudaMemcpyHostToDevice));" << endl; 
	    }
	}

	else if (model.synapseGType[i] == INDIVIDUALID) { // INDIVIDUALID
	    size = (model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]]) / 32 + 1;
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_gp" << model.synapseName[i];
	    os << ", gp" << model.synapseName[i];
	    os << ", " << size << " * sizeof(uint32_t), cudaMemcpyHostToDevice));" << endl;
	}

	size = model.neuronN[model.synapseTarget[i]];
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(d_inSyn" << model.synapseName[i];
	os << ", inSyn" << model.synapseName[i];
	os << ", " << size << " * sizeof(" << model.ftype << "), cudaMemcpyHostToDevice));" << endl; 

	os << CB(1100);
	os << ENDL;
    }


    os << "// ------------------------------------------------------------------------" << endl;
    os << "// copying things from device" << endl << endl;

    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	// neuron state variables
	os << "void pull" << model.neuronName[i] << "StateFromDevice()" << ENDL;
	os << OB(1050);

	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    if (model.neuronVarNeedQueue[i][k]) {
		size = model.neuronN[i] * model.neuronDelaySlots[i];
	    }
	    else {
		size = model.neuronN[i];
	    }
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << nModels[nt].varNames[k] << model.neuronName[i];
	    os << ", d_" << nModels[nt].varNames[k] << model.neuronName[i];
	    os << ", " << size << " * sizeof(" << nModels[nt].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl;
	}

	os << CB(1050);
	os << ENDL;

	// neuron spike variables
	os << "void pull" << model.neuronName[i] << "SpikesFromDevice()" << ENDL;
	os << OB(1060);

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronDelaySlots[i];
	}
	else {
	    size = 1;
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << model.neuronName[i];
	os << ", d_glbSpkCnt" << model.neuronName[i];
	os << ", " << size << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;

	if (model.neuronNeedTrueSpk[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	}
	else {
	    size = model.neuronN[i];
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << model.neuronName[i];
	os << ", d_glbSpk" << model.neuronName[i];
	os << ", " << size << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;

	if (model.neuronNeedSpkEvnt[i]) {
	    size = model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvnt" << model.neuronName[i];
	    os << ", d_glbSpkCntEvnt" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;

	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvnt" << model.neuronName[i];
	    os << ", d_glbSpkEvnt" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;
	}

	if (model.neuronNeedSt[i]) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(sT" << model.neuronName[i];
	    os << ", d_sT" << model.neuronName[i];
	    os << ", " << size << " * sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;
	}

	os << CB(1060);
	os << ENDL;	
    }

    // synapse variables
    for (int i = 0; i < model.synapseGrpN; i++) {
	st = model.synapseType[i];
	pst = model.postSynapseType[i];

	os << "void pull" << model.synapseName[i] << "FromDevice()" << ENDL;
	os << OB(1100);

	if (model.synapseGType[i] == INDIVIDUALG) { // INDIVIDUALG
	    if (model.synapseConnType[i] != SPARSE) {
		os << "size_t size = " << model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]] << ";" << ENDL;
	    }
	    else {
		os << "size_t size = C" << model.synapseName[i] << ".connN;" << ENDL;
	    }
	    for (int k= 0, l= weightUpdateModels[st].varNames.size(); k < l; k++) {
		os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", d_"  << weightUpdateModels[st].varNames[k] << model.synapseName[i];
		os << ", size * sizeof(" << weightUpdateModels[st].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl; 
	    }

	    for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
		size = model.neuronN[model.synapseTarget[i]];
		os << "CHECK_CUDA_ERRORS(cudaMemcpy(" << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", d_"  << postSynModels[pst].varNames[k] << model.synapseName[i];
		os << ", " << size << " * sizeof(" << postSynModels[pst].varTypes[k] << "), cudaMemcpyDeviceToHost));" << endl; 
	    }
	}

	else if (model.synapseGType[i] == INDIVIDUALID) { // INDIVIDUALID
	    size = (model.neuronN[model.synapseSource[i]] * model.neuronN[model.synapseTarget[i]]) / 32 + 1;
	    os << "CHECK_CUDA_ERRORS(cudaMemcpy(gp" << model.synapseName[i];
	    os << ", d_gp" << model.synapseName[i];
	    os << ", " << size << " * sizeof(uint32_t), cudaMemcpyDeviceToHost));" << endl;
	}

	size = model.neuronN[model.synapseTarget[i]];
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(inSyn" << model.synapseName[i];
	os << ", d_inSyn" << model.synapseName[i];
	os << ", " << size << " * sizeof(" << model.ftype << "), cudaMemcpyDeviceToHost));" << endl; 

	os << CB(1100);
	os << ENDL;
    }


    // ------------------------------------------------------------------------
    // copying values to device
    
    os << "void copyStateToDevice()" << endl;
    os << OB(1110);

    for (int i = 0; i < model.neuronGrpN; i++) {
	os << "push" << model.neuronName[i] << "StateToDevice();" << ENDL;
	os << "push" << model.neuronName[i] << "SpikesToDevice();" << ENDL;
    }

    for (int i = 0; i < model.synapseGrpN; i++) {
	os << "push" << model.synapseName[i] << "ToDevice();" << ENDL;
    }

    os << CB(1110);
    os << endl;
    

    // ------------------------------------------------------------------------
    // copying values from device
    
    os << "void copyStateFromDevice()" << endl;
    os << OB(1120);
    
    for (int i = 0; i < model.neuronGrpN; i++) {
	os << "pull" << model.neuronName[i] << "StateFromDevice();" << ENDL;
	os << "pull" << model.neuronName[i] << "SpikesFromDevice();" << ENDL;
    }

    for (int i = 0; i < model.synapseGrpN; i++) {
	os << "pull" << model.synapseName[i] << "FromDevice();" << ENDL;
    }
    
    os << CB(1120);
    os << endl;


    // ------------------------------------------------------------------------
    // copying spikes from device                                            
    
    os << "void copySpikesFromDevice()" << endl;
    os << "{" << endl;

    for (int i = 0; i < model.neuronGrpN; i++) {
	if ((model.neuronNeedTrueSpk[i]) && (model.neuronDelaySlots[i] > 1)) {
	    size = model.neuronN[i] * model.neuronDelaySlots[i];
	}
	else {
	    size = model.neuronN[i];
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpk" << model.neuronName[i];
	os << ", d_glbSpk" << model.neuronName[i] << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;
    }

    os << "}" << endl;
    os << endl;
    

    // ------------------------------------------------------------------------
    // copying spike numbers from device

    os << "void copySpikeNFromDevice()" << endl;
    os << "{" << endl;

    for (int i = 0; i < model.neuronGrpN; i++) {
	if ((model.neuronNeedTrueSpk[i]) && (model.neuronDelaySlots[i] > 1)) {
	    size = model.neuronDelaySlots[i];
	}
	else {
	    size = 1;
	}
	os << "CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCnt" << model.neuronName[i];
	os << ", d_glbSpkCnt" << model.neuronName[i] << ", " << size << "* sizeof(unsigned int), cudaMemcpyDeviceToHost));" << endl;
    }

    os << "}" << endl;
    os << endl;


    // ------------------------------------------------------------------------
    // the actual time stepping procedure
    
    os << "void stepTimeGPU(";
    for (int i= 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    os << model.RNtype << " *rates" << model.neuronName[i];
	    os << ", // pointer to the rates of the Poisson neurons in grp ";
	    os << model.neuronName[i] << endl;
	    os << "unsigned int offset" << model.neuronName[i];
	    os << ", // offset on pointer to the rates in grp ";
	    os << model.neuronName[i] << endl;
	}
	if (model.receivesInputCurrent[i]>=2) {
	    os << model.ftype << " *d_inputI" << model.neuronName[i];
	    os << ", // Explicit input to the neurons in grp ";
	    os << model.neuronName[i] << endl;
	}
    }
    os << model.ftype << " t)" << endl;
    os << "{" << endl;

    if (model.synapseGrpN > 0) { 
	unsigned int synapseGridSz = model.padSumSynapseKrnl[model.synapseGrpN - 1];   
	os << "//model.padSumSynapseTrgN[model.synapseGrpN - 1] is " << model.padSumSynapseKrnl[model.synapseGrpN - 1] << endl; 
	synapseGridSz = synapseGridSz / synapseBlkSz;
	os << "dim3 sThreads(" << synapseBlkSz << ", 1);" << endl;
	os << "dim3 sGrid(" << synapseGridSz << ", 1);" << endl;
	os << endl;
    }
    if (model.lrnGroups > 0) {
	unsigned int learnGridSz = model.padSumLearnN[model.lrnGroups - 1];
	learnGridSz = ceil((float) learnGridSz / learnBlkSz);
	os << "dim3 lThreads(" << learnBlkSz << ", 1);" << endl;
	os << "dim3 lGrid(" << learnGridSz << ", 1);" << endl;
	os << endl;
    }


    unsigned int neuronGridSz = model.padSumNeuronN[model.neuronGrpN - 1];
    neuronGridSz = ceil((float) neuronGridSz / neuronBlkSz);
    os << "dim3 nThreads(" << neuronBlkSz << ", 1);" << endl;
    if (neuronGridSz < deviceProp[theDev].maxGridSize[1]) {
	os << "dim3 nGrid(" << neuronGridSz << ", 1);" << endl;
    }
    else {
	int sqGridSize = ceil((float) sqrt((float) neuronGridSz));
	os << "dim3 nGrid(" << sqGridSize << ","<< sqGridSize <<");" << endl;
    }
    os << endl;
    if (model.synapseGrpN > 0) {
	os << "if (t > 0.0) {" << endl;
	if (model.timing) os << "cudaEventRecord(synapseStart);" << endl; 
	os << "calcSynapses <<< sGrid, sThreads >>> (";
	for (int i= 0; i < model.synapseGrpN; i++) {
	    
	    for (int i=0; i< model.synapseName.size(); i++){
		int st= model.synapseType[i];
		for (int k= 0, l= weightUpdateModels[st].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		    os << weightUpdateModels[st].extraGlobalSynapseKernelParameters[k] << ", " ;
		    os << model.synapseName[i];
		}
	    }
	}
	os << "t);"<< endl;
	if (model.timing) os << "cudaEventRecord(synapseStop);" << endl;
	if (model.lrnGroups > 0) {
	    if (model.timing) os << "cudaEventRecord(learningStart);" << endl;
	    os << "learnSynapsesPost <<< lGrid, lThreads >>> (";         
	    for (int i = 0; i < model.synapseName.size(); i++){
		int st= model.synapseType[i];
		for (int k= 0, l= weightUpdateModels[st].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		    os << weightUpdateModels[st].extraGlobalSynapseKernelParameters[k] << ", ";
		    os << model.synapseName[i];
		}
	    }
	    os << "t);" << endl;
	    if (model.timing) os << "cudaEventRecord(learningStop);" << endl;
	}
	os << "}" << endl;
    }
    if (model.timing) os << "cudaEventRecord(neuronStart);" << endl;
    os << "calcNeurons <<< nGrid, nThreads >>> (";
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
    if (model.timing) {
	os << "cudaEventRecord(neuronStop);" << endl;
	os << "cudaEventSynchronize(neuronStop);" << endl;
	os << "float tmp;" << endl;
	if (model.synapseGrpN > 0) {
	    os << "cudaEventElapsedTime(&tmp, synapseStart, synapseStop);" << endl;
	    os << "synapse_tme+= tmp/1000.0;" << endl;
	}
	if (model.lrnGroups > 0) {
	    os << "cudaEventElapsedTime(&tmp, learningStart, learningStop);" << endl;
	    os << "learning_tme+= tmp/1000.0;" << endl;
	}
	os << "cudaEventElapsedTime(&tmp, neuronStart, neuronStop);" << endl;
	os << "neuron_tme+= tmp/1000.0;" << endl;
    }
    os << "}" << endl;
    os.close();
    //cout << "done with generating GPU runner" << endl;
}
