/*--------------------------------------------------------------------------
  Author: Thomas Nowotny

  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 

  email to:  T.Nowotny@sussex.ac.uk

  initial version: 2010-02-07

  --------------------------------------------------------------------------*/

#include <string>
#include "global.h"
#include "CodeHelper.cc"
//------------------------------------------------------------------------
/*! \file generateKernels.cc

  \brief Contains functions that generate code for CUDA kernels. Part of the code generation section.
*/
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA kernel that simulates all neurons in the model.

  The code generated upon execution of this function is for defining GPU side global variables that will hold model state in the GPU global memory and for the actual kernel function for simulating the neurons for one time step.
*/
//-------------------------------------------------------------------------

unsigned int nt;
short *isGrpVarNeeded;
CodeHelper hlp;

void genNeuronKernel(NNmodel &model, //!< Model description 
		     string &path,  //!< path for code output
		     ostream &mos //!< output stream for messages
    )
{
    //hlp.setVerbose(true);//this will show the generation of bracketing (brace) levels. Helps to debug a bracketing issue

    // write header content
    string name, s, localID;
    ofstream os;
    isGrpVarNeeded = new short[model.neuronGrpN];

    name= path + toString("/") + model.name + toString("_CODE/neuronKrnl.cc");
    os.open(name.c_str());

    writeHeader(os);
    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_neuronKrnl_cc" << ENDL;
    os << "#define _" << model.name << "_neuronKrnl_cc" << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the neuron kernel function." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    // global device variables
    os << "// relevant neuron variables" << ENDL;
    os << "__device__ volatile unsigned int d_done;" << ENDL;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	isGrpVarNeeded[i] = 0;

	//these now hold just true spikes for benefit of the user (raster plots etc)
	os << "__device__ volatile unsigned int d_glbscnt" << model.neuronName[i] << ";" << ENDL;
	os << "__device__ volatile unsigned int d_glbSpk" << model.neuronName[i] << "[" << model.neuronN[i] << "];" << ENDL;


	if (model.neuronDelaySlots[i] == 1) {// no delays
	    os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model.neuronName[i] << ";" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkEvnt" << model.neuronName[i] << "[" << model.neuronN[i] << "];" << ENDL;
	}
	else { // with delays
	    os << "__device__ volatile unsigned int d_spkEvntQuePtr" << model.neuronName[i] << ";" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkEvntCnt" << model.neuronName[i] << "[";
	    os << model.neuronDelaySlots[i] << "];" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkEvnt" << model.neuronName[i] << "[";
	    os << model.neuronN[i] * model.neuronDelaySlots[i] << "];" << ENDL;
	}
	if (model.neuronType[i] != POISSONNEURON) {
	    for (int j= 0; j < model.inSyn[i].size(); j++) {
		os << "__device__ " << model.ftype << " d_inSyn" << model.neuronName[i] << j << "[" << model.neuronN[i] << "];";
		os << "// summed input for neurons in grp" << model.neuronName[i] << ENDL;
	    }
	}
	if (model.neuronNeedSt[i]) {
	    os << "__device__ volatile " << model.ftype << " d_sT" << model.neuronName[i] << "[" << model.neuronN[i] << "];" << ENDL;
	}
    }


    for (int i= 0; i < model.synapseGrpN; i++) {
	if ((model.synapseConnType[i] == SPARSE) && (model.neuronN[model.synapseTarget[i]] > synapseBlkSz)) {
	    isGrpVarNeeded[model.synapseTarget[i]] = 1; //! Binary flag for the sparse synapses to use atomic operations when the number of connections is bigger than the block size, and shared variables otherwise
	}
    }
  
    // kernel header
    os << "__global__ void calcNeurons(" << ENDL;

    for (int i= 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];
	if (nt == POISSONNEURON) {
	    // Note: Poisson neurons only used as input neurons; they do not receive any inputs
	    os << "unsigned int *d_rates" << model.neuronName[i];
	    os << ", // poisson \"rates\" of grp " << model.neuronName[i] << ENDL;
	    os << "unsigned int offset" << model.neuronName[i];
	    os << ", // poisson \"rates\" offset of grp " << model.neuronName[i] << ENDL;
	}
	if (model.receivesInputCurrent[i] > 1) {
	    os << model.ftype << " *d_inputI" << model.neuronName[i];
	    os << ", // explicit input current to grp " << model.neuronName[i] << ENDL;
	}
	for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " *d_" << nModels[nt].varNames[k];
	    os << model.neuronName[i]<< ", " << ENDL;
	}
	for (int k= 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    os << nModels[nt].extraGlobalNeuronKernelParameterTypes[k];
	    os << " " << nModels[nt].extraGlobalNeuronKernelParameters[k];
	    os << model.neuronName[i] << ", " << ENDL;
	}
    }


    for (int i=0; i< model.synapseName.size(); i++){
	int pst= model.postSynapseType[i];
	for (int k= 0, l= postSynModels[pst].varNames.size(); k < l; k++) {
	    os << postSynModels[pst].varTypes[k] << " *d_" << postSynModels[pst].varNames[k];
	    os << model.synapseName[i]<< ", " << ENDL;
	}
    }


    os << model.ftype << " t // absolute time" << ENDL;
    os << ")" << ENDL;

    // kernel code
    os << OB(5);
    unsigned int neuronGridSz = model.padSumNeuronN[model.neuronGrpN - 1];
    neuronGridSz = neuronGridSz / neuronBlkSz;
    if (neuronGridSz < deviceProp[theDev].maxGridSize[1]){
	os << "unsigned int id = " << neuronBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
    }
    else {
      os << "unsigned int id = " << neuronBlkSz << " * (blockIdx.x * " << ceil(sqrt((float) neuronGridSz));
      os << " + blockIdx.y) + threadIdx.x;" << ENDL;
    }
    //these variables deal with high V "spike type events"
    os << "__shared__ volatile unsigned int posSpkEvnt;" << ENDL;
    os << "__shared__ unsigned int shSpkEvnt[" << neuronBlkSz << "];" << ENDL;
    os << "unsigned int spkEvntIdx;" << ENDL;
    os << "__shared__ volatile unsigned int spkEvntCount;" << ENDL; //was scnt

    //these variables now deal only with true spikes , not high V "events"
    os << "__shared__ unsigned int shSpk[" << neuronBlkSz << "];" << ENDL;
    os << "__shared__ volatile unsigned int posSpk;" << ENDL;
    os << "unsigned int spkIdx;" << ENDL; //was sidx
    os << "__shared__ volatile unsigned int spkCount;" << ENDL; //was scnt

    os << ENDL;
    // Reset global spike counting vars here if there are no synapses at all
    if (model.synapseGrpN == 0) {
	os << "if (id == 0)" << OB(6);
	for (int j = 0; j < model.neuronGrpN; j++) {
	    os << "d_glbscnt" << model.neuronName[j] << " = 0;" << ENDL;
	    if (model.neuronDelaySlots[j] != 1) {
		os << "d_spkEvntQuePtr" << model.neuronName[j] << " = (d_spkEvntQuePtr";
		os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		os << "d_glbSpkEvntCnt" << model.neuronName[j] << "[d_spkEvntQuePtr";
		os << model.neuronName[j] << "] = 0;" << ENDL;
	    }
	    else {
		os << "d_glbSpkEvntCnt" << model.neuronName[j] << " = 0;" << ENDL;
	    }
	}
	os << CB(6);
	os << "__threadfence();" << ENDL;
    }
    
    os << "if (threadIdx.x == 0)" << OB(7) ;
    os << "spkEvntCount = 0;" << ENDL ;
    os << CB(7);
    os << "if (threadIdx.x == 1)" << OB(8) ;
    os << "spkCount = 0;" << ENDL ;
    os << CB(8);
    os << "__syncthreads();" << ENDL;

    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	if (i == 0) {
	    os << "if (id < " << model.padSumNeuronN[i] << ")" << OB(10);
	    localID = string("id");
	}
	else {
	    os << "if ((id >= " << model.padSumNeuronN[i-1] << ") && ";
	    os << "(id < " << model.padSumNeuronN[i] << "))" << OB(10);
	    os << "unsigned int lid;" << ENDL;
	    os << "lid = id - " << model.padSumNeuronN[i-1] << ";" << ENDL;
	    localID = string("lid");
	}
	os << "// only do this for existing neurons" << ENDL;
	os << "if (" << localID << " < " << model.neuronN[i] << ")" << OB(20);
	os << "// pull V values in a coalesced access" << ENDL;
	if (nt == POISSONNEURON) {
	    os << "unsigned int lrate = d_rates" << model.neuronName[i];
	    os << "[offset" << model.neuronName[i] << " + " << localID << "]";
	    if (DT != 0.5) {
		os << "*" << DT/0.5;
	    }
	    os << ";" << ENDL;
	}
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k];
	    os << " = d_" <<  nModels[nt].varNames[k] << model.neuronName[i] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "(((d_spkEvntQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
		os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
	    }
	    os << localID << "];" << ENDL;
	}
	if (nt != POISSONNEURON) {
	    os << "// pull inSyn values in a coalesced access" << ENDL;
	    for (int j = 0; j < model.inSyn[i].size(); j++) {
		os << model.ftype << " linSyn" << j << " = d_inSyn" << model.neuronName[i] << j << "[" << localID << "];" << ENDL;
	    }
	    os << model.ftype << " Isyn = 0;" << ENDL;
			
	    if (model.inSyn[i].size() > 0) {
		for (int j = 0; j < model.inSyn[i].size(); j++) {
		    os << "// Synapse " << j << " of Population " << i << ENDL;
		    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames.size(); k < l; k++) {
			os << postSynModels[model.postSynapseType[model.inSyn[i][j]]].varTypes[k] << " lps" << postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] << j;
			os << " = d_" <<  postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] << model.synapseName[model.inSyn[i][j]] << "[";
			os << localID << "];" << ENDL;
		    }
			    
		    os << "Isyn += ";
		    string psCode = postSynModels[model.postSynapseType[model.inSyn[i][j]]].postSyntoCurrent;
			    
		    substitute(psCode, tS("$(inSyn)"), tS("linSyn")+tS(j));
			    
		    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
			substitute(psCode, tS("$(") + nModels[nt].varNames[k] + tS(")"),
				   tS("l") + nModels[nt].varNames[k]);
		    }

		    for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
			substitute(psCode, tS("$(") + nModels[nt].pNames[k] + tS(")"),
				   tS("l") + nModels[nt].pNames[k]);
		    }

		    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].pNames.size(); k < l; k++) {
			substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].pNames[k] + tS(")"),
				   tS(model.postSynapsePara[model.inSyn[i][j]][k]));
		    }

		    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames.size(); k < l; k++) {
			substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] + tS(")"),
				   tS("lps") +tS(postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k])+tS(j));
		    }

		    for (int k = 0; k < postSynModels[model.postSynapseType[model.inSyn[i][j]]].dpNames.size(); ++k)
			substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].dpNames[k] + tS(")"), tS(model.dpsp[model.inSyn[i][j]][k]));

		    os << psCode;
			    
		    os << ";" << ENDL;
			    
			    
			    
		}
	    }
	}
	if (model.receivesInputCurrent[i] == 1) { // receives constant  input
	    os << "Isyn += " << model.globalInp[i] << ";" << ENDL;
	}
	if (model.receivesInputCurrent[i] >= 2) { // receives explicit input from argument
	    os << "Isyn += (" << model.ftype<< ") d_inputI" << model.neuronName[i] << "[" << localID << "];" << ENDL;
	}
	// test whether spike condition was fulfilled previously
	string thcode= nModels[nt].thresholdConditionCode;
	if (thcode  == tS("")) { //no condition provided
	    cerr << "Warning: No thresholdConditionCode for neuron type :  " << model.neuronType[i]  << " used for " << model.name[i] << " was provided. There will be no spikes detected in this population!" << ENDL;
	} 
	else {
	    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
		substitute(thcode, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
	    }
	    for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
		substitute(thcode, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model.neuronPara[i][k]));
	    }
	    for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
		substitute(thcode, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model.dnp[i][k]));
	    }
	    os << "bool oldSpike= (" << thcode << ");" << ENDL;   
	}
	os << "// calculate membrane potential" << ENDL;
	//new way of doing it
	string code = nModels[nt].simCode;
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l") + nModels[nt].varNames[k]);
	}
	substitute(code, tS("$(Isyn)"), tS("Isyn"));
	for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
	    substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model.neuronPara[i][k]));
	}
	for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
	    substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model.dnp[i][k]));
	}
	for (int k = 0, l = nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    substitute(code, tS("$(") + nModels[nt].extraGlobalNeuronKernelParameters[k] + tS(")"), nModels[nt].extraGlobalNeuronKernelParameters[k]+model.neuronName[i]);
	}		
	os << code;
	os << ENDL;

	//test if a spike type event occurred
	os << "if (lV >= " << model.nSpkEvntThreshold[i] << ")" << OB(30);
	os << "// register a spike type event" << ENDL;
	os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << ENDL;
	os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << ENDL;
	os << CB(30);

	//insert condition code provided that tests for a true spike
	if (thcode != tS("")) {
	    os << "if ((" << thcode << ") && !(oldSpike)) " << OB(40);
	    os << "// register a true spike" << ENDL;
	    os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << ENDL;
	    os << "shSpk[spkIdx] = " << localID << ";" << ENDL;
		  
	    //add optional reset code after a true spike, if provided
	    if (nModels[nt].resetCode != tS("")) {
		code = nModels[nt].resetCode;
		for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
		    substitute(code, tS("$(") + nModels[nt].varNames[k] + tS(")"), tS("l")+ nModels[nt].varNames[k]);
		}
		substitute(code, tS("$(Isyn)"), tS("Isyn"));
		for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
		    substitute(code, tS("$(") + nModels[nt].pNames[k] + tS(")"), tS(model.neuronPara[i][k]));
		}
		for (int k = 0, l = nModels[nt].dpNames.size(); k < l; k++) {
		    substitute(code, tS("$(") + nModels[nt].dpNames[k] + tS(")"), tS(model.dnp[i][k]));
		}
		os << "// spike reset code" << ENDL;
		os << code << ENDL;
	    }
	    os << CB(40);
	}

	//store the defined parts of the neuron state into the global state variables d_V.. etc
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << "d_" << nModels[nt].varNames[k] <<  model.neuronName[i] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "(d_spkEvntQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	    }
	    os << localID << "] = l" << nModels[nt].varNames[k] << ";" << ENDL;
	}
	for (int j = 0; j < model.inSyn[i].size(); j++) {

	    string psCode = postSynModels[model.postSynapseType[model.inSyn[i][j]]].postSynDecay;

	    substitute(psCode, tS("$(inSyn)"), tS("linSyn") + tS(j));
	    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].pNames.size(); k < l; k++) {
		substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].pNames[k] + tS(")"), 
			   tS(model.postSynapsePara[model.inSyn[i][j]][k]));
	    }

	    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames.size(); k < l; k++) {
		substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] + tS(")"), 
			   tS("lps") +tS(postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k])+tS(j));
	    }

	    for (int k = 0; k < postSynModels[model.postSynapseType[model.inSyn[i][j]]].dpNames.size(); ++k)
		substitute(psCode, tS("$(") + postSynModels[model.postSynapseType[model.inSyn[i][j]]].dpNames[k] + tS(")"), tS(model.dpsp[model.inSyn[i][j]][k]));

	    for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
		substitute(psCode, tS("$(") + nModels[nt].varNames[k] + tS(")"),
			   tS("l") + nModels[nt].varNames[k]);
	    }

	    for (int k = 0, l = nModels[nt].pNames.size(); k < l; k++) {
		substitute(psCode, tS("$(") + nModels[nt].pNames[k] + tS(")"),
			   tS("l") + nModels[nt].pNames[k]);
	    }


	    os << psCode;
	    os << "d_inSyn"  << model.neuronName[i] << j << "[" << localID << "] = linSyn"<< j << ";" << ENDL;

	    for (int k = 0, l = postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames.size(); k < l; k++) {
		os << "d_" <<  postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] << model.synapseName[model.inSyn[i][j]] << "[";
		os << localID << "] = lps" << postSynModels[model.postSynapseType[model.inSyn[i][j]]].varNames[k] << j << ";"<< ENDL;
	    }
	}
	os << CB(20);

	os << "__syncthreads();" << ENDL; 

	os << "if (threadIdx.x == 0)" << OB(50);
	os << "if (spkEvntCount>0) posSpkEvnt = atomicAdd((unsigned int *) &d_glbSpkEvntCnt" << model.neuronName[i];
	if (model.neuronDelaySlots[i] != 1) {
	    os << "[d_spkEvntQuePtr" << model.neuronName[i] << "]";
	}
	os << ", spkEvntCount);" << ENDL;

	os << "if (spkCount>0) posSpk = atomicAdd((unsigned int *) &d_glbscnt" << model.neuronName[i] << ", spkCount);" << ENDL;

	os << CB(50);  //end if (threadIdx.x == 0)

	os << "__syncthreads();" << ENDL;


	os << "if (threadIdx.x < spkEvntCount)" << OB(60);
	os << "d_glbSpkEvnt" << model.neuronName[i] << "[";

	if (model.neuronDelaySlots[i] != 1) {
	    os << "(d_spkEvntQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	}
	os << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << ENDL;
	os << CB(60);

	os << "if (threadIdx.x < spkCount)" << OB(70);
	os << "d_glbSpk" << model.neuronName[i] << "[";

	os << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << ENDL;
	if (model.neuronNeedSt[i]) {
	    os << "d_sT" << model.neuronName[i] << "[shSpk[threadIdx.x]] = t;" << ENDL;
	}
	os << CB(70);

	os << CB(10);
    }
    os << CB(5);
    os << ENDL;
    os << "#endif" << ENDL;
    os.close();
}


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic spikes or spike type events

*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code(
    ostream &os, //!< output stream for code
    NNmodel &model, //!< the neuronal network model to generate code for
    unsigned int src, //!< the number of the src neuron population
    unsigned int trg, //!< the number of the target neuron population
    int i, //!< the index of the synapse group being processed
    string &localID, //!< the variable name of the local ID of the thread within the synapse group
    string &preSpike, //!< shSpk[j] if true spike, shSpkEvnt[j] if spike event
    string &preSpikeV, //!< shSpkV[j] if true spike, shSpkVEvnt[j] if spike event
    float Epre, //!< the presynaptic threshold potential for spike events
    string &theLG, //! expression for the locally held g value
    unsigned int inSynNo, //!< the ID number of the current synapse population as the incoming population to the target neuron population
    float Vslope, //!< parameter of the graded synapses
    string postfix //!< whether to generate code for true spikes or spike type events
    )
{
    int Evnt= 0;
    if (postfix == tS("Evnt")) {
	Evnt= 1;
    }
    
    // Detect spike events or spikes and do the update
    os << "// process presynaptic events:";
    if (Evnt) {
	os << " Spike type events" << ENDL;
    }
    else {
	os << " True Spikes" << ENDL;
    }
    os << "for (r = 0; r < numSpikeSubsets" << postfix << "; r++)" << OB(90);
    os << "if (r == numSpikeSubsets" << postfix << " - 1) lmax = lscnt" << postfix << " % " << "BLOCKSZ_SYN" << ";" << ENDL;
    os << "else lmax = " << "BLOCKSZ_SYN" << ";" << ENDL;
    os << "if (threadIdx.x < lmax)" << OB(100);
    os << "shSpk" << postfix << "[threadIdx.x] = d_glbSpk" << postfix << "" << model.neuronName[src] << "[";
    if (model.neuronDelaySlots[src] != 1) {
	os << "(delaySlot * " << model.neuronN[src] << ") + ";
    }
    os << "(r * " << "BLOCKSZ_SYN" << ") + threadIdx.x];" << ENDL;
    if (model.neuronType[src] != POISSONNEURON) {                                       //?
	os << "shSpkV" << postfix << "[threadIdx.x] = d_V" << model.neuronName[src] << "[";          //?
	if (model.neuronDelaySlots[src] != 1) {                                         //?
	    os << "(delaySlot * " << model.neuronN[src] << ") + ";                      //?
	}                                                                               //?
	os << "shSpk" << postfix << "[threadIdx.x]];" << ENDL;                                       //? 
    }                                                                                   //?
    os << CB(100);

    if ((model.synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model.synapseTarget[i]] == 0)) {
	os << "if (threadIdx.x < " << neuronBlkSz << ") shLg[threadIdx.x] = 0;" << ENDL;  // what happens if neuronBlkSz > synapseBlkSz
    }
    os << "__syncthreads();" << ENDL;
		
    os << "// only work on existing neurons" << ENDL;
    int maxConnections;
    if ((model.synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model.synapseTarget[i]] == 1)) {
	if(model.maxConn.size()==0) {
	    fprintf(stderr,"Model Generation warning: for every SPARSE synapse group used you must also supply (in your model) a max possible number of connections via the model.setMaxConn() function.");
	    maxConnections= model.neuronN[trg];
	}
	else {
	    maxConnections= model.maxConn[i];
	}
    }
    else {
	maxConnections= model.neuronN[trg];
    }
    os << "if (" << localID << " < " << maxConnections << ")" << OB(110);

    os << "// loop through all incoming spikes" << ENDL;
    os << "for (j = 0; j < lmax; j++)" << OB(120);
    if (model.synapseGType[i] == INDIVIDUALID) {
	os << "unsigned int gid = (shSpk" << postfix << "[j] * " << model.neuronN[trg];
	os << " + " << localID << ");" << ENDL;
    }
    if (model.neuronType[src] != POISSONNEURON) {
	os << "if ";
	if (model.synapseGType[i] == INDIVIDUALID) {
	    // Note: we will just access global mem. For compute >= 1.2
	    // simultaneous access to same global mem in the (half-)warp
	    // will be coalesced - no worries
	    os << "((B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	    os << UIntSz - 1 << ")) && ";
	}
	os << "(shSpkV" << postfix << "[j] > " << Epre << ")";
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << ")";
	}
	os << OB(130);
    }
    else {
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "if (B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	    os << UIntSz - 1 << "))" << OB(135);
	}
    }
    preSpike = tS("shSpk")+postfix+tS("[j]");
    if (model.neuronType[src] == POISSONNEURON) preSpikeV=toString(model.neuronPara[src][2]);
    else preSpikeV = tS("shSpkV")+postfix+tS("[j]"); 
	


    if (model.synapseConnType[i] == SPARSE) {

	os << "npost = d_gp" << model.synapseName[i] << "_indInG[shSpk" << postfix << "[j] + 1] - d_gp";
	os << model.synapseName[i] << "_indInG[shSpk" << postfix << "[j]];" << ENDL;
	os << "if ("<< localID <<" < npost)" << OB(140);
	if (model.synapseGType[i] == INDIVIDUALG){
	    os << "lg = d_gp" << model.synapseName[i] << "[d_gp" << model.synapseName[i] << "_indInG[shSpk" << postfix << "[j]] + "<< localID << "];" << ENDL;
	    //theLG = toString("lg"); //moved above -- leaving this as a comment at the moment to be sure 
	}
	os << "ipost = d_gp" << model.synapseName[i] << "_ind[d_gp";
	os << model.synapseName[i] << "_indInG[shSpk" << postfix << "[j]] + "<< localID <<"];" << ENDL;
		

	if (model.synapseType[i] < MAXSYN){
	    if (isGrpVarNeeded[model.synapseTarget[i]] == 0) {
		os << "shLg[ipost] += " << theLG << ";" << ENDL;
	    }
	    else {
		os << "atomicAdd(&d_inSyn" << model.neuronName[trg] << inSynNo << "[ipost], " << theLG << ");" << ENDL;
	    }
	}
	else{ //user defined synapse
	    unsigned int synt = model.synapseType[i]-MAXSYN;
	    string code;
	    if (Evnt) {
		code = weightUpdateModels[synt].simCodeEvnt;
	    }
	    else {
		code= weightUpdateModels[synt].simCode;
	    }
		   
	    if (isGrpVarNeeded[model.synapseTarget[i]] == 0) {
		substitute(code, tS("$(updatelinsyn)"), tS("$(inSyn)+=$(addtoinSyn)")); //!!!! never used atm		   		
	    }
	    else{
		substitute(code, tS("$(updatelinsyn)"), tS("atomicAdd(&$(inSyn),$(addtoinSyn))"));
	    }

	    for (int k = 0, l = weightUpdateModels[synt].varNames.size(); k < l; k++) {
		substitute(code, tS("$(") + weightUpdateModels[synt].varNames[k] + tS(")"), tS("d_")+weightUpdateModels[synt].varNames[k]+model.synapseName[i]+ tS("[d_gp")+ model.synapseName[i] + tS("_indInG[shSpk")+postfix+tS("[j]] + ") + localID +tS("]]"));
	    }
			
	    for (int k = 0, l = weightUpdateModels[synt].pNames.size(); k < l; k++) {
		substitute(code, tS("$(") + weightUpdateModels[synt].pNames[k] + tS(")"), tS(model.synapsePara[i][k]));
	    }
	    for (int k = 0, l = weightUpdateModels[synt].dpNames.size(); k < l; k++) {
		substitute(code, tS("$(") + weightUpdateModels[synt].dpNames[k] + tS(")"), tS(model.dsp[i][k]));
	    }
	    if (isGrpVarNeeded[model.synapseTarget[i]] == 1) {
		substitute(code, tS("$(inSyn)"), tS("d_inSyn")+model.neuronName[trg]+tS(inSynNo)+tS("[ipost]")); 
	    }
	    else{
		substitute(code, tS("$(inSyn)"), tS("shLg[ipost]"));
	    }
	    substitute(code, tS("$(G)"), theLG);		   
	    substitute(code, tS("$(addtoinSyn)"), tS("addtoinSyn"));
	    substitute(code, tS("$(preSpike)"), preSpike);
	    substitute(code, tS("$(preSpikeV)"), preSpikeV);
	    string sTpost = tS("d_sT")+ model.neuronName[trg] + tS("[") + localID +tS("]");
	    substitute(code, tS("$(sTpost)"),sTpost);
	    string sTpre = tS("d_sT")+ model.neuronName[src] + tS("[") + localID +tS("]");
	    substitute(code, tS("$(sTpre)"),sTpre);
		    		
	    for (int k = 0, l = weightUpdateModels[synt].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		substitute(code, tS("$(") + weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k] + tS(")"), weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k]+model.synapseName[i]);
	    }
	    os << code << ENDL;
	}


	os << CB(140); // end if (id < npost)
	if (model.neuronType[src] != POISSONNEURON) {
	    os << CB(130) << ENDL; // end if (shSpkV" << postfix << "[j]>postthreshold)
	}
	else {
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << CB(135) << ENDL; // end if (B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	    }
	}

	if (isGrpVarNeeded[model.synapseTarget[i]] == 0) {	
	    os << "__syncthreads();" << ENDL;
	}
    }
    else { //not sparse

	
	if (model.synapseGType[i] == INDIVIDUALG) os << "lg = d_gp" << model.synapseName[i] << "[shSpk" << postfix << "[j]*" << model.neuronN[trg] << " + " << localID << "];" << ENDL;		
	
	if (model.synapseType[i] >= MAXSYN) {
	    unsigned int synt = model.synapseType[i]-MAXSYN;
	    string wCode;
	    if (Evnt) {
		wCode= weightUpdateModels[synt].simCodeEvnt;
	    }
	    else {
		wCode= weightUpdateModels[synt].simCode;
	    }
	    substitute(wCode, tS("$(updatelinsyn)"), tS("$(inSyn)+=$(addtoinSyn)"));
	    for (int k = 0, l = weightUpdateModels[synt].varNames.size(); k < l; k++) {
		substitute(wCode, tS("$(") + weightUpdateModels[synt].varNames[k] + tS(")"), tS("d_")+weightUpdateModels[synt].varNames[k]+model.synapseName[i]+tS("[shSpk")+postfix+tS("[j]") + tS("*") + tS(model.neuronN[trg]) + tS("+") + localID + tS(" ]"));
	    }
			
	    for (int k = 0, l = weightUpdateModels[synt].pNames.size(); k < l; k++) {
		substitute(wCode, tS("$(") + weightUpdateModels[synt].pNames[k] + tS(")"), tS(model.synapsePara[i][k]));
	    }
			
	    for (int k = 0, l = weightUpdateModels[synt].dpNames.size(); k < l; k++) {
		substitute(wCode, tS("$(") + weightUpdateModels[synt].dpNames[k] + tS(")"), tS(model.dsp[i][k]));
	    }
			
	    for (int k = 0, l = weightUpdateModels[synt].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		substitute(wCode, tS("$(") + weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k] + tS(")"), weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k]+model.synapseName[i]);
	    }
	    substitute(wCode, tS("$(inSyn)"), tS("linSyn"));
	    substitute(wCode, tS("$(G)"), theLG);		   
	    substitute(wCode, tS("$(addtoinSyn)"), tS("addtoinSyn"));
	    substitute(wCode, tS("$(preSpike)"), preSpike);
	    substitute(wCode, tS("$(preSpikeV)"), preSpikeV);
	    string sTpost = tS("d_sT")+ model.neuronName[trg] + tS("[") + localID +tS("]");
	    substitute(wCode, tS("$(sTpost)"),sTpost);
	    string sTpre = tS("d_sT")+ model.neuronName[src] + tS("[") + localID +tS("]");
	    substitute(wCode, tS("$(sTpre)"),sTpre);		    
	    os << wCode;
	    
       if (model.usesPostLearning[i]==TRUE){ //!TODO check if this is correct. setting back the g valiue if learning 
			os << "d_gp" << model.synapseName[i] << "[" << localID << " + " << model.neuronN[trg] << " * shSpk[j]] =" << theLG << ";" ;
		}		
	}
    }		
    os << ENDL;

    if (model.synapseConnType[i] != SPARSE) {
	if ((model.synapseType[i] == NSYNAPSE) || (model.synapseType[i] == LEARN1SYNAPSE)) {
	    os << "linSyn = linSyn + " << theLG << "; //1" << ENDL;
	}
	if (model.synapseType[i] == NGRADSYNAPSE) {
	    if (model.neuronType[src] == POISSONNEURON) {
		os << "linSyn = linSyn + " << theLG << " * tanh((";
		os << SAVEP(model.neuronPara[src][2]) << " - " << SAVEP(Epre);
	    }
	    else {
		os << "linSyn = linSyn + " << theLG << " * tanh((shSpkV" << postfix << "[j] - " << SAVEP(Epre);
	    }
	    os << ")*DT*2 / " << Vslope << ");" << ENDL;
	}
    }
    if ((isGrpVarNeeded[model.synapseTarget[i]] == 0) && (model.synapseConnType[i] == SPARSE) && (model.synapseType[i] >= MAXSYN)) {
	os << "linSyn+=shLg[" << localID << "];" << ENDL;
    }
    
    if (!Evnt) {
	// if needed, do some learning (this is for pre-synaptic spikes)
	if (model.synapseType[i] == LEARN1SYNAPSE) {  
	    // simply assume INDIVIDUALG for now
	    os << "lg = d_grawp" << model.synapseName[i] << "[shSpk[j] * " << model.neuronN[trg] << " + " << localID << "];" << ENDL;
	    os << model.ftype << " dt = d_sT" << model.neuronName[trg] << "[" << localID << "] - t - ";
	    os << SAVEP(model.synapsePara[i][11]) << ";" << ENDL;
	    os << "if (dt > " << model.dsp[i][1] << ")" << OB(150);
	    os << "dt = - " << SAVEP(model.dsp[i][5]) << ";" << ENDL;
	    os << CB(150);
	    os << "else if (dt > 0.0)" << OB(160);
	    os << "dt = " << SAVEP(model.dsp[i][3]) << " * dt + " << SAVEP(model.dsp[i][6]) << ";" << ENDL;
	    os << CB(160);
	    os << "else if (dt > " << model.dsp[i][2] << ")" << OB(170);
	    os << "dt = " << SAVEP(model.dsp[i][4]) << " * dt + " << SAVEP(model.dsp[i][6]) << ";" << ENDL;
	    os << CB(170);
	    os << "else" << OB(180);
	    os << "dt = - " << SAVEP(model.dsp[i][7]) << ";" << ENDL;
	    os << CB(180);
	    os << "lg = lg + dt;" << ENDL;
	    os << "d_grawp" << model.synapseName[i] << "[shSpk[j] * " << model.neuronN[trg] << " + " << localID << "] = lg;" << ENDL;
	    os << "d_gp" << model.synapseName[i] << "[shSpk[j] * " << model.neuronN[trg] << " + " << localID << "] = ";
	    os << "gFunc" << model.synapseName[i] << "(lg);" << ENDL;
	}
    }
    if (model.synapseConnType[i] != SPARSE) {
	if (model.neuronType[src] != POISSONNEURON) {
	    os << CB(130) << ENDL; // end if (shSpkV" << postfix << "[j]>postthreshold)
	}
	else {
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << CB(135) << ENDL; // end if (B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	    }
	}
    }
    
    os << CB(120) << ENDL;
    os << CB(110) << ENDL;
    os << CB(90) << ENDL;
}




//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are synapse-related and the actual CUDA kernel for simulating one time step of the synapses.
*/
//-------------------------------------------------------------------------

void genSynapseKernel(NNmodel &model, //!< Model description 
		      string &path, //!< Path for code output
		      ostream &mos //!< output stream for messages
    )
{
    string name, 
	s, 
	localID, //!< "id" if first synapse group, else "lid". lid =(thread index- last thread of the last synapse group)
	theLG, //!< Conductance to be used. It can replace a float if GLOBALG is used, or an element of an array in the shared or global memory depending on the connectivity patterns. 
	preSpike, //!< shSpk[j] if true spike, shSpkEvnt[j] if spike event
	preSpikeV; //!< shSpkV[j] if true spike, shSpkVEvnt[j] if spike event
    ofstream os;
    unsigned int numOfBlocks,trgN;
    // count how many neuron blocks to use: one thread for each synapse target
    // targets of several input groups are counted multiply
    numOfBlocks = model.padSumSynapseKrnl[model.synapseGrpN - 1] / synapseBlkSz;

    name = path + toString("/") + model.name + toString("_CODE/synapseKrnl.cc");
    os.open(name.c_str());
    writeHeader(os);  // write header content
    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_synapseKrnl_cc" << ENDL;
    os << "#define _" << model.name << "_synapseKrnl_cc" << ENDL;
    os << "#define BLOCKSZ_SYN " << synapseBlkSz << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file synapseKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name;
    os << " containing the synapse kernel and learning kernel functions." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;
    os << ENDL;

    // Kernel header
    unsigned int src;
    os << "__global__ void calcSynapses(" << ENDL;
    for (int i = 0; i < model.synapseGrpN; i++) {
	if (model.synapseGType[i] == INDIVIDUALG) {
	    os << " " << model.ftype << " * d_gp" << model.synapseName[i] << "," << ENDL;
	}
	if (model.synapseConnType[i] == SPARSE){
	    os << "unsigned int * d_gp" << model.synapseName[i] << "_ind," << ENDL;
	    os << "unsigned int * d_gp" << model.synapseName[i] << "_indInG," << ENDL;
	    trgN = model.neuronN[model.synapseTarget[i]];
	}
	if (model.synapseGType[i] == (INDIVIDUALID )) {
	    os << "unsigned int *d_gp" << model.synapseName[i] << "," << ENDL;
	}
	if (model.synapseType[i] == LEARN1SYNAPSE) {
	    os << model.ftype << " * d_grawp" << model.synapseName[i] << "," << ENDL;
	}
    }
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];
	os << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model.neuronName[i]; // Vm
	if (i < (model.neuronGrpN - 1)) 
	    os << "," << ENDL;
    }
	
	
    for (int i=0; i< model.synapseName.size(); i++){
	int st= model.synapseType[i];
	if (st >= MAXSYN){
	    for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		os << ", " << weightUpdateModels[st-MAXSYN].varTypes[k] << " *d_" << weightUpdateModels[st-MAXSYN].varNames[k];
		os << model.synapseName[i];
	    }
	    for (int k= 0, l= weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		os << ", " << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameterTypes[k] << " " << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters[k];
		os << model.synapseName[i];
	    }
	}	
    }
	
    if (model.needSt) {
	os << ", " << model.ftype << " t";
    }
    os << ENDL << ")";
    os << ENDL;

    // kernel code
    os << OB(75);
    
    //common variables for all cases
    os << "unsigned int id = " << "BLOCKSZ_SYN" << " * blockIdx.x + threadIdx.x;" << ENDL;
    os << "__shared__ unsigned int shSpkEvnt[" << "BLOCKSZ_SYN" << "];" << ENDL;
    os << "__shared__ " << model.ftype << " shSpkVEvnt[" << "BLOCKSZ_SYN" << "];" << ENDL;
    os << "volatile __shared__ " << model.ftype << " shLg[" << neuronBlkSz << "];" << ENDL;
    os << "unsigned int lscntEvnt, numSpikeSubsetsEvnt, lmax, j, r;" << ENDL;
    
    
    //case-dependent variables
    for (int i = 0; i < model.synapseGrpN; i++) {
	if ((model.synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model.synapseTarget[i]] == 0) || (model.synapseType[i] >= MAXSYN)){
	    os << model.ftype << " linSyn, lg;" << ENDL;
	    break;
	}
    }
    for (int i = 0; i < model.synapseGrpN; i++) {  
	if (model.synapseConnType[i] == SPARSE){
	    os << "unsigned int  ipost, npost; " << ENDL;		
	    os << "ipost = 0;" << ENDL;
	    break; 
	}    
    }  

    for (int i = 0; i < model.synapseGrpN; i++) {  
	if (model.synapseType[i]>= MAXSYN){
	    os << model.ftype << " addtoinSyn;" << ENDL;  
	    break;
	}
    }
    for (int i = 0; i < model.synapseGrpN; i++) {
	if (model.usesTrueSpikes[i] == TRUE || model.usesPostLearning[i] == TRUE) {  
	    os << "__shared__ unsigned int shSpk[" << "BLOCKSZ_SYN" << "];" << ENDL;
	    os << "__shared__ " << model.ftype << " shSpkV[" << "BLOCKSZ_SYN" << "];" << ENDL;
	    os << "unsigned int lscnt, numSpikeSubsets;" << ENDL;
	    break;
	}
    }
    if (model.needSynapseDelay == 1) {
	os << "int delaySlot;" << ENDL;
    }
    os << ENDL;
    for (int i = 0; i < model.synapseGrpN; i++) {
	unsigned int inSynNo = model.synapseInSynNo[i];
	if (i == 0) {
	    os << "if (id < " << model.padSumSynapseKrnl[i] << ") "  << OB(77);
	    os << "//synapse group " << model.synapseName[i] << ENDL;
	    localID = string("id");
	}
	else {
	    os << "if ((id >= " << model.padSumSynapseKrnl[i - 1] << ") && ";
	    os << "(id < " << model.padSumSynapseKrnl[i] << ")) "  << OB(77);
	    os << "//synapse group " << model.synapseName[i] << ENDL;
	    os << "unsigned int lid;" << ENDL;
	    os << "lid = id - " << model.padSumSynapseKrnl[i - 1] << ";" << ENDL;
	    localID = string("lid");
	}
	unsigned int trg = model.synapseTarget[i];
	unsigned int nN = model.neuronN[trg];
	src = model.synapseSource[i];
	float Epre = 0;
	if (model.synapseType[i]< MAXSYN) Epre = model.synapsePara[i][1];    //
	else{
	    unsigned int synt = model.synapseType[i]-MAXSYN;
	    for (int k = 0, l = weightUpdateModels[synt].pNames.size(); k < l; k++) {
		if (weightUpdateModels[synt].pNames[k] == "Epre") {
		    Epre = model.synapsePara[i][k];
		    break;
	    	}
	    	if (k == l-1) mos << "!!! WARNING: You did not provide a synapse parameter named Epre. Presynaptic threshold potential is set to 0" << ENDL;
	    }
	}
	float Vslope;                                                        //
	if (model.synapseType[i] == NGRADSYNAPSE) {                          //
	    Vslope = model.synapsePara[i][3];                                //
	}                                                                    //

	if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot = (d_spkEvntQuePtr" << model.neuronName[src] << " + ";
	    os << (int) (model.neuronDelaySlots[src] - model.synapseDelay[i] + 1);
	    os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
	}
	if ((model.synapseConnType[i] != SPARSE) || (isGrpVarNeeded[model.synapseTarget[i]] == 0)){
	    os << "// only do this for existing neurons" << ENDL;
	    os << "if (" << localID << " < " << nN <<")" << OB(80);
	    os << "linSyn = d_inSyn" << model.neuronName[trg] << inSynNo << "[" << localID << "];" << ENDL;

	    os << CB(80);
	}
	os << "lscntEvnt = d_glbSpkEvntCnt" << model.neuronName[src];
	if (model.neuronDelaySlots[src] != 1) {
	    os << "[delaySlot]";
	}
	os << ";" << ENDL;
	os << "numSpikeSubsetsEvnt = (unsigned int) (ceilf((float) lscntEvnt / " << "((float)BLOCKSZ_SYN)" << "));" << ENDL;

	os << "lscnt = d_glbscnt" << model.neuronName[src];
	if (model.neuronDelaySlots[src] != 1) os << "[d_spkQuePtr" << model.neuronName[src] << "]";
	os << ";" << ENDL;

	os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << "((float)BLOCKSZ_SYN)" << "));" << ENDL;
	
	//set theLG	
	if ((model.synapseGType[i] == GLOBALG) || (model.synapseGType[i] == INDIVIDUALID)) {
	    theLG = toString(model.g0[i]);
	}
  	else{
	    // if ((model.synapseConnType[i] == SPARSE) || (model.synapseGType[i] == INDIVIDUALG)) {
	    theLG = toString("lg");
		//}
	}

	// generate the code for processing spike like events
	if (model.usesSpikeEvents[i] == TRUE) {	
	    generate_process_presynaptic_events_code(os, model, src, trg, i, localID, preSpike, preSpikeV, Epre, theLG, inSynNo, Vslope, tS("Evnt"));
	}
	// generate the code for processing true spike events
	if (model.usesTrueSpikes[i] == TRUE) {
	    generate_process_presynaptic_events_code(os, model, src, trg, i, localID, preSpike, preSpikeV, Epre, theLG, inSynNo, Vslope, tS(""));
	}       
	   
	if ((model.synapseConnType[i] != SPARSE)|| (isGrpVarNeeded[model.synapseTarget[i]] == 0)) {

	    os << "// only do this for existing neurons" << ENDL;
	    os << "if (" << localID << " < " << model.neuronN[trg] <<")" << OB(190);
	    os << "d_inSyn" << model.neuronName[trg] << inSynNo << "[" << localID << "] = linSyn;" << ENDL;

	    os << CB(190);
	}
	if (model.lrnGroups == 0) {
	    os << "if (threadIdx.x == 0)" << OB(200);
	    os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
	    os << "if (j == " << numOfBlocks - 1 << ")" << OB(210);
	    for (int j = 0; j < model.neuronGrpN; j++) {
		os << "d_glbscnt" << model.neuronName[j] << " = 0;" << ENDL;
		if (model.neuronDelaySlots[j] != 1) { // with delays
		    os << "d_spkEvntQuePtr" << model.neuronName[j] << " = (d_spkEvntQuePtr";
		    os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		    os << "d_glbSpkEvntCnt" << model.neuronName[j] << "[d_spkEvntQuePtr";
		    os << model.neuronName[j] << "] = 0;" << ENDL;
		}
		else { // no delays
		    os << "d_glbSpkEvntCnt" << model.neuronName[j] << " = 0;" << ENDL;
		}
	    }
	    os << "d_done = 0;" << ENDL;
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

    if (model.lrnGroups > 0) {

	// count how many learn blocks to use: one thread for each synapse source
	// sources of several output groups are counted multiply
	numOfBlocks = model.padSumLearnN[model.lrnGroups - 1] / learnBlkSz;
  
	// Kernel header
	os << "__global__ void learnSynapsesPost(" << ENDL;
	for (int i = 0; i < model.synapseGrpN; i++) {
	    if (model.synapseGType[i] == (INDIVIDUALG )) {
		os << model.ftype << " *d_gp" << model.synapseName[i] << "," << ENDL;
	    }
	    if (model.synapseGType[i] == (INDIVIDUALID )) {
		os << "unsigned int *d_gp" << model.synapseName[i] << "," << ENDL;
	    }
	    if (model.synapseType[i] == LEARN1SYNAPSE) {
		os << model.ftype << " *d_grawp"  << model.synapseName[i] << "," << ENDL;
	    }
	}
	for (int i = 0; i < model.neuronGrpN; i++) {
	    nt = model.neuronType[i];
	    os << nModels[nt].varTypes[0] << " *d_" << nModels[nt].varNames[0] << model.neuronName[i]; // Vm
	    if (i < model.neuronGrpN-1) {
		os << ",";
	    }  
	}
	
	for (int i=0; i< model.synapseName.size(); i++){
	    int st= model.synapseType[i];
	    if (st >= MAXSYN){
		for (int k= 0, l= weightUpdateModels[st-MAXSYN].varNames.size(); k < l; k++) {
		    os << ", " << weightUpdateModels[st-MAXSYN].varTypes[k] << " *d_" << weightUpdateModels[st-MAXSYN].varNames[k];
		    os << model.synapseName[i];
		}
		for (int k= 0, l= weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		    os << ", " << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameterTypes[k] << " " << weightUpdateModels[st-MAXSYN].extraGlobalSynapseKernelParameters[k];
		    os << model.synapseName[i];
		}
	    }	
	}
	
	
	if (model.needSt) {
	    os << ", " << model.ftype << " t";
	}
	os << ")" << ENDL;

	// kernel code
	os << OB(215);
	os << "unsigned int id = " << learnBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
	os << "__shared__ unsigned int shSpk[" << learnBlkSz << "];" << ENDL;
	os << "__shared__ " << model.ftype << " shSpkV[" << learnBlkSz << "];" << ENDL;
	os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << ENDL;
	os << model.ftype << " lg;" << ENDL;
	os << ENDL;

	for (int i = 0; i < model.lrnGroups; i++) {
	    if (i == 0) {
		os << "if (id < " << model.padSumLearnN[i] << ")" << OB(220);
		localID = string("id");
	    }
	    else {
		os << "if ((id >= " << model.padSumLearnN[i - 1] << ") && ";
		os << "(id < " << model.padSumLearnN[i] << "))" << OB(220);
		os << "unsigned int lid;" << ENDL;
		os << "lid = id - " << model.padSumLearnN[i - 1] << ";" << ENDL;
		localID = string("lid");
	    }
	    unsigned int k = model.lrnSynGrp[i];
	    unsigned int src = model.synapseSource[k];
	    unsigned int nN = model.neuronN[src];
	    unsigned int trg = model.synapseTarget[k];
	    float Epre = 0;
	    unsigned int inSynNo = model.synapseInSynNo[k];
	    if (model.synapseType[i]< MAXSYN) Epre = model.synapsePara[k][1];
	    else{
		unsigned int synt = model.synapseType[i]-MAXSYN;
		for (int p = 0, l = weightUpdateModels[synt].pNames.size(); p < l; p++) {
		    if (weightUpdateModels[synt].pNames[p] == "Epre") {
			Epre = model.synapsePara[k][p];
			break;
		    }
		    if (p==l-1) mos << "!!! WARNING: You did not provide a synapse parameter named Epre. Presynaptic threshold potential is set to 0" << ENDL;
		}
	    }
		
	    //set theLG	
	    if ((model.synapseGType[k] == GLOBALG) || (model.synapseGType[k] == INDIVIDUALID)) {
		theLG = toString(model.g0[k]);
	    }
	    else{
		if ((model.synapseConnType[k] == SPARSE) || (model.synapseGType[k] == INDIVIDUALG)) {
		    //if sparse, defined below as os << "lg = d_gp" << model.synapseName[i] << "[d_gp" << model.synapseName[i] << "_indInG[shSpk[j]] + "<< localID << "];" << ENDL;
		    //else, os << "lg = d_gp" << model.synapseName[i] << "[shSpkEvnt[j]*" << model.neuronN[trg] << " + " << localID << "];" << ENDL;	 
		    theLG = toString("lg");
		}
	    }
  
  
	    os << "lscnt = d_glbscnt" << model.neuronName[trg];
	    if (model.neuronDelaySlots[trg] != 1) os << "[d_spkQuePtr" << model.neuronName[trg] << "]";
	    os << ";" << ENDL;

	    os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << learnBlkSz << ".0f));" << ENDL;
	    os << "for (r = 0; r < numSpikeSubsets; r++)" << OB(230);
	    os << "if (r == numSpikeSubsets - 1) lmax = lscnt % " << learnBlkSz << ";" << ENDL;
	    os << "else lmax = " << learnBlkSz << ";" << ENDL;
	    os << "if (threadIdx.x < lmax)" << OB(240);
	    os << "shSpk[threadIdx.x] = d_glbSpk" << model.neuronName[trg] << "[";
	    if (model.neuronDelaySlots[trg] != 1) {
		os << "(d_spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
	    }
	    os << "(r * " << learnBlkSz << ") + threadIdx.x];" << ENDL;
	    os << "shSpkV[threadIdx.x] = d_V" << model.neuronName[trg] << "[";
	    if (model.neuronDelaySlots[trg] != 1) {
		os << "(d_spkQuePtr" << model.neuronName[trg] << " * " << model.neuronN[trg] << ") + ";
	    }
	    os << "shSpk[threadIdx.x]];" << ENDL;
	    os << CB(240);
	    os << "__syncthreads();" << ENDL;
	    os << "// only work on existing neurons" << ENDL;
	    os << "if (" << localID << " < " << model.neuronN[src] << ")" << OB(250);
	    os << "// loop through all incoming spikes for learning" << ENDL;
	    os << "for (j = 0; j < lmax; j++)" << OB(260);
	    os << "if (shSpkV[j] > " << Epre << ")" << OB(270); 
	    
	    if (model.synapseType[k] ==LEARN1SYNAPSE){
		os << "lg = d_grawp" << model.synapseName[k] << "[" << localID << " * ";
		os << model.neuronN[trg] << " + shSpk[j]];" << ENDL;
		os << model.ftype << " dt = t - d_sT" << model.neuronName[src] << "[" << localID << "]";
		if (model.neuronDelaySlots[src] != 1) {
		    os << " + " << (DT * model.synapseDelay[k]);
		}
		os << " - " << SAVEP(model.synapsePara[k][11]) << ";" << ENDL;
		os << "if (dt > " << model.dsp[k][1] << ")" << OB(280);
		os << "dt = - " << SAVEP(model.dsp[k][5]) << ";" << ENDL;
		os << CB(280);
		os << "else if (dt > 0.0)" << OB(290);
		os << "dt = " << SAVEP(model.dsp[k][3]) << " * dt + " << SAVEP(model.dsp[k][6]) << ";" << ENDL;
		os << CB(290);
		os << "else if (dt > " << model.dsp[k][2] << ")" << OB(300);
		os << "dt = " << SAVEP(model.dsp[k][4]) << " * dt + " << SAVEP(model.dsp[k][6]) << ";" << ENDL;
		os << CB(300);
		os << "else" << OB(310);
		os << "dt = - " << SAVEP(model.dsp[k][7]) << ";" << ENDL;
		os << CB(310);
		os << "lg = lg + dt;" << ENDL;
		os << "d_grawp" << model.synapseName[k] << "[" << localID << " * ";
		os << model.neuronN[trg] << " + shSpk[j]] = lg;" << ENDL;
		os << "d_gp" << model.synapseName[k] << "[" << localID << " * ";
		os << model.neuronN[trg] << " + shSpk[j]] = gFunc" << model.synapseName[k] << "(lg);" << ENDL;
	    }
	    if (model.synapseType[k] >= MAXSYN) {
		if (model.synapseGType[k] == INDIVIDUALG){
		    if (model.synapseConnType[k] == SPARSE){
			os << "lg = d_gp" << model.synapseName[i] << "[d_gp" << model.synapseName[i] << "_indInG[shSpk[j]] + "<< localID <<"]" << ENDL;
		    }
		    else{
			os << "lg = d_gp" << model.synapseName[k] << "[shSpk[j] +" << model.neuronN[trg] << " * " << localID << "];" << ENDL;
		    }
		    //theLG = toString("lg");
		}
		unsigned int synt = model.synapseType[k]-MAXSYN;
		string code = weightUpdateModels[synt].simLearnPost;
		for (int p = 0, l = weightUpdateModels[synt].varNames.size(); p < l; p++) {
		    //os << "linSyn=d_inSyn" << model.neuronName[trg] << inSynNo << "[ipost];";
		    if (model.synapseConnType[k] == SPARSE){
		      	substitute(code, tS("$(") + weightUpdateModels[synt].varNames[p] + tS(")"), tS("d_")+weightUpdateModels[synt].varNames[p]+model.synapseName[k]+ tS("[d_gp")+ model.synapseName[k] + tS("_indInG[shSpk[j]] + ") + localID +tS("]]"));
		    }
		    else{ 
		      	substitute(code, tS("$(") + weightUpdateModels[synt].varNames[p] + tS(")"), tS("d_")+weightUpdateModels[synt].varNames[p]+model.synapseName[k] + tS("[")+ localID + tS(" * ") + tS(model.neuronN[trg]) +tS(" + shSpk[j]]"));
		    }
		    substitute(code, tS("$(inSyn)"), tS("linSyn"));
		}
			
		for (int p = 0, l = weightUpdateModels[synt].pNames.size(); p < l; p++) {
		    substitute(code, tS("$(") + weightUpdateModels[synt].pNames[p] + tS(")"), tS(model.synapsePara[k][p]));
		}
		for (int p = 0, l =  weightUpdateModels[synt].dpNames.size(); p < l; p++) {
		    substitute(code, tS("$(") + weightUpdateModels[synt].dpNames[p] + tS(")"), tS(model.dsp[k][p]));
		}		
		for (int p = 0, l = weightUpdateModels[synt].extraGlobalSynapseKernelParameters.size(); p < l; p++) {
		    substitute(code, tS("$(") + weightUpdateModels[synt].extraGlobalSynapseKernelParameters[p] + tS(")"), weightUpdateModels[synt].extraGlobalSynapseKernelParameters[p]+model.synapseName[k]);
		}
		substitute(code, tS("$(inSyn)"), tS("d_inSyn")+model.neuronName[trg]+tS(inSynNo)+tS("[ipost]"));
		substitute(code, tS("$(G)"), theLG);
		substitute(code, tS("$(preSpike)"), preSpike);
		substitute(code, tS("$(preSpikeV)"), preSpikeV);
		string sTpost = tS("d_sT")+ model.neuronName[trg] + tS("[") + localID +tS("]");
		substitute(code, tS("$(sTpost)"),sTpost);
		string sTpre = tS("d_sT")+ model.neuronName[src] + tS("[") + localID +tS("]");
		substitute(code, tS("$(sTpre)"),sTpre);

		os << code;
	  
		if (model.usesPostLearning[k]==TRUE){ //!TODO check if this is correct. setting back the g valiue if learning 
		    if (model.synapseGType[k] == INDIVIDUALG){
			os << "d_gp" << model.synapseName[k] << "[" << localID << " * " << model.neuronN[trg] << " + shSpk[j]] =" << theLG << ";" ;
		    }
		}		    	
	    }
	    os << CB(270);
	    os << CB(260);
	    os << CB(250);
	    os << CB(230);
	    os << "__threadfence();" << ENDL;
	    os << "if (threadIdx.x == 0)" << OB(320);
	    os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
	    os << "if (j == " << numOfBlocks - 1 << ")" << OB(330);
	    for (int j = 0; j < model.neuronGrpN; j++) {
		os << "d_glbscnt" << model.neuronName[j] << " = 0;" << ENDL;
		if (model.neuronDelaySlots[j] != 1) { // with delay
		    os << "d_spkQuePtr" << model.neuronName[j] << " = (d_spkQuePtr";
		    os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		    os << "d_glbSpkEvntCnt" << model.neuronName[j] << "[d_spkQuePtr";
		    os << model.neuronName[j] << "] = 0;" << ENDL;
		}
		else { // no delay
		    os << "d_glbSpkEvntCnt" << model.neuronName[j] << " = 0;" << ENDL;
		}
	    }
	    os << "d_done = 0;" << ENDL;
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
