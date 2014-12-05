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

//    os << "#include <cfloat>" << ENDL << ENDL;
    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronKrnl.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the neuron kernel function." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

//    os << "__device__ __host__ float exp(int i) { return exp((float) i); }" << endl;

    // global device variables
    os << "// relevant neuron variables" << ENDL;
    os << "__device__ volatile unsigned int d_done;" << ENDL;
    for (int i= 0; i < model.neuronGrpN; i++) {
	nt= model.neuronType[i];
	isGrpVarNeeded[i] = 0;

	//these now hold just true spikes for benefit of the user (raster plots etc)
	os << "__device__ volatile unsigned int d_glbSpkCnt" << model.neuronName[i] << ";" << ENDL;
	os << "__device__ volatile unsigned int d_glbSpk" << model.neuronName[i] << "[" << model.neuronN[i] << "];" << ENDL;

	if (model.neuronDelaySlots[i] == 1) {// no delays
	    os << "__device__ volatile unsigned int d_glbSpkCntEvnt" << model.neuronName[i] << ";" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkEvnt" << model.neuronName[i] << "[" << model.neuronN[i] << "];" << ENDL;
	}
	else { // with delays
	    os << "__device__ volatile unsigned int d_spkQuePtr" << model.neuronName[i] << ";" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkCntEvnt" << model.neuronName[i] << "[";
	    os << model.neuronDelaySlots[i] << "];" << ENDL;
	    os << "__device__ volatile unsigned int d_glbSpkEvnt" << model.neuronName[i] << "[";
	    os << model.neuronN[i] * model.neuronDelaySlots[i] << "];" << ENDL;
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
	    os << model.RNtype << " *d_rates" << model.neuronName[i];
	    os << ", // poisson \"rates\" of grp " << model.neuronName[i] << ENDL;
	    os << "unsigned int offset" << model.neuronName[i];
	    os << ", // poisson \"rates\" offset of grp " << model.neuronName[i] << ENDL;
	}
	if (model.receivesInputCurrent[i] > 1) {
	    os << model.ftype << " *d_inputI" << model.neuronName[i];
	    os << ", // explicit input current to grp " << model.neuronName[i] << ENDL;
	}
	for (int k= 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); k < l; k++) {
	    os << nModels[nt].extraGlobalNeuronKernelParameterTypes[k];
	    os << " " << nModels[nt].extraGlobalNeuronKernelParameters[k];
	    os << model.neuronName[i] << ", " << ENDL;
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
    for (int j = 0; j < model.neuronGrpN; j++) {
      if (model.neuronNeedSpkEvnt[j]){
        os << "__shared__ volatile unsigned int posSpkEvnt;" << ENDL;
        os << "__shared__ unsigned int shSpkEvnt[" << neuronBlkSz << "];" << ENDL;
        os << "unsigned int spkEvntIdx;" << ENDL;
        os << "__shared__ volatile unsigned int spkEvntCount;" << ENDL; //was scnt
        break;
      }
    }
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
	    os << "d_glbSpkCnt" << model.neuronName[j] << " = 0;" << ENDL;
	    if (model.neuronDelaySlots[j] != 1) {
		os << "d_spkQuePtr" << model.neuronName[j] << " = (d_spkQuePtr";
		os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		os << "d_glbSpkCntEvnt" << model.neuronName[j] << "[d_spkQuePtr";
		os << model.neuronName[j] << "] = 0;" << ENDL;
	    }
	    else {
		os << "d_glbSpkCntEvnt" << model.neuronName[j] << " = 0;" << ENDL;
	    }
	}
	os << CB(6);
	os << "__threadfence();" << ENDL;
    }

    os << "if (threadIdx.x == 0)" << OB(8) ;
    os << "spkCount = 0;" << ENDL ;
    os << CB(8);

    for (int j = 0; j < model.neuronGrpN; j++) {
      if (model.neuronNeedSpkEvnt[j]){
        os << "if (threadIdx.x == 1)" << OB(7) ;
        os << "spkEvntCount = 0;" << ENDL ;
        os << CB(7);
        break;
      }
    }

    os << "__syncthreads();" << ENDL;

    /*for (int i= 0; i < model.neuronGrpN; i++) {
	cerr <<  model.padSumNeuronN[i] << endl;
    }*/
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
	    os << model.RNtype << " lrate = d_rates" << model.neuronName[i];
	    os << "[offset" << model.neuronName[i] << " + " << localID << "]";
	    os << ";" << ENDL;
	}
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k];
	    os << " = dd_" <<  nModels[nt].varNames[k] << model.neuronName[i] << "[";
	    if (model.neuronDelaySlots[i] != 1) {
		os << "(((d_spkQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
		os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
	    }
	    os << localID << "];" << ENDL;
	}
        os << "// execute internal synapse dynamics if any" << ENDL;
	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    unsigned int synPopID=  model.inSyn[i][j];
	    unsigned int synt= model.synapseType[synPopID];
	    if (weightUpdateModels[synt].synapseDynamics != tS("")) {
                // there is some internal synapse dynamics
		weightUpdateModel wu= weightUpdateModels[synt];
		string code= wu.synapseDynamics;
		unsigned int srcno= model.neuronN[model.synapseSource[synPopID]]; 
		string synapseName= model.synapseName[synPopID];
		if (model.synapseConnType[synPopID] == SPARSE) {
		    os << "for (int k= 0; k <" <<  srcno << "; k++) " << OB(24) << ENDL;
		    os << "if (" << localID << " < dd_indInG" << synapseName << "[k+1] - dd_indInG" << synapseName << "[k])";
		    os << OB(24) << " // all threads participate that can work on a post-synaptic neuron" << ENDL;
		    if (model.synapseGType[synPopID] == INDIVIDUALG) {
		      name_substitutions(code, tS("dd_"), wu.varNames, synapseName + tS("[")+localID+tS(" + dd_indInG") + synapseName + tS("[k]]"));
		    }
		    else {
			value_substitutions(code, wu.varNames, model.synapseIni[synPopID]);
		    }
		    value_substitutions(code, wu.pNames, model.synapsePara[synPopID]);
		    value_substitutions(code, wu.dpNames, model.dsp_w[synPopID]);
		    os << ensureFtype(code, model.ftype) << ENDL;
		    os << CB(24);
		}
		else { // DENSE
		    os << "for (int k= 0; k <" <<  srcno << "; k++) " << OB(25) << ENDL;
		    if (model.synapseGType[synPopID] == INDIVIDUALG) {
		      name_substitutions(code, tS("dd_"), wu.varNames, synapseName + tS("[")+localID+tS(" + k*")+tS(model.neuronN[i])+tS("]"));
		    }
		    else {
		      value_substitutions(code, wu.varNames, model.synapseIni[synPopID]);
		    }
		    value_substitutions(code, wu.pNames, model.synapsePara[synPopID]);
		    value_substitutions(code, wu.dpNames, model.dsp_w[synPopID]);
		    os << ensureFtype(code, model.ftype) << ENDL;
		    os << CB(25);
		}
	    }
	}
	if (nt != POISSONNEURON) {
	    os << model.ftype << " Isyn = 0;" << ENDL;
	    
	    for (int j = 0; j < model.inSyn[i].size(); j++) {
		unsigned int synPopID= model.inSyn[i][j]; // number of (post)synapse group 
		postSynModel psm= postSynModels[model.postSynapseType[synPopID]];
		string sName= model.synapseName[synPopID];
		os << "// pull inSyn values in a coalesced access" << ENDL;
		os << model.ftype << " linSyn" << sName << " = dd_inSyn" << sName << "[" << localID << "];" << ENDL;
		if (model.synapseGType[synPopID] == INDIVIDUALG) {
		    for (int k = 0, l = psm.varNames.size(); k < l; k++) {
			os << psm.varTypes[k] << " lps" << psm.varNames[k] << sName;
			os << " = dd_" <<  psm.varNames[k] << sName << "[";
			os << localID << "];" << ENDL;
		    }
		}
		os << "Isyn += ";
		string psCode = psm.postSyntoCurrent;
		substitute(psCode, tS("$(inSyn)"), tS("linSyn")+sName);
		name_substitutions(psCode, tS("l"), nModels[nt].varNames, tS(""));
		value_substitutions(psCode, nModels[nt].pNames, model.neuronPara[i]);
		value_substitutions(psCode, nModels[nt].dpNames, model.dnp[i]);
		if (model.synapseGType[synPopID] == INDIVIDUALG) {
		    name_substitutions(psCode, tS("lps"), psm.varNames, sName);
		}
		else {
		    value_substitutions(psCode, psm.varNames, model.postSynIni[synPopID]);
		}
		value_substitutions(psCode, psm.pNames, model.postSynapsePara[synPopID]);
		value_substitutions(psCode, psm.dpNames, model.dpsp[synPopID]);
		os << ensureFtype(psCode, model.ftype);
		os << ";" << ENDL;	    
	    }
	}
	if (model.receivesInputCurrent[i] == 1) { // receives constant  input
	    os << "Isyn += " << model.globalInp[i] << ";" << ENDL;
	}
	if (model.receivesInputCurrent[i] >= 2) { // receives explicit input from argument
	    os << "Isyn += (" << model.ftype<< ") d_inputI" << model.neuronName[i] << "[" << localID << "];" << ENDL;
	}
	os << "// test whether spike condition was fulfilled previously" << ENDL;
	string thCode= nModels[nt].thresholdConditionCode;
	if (thCode  == tS("")) { //no condition provided
	    cerr << "Warning: No thresholdConditionCode for neuron type :  " << model.neuronType[i]  << " used for " << model.name[i] << " was provided. There will be no spikes detected in this population!" << ENDL;
	} 
	else {
	    name_substitutions(thCode, tS("l"), nModels[nt].varNames, tS(""));
	    value_substitutions(thCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(thCode, nModels[nt].dpNames, model.dnp[i]);
	    os << "bool oldSpike= (" << ensureFtype(thCode, model.ftype) << ");" << ENDL;   
	}
	os << "// calculate membrane potential" << ENDL;
	string code = nModels[nt].simCode;
	name_substitutions(code, tS("l"), nModels[nt].varNames, tS(""));
	value_substitutions(code, nModels[nt].pNames, model.neuronPara[i]);
	value_substitutions(code, nModels[nt].dpNames, model.dnp[i]);
	name_substitutions(code, tS(""), nModels[nt].extraGlobalNeuronKernelParameters, model.neuronName[i]);
	substitute(code, tS("$(Isyn)"), tS("Isyn"));
	os << ensureFtype(code, model.ftype);
	os << ENDL;

	if (model.neuronNeedSpkEvnt[i]) {
	    os << "// test if a spike event occurred" << ENDL;
	    os << "if ";
	    string eCode= model.neuronSpkEvntCondition[i];
	    // code substitutions ----
	    extended_name_substitutions(eCode, tS("l"), nModels[model.neuronType[i]].varNames, tS("_pre"), tS(""));
	    for (int j= 0; j < model.outSyn[i].size(); j++) {
		unsigned int synPopID= model.outSyn[i][j];
		unsigned int synt= model.synapseType[synPopID];
		value_substitutions(eCode, weightUpdateModels[synt].pNames, model.synapsePara[synPopID]);
		value_substitutions(eCode, weightUpdateModels[synt].dpNames, model.dsp_w[synPopID]);
		name_substitutions(eCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[synPopID]);
	    }
	    // end code substitutions ----
	    os << "(" + ensureFtype(eCode, model.ftype) + ")" << OB(30);
	    os << "// register a spike type event" << ENDL;
	    os << "spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);" << ENDL;
	    os << "shSpkEvnt[spkEvntIdx] = " << localID << ";" << ENDL;
	    os << CB(30);
	}
        // test for true spikes if condition is provided
	if (thCode != tS("")) {
	    os << "// test for a true spike" << ENDL;
	    os << "if ((" << ensureFtype(thCode, model.ftype) << ") && !(oldSpike)) " << OB(40);
	    os << "// register a true spike" << ENDL;
	    os << "spkIdx = atomicAdd((unsigned int *) &spkCount, 1);" << ENDL;
	    os << "shSpk[spkIdx] = " << localID << ";" << ENDL;
	    // add after-spike reset if provided
	    if (nModels[nt].resetCode != tS("")) {
		code = nModels[nt].resetCode;
		name_substitutions(code, tS("l"), nModels[nt].varNames, tS(""));
		value_substitutions(code, nModels[nt].pNames, model.neuronPara[i]);
		value_substitutions(code, nModels[nt].dpNames, model.dnp[i]);
		substitute(code, tS("$(Isyn)"), tS("Isyn"));
		os << "// spike reset code" << ENDL;
		os << ensureFtype(code, model.ftype) << ENDL;
	    }
	    os << CB(40);
	}

	//store the defined parts of the neuron state into the global state variables dd_V etc
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << "dd_" << nModels[nt].varNames[k] <<  model.neuronName[i] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "(d_spkQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	    }
	    os << localID << "] = l" << nModels[nt].varNames[k] << ";" << ENDL;
	}
	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    postSynModel psModel= postSynModels[model.postSynapseType[model.inSyn[i][j]]];
	    string sName= model.synapseName[model.inSyn[i][j]];
	    string psCode = psModel.postSynDecay;
	    substitute(psCode, tS("$(inSyn)"), tS("linSyn") + sName);
	    name_substitutions(psCode, tS("lps"), psModel.varNames, sName);
	    value_substitutions(psCode, psModel.pNames, model.postSynapsePara[model.inSyn[i][j]]);
	    value_substitutions(psCode, psModel.dpNames, model.dpsp[model.inSyn[i][j]]);
	    name_substitutions(psCode, tS("l"), nModels[nt].varNames, tS(""));
	    value_substitutions(psCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(psCode, nModels[nt].dpNames, model.dnp[i]);
	    os << "// the post-synaptic dynamics" << ENDL;
	    os << ensureFtype(psCode, model.ftype) << ENDL;
	    os << "dd_inSyn"  << sName << "[" << localID << "] = linSyn"<< sName << ";" << ENDL;
	    for (int k = 0, l = psModel.varNames.size(); k < l; k++) {
		os << "dd_" <<  psModel.varNames[k] << model.synapseName[model.inSyn[i][j]] << "[";
		os << localID << "] = lps" << psModel.varNames[k] << sName << ";"<< ENDL;
	    }
	}
	os << CB(20);
	os << "__syncthreads();" << ENDL; 

        os << "if (threadIdx.x == 0)" << OB(51);
	os << "if (spkCount>0) posSpk = atomicAdd((unsigned int *) &d_glbSpkCnt" << model.neuronName[i] << ", spkCount);" << ENDL;
	os << CB(51);  // end if (threadIdx.x == 1)
	
        if (model.neuronNeedSpkEvnt[i]){
          os << "if (threadIdx.x == 1)" << OB(50);
          os << "if (spkEvntCount>0) posSpkEvnt = atomicAdd((unsigned int *) &d_glbSpkCntEvnt" << model.neuronName[i];
          if (model.neuronDelaySlots[i] != 1) {
            os << "[d_spkQuePtr" << model.neuronName[i] << "]";
          }
          os << ", spkEvntCount);" << ENDL;
          os << CB(50);  // end if (threadIdx.x == 0)
        }

        os << "__syncthreads();" << ENDL;

	if (model.neuronNeedSpkEvnt[i]) {
          os << "if (threadIdx.x < spkEvntCount)" << OB(60);
	  os << "d_glbSpkEvnt" << model.neuronName[i] << "[";
	  if (model.neuronDelaySlots[i] != 1) {
	    os << "(d_spkQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	  }
	  os << "posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];" << ENDL;
	  os << CB(60);
        }
	os << "if (threadIdx.x < spkCount)" << OB(70);
	os << "d_glbSpk" << model.neuronName[i] << "[";
	os << "posSpk + threadIdx.x] = shSpk[threadIdx.x];" << ENDL;
	if (model.neuronNeedSt[i]) {
	    os << "d_sT" << model.neuronName[i] << "[shSpk[threadIdx.x]] = t;" << ENDL;
	}
	os << CB(70); // end if (threadIdx.x < spkCount)

	os << CB(10); // end if (id < model.padSumNeuronN[i] )
    }
    os << CB(5); // end of neuron kernel
    os << ENDL;
    os << "#endif" << ENDL;
    os.close();
}


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic 
  spikes or spike type events

*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code(
    ostream &os, //!< output stream for code
    NNmodel &model, //!< the neuronal network model to generate code for
    unsigned int src, //!< the number of the src neuron population
    unsigned int trg, //!< the number of the target neuron population
    int i, //!< the index of the synapse group being processed
    string &localID, //!< the variable name of the local ID of the thread within the synapse group
    unsigned int inSynNo, //!< the ID number of the current synapse population as the incoming population to the target neuron population
    string postfix //!< whether to generate code for true spikes or spike type events
    )
{
    bool Evnt= FALSE;
    if (postfix == tS("Evnt")) {
	Evnt= TRUE;
    }
    unsigned int synt = model.synapseType[i];
    bool needSpkEvnt= (weightUpdateModels[synt].evntThreshold != tS(""));
    if ((Evnt && needSpkEvnt) || !Evnt) {
	string dSlot;
	if (model.neuronDelaySlots[src] != 1) { 
	    dSlot= tS("(delaySlot * ") + tS(model.neuronN[src]) + tS(") +");
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
	os << "shSpk" << postfix << "[threadIdx.x] = d_glbSpk" << postfix << "" << model.neuronName[src] << "[" << dSlot << "(r * BLOCKSZ_SYN) + threadIdx.x];" << ENDL;
	if (Evnt && (model.neuronType[src] != POISSONNEURON)) {     
	    vector<string> vars= model.synapseSpkEvntVars[i];
	    for (int j= 0; j < vars.size(); j++) {
		os << "shSpkEvnt" << vars[j] << "[threadIdx.x] = dd_" << vars[j] << model.neuronName[src] << "[" << dSlot << " shSpk" << postfix << "[threadIdx.x]];" << ENDL;
	    }                                                                                   
	}
	os << CB(100);

	if ((model.synapseConnType[i] == SPARSE) && (!isGrpVarNeeded[model.synapseTarget[i]])) {
	    os << "if (threadIdx.x < " << model.neuronN[model.synapseTarget[i]] << ") shLg[threadIdx.x] = 0;" << ENDL;  // set shLg to 0 for all postsynaptic neurons; is ok as model.neuronN[model.synapseTarget[i]] <= synapseBlkSz
	}
	os << "__syncthreads();" << ENDL;
		
	os << "// only work on existing neurons" << ENDL;
	int maxConnections;
	if ((model.synapseConnType[i] == SPARSE) && (isGrpVarNeeded[model.synapseTarget[i]])) {
	    if(model.maxConn.size()==0) {
		fprintf(stderr,"Model Generation warning: for every SPARSE synapse group used you must also supply (in your model)\
 a max possible number of connections via the model.setMaxConn() function.");
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
	if ((Evnt) && (model.neuronType[src] != POISSONNEURON)) { // cosider whether POISSON Neurons should be allowed to throw events
	    os << "if ";
	    if (model.synapseGType[i] == INDIVIDUALID) {
		// Note: we will just access global mem. For compute >= 1.2 simultaneous access to same global mem in the (half-)warp
		// will be coalesced - no worries
		os << "((B(dd_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
		os << UIntSz - 1 << ")) && ";
	    }
	    string eCode= weightUpdateModels[synt].evntThreshold;
	    // code substitutions ----
	    extended_name_substitutions(eCode, tS("shSpkEvnt"), model.synapseSpkEvntVars[i], tS("_pre"), tS("[j]"));
	    value_substitutions(eCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
	    value_substitutions(eCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
	    name_substitutions(eCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
	    // end code substitutions ----
	    os << "(" << ensureFtype(eCode, model.ftype) << ")"; 
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << ")";
	    }
	    os << OB(130);
	}
	else {
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << "if (B(dd_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
		os << UIntSz - 1 << "))" << OB(135);
	    }
	}

	bool sparse= (model.synapseConnType[i] == SPARSE);
	if (sparse) {
	    os << "prePos= dd_indInG" << model.synapseName[i] << "[shSpk" << postfix << "[j]];" << ENDL;
	    os << "npost = dd_indInG" << model.synapseName[i] << "[shSpk" << postfix << "[j] + 1] - prePos;" << ENDL;
	    os << "if ("<< localID <<" < npost)" << OB(140);
	    os << "prePos+= " << localID << ";" << ENDL;
	    os << "ipost = dd_ind" << model.synapseName[i] << "[prePos];" << ENDL;
	}
	else {
	    os << "ipost= " << localID << ";" << ENDL;
	}
	string wCode;
	if (Evnt) {
	    wCode = weightUpdateModels[synt].simCodeEvnt;
	}
	else {
	    wCode= weightUpdateModels[synt].simCode;
	}
	// Code substitutions ----------------------------------------------------------------------------------
	if (sparse) { // SPARSE
	    if (isGrpVarNeeded[model.synapseTarget[i]]) { // SPARSE using atomicAdd
		substitute(wCode, tS("$(updatelinsyn)"), tS("atomicAdd(&$(inSyn),$(addtoinSyn))"));
		substitute(wCode, tS("$(inSyn)"), tS("dd_inSyn")+model.synapseName[i]+tS("[ipost]")); 
	    }
	    else{ // SPARSE using shared memory
		substitute(wCode, tS("$(updatelinsyn)"), tS("$(inSyn)+=$(addtoinSyn)")); 		   		
		substitute(wCode, tS("$(inSyn)"), tS("shLg[ipost]"));
	    }
	    if (model.synapseGType[i] == INDIVIDUALG) {
		name_substitutions(wCode, tS("dd_"), weightUpdateModels[synt].varNames, model.synapseName[i]+ tS("[prePos]"));
	    }
	    else {
		value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
	    }
	}
	else { // DENSE
	    substitute(wCode, tS("$(updatelinsyn)"), tS("$(inSyn)+=$(addtoinSyn)")); 		   		
	    substitute(wCode, tS("$(inSyn)"), tS("linSyn"));
	    if (model.synapseGType[i] == INDIVIDUALG) {
		name_substitutions(wCode, tS("dd_"), weightUpdateModels[synt].varNames, model.synapseName[i]+tS("[shSpk")+
				   postfix+tS("[j]") + tS("*") + tS(model.neuronN[trg]) + tS("+") + localID + tS(" ]"));
	    }
	    else {
		value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
	    }
	}
	value_substitutions(wCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
	value_substitutions(wCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
	name_substitutions(wCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
	substitute(wCode, tS("$(addtoinSyn)"), tS("addtoinSyn"));
	// presynaptic neuron variables and parameters
	unsigned int nt_pre= model.neuronType[src];
	if (model.neuronType[src] == POISSONNEURON) substitute(wCode, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
	substitute(wCode, tS("$(sT_pre)"), tS("d_sT")+model.neuronName[src]+tS("[shSpk") + postfix + tS("[j]]"));
	extended_name_substitutions(wCode, tS("dd_"), nModels[nt_pre].varNames, tS("_pre"), tS(model.neuronName[src])+tS("[shSpk") + postfix + tS("[j]]"));
	extended_value_substitutions(wCode, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
	extended_value_substitutions(wCode, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);
	// postsynaptic neuron variables and parameters
	unsigned int nt_post= model.neuronType[trg];
	substitute(wCode, tS("$(sT_post)"), tS("d_sT")+model.neuronName[src]+tS("[ipost]"));
	extended_name_substitutions(wCode, tS("dd_"), nModels[nt_post].varNames, tS("_post"), tS(model.neuronName[trg])+tS("[ipost]"));
	extended_value_substitutions(wCode, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
	extended_value_substitutions(wCode, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
	// end Code substitutions ------------------------------------------------------------------------- 
	os << ensureFtype(wCode, model.ftype) << ENDL;
	if (sparse) {
	    os << CB(140); // end if (id < npost)
	} 
	if ((Evnt) && (model.neuronType[src] != POISSONNEURON)) {
	    os << CB(130) << ENDL; // end if (eCode) 
	}
	else {
	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << CB(135) << ENDL; // end if (B(dd_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	    }
	}
	if ((sparse) && (!isGrpVarNeeded[model.synapseTarget[i]])) {	
	    os << "__syncthreads();" << ENDL;
	    os << "if (threadIdx.x < " << model.neuronN[model.synapseTarget[i]] << ")" << OB(136); // need to write back results
	    os << "linSyn += shLg[" << localID << "];" << ENDL;
	    os << "shLg[" << localID << "] = 0;" << ENDL;
	    os << CB(136) << ENDL;
	    os << "__syncthreads();" << ENDL;
// This seems a duplication that is not correct:
//	    os << "linSyn+=shLg[" << localID << "];" << ENDL;
	}
	os << CB(120) << ENDL;
	os << CB(110) << ENDL;
	os << CB(90) << ENDL;
    } // end if ((Evnt && needSpkEvnt) || !Evnt)
}
    



//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are 
  synapse-related and the actual CUDA kernel for simulating one time step of 
  the synapses.
*/
//-------------------------------------------------------------------------

void genSynapseKernel(NNmodel &model, //!< Model description 
		      string &path, //!< Path for code output
		      ostream &mos //!< output stream for messages
    )
{
    cout << "entering genSynapseKernel ..." << endl;
    string name, 
	s, 
	localID; //!< "id" if first synapse group, else "lid". lid =(thread index- last thread of the last synapse group)
    ofstream os;
    unsigned int numOfBlocks;
    // count how many neuron blocks to use: one thread for each synapse target
    // targets of several input groups are counted multiply
    //cerr << model.padSumSynapseKrnl.size() << " " << model.synapseGrpN - 1 << endl;
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

    // synapse kernel header
    unsigned int src;
    os << "__global__ void calcSynapses(" << ENDL;
    for (int i=0; i< model.synapseGrpN; i++){
	int synt= model.synapseType[i];
	for (int k= 0, l= weightUpdateModels[synt].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
	    os << weightUpdateModels[synt].extraGlobalSynapseKernelParameterTypes[k] << " ";
	    os << weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k];
	    os << model.synapseName[i] << ", ";
	}
    }	
    os << model.ftype << " t";
    os << ENDL << ")";  // end of synapse kernel header
    os << ENDL;
    

    // kernel code
    os << OB(75);
    //common variables for all cases
    os << "unsigned int id = " << "BLOCKSZ_SYN" << " * blockIdx.x + threadIdx.x;" << ENDL;
    os << "volatile __shared__ " << model.ftype << " shLg[" << neuronBlkSz << "];" << ENDL;
    os << "unsigned int lmax, j, r;" << ENDL;
    os << model.ftype << " addtoinSyn;" << ENDL;  

    //case-dependent variables
    for (int i = 0; i < model.synapseGrpN; i++) { 
      if ((model.synapseConnType[i] != SPARSE) || (!isGrpVarNeeded[model.synapseTarget[i]])){
        os << model.ftype << " linSyn;" << ENDL;
        break;
      }
    }
    // we need ipost in any case, and we need npost if there are any SPARSE connections
    os << "unsigned int ipost;" << ENDL;
    for (int i = 0; i < model.synapseGrpN; i++) {  
	    if (model.synapseConnType[i] == SPARSE){
	      os << "unsigned int prePos; " << ENDL;		
	      os << "unsigned int npost; " << ENDL;		
	      break; 
	    }    
    }  
    
   for (int i = 0; i < model.synapseGrpN; i++) {
	if (model.usesTrueSpikes[i] || model.usesPostLearning[i]) {  
	    os << "__shared__ unsigned int shSpk[" << "BLOCKSZ_SYN" << "];" << ENDL;
//	    os << "__shared__ " << model.ftype << " shSpkV[" << "BLOCKSZ_SYN" << "];" << ENDL;
	    os << "unsigned int lscnt, numSpikeSubsets;" << ENDL;
	    break;
	}
    }
   
    for (int i = 0; i < model.synapseGrpN; i++) {
	if(model.usesSpikeEvents[i]){
	    os << "__shared__ unsigned int shSpkEvnt[" << "BLOCKSZ_SYN" << "];" << ENDL;
	    vector<string> vars= model.synapseSpkEvntVars[i];
	    for (int j= 0; j < vars.size(); j++) {
		os << "__shared__ " << model.ftype << " shSpkEvnt" << vars[j] << "[" << "BLOCKSZ_SYN" << "];" << ENDL; 
	    }
	    os << "unsigned int lscntEvnt, numSpikeSubsetsEvnt;" << ENDL;    
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
	unsigned int synt = model.synapseType[i];
	if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot = (d_spkQuePtr" << model.neuronName[src] << " + ";
	    os << (int) (model.neuronDelaySlots[src] - model.synapseDelay[i] + 1);
	    os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
	}
	if ((model.synapseConnType[i] != SPARSE) || (!isGrpVarNeeded[model.synapseTarget[i]])){
	    os << "// only do this for existing neurons" << ENDL;
	    os << "if (" << localID << " < " << nN <<")" << OB(80);
	    os << "linSyn = dd_inSyn" << model.synapseName[i] << "[" << localID << "];" << ENDL;

	    os << CB(80);
	}
	if (model.usesSpikeEvents[i] == TRUE){
	    os << "lscntEvnt = d_glbSpkCntEvnt" << model.neuronName[src];
	    if (model.neuronDelaySlots[src] != 1) {
		os << "[delaySlot]";
	    }
	    os << ";" << ENDL;
	    os << "numSpikeSubsetsEvnt = (unsigned int) (ceilf((float) lscntEvnt / " << "((float)BLOCKSZ_SYN)" << "));" << ENDL;
	}
  
	if ((model.usesTrueSpikes[i]) || (model.usesPostLearning[i])){
	    os << "lscnt = d_glbSpkCnt" << model.neuronName[src];
	    if (model.neuronDelaySlots[src] != 1) os << "[d_spkQuePtr" << model.neuronName[src] << "]";
	    os << ";" << ENDL;
	    os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << "((float)BLOCKSZ_SYN)" << "));" << ENDL;
	}

	// generate the code for processing spike like events
	if (model.usesSpikeEvents[i] == TRUE) {	
	    generate_process_presynaptic_events_code(os, model, src, trg, i, localID, inSynNo, tS("Evnt"));
	}
	// generate the code for processing true spike events
	if (model.usesTrueSpikes[i] == TRUE) {
	    generate_process_presynaptic_events_code(os, model, src, trg, i, localID, inSynNo, tS(""));
	}       
	   
	if ((model.synapseConnType[i] != SPARSE) || (!isGrpVarNeeded[model.synapseTarget[i]])) {
	    os << "// only do this for existing neurons" << ENDL;
	    os << "if (" << localID << " < " << model.neuronN[trg] <<")" << OB(190);
	    os << "dd_inSyn" << model.synapseName[i] << "[" << localID << "] = linSyn;" << ENDL;
	    os << CB(190);
	}
	if (model.lrnGroups == 0) { // need to do reset operations in this kernel (no learning kernel)
	    os << "if (threadIdx.x == 0)" << OB(200);
	    os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
	    os << "if (j == " << numOfBlocks - 1 << ")" << OB(210);
	    for (int j = 0; j < model.neuronGrpN; j++) {
		os << "d_glbSpkCnt" << model.neuronName[j] << " = 0;" << ENDL;
		if (model.neuronDelaySlots[j] != 1) { // with delays
		    os << "d_spkQuePtr" << model.neuronName[j] << " = (d_spkQuePtr";
		    os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		    os << "d_glbSpkCntEvnt" << model.neuronName[j] << "[d_spkQuePtr";
		    os << model.neuronName[j] << "] = 0;" << ENDL;
		}
		else { // no delays
		    os << "d_glbSpkCntEvnt" << model.neuronName[j] << " = 0;" << ENDL;
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
	for (int i=0; i< model.synapseName.size(); i++){
	    unsigned int synt= model.synapseType[i];
	    for (int k= 0, l= weightUpdateModels[synt].extraGlobalSynapseKernelParameters.size(); k < l; k++) {
		os << weightUpdateModels[synt].extraGlobalSynapseKernelParameterTypes[k] << " ";
		os << weightUpdateModels[synt].extraGlobalSynapseKernelParameters[k];
		os << model.synapseName[i] << ", ";
	    }
	}
	os << model.ftype << " t";
	os << ")" << ENDL;

	// kernel code
	os << OB(215);
	os << "unsigned int id = " << learnBlkSz << " * blockIdx.x + threadIdx.x;" << ENDL;
	os << "__shared__ unsigned int shSpk[" << learnBlkSz << "];" << ENDL;
//	os << "__shared__ " << model.ftype << " shSpkV[" << learnBlkSz << "];" << ENDL;
	os << "unsigned int lscnt, numSpikeSubsets, lmax, j, r;" << ENDL;
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
	    unsigned int inSynNo = model.synapseInSynNo[k];
	    unsigned int synt = model.synapseType[k];
	    bool sparse= (model.synapseConnType[k] == SPARSE);

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM		
 	    os << "lscnt = d_glbSpkCnt" << model.neuronName[trg] << ";" << ENDL;

	    os << "numSpikeSubsets = (unsigned int) (ceilf((float) lscnt / " << learnBlkSz << ".0f));" << ENDL;
	    os << "for (r = 0; r < numSpikeSubsets; r++)" << OB(230);
	    os << "if (r == numSpikeSubsets - 1) lmax = lscnt % " << learnBlkSz << ";" << ENDL;
	    os << "else lmax = " << learnBlkSz << ";" << ENDL;
	    os << "if (threadIdx.x < lmax)" << OB(240);
	    os << "shSpk[threadIdx.x] = d_glbSpk" << model.neuronName[trg] << "[";
	    os << "(r * " << learnBlkSz << ") + threadIdx.x];" << ENDL;
//	    os << "shSpkV[threadIdx.x] = dd_V" << model.neuronName[trg] << "[";
//	    os << "shSpk[threadIdx.x]];" << ENDL;
	    os << CB(240);
	    os << "__syncthreads();" << ENDL;
	    os << "// only work on existing neurons" << ENDL;
	    os << "if (" << localID << " < " << model.neuronN[src] << ")" << OB(250);
	    os << "// loop through all incoming spikes for learning" << ENDL;
	    os << "for (j = 0; j < lmax; j++)" << OB(260) << ENDL;
	    if (sparse) {
		os << "unsigned int iprePos = dd_revIndInG" <<  model.synapseName[k];
		os << "[shSpk[j]] + " << localID << ";" << ENDL;
	    }
	    // for DENSE, ipre == localID
	    string code = weightUpdateModels[synt].simLearnPost;
	    // Code substitutions ----------------------------------------------------------------------------------
	    if (sparse) { // SPARSE
		name_substitutions(code, tS("dd_"), weightUpdateModels[synt].varNames, model.synapseName[k] 
				   + tS("[dd_remap") + model.synapseName[k] + tS("[iprePos]]"));
	    }
	    else{ // DENSE
		name_substitutions(code, tS("dd_"), weightUpdateModels[synt].varNames, model.synapseName[k] 
				   + tS("[")+ localID + tS(" * ") + tS(model.neuronN[trg]) +tS(" + shSpk[j]]"));
	    }
	    value_substitutions(code, weightUpdateModels[synt].pNames, model.synapsePara[k]);
	    value_substitutions(code, weightUpdateModels[synt].dpNames, model.dsp_w[k]);
	    name_substitutions(code, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[k]);
	    // presynaptic neuron variables and parameters
	    unsigned int nt_pre= model.neuronType[src];
	    if (model.neuronType[src] == POISSONNEURON) substitute(code, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
	    if (sparse) {
		substitute(code, tS("$(sT_pre)"), tS("d_sT") + tS(model.neuronName[src])
					    +tS("[dd_revInd" + model.synapseName[k] + "[iprePos]]"));
		extended_name_substitutions(code, tS("dd_"), nModels[nt_pre].varNames, tS("_pre"), tS(model.neuronName[src])
					    +tS("[dd_revInd" + model.synapseName[k] + "[iprePos]]"));
	    }
	    else {
		substitute(code, tS("$(sT_pre)"), tS("d_sT") + tS(model.neuronName[src])
			   +tS("[")+localID+tS("]"));
		extended_name_substitutions(code, tS("dd_"), nModels[nt_pre].varNames, tS("_pre"), model.neuronName[src]+tS("["+localID+"]"));
	    }
	    extended_value_substitutions(code, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
	    extended_value_substitutions(code, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);
	    // postsynaptic neuron variables and parameters
	    unsigned int nt_post= model.neuronType[trg];
	    substitute(code, tS("$(sT_post)"), tS("d_sT")+ model.neuronName[trg]+tS("[ShSpk[j]]"));
	    extended_name_substitutions(code, tS("dd_"), nModels[nt_post].varNames, tS("_post"), model.neuronName[trg]+tS("[ShSpk[j]]"));
	    extended_value_substitutions(code, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
	    extended_value_substitutions(code, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
	    // end Code substitutions ------------------------------------------------------------------------- 
	    os << ensureFtype(code, model.ftype) << ENDL;
	    os << CB(260);
	    os << CB(250);
	    os << CB(230);
	    os << "__threadfence();" << ENDL;
	    os << "if (threadIdx.x == 0)" << OB(320);
	    os << "j = atomicAdd((unsigned int *) &d_done, 1);" << ENDL;
	    os << "if (j == " << numOfBlocks - 1 << ")" << OB(330);
	    for (int j = 0; j < model.neuronGrpN; j++) {
		os << "d_glbSpkCnt" << model.neuronName[j] << " = 0;" << ENDL;
		if (model.neuronDelaySlots[j] != 1) { // with delay
		    os << "d_spkQuePtr" << model.neuronName[j] << " = (d_spkQuePtr";
		    os << model.neuronName[j] << " + 1) % " << model.neuronDelaySlots[j] << ";" << ENDL;
		    os << "d_glbSpkCntEvnt" << model.neuronName[j] << "[d_spkQuePtr";
		    os << model.neuronName[j] << "] = 0;" << ENDL;
		}
		else { // no delay
		    os << "d_glbSpkCntEvnt" << model.neuronName[j] << " = 0;" << ENDL;
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
