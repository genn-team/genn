/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "global.h"
#include "utils.h"
#include <string>

//------------------------------------------------------------------------
/*! \file generateKernels.cc

  \brief Contains functions that generate code for CUDA kernels. Part of the code generation section.
*/
//-------------------------------------------------------------------------


//! Macro for a "safe" output of a parameter into generated code by essentially just adding a bracket around the parameter value in the generated code.
#define SAVEP(X) "(" << X << ")" 


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA kernel that simulates all neurons in the model.

The code generated upon execution of this function is for defining GPU side global variables that will hold model state in the GPU global memory and for the actual kernel function for simulating the neurons for one time step.
*/
//-------------------------------------------------------------------------

void genNeuronKernel(NNmodel &model, //!< Model description 
		     string &path,  //!< path for code output
		     ostream &mos //!< output stream for messages
		     )
{
   // write header content
  string name, s, localID;
  unsigned int nt;
  ofstream os;

  name= path + toString("/") + model.name + toString("_CODE/neuronKrnl.cc");
  os.open(name.c_str());
  
  writeHeader(os);
  // compiler/include control (include once)
  os << "#ifndef _" << model.name << "_neuronKrnl_cc" << endl;
  os << "#define _" << model.name << "_neuronKrnl_cc" << endl;
  
  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file neuronKrnl.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model.name << " containing the neuron kernel function." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;

  // global device variables
 
  os << "// relevant neuron variables" << endl;
  os << "__device__ volatile unsigned int d_done;" << endl;
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    os << "__device__ volatile unsigned int d_glbscnt";
    os << model.neuronName[i] << ";" << endl;
    os << "__device__ volatile unsigned int d_glbSpk" << model.neuronName[i] << "[";
    os << model.neuronN[i] << "];" << endl;
    os << endl;
    if (model.neuronType[i] != POISSONNEURON) {
      for (int j= 0; j < model.inSyn[i].size(); j++) {
	os << "__device__ float d_inSyn" << model.neuronName[i] << j;
	os << "[" << model.neuronN[i] << "];";
	os << "    // summed input for neurons" << " in grp" << model.neuronName[i] << endl;
      }
    }
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "__device__ " << nModels[nt].varTypes[k];
      os << " d_" << nModels[nt].varNames[k] << model.neuronName[i];
      os << "[" << model.neuronN[i] << "]; " << endl;
    }
    os << endl;
    if (model.neuronNeedSt[i]) {
      os << "__device__ volatile float d_sT" << model.neuronName[i] << "[";
      os << model.neuronN[i] << "];" << endl;
    }
  }
  
  // Kernel for calculating neuron states
  // kernel header
  os << "__global__ void" << endl;
  os << "calcNeurons(";
  for (int i= 0; i < model.neuronGrpN; i++) {
    if (model.neuronType[i] == POISSONNEURON) {
      // Note: Poisson neurons only used as input neurons; they do not 
      // receive any inputs
      os << "unsigned int *d_rates" << model.neuronName[i] << ", // poisson ";
      os << "\"rates\" of grp " << model.neuronName[i] << endl;
      os << "unsigned int offset" << model.neuronName[i] << ", // poisson ";
      os << "\"rates\" offset of grp " << model.neuronName[i] << endl;
    }
  }
  os << "float t // absolute time" << endl; 
  os << ")" << endl;
  os << "{" << endl;
  os << "  unsigned int id= " << neuronBlkSz << "*blockIdx.x + threadIdx.x;" << endl;
  os << "  __shared__ volatile unsigned int scnt;" << endl;
  os << "  __shared__ volatile unsigned int pos;" << endl;
  os << "  __shared__ unsigned int shSpk[" << neuronBlkSz << "];" << endl;
  os << "  unsigned int sidx;" << endl;
  os << endl;

  os << "  if (threadIdx.x == 0) scnt= 0;" << endl;
  for (int i= 0; i < model.neuronGrpN; i++) {
    nt= model.neuronType[i];
    if (i == 0) {
      os << "  if (id < " << model.padSumNeuronN[i] << ") {" << endl;
      localID= string("id");
    }
    else {
      os << "  if ((id >= " << model.padSumNeuronN[i-1] << ") &&";
      os << " (id < " << model.padSumNeuronN[i] << ")) {" << endl;
      os << "    unsigned int lid;" << endl;
      os << "    lid= id-" << model.padSumNeuronN[i-1] << ";" << endl;
      localID= string("lid");
    }
    os << "    // only do this for existing neurons" << endl;
    os << "    if (" << localID << " <" << model.neuronN[i] << ") {" << endl;
    os << "    // pull V values in a coalesced access" << endl;
    if (nt == POISSONNEURON) {
      os << "    unsigned int lrate= d_rates"  << model.neuronName[i];
      os << "[offset" << model.neuronName[i] << "+" << localID << "];" << endl;
    }
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "    " << nModels[nt].varTypes[k] << " ";
      os << "l" << nModels[nt].varNames[k] << "= d_";
      os <<  nModels[nt].varNames[k] << model.neuronName[i];
      os << "[" << localID << "];" << endl;
    }
    if (nt != POISSONNEURON) {
      os << "    // pull inSyn values in a coalesced access" << endl;
      for (int j= 0; j < model.inSyn[i].size(); j++) {
	os << "    float linSyn" << j << "= d_inSyn" << model.neuronName[i] << j << "[";
	os << localID << "];" << endl;
      }
      if (model.inSyn[i].size() > 0) {
	os << "    float Isyn=";
	for (int j= 0; j < model.inSyn[i].size(); j++) {
	  os << " linSyn" << j << "*(";
	  os << SAVEP(model.synapsePara[model.inSyn[i][j]][0]);
	  os << "-lV)";
	  if (j < model.inSyn[i].size()-1) os << "+";
	  else os << ";" << endl;
	}
      }
    }
    os << "    // calculate membrane potential" << endl;
    //new way of doing it
    string code= nModels[nt].simCode;
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      substitute(code, tS("$(")+nModels[nt].varNames[k]+tS(")"), 
		 tS("l")+ nModels[nt].varNames[k]);
    }
    substitute(code, tS("$(Isyn)"), tS("Isyn"));
    for (int k= 0, l= nModels[nt].pNames.size(); k < l; k++) {
      substitute(code, tS("$(")+nModels[nt].pNames[k]+tS(")"), 
		 tS(model.neuronPara[i][k]));
    }
    for (int k= 0, l= nModels[nt].dpNames.size(); k < l; k++) {
      substitute(code, tS("$(")+nModels[nt].dpNames[k]+tS(")"), 
		 tS(model.dnp[i][k]));
    }
    os << code;
    os << "    if (lV >= " << model.nThresh[i] << ") {" << endl;
    os << "      // register a spike type event" << endl;
    os << "      sidx= atomicAdd((unsigned int*)&scnt,1);" << endl;
    os << "      shSpk[sidx]= " << localID << ";" << endl;
    os << "    }" << endl;
    for (int k= 0, l= nModels[nt].varNames.size(); k < l; k++) {
      os << "    d_" << nModels[nt].varNames[k] <<  model.neuronName[i];
      os << "[" << localID << "]= l" << nModels[nt].varNames[k] << ";" << endl;
    }
    for (int j= 0; j < model.inSyn[i].size(); j++) {
      os << "    d_inSyn"  << model.neuronName[i] << j << "[" << localID << "]= linSyn";
      unsigned int synID= model.inSyn[i][j];
      os << j << "*" << SAVEP(model.dsp[synID][0]) << ";" << endl;
    }
    os << "  }" << endl;
    os << "  __syncthreads();" << endl;
    os << "  if (threadIdx.x == 0) {" << endl;
    os << "    pos= atomicAdd((unsigned int*)&d_glbscnt" << model.neuronName[i] << ", scnt);" << endl;
    os << "  }" << endl;
    os << "  __syncthreads();" << endl;
    os << "  if (threadIdx.x < scnt) {" << endl;
    os << "    d_glbSpk" << model.neuronName[i] << "[pos+threadIdx.x]= shSpk[threadIdx.x];" << endl;
    if (model.neuronNeedSt[i]) {
      os << "    d_sT" << model.neuronName[i] << "[shSpk[threadIdx.x]]= t;" << endl;
    }
    os << "  }" << endl;
    os << "  }" << endl;
    os << endl;
  }
  os << "}" << endl << endl;
  os << "#endif" << endl;
  os.close();
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
  string name, s, localID, theLG;
  unsigned int BlkNo;
  ofstream os;

  // count how many blocks to use: one thread for each synapse target
  // targets of several input groups are counted multiply
  BlkNo= model.padSumSynapseTrgN[model.synapseGrpN-1] >> logSynapseBlkSz;

  name= path + toString("/") + model.name + toString("_CODE/synapseKrnl.cc");
  os.open(name.c_str());
  // write header content
  writeHeader(os);

  // compiler/include control (include once)
  os << "#ifndef _" << model.name << "_synapseKrnl_cc" << endl;
  os << "#define _" << model.name << "_synapseKrnl_cc" << endl;
  os << endl;

  // write doxygen comment
  os << "//-------------------------------------------------------------------------" << endl;
  os << "/*! \\file synapseKrnl.cc" << endl << endl;
  os << "\\brief File generated from GeNN for the model " << model.name << " containing the synapse kernel and learning kernel functions." << endl;
  os << "*/" << endl;
  os << "//-------------------------------------------------------------------------" << endl << endl;

  for (int i= 0; i < model.synapseGrpN; i++) {
    if (model.synapseGType[i] == INDIVIDUALG) {
      // (cases necessary here when considering sparse reps as well)
      os << "__device__ float d_gp" << model.synapseName[i] << "[";
      os << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
      os << "];     // synaptic conductances of group " << model.synapseName[i];
      os << endl;
      if (model.synapseType[i] == LEARN1SYNAPSE) {
	os << "__device__ float d_grawp" << model.synapseName[i] << "[";
	os << model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
	os << "];     // raw synaptic conductances of group " << model.synapseName[i];
	os << endl;
      }
    }
    // note, if GLOBALG we put the value at compile time
    if (model.synapseGType[i] == INDIVIDUALID) {
      os << "__device__ unsigned int d_gp" << model.synapseName[i] << "[";
      unsigned int tmp= model.neuronN[model.synapseSource[i]]*model.neuronN[model.synapseTarget[i]];
      unsigned int size= tmp >> logUIntSz;
      if (tmp > (size << logUIntSz)) size++;
      os << size;
      os << "];     // synaptic connectivity of group " << model.synapseName[i];
      os << endl;
    }
  }
  os << endl;

  // Kernel for calculating synapse input to neurons
  // Kernel header
  os << "__global__ void" << endl;
  os << "calcSynapses(";
  if (model.needSt) {
    os << "float t";
  }
  os << ")" << endl;
  os << "{" << endl;
  os << "  unsigned int id= " << synapseBlkSz << "*blockIdx.x + threadIdx.x;" << endl;

  os << "  __shared__ unsigned int shSpk[" << synapseBlkSz << "];" << endl;
  os << "  __shared__ float shSpkV[" << synapseBlkSz << "];" << endl;
  os << "  unsigned int lscnt, lrpt, lmax, j, r;" << endl;
  os << "  float linSyn, lg;" << endl;
  os << endl;

  for (int i= 0; i < model.synapseGrpN; i++) {
    if (i == 0) {
      os << "  if (id < " << model.padSumSynapseTrgN[i] << ") { " << endl;
      localID= string("id");
    }
    else {
      os << "  if ((id >= " << model.padSumSynapseTrgN[i-1] << ") &&";
      os << " (id < " << model.padSumSynapseTrgN[i] << ")) {" << endl;
      os << "    unsigned int lid;" << endl;
      os << "    lid= id-" << model.padSumSynapseTrgN[i-1] << ";" << endl;
      localID= string("lid");
    }
    unsigned int trg= model.synapseTarget[i];
    unsigned int nN= model.neuronN[trg];
    unsigned int src= model.synapseSource[i];
    float Epre= model.synapsePara[i][1];
    float Vslope;
    if (model.synapseType[i] == NGRADSYNAPSE) {
	   Vslope= model.synapsePara[i][3]; //fails here
	}
    unsigned int inSynNo= model.synapseInSynNo[i];

    os << "    // only do this for existing neurons" << endl;
    os << "    if (" << localID << " < " << nN <<") {" << endl;
    os << "      linSyn= d_inSyn" << model.neuronName[trg] << inSynNo << "[" << localID << "];" << endl;
    os << "    }" << endl;
    os << "    lscnt= d_glbscnt" << model.neuronName[src] << ";" << endl;
    os << "    lrpt= lscnt >> " << logSynapseBlkSz << ";" << endl;
    os << "    if ((lrpt << " << logSynapseBlkSz <<") < lscnt) lrpt= lrpt+1;" << endl;
    os << "    for (r= 0; r < lrpt; r++) {" << endl;
    os << "      if (r == lrpt-1) lmax= lscnt-((lrpt-1) << " << logSynapseBlkSz << ");";
    os << endl;
    os << "      else lmax= " << synapseBlkSz << ";" << endl; 
    os << "      if (threadIdx.x < lmax) {" << endl;
    os << "        shSpk[threadIdx.x]= d_glbSpk" << model.neuronName[src];
    os << "[(r << " << logSynapseBlkSz << ") + threadIdx.x];" << endl;
    if (model.neuronType[src] != POISSONNEURON) {
      os << "        shSpkV[threadIdx.x]= d_V" << model.neuronName[src];
      os << "[shSpk[threadIdx.x]];" << endl;
    }
    os << "      }" << endl;
    os << "      __syncthreads();" << endl;
    os << "      // only work on existing neurons" << endl;
    os << "      if (" << localID << " < " << model.neuronN[trg] << ") {" << endl;
    os << "        // loop through all incoming spikes" << endl;
    os << "        for (j= 0; j < lmax; j++) {" << endl;
    if (model.synapseGType[i] == INDIVIDUALID) {
      os << "          unsigned int gid= (shSpk[j]*" << model.neuronN[trg];
      os << " + " << localID << ");" << endl;
    }
    if (model.neuronType[src] != POISSONNEURON) {
      os << "          if ";
      if (model.synapseGType[i] == INDIVIDUALID) {
	// Note: we will just access global mem. For compute >= 1.2
	// simultaneous access to same global mem in the (half-)warp
	// will be coalesced - no worries
	os << "((B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz-1 << ")) && ";
      } 
      os << "(shSpkV[j] > " << Epre << ")";
      if (model.synapseGType[i] == INDIVIDUALID) {
	os << ")";
      }
      os << " {" << endl;
    }
    else {
      if (model.synapseGType[i] == INDIVIDUALID) {
	os << "          if (B(d_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";
	os << UIntSz-1 << ")) {" << endl;
      }
    }
    if (model.synapseGType[i] == INDIVIDUALG) {
      os << "            lg= d_gp" << model.synapseName[i] << "[shSpk[j]*" << model.neuronN[trg] << " + " << localID << "];";
      os << endl;
      theLG= toString("lg");
    }
    if ((model.synapseGType[i] == GLOBALG) ||
	(model.synapseGType[i] == INDIVIDUALID)) {
      theLG= toString(model.g0[i]);
    }
    if ((model.synapseType[i] == NSYNAPSE) || 
	(model.synapseType[i] == LEARN1SYNAPSE)) {
      os << "              linSyn= linSyn+" << theLG << ";" << endl;
    }
    if (model.synapseType[i] == NGRADSYNAPSE) {
      if (model.neuronType[src] == POISSONNEURON) {
	os << "            linSyn= linSyn+" << theLG << "*tanh((float)((";
	os << SAVEP(model.neuronPara[src][2]) << "-" << SAVEP(Epre);
      }
      else {
	os << "            linSyn= linSyn+" << theLG << "*tanh((float)((shSpkV[j]-" << SAVEP(Epre);
      }
      os << ")/" << Vslope << "));" << endl;
    }
    // if needed, do some learning (this is for pre-synaptic spikes)
    if (model.synapseType[i] == LEARN1SYNAPSE) {
      // simply assume INDIVIDUALG for now
      os << "            lg= d_grawp" << model.synapseName[i] << "[shSpk[j]*" << model.neuronN[trg] << " + " << localID << "];" << endl;
      os << "            float dt= d_sT" << model.neuronName[trg] << "[" << localID << "] - t - ";
      os << SAVEP(model.synapsePara[i][11]) << ";" << endl;
      os << "            if (dt > " << model.dsp[i][1] << ") {" << endl;
      os << "              dt= -" << SAVEP(model.dsp[i][5]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > 0.0) {" << endl;
      os << "              dt= " << SAVEP(model.dsp[i][3]) << "*dt+" << SAVEP(model.dsp[i][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > " << model.dsp[i][2] << ") {" << endl;
      os << "              dt= " << SAVEP(model.dsp[i][4]) << "*dt+" << SAVEP(model.dsp[i][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else {" << endl;
      os << "              dt= -" << SAVEP(model.dsp[i][7]) << ";" << endl;
      os << "            }" << endl;
      os << "            lg= lg+dt;" << endl;
      os << "            d_grawp" << model.synapseName[i] << "[shSpk[j]*" << model.neuronN[trg] << " + " << localID << "]= ";
      os << "lg;" << endl;
      os << "            d_gp" << model.synapseName[i] << "[shSpk[j]*" << model.neuronN[trg] << " + " << localID << "]= ";
      os << "gFunc" << model.synapseName[i] << "(lg);" << endl; 
    }
    if ((model.neuronType[src] != POISSONNEURON) ||
	(model.synapseGType[i] == INDIVIDUALID)) {
      os << "          }" << endl;
    }
    os << "        }" << endl;
    os << "      }" << endl;
    os << "    }" << endl;
    os << "    // only do this for existing neurons" << endl;
    os << "    if (" << localID << " < " << model.neuronN[trg] <<") {" << endl;
    os << "      d_inSyn" << model.neuronName[trg] << inSynNo << "[" << localID << "]= linSyn;" << endl;
    os << "    }" << endl;
    os << "    __syncthreads();" << endl;
    os << "    if ( threadIdx.x == 0) {" << endl;
    os << "      j= atomicAdd((unsigned int*)&d_done, 1);" << endl;
    os << "      if (j == " << BlkNo-1 << ") {" << endl;
    for (int j= 0; j < model.neuronGrpN; j++) {
      os << "        d_glbscnt" << model.neuronName[j] << "= 0;" << endl;
    }
    os << "          d_done= 0;" << endl;
    os << "      }" << endl;
    os << "    }" << endl;
    os << "  }" << endl;
    os << endl;
  }
  os << "}" << endl << endl;

  if (model.lrnGroups > 0) {
    // Kernel for learning synapses, post-synaptic spikes
    // Kernel header
    os << "__global__ void" << endl;
    os << "learnSynapsesPost(float t)" << endl;
    os << "{" << endl;
    os << "  unsigned int id= " << learnBlkSz << "*blockIdx.x + threadIdx.x;" << endl;
    os << "  __shared__ unsigned int shSpk[" << learnBlkSz << "];" << endl;
    os << "  __shared__ float shSpkV[" << learnBlkSz << "];" << endl;
    os << "  unsigned int lscnt, lrpt, lmax, j, r;" << endl;
    os << "  float lg;" << endl;
    os << endl;

    for (int i= 0; i < model.lrnGroups; i++) {
      if (i == 0) {
	os << "  if (id < " << model.padSumLearnN[i] << ") {" << endl;
	localID= string("id");
      }
      else {
	os << "  if ((id >= " << model.padSumLearnN[i-1] << ") &&";
	os << " (id < " << model.padSumLearnN[i] << ")) {" << endl;
	os << "    unsigned int lid;" << endl;
	os << "    lid= id-" << model.padSumLearnN[i-1] << ";" << endl;
	localID= string("lid");
      }
      unsigned int k= model.lrnSynGrp[i];
      unsigned int src= model.synapseSource[k];
      unsigned int nN= model.neuronN[src];
      unsigned int trg= model.synapseTarget[k];
      float Epre= model.synapsePara[k][1];
      os << "    lscnt= d_glbscnt" << model.neuronName[trg] << ";" << endl;
      os << "    lrpt= lscnt >> " << logLearnBlkSz << ";" << endl;
      os << "    if ((lrpt << " << logLearnBlkSz <<") < lscnt) lrpt= lrpt+1;" << endl;
      os << "    for (r= 0; r < lrpt; r++) {" << endl;
      os << "      if (r == lrpt-1) lmax= lscnt-((lrpt-1) << " << logLearnBlkSz << ");";
      os << endl;
      os << "      else lmax= " << learnBlkSz << ";" << endl; 
      os << "      if (threadIdx.x < lmax) {" << endl;
      os << "        shSpk[threadIdx.x]= d_glbSpk" << model.neuronName[trg];
      os << "[(r << " << logLearnBlkSz << ") + threadIdx.x];" << endl;
      os << "        shSpkV[threadIdx.x]= d_V" << model.neuronName[trg];
      os << "[shSpk[threadIdx.x]];" << endl;
      os << "      }" << endl;
      os << "      __syncthreads();" << endl;
      os << "      // only work on existing neurons" << endl;
      os << "      if (" << localID << " < " << model.neuronN[src] << ") {" << endl;
      os << "        // loop through all incoming spikes" << endl;
      os << "        for (j= 0; j < lmax; j++) {" << endl;
      os << "          if (shSpkV[j] > " << Epre << ") {" << endl;
      os << "            lg= d_grawp" << model.synapseName[k] << "[" << localID << "*";
      os << model.neuronN[trg] << " + shSpk[j]];" << endl;
      os << "            float dt= t - d_sT" << model.neuronName[src] << "[" << localID << "] - ";
      os << SAVEP(model.synapsePara[k][11]) << ";" << endl;
      os << "            if (dt > " << model.dsp[k][1] << ") {" << endl;
      os << "              dt= -" << SAVEP(model.dsp[k][5]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > 0.0) {" << endl;
      os << "              dt= " << SAVEP(model.dsp[k][3]) << "*dt+" << SAVEP(model.dsp[k][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else if (dt > " << model.dsp[k][2] << ") {" << endl;
      os << "              dt= " << SAVEP(model.dsp[k][4]) << "*dt+" << SAVEP(model.dsp[k][6]) << ";" << endl;
      os << "            }" << endl;
      os << "            else {" << endl;
      os << "              dt= -" << SAVEP(model.dsp[k][7]) << ";" << endl;
      os << "            }" << endl;
      os << "            lg= lg+dt;" << endl;
      os << "            d_grawp" << model.synapseName[k] << "[" << localID << "*";
      os << model.neuronN[trg] << " + shSpk[j]]= lg;" << endl;
      os << "            d_gp" << model.synapseName[k] << "[" << localID << "*";
      os << model.neuronN[trg] << " + shSpk[j]]= gFunc" << model.synapseName[k] << "(lg);" << endl; 
      os << "          }" << endl;
      os << "        }" << endl;
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

