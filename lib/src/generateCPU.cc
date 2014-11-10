/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateCPU.cc 

  \brief Functions for generating code that will run the neuron and synapse simulations on the CPU. Part of the code generation section.

*/
//--------------------------------------------------------------------------

#include <string>
#include "CodeHelper.cc"
//--------------------------------------------------------------------------
/*!
  \brief Function that generates the code of the function the will simulate all neurons on the CPU.

*/
//--------------------------------------------------------------------------

void genNeuronFunction(NNmodel &model, //!< Model description 
		       string &path, //!< output stream for code
		       ostream &mos //!< output stream for messages
    )
{
    string name, s, localID;
    unsigned int nt;
    ofstream os;

    name = path + toString("/") + model.name + toString("_CODE/neuronFnct.cc");
    os.open(name.c_str());
    // write header content
    writeHeader(os);
    os << ENDL;
    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_neuronFnct_cc" << ENDL;
    os << "#define _" << model.name << "_neuronFnct_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronFnct.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name;
    os << " containing the the equivalent of neuron kernel function for the CPU-only version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;
    // header
    os << "void calcNeuronsCPU(";
    for (int i = 0; i < model.neuronGrpN; i++) {
	if (model.neuronType[i] == POISSONNEURON) {
	    // Note: Poisson neurons only used as input neurons; they do not 
	    // receive any inputs
	    os << model.RNtype << " *rates" << model.neuronName[i] << ", // poisson ";
	    os << "\"rates\" of grp " << model.neuronName[i] << ENDL;
	    os << "unsigned int offset" << model.neuronName[i] << ", // poisson ";
	    os << "\"rates\" offset of grp " << model.neuronName[i] << ENDL;
	}
	if (model.receivesInputCurrent[i] >= 2) {
	  os << model.ftype << " *inputI" << model.neuronName[i] << ", // input current of grp " << model.neuronName[i] << ENDL;
	}
    }
    os << model.ftype << " t)" << ENDL;
    os << OB(51);
    for (int i = 0; i < model.neuronGrpN; i++) {
	    nt = model.neuronType[i];
	    os << "glbSpkCnt" << model.neuronName[i] << " = 0;" << ENDL;
	    if (model.neuronDelaySlots[i] == 1) {
	      os << "glbSpkCntEvnt" << model.neuronName[i] << " = 0;" << ENDL;
    	}
	    else {
	      os << "spkEvntQuePtr" << model.neuronName[i] << " = (spkEvntQuePtr" << model.neuronName[i] << " + 1) % ";
	      os << model.neuronDelaySlots[i] << ";" << ENDL;
	      os << "glbSpkCntEvnt" << model.neuronName[i] << "[spkEvntQuePtr" << model.neuronName[i] << "] = 0;" << ENDL;
	    }
	os << "for (int n = 0; n < " <<  model.neuronN[i] << "; n++)" << OB(10);
	for (int k = 0; k < nModels[nt].varNames.size(); k++) {
	    os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k] << " = ";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "(((spkEvntQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
		os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
	    }
	    os << "n];" << ENDL;
	}

 	if ((model.inSyn[i].size() > 0) || (model.receivesInputCurrent[i] > 0)) {
	    os << model.ftype << " Isyn = 0;" << ENDL;
	}

	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    unsigned int synPopID=  model.inSyn[i][j];
	    unsigned int synt= model.synapseType[synPopID];
	    if (weightUpdateModels[synt].synapseDynamics != tS("")) {
		weightUpdateModel wu= weightUpdateModels[synt];
		string code= wu.synapseDynamics;
		unsigned int srcno= model.neuronN[model.synapseSource[synPopID]]; 
		if (model.synapseConnType[synPopID] == SPARSE) { // SPARSE
		    os << "for (int k= 0; k < C" <<  model.synapseName[synPopID] << ".indInG[" << srcno << "]; k++) " << OB(24) << ENDL;
		    os << "// loop over all synapses" << ENDL;
		    if (model.synapseGType[synPopID] == INDIVIDUALG) {
		      name_substitutions(code, tS(""), wu.varNames, model.synapseName[synPopID] + tS("[k]"));
		    }
		    else {
		      value_substitutions(code, wu.varNames, model.synapseIni[synPopID]);
		    }
		    value_substitutions(code, wu.pNames, model.synapsePara[synPopID]);
		    value_substitutions(code, wu.dpNames, model.dsp_w[synPopID]);
		    os << code << ENDL;
		    os << CB(24);
		}
		else { // DENSE
		    unsigned int srcno= model.neuronN[model.synapseSource[synPopID]]; 
		    unsigned int trgno= model.neuronN[model.synapseTarget[synPopID]]; 
		    os << "for (int k= 0; k <" <<  srcno*trgno << "; k++) " << OB(25) << ENDL;
		    if (model.synapseGType[synPopID] == INDIVIDUALG) {
		      name_substitutions(code, tS(""), wu.varNames, model.synapseName[synPopID] + tS("[k]"));
		    }
		    else {
		      value_substitutions(code, wu.varNames, model.synapseIni[synPopID]);
		    }
		    value_substitutions(code, wu.pNames, model.synapsePara[synPopID]);
		    value_substitutions(code, wu.dpNames, model.dsp_w[synPopID]);
		    os << code << ENDL;
		    os << CB(25);
		}
	    }
	}
   
	if (model.inSyn[i].size() > 0) {
	    //os << "    Isyn = ";
	    for (int j = 0; j < model.inSyn[i].size(); j++) {
		unsigned int synPopID= model.inSyn[i][j]; // number of (post)synapse group
		postSynModel psm= postSynModels[model.postSynapseType[synPopID]];
		string sName= model.synapseName[synPopID];
		if (model.synapseGType[synPopID] == INDIVIDUALG) {
		  for (int k = 0, l = psm.varNames.size(); k < l; k++) {
		      os << psm.varTypes[k] << " lps" << psm.varNames[k] << sName;
		    os << " =" <<  psm.varNames[k] << sName << "[n];" << ENDL;
		    
		  }
		}
		os << "Isyn+= ";
      		string psCode = psm.postSyntoCurrent;
		substitute(psCode, tS("$(inSyn)"), tS("inSyn") + sName + tS("[n]"));
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
		os << psCode;
		os << ";" << ENDL;
	    }
	}
	if (model.receivesInputCurrent[i] == 1) { //receives constant input
	    if (model.synapseGrpN == 0) {
		os << "Isyn= " << model.globalInp[i] << ";" << ENDL;
	    }
	    else {
		os << "Isyn+= " << model.globalInp[i] << ";" << ENDL;
	    }
	}    	
	if (model.receivesInputCurrent[i] >= 2) {
	    if (model.synapseGrpN == 0) {
		os << "Isyn = (" << model.ftype << ") inputI" << model.neuronName[i] << "[n];" << ENDL;
	    }
	    else {
		os << "Isyn += (" << model.ftype << ") inputI" << model.neuronName[i] << "[n];" << ENDL;
	    }
	}
	string thCode= nModels[nt].thresholdConditionCode;
	if (nModels[nt].thresholdConditionCode  == tS("")) { //no condition provided
	    cerr << "Warning: No thresholdConditionCode for neuron type :  " << model.neuronType[i]  << " used for " << model.name[i] << " was provided. There will be no spikes detected in this population!" << ENDL;
	}
	else {
	    name_substitutions(thCode, tS("l"), nModels[nt].varNames, tS(""));
	    value_substitutions(thCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(thCode, nModels[nt].dpNames, model.dnp[i]);
	    substitute(thCode, tS("$(Isyn)"), tS("Isyn"));
	    os << "bool oldSpike= (" << thCode << ");" << ENDL;  
	}
	os << "// calculate membrane potential" << ENDL;
	string code = nModels[nt].simCode;
	name_substitutions(code, tS("l"), nModels[nt].varNames, tS(""));
	value_substitutions(code, nModels[nt].pNames, model.neuronPara[i]);
	value_substitutions(code, nModels[nt].dpNames, model.dnp[i]);
	name_substitutions(code, tS(""), nModels[nt].extraGlobalNeuronKernelParameters, model.neuronName[i]);
	if (nt == POISSONNEURON) {
	    substitute(code, tS("lrate"), 
		       tS("rates") + model.neuronName[i] + tS("[n + offset") + model.neuronName[i] + tS("]"));
	}
	substitute(code, tS("$(Isyn)"), tS("Isyn"));
	os << code << ENDL;

	if (model.neuronNeedSpkEvnt[i]) {
	    // look for spike type events first.
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
	    os << "(" + eCode + ")" << OB(30);
	    os << "// register a spike type event " << ENDL;
	    os << "glbSpkEvnt" << model.neuronName[i] << "[";
	    if (model.neuronDelaySlots[i] != 1) {
		os << "(spkEvntQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	    }
	    os << "glbSpkCntEvnt" << model.neuronName[i];
	    if (model.neuronDelaySlots[i] != 1) {
		os << "[spkEvntQuePtr" << model.neuronName[i] << "]";
	    }
	    os << "++] = n;" << ENDL;
	    os << CB(30) << ENDL;
	}
	
	if (thCode != tS("")) {
	    os << "if ((" << thCode << ") && !(oldSpike)) " << OB(40) << ENDL;
	    os << "// register a true spike" << ENDL;
	    os << "glbSpk" << model.neuronName[i] << "[";
	    os << "glbSpkCnt" << model.neuronName[i] << "++] = n;" << ENDL;
	    if (model.neuronNeedSt[i]) {
		os << "sT" << model.neuronName[i] << "[n] = t;" << ENDL;
	    }
	    if (nModels[nt].resetCode != tS("")) {
		code = nModels[nt].resetCode;
		name_substitutions(code, tS("l"), nModels[nt].varNames, tS(""));
		value_substitutions(code, nModels[nt].pNames, model.neuronPara[i]);
		value_substitutions(code, nModels[nt].dpNames, model.dnp[i]);
		substitute(code, tS("$(Isyn)"), tS("Isyn"));
		os << "// spike reset code" << ENDL;
		os << code << ENDL;
	    }
	    os << CB(40);
	}
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    os << nModels[nt].varNames[k] <<  model.neuronName[i] << "[";
	    if ((nModels[nt].varNames[k] == "V") && (model.neuronDelaySlots[i] != 1)) {
		os << "(spkEvntQuePtr" << model.neuronName[i] << " * " << model.neuronN[i] << ") + ";
	    };
	    os << "n] = l" << nModels[nt].varNames[k] << ";" << ENDL;
	}
	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    postSynModel psModel= postSynModels[model.postSynapseType[model.inSyn[i][j]]];
	    string psCode = psModel.postSynDecay;
	    substitute(psCode, tS("$(inSyn)"), tS("inSyn") + model.synapseName[model.inSyn[i][j]] +tS("[n]"));
	    name_substitutions(psCode, tS("lps"), psModel.varNames, model.synapseName[model.inSyn[i][j]]);
	    value_substitutions(psCode, psModel.pNames, model.postSynapsePara[model.inSyn[i][j]]);
	    value_substitutions(psCode, psModel.dpNames, model.dpsp[model.inSyn[i][j]]);
	    name_substitutions(psCode, tS("l"), nModels[nt].varNames, tS(""));
	    value_substitutions(psCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(psCode, nModels[nt].dpNames, model.dnp[i]);
	    os << "// the post-synaptic dynamics" << ENDL;
	    os << psCode;
	    for (int k = 0, l = psModel.varNames.size(); k < l; k++) {
		os << psModel.varNames[k] << model.synapseName[model.inSyn[i][j]] << "[n]" << " = lps" << psModel.varNames[k] << model.synapseName[model.inSyn[i][j]] << ";" << ENDL;
	    }
	}
	os << CB(10) << ENDL;
	os << ENDL;
    }
    os << CB(51) << ENDL << ENDL;
    os << "#endif" << ENDL;
    os.close();
} 


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic 
  spikes or spike type events

*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code_CPU(
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
    int Evnt= 0;
    if (postfix == tS("Evnt")) {
	Evnt= 1;
    }
    unsigned int synt = model.synapseType[i];
    bool needSpkEvnt= (weightUpdateModels[synt].evntThreshold != tS(""));
    if ((Evnt && needSpkEvnt) || !Evnt) {
    // Detect spike events or spikes and do the update
    os << "// process presynaptic events:";
    if (Evnt) {
	os << " Spike type events" << ENDL;
    }
    else {
	os << " True Spikes" << ENDL;
    }

    os << "// loop through all incoming spikes" << ENDL;
    os << "for (int j = 0; j < glbSpkCnt" << postfix << model.neuronName[src];
    if (model.neuronDelaySlots[src] != 1) {
	os << "[delaySlot]";
    }
    os << "; j++) " << OB(201);

    os << "unsigned int lSpk= glbSpk" << postfix << model.neuronName[src] << "[";
    if (model.neuronDelaySlots[src] != 1) {
	os << "delaySlot * " << model.neuronN[src] << " + ";
    }
    os << "j];" << ENDL;
	
    if (model.synapseConnType[i] == SPARSE) {
	os << "npost = C" << model.synapseName[i] << ".indInG[ lSpk + 1] - C" << model.synapseName[i] << ".indInG[lSpk];" << ENDL;
	os << "for (int l = 0; l < npost; l++)" << OB(202);
	os << "ipost = C" << model.synapseName[i] << ".ind[C" << model.synapseName[i];
	os << ".indInG[lSpk] + l];" << ENDL;
    }
    else {
	os << "for (int n = 0; n < " << model.neuronN[trg] << "; n++)" << OB(202);
	os << "ipost= n;" << endl;
    }
    
    if (model.synapseGType[i] == INDIVIDUALID) {
	os << "unsigned int gid = (lSpk * " << model.neuronN[i] << " + n);" << ENDL;
    }
    if ((Evnt) && (model.neuronType[src] != POISSONNEURON)) { // consider whether POISSON Neurons should be allowed to throw events
	
	string extension= model.neuronName[src] + tS("[");
	if (model.neuronDelaySlots[src] != 1) {
	    extension+= tS("delaySlot *") + tS(model.neuronN[src]) + tS(" + ");;
	}
	extension+= tS("lSpk]");
	os << "if ";
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "((B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & "; 
	    os << UIntSz - 1 << ")) && ";
	}
	string eCode= weightUpdateModels[synt].evntThreshold;
	// code substitutions ----
	extended_name_substitutions(eCode, tS(""), model.synapseSpkEvntVars[i], tS("_pre"), extension);
	value_substitutions(eCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
	value_substitutions(eCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
	name_substitutions(eCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
	// end code substitutions ----
	os << "(" << eCode << ")"; 
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << ")";
	}
	os << OB(2041);
    }
    else {
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "if (B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & ";				  
	    os << UIntSz - 1 << "))" << OB(2041);
	}
    }

    bool sparse= (model.synapseConnType[i] == SPARSE);
    unsigned int synt = model.synapseType[i];
    string wCode;
    if (Evnt) {
	wCode = weightUpdateModels[synt].simCodeEvnt;
    }
    else {
	wCode= weightUpdateModels[synt].simCode;
    }
    // Code substitutions ----------------------------------------------------------------------------------
    substitute(wCode, tS("$(updatelinsyn)"), tS("$(inSyn)+=$(addtoinSyn)")); 		   		
    if (sparse) { // SPARSE
      if (model.synapseGType[i] == INDIVIDUALG) {
	  name_substitutions(wCode, tS(""), weightUpdateModels[synt].varNames, model.synapseName[i]+ tS("[C")+ model.synapseName[i] + tS(".indInG[lSpk]")  + tS("+ l]"));
      }
      else {
	value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
      }
    }
    else { // DENSE
      if (model.synapseGType[i] == INDIVIDUALG) {
	name_substitutions(wCode, tS(""), weightUpdateModels[synt].varNames, model.synapseName[i]+tS("[lSpk") + tS("*") + tS(model.neuronN[trg]) + tS("+ ipost ]"));
      }
      else {
	value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
      }      
    }
    substitute(wCode, tS("$(inSyn)"), tS("inSyn")+  model.synapseName[i] + tS("[ipost]"));
    value_substitutions(wCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
    value_substitutions(wCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
    name_substitutions(wCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
    substitute(wCode, tS("$(addtoinSyn)"), tS("addtoinSyn"));
    // presynaptic neuron variables and parameters
    unsigned int nt_pre= model.neuronType[src];
    if (model.neuronType[src] == POISSONNEURON) substitute(wCode, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
    substitute(wCode, tS("$(sT_pre)"), tS("sT")+model.neuronName[src]+tS("[lSpk]"));
    extended_name_substitutions(wCode, tS(""), nModels[nt_pre].varNames, tS("_pre"), tS(model.neuronName[src])+tS("[glbSpk") + postfix + model.neuronName[src] + tS("[j]]"));
    extended_value_substitutions(wCode, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
    extended_value_substitutions(wCode, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);
    // postsynaptic neuron variables and parameters
    unsigned int nt_post= model.neuronType[trg];
    substitute(wCode, tS("$(sT_post)"), tS("sT")+model.neuronName[trg]+tS("[ipost]"));
    extended_name_substitutions(wCode, tS(""), nModels[nt_post].varNames, tS("_post"), tS(model.neuronName[trg])+tS("[ipost]"));
    extended_value_substitutions(wCode, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
    extended_value_substitutions(wCode, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
    // end Code substitutions ------------------------------------------------------------------------- 
    os << wCode << ENDL;
    if ((Evnt) && (model.neuronType[src] != POISSONNEURON)) {
	os << CB(2041) << ENDL; // end if (shSpkV" << postfix << "[j]>postthreshold)
    }
    else {
	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << CB(2041) << ENDL; // end if (B(dd_gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	}
    }
    os << CB(202) << ENDL;
    os << CB(201) << ENDL;
    }
}


//--------------------------------------------------------------------------
/*!
  \brief Function that generates code that will simulate all synapses of the model on the CPU.

*/
//--------------------------------------------------------------------------

void genSynapseFunction(NNmodel &model, //!< Model description
			string &path, //!< Path for code generation
			ostream &mos //!< output stream for messages
    )
{
    string name, s, localID, theLG, preSpike, preSpikeV, sTpost, sTpre;
    ofstream os;

    cerr << "entering genSynapseFunction" << endl;
    name = path + toString("/") + model.name + toString("_CODE/synapseFnct.cc");
    os.open(name.c_str());
    // write header content
    writeHeader(os);
    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_synapseFnct_cc" << ENDL;
    os << "#define _" << model.name << "_synapseFnct_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file synapseFnct.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL;

    // Function for calculating synapse input to neurons
    // Function header
    os << "void calcSynapsesCPU(" << model.ftype << " t)" << ENDL;
    os << OB(1001);
    os << "unsigned int ipost, npost";
    if (model.needSynapseDelay) {
	os << ", delaySlot";
    }
    os << ";" << ENDL;
    
    os << model.ftype << " addtoinSyn;" << ENDL;  
    os << ENDL;
    for (int i = 0; i < model.synapseGrpN; i++) {
	unsigned int src = model.synapseSource[i];
	unsigned int trg = model.synapseTarget[i];
	    
	os << "//synapse group " << model.synapseName[i] << ENDL;
	unsigned int synt = model.synapseType[i];
	unsigned int inSynNo = model.synapseInSynNo[i];
	if (model.neuronDelaySlots[src] != 1) {
	    os << "delaySlot = (spkEvntQuePtr" << model.neuronName[src] << " + ";
	    os << (int) (model.neuronDelaySlots[src] - model.synapseDelay[i] + 1);
	    os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
	}	

	//Detect spike events and do the update
	if (model.usesSpikeEvents[i] == TRUE) {	
	    generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, localID, inSynNo, tS("Evnt"));
	}	    
        //Detect true spikes and do the update
	if (model.usesTrueSpikes[i] == 1) {
	    generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, localID, inSynNo, tS(""));
	}
    }
    os << CB(1001);

    if (model.lrnGroups > 0) {
	// function for learning synapses, post-synaptic spikes
	// function header
	os << "void learnSynapsesPostHost(" << model.ftype << " t)" << ENDL;
	os << OB(811);
	os << "unsigned int npre;" <<ENDL;
	os << ENDL;

	for (int i = 0; i < model.lrnGroups; i++) {
	  
	  unsigned int k = model.lrnSynGrp[i];
	  unsigned int src = model.synapseSource[k];
	  unsigned int nN = model.neuronN[src];
	  unsigned int trg = model.synapseTarget[k];
	  unsigned int inSynNo = model.synapseInSynNo[k];
	  unsigned int synt = model.synapseType[k];
	  bool sparse= (model.synapseConnType[k] == SPARSE);
// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM
	  os << "for (int j = 0; j < glbSpkCnt" << model.neuronName[trg];
	  if (model.neuronDelaySlots[trg] != 1) {
	      os << "[spkQuePtr" << model.neuronName[trg] << "]" << ENDL;
	  }
	  os << "; j++)" << OB(910);
	  if (sparse) { // SPARSE
	      // TODO: THIS NEEDS CHECKING AND FUNCTIONAL C.POST* ARRAYS
	      os << "npre = C" << model.synapseName[k] << ".revIndInG[glbSpk" << model.neuronName[trg] << "[j] + 1] - C";
	      os << model.synapseName[k] << ".revIndInG[glbSpk" << model.neuronName[trg] << "[j]];" << ENDL;
	      os << "for (int l = 0; l < npre; l++)" << OB(121);
	      os << "unsigned int iprePos = C" << model.synapseName[k];
	      os << ".revIndInG[glbSpk" << model.neuronName[trg] << "[j]] + l;" << ENDL;
	  }
	  else{ // DENSE
	      os << "for (int n = 0; n < " << model.neuronN[src] << "; n++)" << OB(121);
	  }
	  
	  string code = weightUpdateModels[synt].simLearnPost;
	  // Code substitutions ----------------------------------------------------------------------------------
	  if (sparse){ // SPARSE
	      name_substitutions(code, tS(""), weightUpdateModels[synt].varNames, model.synapseName[k] 
				 + tS("[C") + model.synapseName[k] + tS(".remap[iprePos]]"));
	  }
	  else{ // DENSE
	      name_substitutions(code, tS(""), weightUpdateModels[synt].varNames, model.synapseName[k] 
				 + tS("[glbSpk")+ model.neuronName[trg] + tS("[j] + ") + tS(model.neuronN[trg]) + tS("* n]"));
	  }
	  value_substitutions(code, weightUpdateModels[synt].pNames, model.synapsePara[k]);
	  value_substitutions(code, weightUpdateModels[synt].dpNames, model.dsp_w[k]);
	  name_substitutions(code, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[k]);
	  // presynaptic neuron variables and parameters
	  unsigned int nt_pre= model.neuronType[src];
	  if (model.neuronType[src] == POISSONNEURON) substitute(code, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
	  if (sparse) {
	      substitute(code, tS("$(sT_pre)"), tS("sT") + tS(model.neuronName[src])
			 +tS("[C") + model.synapseName[k] + tS(".revInd[iprePos]]"));
	      extended_name_substitutions(code, tS(""), nModels[nt_pre].varNames, tS("_pre"), tS(model.neuronName[src])+tS("[C") 
					  + model.synapseName[k] + tS(".revInd[iprePos]]"));
	  }
	  else {
	      substitute(code, tS("$(sT_pre)"), tS("sT") + tS(model.neuronName[src])
			 +tS("[n]"));
	      extended_name_substitutions(code, tS(""), nModels[nt_pre].varNames, tS("_pre"), tS(model.neuronName[src])+tS("[n]"));
	  }
	  extended_value_substitutions(code, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
	  extended_value_substitutions(code, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);
	  // postsynaptic neuron variables and parameters
	  unsigned int nt_post= model.neuronType[trg];
	  substitute(code, tS("$(sT_post)"), tS("d_sT")+ model.neuronName[trg]+tS("[glbSpk") + model.neuronName[trg]+ tS("[j]]"));
	  extended_name_substitutions(code, tS(""), nModels[nt_post].varNames, tS("_post"), tS(model.neuronName[trg])+tS("[glbSpk") + model.neuronName[trg] + tS("[j]]"));
	  extended_value_substitutions(code, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
	  extended_value_substitutions(code, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
	  // end Code substitutions ------------------------------------------------------------------------- 
	  os << code;
      	  
	  os << CB(121);
	  os << CB(910);
	}
	os << CB(811);
    }
    os << ENDL;
    os << "#endif" << ENDL;
    os.close();
    cerr << "exiting genSynapseFunction" << endl;

}
