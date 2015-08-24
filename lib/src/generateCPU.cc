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
#include <cfloat>

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

    os << "// include the support codes provided by the user for neuron or synaptic models" << ENDL;
    os << "#include \"support_code.h\"" << ENDL << ENDL; 

    // function header
    os << "void calcNeuronsCPU(" << model.ftype << " t)" << ENDL;
    os << OB(51);

    // function code
    for (int i = 0; i < model.neuronGrpN; i++) {
	nt = model.neuronType[i];

	string queueOffset = (model.neuronDelaySlots[i] > 1 ? "(spkQuePtr" + model.neuronName[i] + " * " + tS(model.neuronN[i]) + ") + " : "");
	string queueOffsetTrueSpk = (model.neuronNeedTrueSpk[i] ? queueOffset : "");

	// increment spike queue pointer and reset spike count
	if (model.neuronDelaySlots[i] > 1) { // with delay
	    os << "spkQuePtr" << model.neuronName[i] << " = (spkQuePtr" << model.neuronName[i] << " + 1) % " << model.neuronDelaySlots[i] << ";" << ENDL;
	    if (model.neuronNeedSpkEvnt[i]) {
		os << "glbSpkCntEvnt" << model.neuronName[i] << "[spkQuePtr" << model.neuronName[i] << "] = 0;" << ENDL;
	    }
	    if (model.neuronNeedTrueSpk[i]) {
		os << "glbSpkCnt" << model.neuronName[i] << "[spkQuePtr" << model.neuronName[i] << "] = 0;" << ENDL;
	    }
	    else {
		os << "glbSpkCnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
	    }
	}
	else { // no delay
	    if (model.neuronNeedSpkEvnt[i]) {
		os << "glbSpkCntEvnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
	    }
	    os << "glbSpkCnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
	}
	os << ENDL;

	os << "for (int n = 0; n < " <<  model.neuronN[i] << "; n++)" << OB(10);
	for (int k = 0; k < nModels[nt].varNames.size(); k++) {
	    os << nModels[nt].varTypes[k] << " l" << nModels[nt].varNames[k] << " = ";
	    os << nModels[nt].varNames[k] << model.neuronName[i] << "[";
	    if ((model.neuronVarNeedQueue[i][k]) && (model.neuronDelaySlots[i] > 1)) {
		os << "(((spkQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
		os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
	    }
	    os << "n];" << ENDL;
	}
	if ((nModels[nt].simCode.find(tS("$(sT)")) != string::npos)
	    || (nModels[nt].thresholdConditionCode.find(tS("$(sT)")) != string::npos)
	    || (nModels[nt].resetCode.find(tS("$(sT)")) != string::npos)) { // load sT into local variable
	    os << model.ftype << " lsT= sT" <<  model.neuronName[i] << "[";
	    if (model.neuronDelaySlots[i] > 1) {
		os << "(((dd_spkQuePtr" << model.neuronName[i] << " + " << (model.neuronDelaySlots[i] - 1) << ") % ";
		os << model.neuronDelaySlots[i] << ") * " << model.neuronN[i] << ") + ";
	    }
	    os << "n];" << ENDL;
	}

	if (model.needSynapseDelay) {
	    os << "unsigned int delaySlot;" << ENDL;
	}
	if ((model.inSyn[i].size() > 0) || (nModels[nt].simCode.find(tS("Isyn")) != string::npos)) {
	    os << model.ftype << " Isyn = 0;" << ENDL;
	} 
	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    unsigned int synPopID= model.inSyn[i][j]; // number of (post)synapse group
	    postSynModel psm= postSynModels[model.postSynapseType[synPopID]];
	    string sName= model.synapseName[synPopID];
	    
	    if (model.synapseGType[synPopID] == INDIVIDUALG) {
		for (int k = 0, l = psm.varNames.size(); k < l; k++) {
		    os << psm.varTypes[k] << " lps" << psm.varNames[k] << sName;
		    os << " = " <<  psm.varNames[k] << sName << "[n];" << ENDL;
		    
		}
	    }
	    if (psm.supportCode != tS("")) {
		os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;
	    }
	    os << "Isyn += ";
	    string psCode = psm.postSyntoCurrent;
	    substitute(psCode, tS("$(id)"), tS("n"));
	    substitute(psCode, tS("$(t)"), tS("t"));
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
	    psCode= ensureFtype(psCode, model.ftype);
	    checkUnreplacedVariables(psCode, tS("postSyntoCurrent"));
	    os << psCode << ";" << ENDL;
	    if (psm.supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << ENDL;
	    }
	}
    
    
	os << "// test whether spike condition was fulfilled previously" << ENDL;
	string thCode= nModels[nt].thresholdConditionCode;
	if (thCode == tS("")) { // no condition provided
	    cerr << "Warning: No thresholdConditionCode for neuron type " << model.neuronType[i] << " used for population \"" << model.neuronName[i] << "\" was provided. There will be no spikes detected in this population!" << endl;
	}
	else {
	    substitute(thCode, tS("$(id)"), tS("n"));
	    substitute(thCode, tS("$(t)"), tS("t"));
	    name_substitutions(thCode, tS("l"), nModels[nt].varNames, tS(""));
	    substitute(thCode, tS("$(sT)"), tS("lsT"));
	    value_substitutions(thCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(thCode, nModels[nt].dpNames, model.dnp[i]);
	    substitute(thCode, tS("$(Isyn)"), tS("Isyn"));
	    thCode= ensureFtype(thCode, model.ftype);
	    checkUnreplacedVariables(thCode, tS("thresholdConditionCode"));
	    if (GENN_PREFERENCES::autoRefractory) {
		if (nModels[nt].supportCode != tS("")) {
		    os << OB(29) << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;
		}
		os << "bool oldSpike= (" << thCode << ");" << ENDL;  
		if (nModels[nt].supportCode != tS("")) {
		    os << CB(29) << " // namespace bracket closed" << endl;
		}
	    }
	}

	os << "// calculate membrane potential" << ENDL;
	string sCode = nModels[nt].simCode;
	substitute(sCode, tS("$(id)"), tS("n"));
	substitute(sCode, tS("$(t)"), tS("t"));
	name_substitutions(sCode, tS("l"), nModels[nt].varNames, tS(""));
	value_substitutions(sCode, nModels[nt].pNames, model.neuronPara[i]);
	value_substitutions(sCode, nModels[nt].dpNames, model.dnp[i]);
	name_substitutions(sCode, tS(""), nModels[nt].extraGlobalNeuronKernelParameters, model.neuronName[i]);
	if (nt == POISSONNEURON) {
	    substitute(sCode, tS("lrate"), tS("rates") + model.neuronName[i] + tS("[n + offset") + model.neuronName[i] + tS("]"));
	}
	substitute(sCode, tS("$(Isyn)"), tS("Isyn"));
	substitute(sCode, tS("$(sT)"), tS("lsT"));
	sCode= ensureFtype(sCode, model.ftype);
	checkUnreplacedVariables(sCode,tS("neuron simCode"));
	if (nModels[nt].supportCode != tS("")) {
	    os << OB(29) << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;
	}
	os << sCode << ENDL;
	if (nModels[nt].supportCode != tS("")) {
	    os << CB(29) << " // namespace bracket closed" << endl;
	}

	// look for spike type events first.
	if (model.neuronNeedSpkEvnt[i]) {
	    string eCode= model.neuronSpkEvntCondition[i];
	    // code substitutions ----
	    extended_name_substitutions(eCode, tS("l"), nModels[model.neuronType[i]].varNames, tS("_pre"), tS(""));
	    for (int j= 0; j < model.outSyn[i].size(); j++) {
		unsigned int synPopID= model.outSyn[i][j];
		unsigned int synt= model.synapseType[synPopID];
		substitute(eCode, tS("$(id)"), tS("n"));
		substitute(eCode, tS("$(t)"), tS("t"));
		value_substitutions(eCode, weightUpdateModels[synt].pNames, model.synapsePara[synPopID]);
		value_substitutions(eCode, weightUpdateModels[synt].dpNames, model.dsp_w[synPopID]);
		name_substitutions(eCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[synPopID]);
	    }
	    eCode= ensureFtype(eCode, model.ftype);
	    checkUnreplacedVariables(eCode, tS("neuronSpkEvntCondition"));
	    // end code substitutions ----

	    os << "// test for and register a spike-like event" << ENDL;
	    if (nModels[nt].supportCode != tS("")) {
		os << OB(29) << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;	
	    }
	    os << "if (" + eCode + ")" << OB(30);
	    os << "glbSpkEvnt" << model.neuronName[i] << "[" << queueOffset << "glbSpkCntEvnt" << model.neuronName[i];
	    if (model.neuronDelaySlots[i] > 1) { // WITH DELAY
		os << "[spkQuePtr" << model.neuronName[i] << "]++] = n;" << ENDL;
	    }
	    else { // NO DELAY
		os << "[0]++] = n;" << ENDL;
	    }
	    os << CB(30);
	    if (nModels[nt].supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << endl;
	    }
	}

        // test for true spikes if condition is provided
	if (thCode != tS("")) {
	    os << "// test for and register a true spike" << ENDL;
	    if (nModels[nt].supportCode != tS("")) {
		os << OB(29) << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;	
	    }
	    if (GENN_PREFERENCES::autoRefractory) {
	      os << "if ((" << thCode << ") && !(oldSpike))" << OB(40);
	    }
	    else{
	      os << "if (" << thCode << ") " << OB(40);
	    }
	    os << "glbSpk" << model.neuronName[i] << "[" << queueOffsetTrueSpk << "glbSpkCnt" << model.neuronName[i];
	    if ((model.neuronDelaySlots[i] > 1) && (model.neuronNeedTrueSpk[i])) { // WITH DELAY
		os << "[spkQuePtr" << model.neuronName[i] << "]++] = n;" << ENDL;
	    }
	    else { // NO DELAY
		os << "[0]++] = n;" << ENDL;
	    }
	    if (model.neuronNeedSt[i]) {
		os << "sT" << model.neuronName[i] << "[" << queueOffset << "n] = t;" << ENDL;
	    }

	    // add after-spike reset if provided
	    if (nModels[nt].resetCode != tS("")) {
		string rCode = nModels[nt].resetCode;
		substitute(rCode, tS("$(id)"), tS("n"));
		substitute(rCode, tS("$(t)"), tS("t"));
		name_substitutions(rCode, tS("l"), nModels[nt].varNames, tS(""));
		value_substitutions(rCode, nModels[nt].pNames, model.neuronPara[i]);
		value_substitutions(rCode, nModels[nt].dpNames, model.dnp[i]);
		substitute(rCode, tS("$(Isyn)"), tS("Isyn"));
		substitute(rCode, tS("$(sT)"), tS("lsT"));
		os << "// spike reset code" << ENDL;
		rCode= ensureFtype(rCode, model.ftype);
		checkUnreplacedVariables(rCode, tS("resetCode"));
		os << rCode << ENDL;
	    }
	    os << CB(40);
	    if (nModels[nt].supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << endl;
	    }
	}

	// store the defined parts of the neuron state into the global state variables V etc
	for (int k = 0, l = nModels[nt].varNames.size(); k < l; k++) {
	    if (model.neuronVarNeedQueue[i][k]) {
		os << nModels[nt].varNames[k] << model.neuronName[i] << "[" << queueOffset << "n] = l" << nModels[nt].varNames[k] << ";" << ENDL;
	    }
	    else {
		os << nModels[nt].varNames[k] << model.neuronName[i] << "[n] = l" << nModels[nt].varNames[k] << ";" << ENDL;
	    }
	}

	for (int j = 0; j < model.inSyn[i].size(); j++) {
	    postSynModel psModel= postSynModels[model.postSynapseType[model.inSyn[i][j]]];
	    string sName= model.synapseName[model.inSyn[i][j]];
	    string pdCode = psModel.postSynDecay;
	    substitute(pdCode, tS("$(id)"), tS("n"));
	    substitute(pdCode, tS("$(t)"), tS("t"));
	    substitute(pdCode, tS("$(inSyn)"), tS("inSyn") + sName + tS("[n]"));
	    name_substitutions(pdCode, tS("lps"), psModel.varNames, sName);
	    value_substitutions(pdCode, psModel.pNames, model.postSynapsePara[model.inSyn[i][j]]);
	    value_substitutions(pdCode, psModel.dpNames, model.dpsp[model.inSyn[i][j]]);
	    name_substitutions(pdCode, tS("l"), nModels[nt].varNames, tS(""));
	    value_substitutions(pdCode, nModels[nt].pNames, model.neuronPara[i]);
	    value_substitutions(pdCode, nModels[nt].dpNames, model.dnp[i]);
	    os << "// the post-synaptic dynamics" << ENDL;
	    pdCode= ensureFtype(pdCode, model.ftype);
	    checkUnreplacedVariables(pdCode, tS("postSynDecay"));
	    if (psModel.supportCode != tS("")) {
		os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;	
	    }
	    os << pdCode << ENDL;
	    if (psModel.supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << endl;
	    }
	    for (int k = 0, l = psModel.varNames.size(); k < l; k++) {
		os << psModel.varNames[k] << sName << "[n]" << " = lps" << psModel.varNames[k] << sName << ";" << ENDL;
	    }
	}
	os << CB(10) << ENDL;
	os << ENDL;
    }
    os << CB(51) << ENDL;
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
    bool evnt = postfix == tS("Evnt");

    if ((evnt && model.synapseUsesSpikeEvents[i]) || (!evnt && model.synapseUsesTrueSpikes[i])) {
	unsigned int synt = model.synapseType[i];
	bool sparse = model.synapseConnType[i] == SPARSE;

	unsigned int nt_pre = model.neuronType[src];
	bool delayPre = model.neuronDelaySlots[src] > 1;
	string offsetPre = (delayPre ? "(delaySlot * " + tS(model.neuronN[src]) + ") + " : "");

	unsigned int nt_post = model.neuronType[trg];
	bool delayPost = model.neuronDelaySlots[trg] > 1;
	string offsetPost = (delayPost ? "(spkQuePtr" + model.neuronName[trg] + " * " + tS(model.neuronN[trg]) + ") + " : "");

	// Detect spike events or spikes and do the update
	os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << ENDL;
	if (delayPre) {
	    os << "for (int i = 0; i < glbSpkCnt" << postfix << model.neuronName[src] << "[delaySlot]; i++)" << OB(201);
	}
	else {
	    os << "for (int i = 0; i < glbSpkCnt" << postfix << model.neuronName[src] << "[0]; i++)" << OB(201);
	}

	os << "ipre = glbSpk" << postfix << model.neuronName[src] << "[" << offsetPre << "i];" << ENDL;

	if (sparse) { // SPARSE
	    os << "npost = C" << model.synapseName[i] << ".indInG[ipre + 1] - C" << model.synapseName[i] << ".indInG[ipre];" << ENDL;
	    os << "for (int j = 0; j < npost; j++)" << OB(202);
	    os << "ipost = C" << model.synapseName[i] << ".ind[C" << model.synapseName[i] << ".indInG[ipre] + j];" << ENDL;
	}
	else { // DENSE
	    os << "for (ipost = 0; ipost < " << model.neuronN[trg] << "; ipost++)" << OB(202);
	}

	if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "unsigned int gid = (ipre * " << model.neuronN[i] << " + ipost);" << ENDL;
	}

	if (weightUpdateModels[synt].simCode_supportCode != tS("")) {
	    os << OB(29) << " using namespace " << model.synapseName[i] << "_weightupdate_simCode;" << ENDL;	
	}
	if (evnt) { 
	    // code substitutions ----
	    string eCode = weightUpdateModels[synt].evntThreshold;
	    substitute(eCode, tS("$(id)"), tS("n"));
	    substitute(eCode, tS("$(t)"), tS("t"));
	    extended_name_substitutions(eCode, tS(""), model.synapseSpkEvntVars[i], tS("_pre"), model.neuronName[src] + tS("[") + offsetPre + tS("ipre]"));
	    value_substitutions(eCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
	    value_substitutions(eCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
	    name_substitutions(eCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
	    eCode= ensureFtype(eCode, model.ftype);
	    checkUnreplacedVariables(eCode, tS("evntThreshold"));
	    // end code substitutions ----

	    if (model.synapseGType[i] == INDIVIDUALID) {
		os << "if ((B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1;
		os << ")) && (" << eCode << "))" << OB(2041);
	    }
	    else {
		os << "if (" << eCode << ")" << OB(2041);
	    }
	}
	else if (model.synapseGType[i] == INDIVIDUALID) {
	    os << "if (B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << OB(2041);
	}

	// Code substitutions ----------------------------------------------------------------------------------
	string wCode = (evnt ? weightUpdateModels[synt].simCodeEvnt : weightUpdateModels[synt].simCode);
	substitute(wCode, tS("$(updatelinsyn)"), tS("$(inSyn) += $(addtoinSyn)"));
	substitute(wCode, tS("$(t)"), tS("t"));
	if (sparse) { // SPARSE
	    if (model.synapseGType[i] == INDIVIDUALG) {
		name_substitutions(wCode, tS(""), weightUpdateModels[synt].varNames, model.synapseName[i] + tS("[C") + model.synapseName[i] + tS(".indInG[ipre] + j]"));
	    }
	    else {
		value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
	    }
	}
	else { // DENSE
	    if (model.synapseGType[i] == INDIVIDUALG) {
		name_substitutions(wCode, tS(""), weightUpdateModels[synt].varNames, model.synapseName[i] + tS("[ipre * ") + tS(model.neuronN[trg]) + tS(" + ipost]"));
	    }
	    else {
		value_substitutions(wCode, weightUpdateModels[synt].varNames, model.synapseIni[i]);
	    }      
	}
	substitute(wCode, tS("$(inSyn)"), tS("inSyn") + model.synapseName[i] + tS("[ipost]"));
	value_substitutions(wCode, weightUpdateModels[synt].pNames, model.synapsePara[i]);
	value_substitutions(wCode, weightUpdateModels[synt].dpNames, model.dsp_w[i]);
	name_substitutions(wCode, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[i]);
	substitute(wCode, tS("$(addtoinSyn)"), tS("addtoinSyn"));

	// presynaptic neuron variables and parameters
	if (model.neuronType[src] == POISSONNEURON) substitute(wCode, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
	substitute(wCode, tS("$(sT_pre)"), tS("sT") + model.neuronName[src] + tS("[") + offsetPre + tS("ipre]"));
	for (int j = 0; j < nModels[nt_pre].varNames.size(); j++) {
	    if (model.neuronVarNeedQueue[src][j]) {
		substitute(wCode, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"), nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[") + offsetPre + tS("ipre]"));
	    }
	    else {
		substitute(wCode, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"), nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[ipre]"));
	    }
	}
	extended_value_substitutions(wCode, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
	extended_value_substitutions(wCode, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);

	// postsynaptic neuron variables and parameters
	substitute(wCode, tS("$(sT_post)"), tS("sT") + model.neuronName[trg] + tS("[") + offsetPost + tS("ipost]"));
	for (int j = 0; j < nModels[nt_post].varNames.size(); j++) {
	    if (model.neuronVarNeedQueue[trg][j]) {
		substitute(wCode, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"), nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[") + offsetPost + tS("ipost]"));
	    }
	    else {
		substitute(wCode, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"), nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[ipost]"));
	    }
	}
	extended_value_substitutions(wCode, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
	extended_value_substitutions(wCode, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
	wCode= ensureFtype(wCode, model.ftype);
	checkUnreplacedVariables(wCode, tS("simCode")+postfix);
	// end Code substitutions ------------------------------------------------------------------------- 
	os << wCode << ENDL;

	if (evnt) {
	    os << CB(2041); // end if (eCode)
	}
	else if (model.synapseGType[i] == INDIVIDUALID) {
	    os << CB(2041); // end if (B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid 
	}
	if (weightUpdateModels[synt].simCode_supportCode != tS("")) {
	    os << CB(29) << " // namespace bracket closed" << ENDL;
	}
	os << CB(202);
	os << CB(201);
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
    unsigned int k, src, trg, synt, inSynNo;
    ofstream os;

//    cout << "entering genSynapseFunction" << endl;
    name = path + toString("/") + model.name + toString("_CODE/synapseFnct.cc");
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_synapseFnct_cc" << ENDL;
    os << "#define _" << model.name << "_synapseFnct_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file synapseFnct.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    // synapse dynamics function
    

    os << "void calcSynapseDynamicsCPU(" << model.ftype << " t)" << ENDL;
    os << OB(1000);
    os << "// execute internal synapse dynamics if any" << ENDL;

    if (model.needSynapseDelay) {
	os << "unsigned int delaySlot;" << ENDL;
    }
    for (int i = 0; i < model.synapseGrpN; i++) {
	unsigned int synt= model.synapseType[i];
	string synapseName= model.synapseName[i];
	
	if (weightUpdateModels[synt].synapseDynamics != tS("")) {
	    // there is some internal synapse dynamics
	    weightUpdateModel wu= weightUpdateModels[synt];
	    if (wu.synapseDynamics_supportCode != tS("")) {
		os << OB(29) << " using namespace " << model.synapseName[i] << "_weightupdate_synapseDynamics;" << ENDL;	
	    }
	    string SDcode= wu.synapseDynamics;
	    substitute(SDcode, tS("$(t)"), tS("t"));
	    unsigned int srcno= model.neuronN[model.synapseSource[i]]; 
	    unsigned int trgno= model.neuronN[model.synapseTarget[i]];
	    int src= model.synapseSource[i];
	    int trg= model.synapseTarget[i];
	    int nt_pre= model.neuronType[src];
	    int nt_post= model.neuronType[trg];
	    bool delayPre = model.neuronDelaySlots[src] > 1;
	    bool delayPost = model.neuronDelaySlots[trg] > 1;
	    string offsetPre = (delayPre ? "(delaySlot * " + tS(model.neuronN[src]) + ") + " : "");
	    string offsetPost = (delayPost ? "(dd_spkQuePtr" + model.neuronName[trg] +" * " + tS(model.neuronN[trg]) + ") + " : "");
	    if (model.neuronDelaySlots[src] > 1) {
		os << "delaySlot = (spkQuePtr" << model.neuronName[src];
		os << " + " << tS(model.neuronDelaySlots[src] - model.synapseDelay[i]);
		os << ") % " << tS(model.neuronDelaySlots[src]) << ";" << ENDL;
	    }
	    if (model.synapseConnType[i] == SPARSE) { // SPARSE
		os << "for (int n= 0; n < C" << model.synapseName[i] << ".connN; n++)" << OB(24) << ENDL; 
		if (model.synapseGType[i] == INDIVIDUALG) {
		    // name substitute synapse var names in synapseDynamics code
		    name_substitutions(SDcode, tS(""), wu.varNames, synapseName + tS("[n]"));
		}
		else {
		    // substitute initial values as constants for synapse var names in synapseDynamics code
		    value_substitutions(SDcode, wu.varNames, model.synapseIni[i]);
		}
		// substitute pre-synaptic variable names in synapseDynamics code
		// !!! this functionality needs testing !!!    
		string theOffset;
		for (int k= 0, l= nModels[nt_pre].varNames.size(); k < l; k++) {
		    if (model.neuronVarNeedQueue[src][k]) {
			theOffset= offsetPre;
		    }
		    else {
			theOffset= tS("");
		    }
		    substitute(SDcode, tS("$(") + nModels[nt_pre].varNames[k] + tS("_pre)"), tS("")+nModels[nt_pre].varNames[k]+ model.neuronName[src]+tS("[")+theOffset+tS("C")+ synapseName+ tS(".preInd[n]]"));
		}
		for (int k= 0, l= nModels[nt_post].varNames.size(); k < l; k++) {
		    // substitute post-synaptic variable names in synapseDynamics code
		    if (model.neuronVarNeedQueue[trg][k]) {
			theOffset= offsetPost;
		    }
		    else {
			theOffset= tS("");
		    }
		    substitute(SDcode, tS("$(") + nModels[nt_post].varNames[k] + tS("_post)"), tS("")+nModels[nt_post].varNames[k]+ model.neuronName[trg]+tS("[")+theOffset+tS("C")+synapseName+tS(".ind[n]]"));
		}
		// !!! end of new mechanism !!!
		// substitute parameter values for parameters in synapseDynamics code
		value_substitutions(SDcode, wu.pNames, model.synapsePara[i]);
		// substitute values for derived parameters in synapseDynamics code
		value_substitutions(SDcode, wu.dpNames, model.dsp_w[i]);
		SDcode= ensureFtype(SDcode, model.ftype);
		checkUnreplacedVariables(SDcode, tS("synapseDynamics"));
		os << SDcode << ENDL;
		os << CB(24);
	    }
	    else { // DENSE
		os << "for (int i = 0; i < " <<  srcno << "; i++)" << OB(25);
		os << "for (int j = 0; j < " <<  trgno << "; j++)" << OB(26);
		os << "// loop through all synapses" << endl;
		// substitute initial values as constants for synapse var names in synapseDynamics code
		if (model.synapseGType[i] == INDIVIDUALG) {
		    name_substitutions(SDcode, tS(""), wu.varNames, synapseName + tS("[i*") + tS(trgno) + tS("+j]"));
		}
		else {
		    // substitute initial values as constants for synapse var names in synapseDynamics code
		    value_substitutions(SDcode, wu.varNames, model.synapseIni[i]);
		}
		// substitute pre-synaptic variable names in synapseDynamics code
		// !!! new - needs testing
		string theOffset;
		for (int k= 0, l= nModels[nt_pre].varNames.size(); k < l; k++) {
		    if (model.neuronVarNeedQueue[src][k]) {
			theOffset= offsetPre;
		    }
		    else {
			theOffset= tS("");
		    }
		    substitute(SDcode, tS("$(") + nModels[nt_pre].varNames[k] + tS("_pre)"), nModels[nt_pre].varNames[k]+model.neuronName[src]+tS("[")+theOffset+tS("i]"));
		}
		for (int k= 0, l= nModels[nt_post].varNames.size(); k < l; k++) {
		    if (model.neuronVarNeedQueue[trg][k]) {
			theOffset= offsetPost;
		    }
		    else {
			theOffset= tS("");
		    }
		    substitute(SDcode, tS("$(") + nModels[nt_post].varNames[k] + tS("_post)"), nModels[nt_post].varNames[k]+model.neuronName[trg]+tS("[")+theOffset+tS("j]"));
		}
		// !!! end of new mechanism !!!
                // substitute parameter values for parameters in synapseDynamics code
		value_substitutions(SDcode, wu.pNames, model.synapsePara[i]);
		// substitute values for derived parameters in synapseDynamics code
		value_substitutions(SDcode, wu.dpNames, model.dsp_w[i]);
		SDcode= ensureFtype(SDcode, model.ftype);
		checkUnreplacedVariables(SDcode, tS("synapseDynamics"));
		os << SDcode << ENDL;
		os << CB(26);
		os << CB(25);
	    }
	    if (weightUpdateModels[synt].synapseDynamics_supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << ENDL;
	    }
	}
    }
    os << CB(1000);

    // synapse function header
    os << "void calcSynapsesCPU(" << model.ftype << " t)" << ENDL;

    // synapse function code
    os << OB(1001);

    os << "unsigned int ipost;" << ENDL;
    os << "unsigned int ipre;" << ENDL;
    for (int i = 0; i < model.synapseGrpN; i++) {  
	if (model.synapseConnType[i] == SPARSE) {
	    os << "unsigned int npost;" << ENDL;
	    break;
	}
    }
    if (model.needSynapseDelay) {
	os << "unsigned int delaySlot;" << ENDL;
    }
    os << model.ftype << " addtoinSyn;" << ENDL;  
    os << ENDL;

    for (int i = 0; i < model.synapseGrpN; i++) {
	src = model.synapseSource[i];
	trg = model.synapseTarget[i];
	synt = model.synapseType[i];
	inSynNo = model.synapseInSynNo[i];

	os << "// synapse group " << model.synapseName[i] << ENDL;

	if (model.neuronDelaySlots[src] > 1) {
	    os << "delaySlot = (spkQuePtr" << model.neuronName[src];
	    os << " + " << tS(model.neuronDelaySlots[src] - model.synapseDelay[i]);
	    os << ") % " << tS(model.neuronDelaySlots[src]) << ";" << ENDL;
	}

	// generate the code for processing spike-like events
	if (model.synapseUsesSpikeEvents[i]) {
	    generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, localID, inSynNo, tS("Evnt"));
	}

        // generate the code for processing true spike events
	if (model.synapseUsesTrueSpikes[i]) {
	    generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, localID, inSynNo, tS(""));
	}

	os << ENDL;
    }
    os << CB(1001);
    os << ENDL;


    //////////////////////////////////////////////////////////////
    // function for learning synapses, post-synaptic spikes

    if (model.lrnGroups > 0) {

	os << "void learnSynapsesPostHost(" << model.ftype << " t)" << ENDL;
	os << OB(811);

	os << "unsigned int ipost;" << ENDL;
	os << "unsigned int ipre;" << ENDL;
	os << "unsigned int lSpk;" << ENDL;
	for (int i = 0; i < model.synapseGrpN; i++) {  
	    if (model.synapseConnType[i] == SPARSE) {
		os << "unsigned int npre;" << ENDL;
		break;
	    }
	}
	if (model.needSynapseDelay) {
	    os << "unsigned int delaySlot;" << ENDL;
	}
	os << ENDL;

	for (int i = 0; i < model.lrnGroups; i++) {
	    k = model.lrnSynGrp[i];
	    src = model.synapseSource[k];
	    trg = model.synapseTarget[k];
	    synt = model.synapseType[k];
	    inSynNo = model.synapseInSynNo[k];
	    unsigned int nN = model.neuronN[src];
	    bool sparse = model.synapseConnType[k] == SPARSE;

	    unsigned int nt_pre = model.neuronType[src];
	    bool delayPre = model.neuronDelaySlots[src] > 1;
	    string offsetPre = (delayPre ? "(delaySlot * " + tS(model.neuronN[src]) + ") + " : "");
	    string offsetTrueSpkPre = (model.neuronNeedTrueSpk[src] ? offsetPre : "");

	    unsigned int nt_post = model.neuronType[trg];
	    bool delayPost = model.neuronDelaySlots[trg] > 1;
	    string offsetPost = (delayPost ? "(spkQuePtr" + model.neuronName[trg] + " * " + tS(model.neuronN[trg]) + ") + " : "");
	    string offsetTrueSpkPost = (model.neuronNeedTrueSpk[trg] ? offsetPost : "");

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM

	    os << "// synapse group " << model.synapseName[k] << ENDL;
	    if (weightUpdateModels[synt].simLearnPost_supportCode != tS("")) {
		os << OB(29) << " using namespace " << model.synapseName[k] << "_weightupdate_simLearnPost;" << ENDL;	
	    }

	    if (delayPre) {
		os << "delaySlot = (spkQuePtr" << model.neuronName[src];
		os << " + " << tS(model.neuronDelaySlots[src] - model.synapseDelay[k]);
		os << ") % " << tS(model.neuronDelaySlots[src]) << ";" << ENDL;
	    }

	    if (delayPost && model.neuronNeedTrueSpk[trg]) {
		os << "for (ipost = 0; ipost < glbSpkCnt" << model.neuronName[trg] << "[spkQuePtr" << model.neuronName[trg] << "]; ipost++)" << OB(910);
	    }
	    else {
		os << "for (ipost = 0; ipost < glbSpkCnt" << model.neuronName[trg] << "[0]; ipost++)" << OB(910);
	    }

	    os << "lSpk = glbSpk" << model.neuronName[trg] << "[" << offsetTrueSpkPost << "ipost];" << ENDL;

	    if (sparse) { // SPARSE
	        // TODO: THIS NEEDS CHECKING AND FUNCTIONAL C.POST* ARRAYS
	        os << "npre = C" << model.synapseName[k] << ".revIndInG[lSpk + 1] - C" << model.synapseName[k] << ".revIndInG[lSpk];" << ENDL;
		os << "for (int l = 0; l < npre; l++)" << OB(121);
		os << "ipre = C" << model.synapseName[k] << ".revIndInG[lSpk] + l;" << ENDL;
	    }
	    else { // DENSE
		os << "for (ipre = 0; ipre < " << model.neuronN[src] << "; ipre++)" << OB(121);
	    }

	    string code = weightUpdateModels[synt].simLearnPost;
	    substitute(code, tS("$(t)"), tS("t"));
	    // Code substitutions ----------------------------------------------------------------------------------
	    if (sparse) { // SPARSE
		name_substitutions(code, tS(""), weightUpdateModels[synt].varNames, model.synapseName[k] + tS("[C") + model.synapseName[k] + tS(".remap[ipre]]"));
	    }
	    else { // DENSE
		name_substitutions(code, tS(""), weightUpdateModels[synt].varNames, model.synapseName[k] + tS("[lSpk + ") + tS(model.neuronN[trg]) + tS(" * ipre]"));
	    }
	    value_substitutions(code, weightUpdateModels[synt].pNames, model.synapsePara[k]);
	    value_substitutions(code, weightUpdateModels[synt].dpNames, model.dsp_w[k]);
	    name_substitutions(code, tS(""), weightUpdateModels[synt].extraGlobalSynapseKernelParameters, model.synapseName[k]);

	    // presynaptic neuron variables and parameters
	    if (model.neuronType[src] == POISSONNEURON) substitute(code, tS("$(V_pre)"), tS(model.neuronPara[src][2]));
	    if (sparse) { // SPARSE
		substitute(code, tS("$(sT_pre)"), tS("sT") + model.neuronName[src] + tS("[") + offsetPre + tS("C") + model.synapseName[k] + tS(".revInd[ipre]]"));
		for (int j = 0; j < nModels[nt_pre].varNames.size(); j++) {
		    if (model.neuronVarNeedQueue[src][j]) {
			substitute(code, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
				   nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[") + offsetPre + tS("C") + model.synapseName[k] + tS(".revInd[ipre]]"));
		    }
		    else {
			substitute(code, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
				   nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[C") + model.synapseName[k] + tS(".revInd[ipre]]"));
		    }
		}
	    }
	    else { // DENSE
		substitute(code, tS("$(sT_pre)"), tS("sT") + model.neuronName[src] + tS("[") + offsetPre + tS("ipre]"));
		for (int j = 0; j < nModels[nt_pre].varNames.size(); j++) {
		    if (model.neuronVarNeedQueue[src][j]) {
			substitute(code, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
				   nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[") + offsetPre + tS("ipre]"));
		    }
		    else {
			substitute(code, tS("$(") + nModels[nt_pre].varNames[j] + tS("_pre)"),
				   nModels[nt_pre].varNames[j] + model.neuronName[src] + tS("[ipre]"));
		    }
		}
	    }
	    extended_value_substitutions(code, nModels[nt_pre].pNames, tS("_pre"), model.neuronPara[src]);
	    extended_value_substitutions(code, nModels[nt_pre].dpNames, tS("_pre"), model.dnp[src]);

	    // postsynaptic neuron variables and parameters
	    substitute(code, tS("$(sT_post)"), tS("sT") + model.neuronName[trg] + tS("[") + offsetPost + tS("lSpk]"));
	    for (int j = 0; j < nModels[nt_post].varNames.size(); j++) {
		if (model.neuronVarNeedQueue[trg][j]) {
		    substitute(code, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"),
			       nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[") + offsetPost + tS("lSpk]"));
		}
		else {
		    substitute(code, tS("$(") + nModels[nt_post].varNames[j] + tS("_post)"),
			       nModels[nt_post].varNames[j] + model.neuronName[trg] + tS("[lSpk]"));
		}
	    }
	    extended_value_substitutions(code, nModels[nt_post].pNames, tS("_post"), model.neuronPara[trg]);
	    extended_value_substitutions(code, nModels[nt_post].dpNames, tS("_post"), model.dnp[trg]);
	    code= ensureFtype(code, model.ftype);
	    checkUnreplacedVariables(code, tS("simLearnPost"));
	    // end Code substitutions ------------------------------------------------------------------------- 
	    os << code << ENDL;

	    os << CB(121);
	    os << CB(910);
	    if (weightUpdateModels[synt].simLearnPost_supportCode != tS("")) {
		os << CB(29) << " // namespace bracket closed" << ENDL;
	    }
	}
	os << CB(811);
    }
    os << ENDL;


    os << "#endif" << ENDL;
    os.close();

//  cout << "exiting genSynapseFunction" << endl;
}
