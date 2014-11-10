/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef _MODELSPEC_CC_
#define _MODELSPEC_CC_ //!< macro for avoiding multiple inclusion during compilation

#include "utils.h"

// ------------------------------------------------------------------------
// Code for GeNN initialisation
    
void initGeNN()
{
    prepareStandardModels();
    preparePostSynModels();
    prepareWeightUpdateModels();
    GeNNReady= 1;
}

// class NNmodel for specifying a neuronal network model

NNmodel::NNmodel() 
{
  valid= 0;
  neuronGrpN= 0;
  synapseGrpN= 0;
  lrnGroups= 0;
  needSt= 0;
  needSynapseDelay = 0;
  setPrecision(0);
  RNtype= tS("uint64_t");
  setGPUDevice(AUTODEVICE);
  setSeed(0);
  needSpkEvnt= FALSE;
}

NNmodel::~NNmodel() 
{
}

void NNmodel::setName(const string inname)
{
  name= toString(inname);
}


//--------------------------------------------------------------------------
/*! \brief Method for calculating dependent parameter values from independent parameters.

This method is to be invoked when all independent parameters have been set.
It should also should only be called once and right after a population has been added. The method appends the derived values of dependent parameters to the corresponding vector (dnp) without checking for multiple calls. If called repeatedly, multiple copies of dependent parameters would be added leading to potential errors in the model execution.

This method also saves the neuron numbers of the populations rounded to the next multiple of the block size as well as the sums s(i) = sum_{1...i} n_i of the rounded population sizes. These are later used to determine the branching structure for the generated neuron kernel code. 
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedNeuronPara(unsigned int i /**< index of the neuron population */)
{
    vector<float> tmpP;
    int numDpNames = nModels[neuronType[i]].dpNames.size();
    for (int j=0; j < nModels[neuronType[i]].dpNames.size(); ++j) {	
	float retVal = nModels[neuronType[i]].dps->calculateDerivedParameter(j, neuronPara[i], DT);
	tmpP.push_back(retVal);
    }
    dnp.push_back(tmpP);
}


void NNmodel::initNeuronSpecs(unsigned int i /**< index of the neuron population */)
{
    // padnN is the lowest multiple of neuronBlkSz >= neuronN[i]
    unsigned int padnN = ceil((float) neuronN[i] / (float) neuronBlkSz) * (float) neuronBlkSz;
    if (i == 0) {
	sumNeuronN.push_back(neuronN[i]);
	padSumNeuronN.push_back(padnN);
    }
    else {
	sumNeuronN.push_back(sumNeuronN[i - 1] + neuronN[i]); 
	padSumNeuronN.push_back(padSumNeuronN[i - 1] + padnN); 
    }
}

//--------------------------------------------------------------------------
/*! \brief This function calculates derived synapse parameters from primary synapse parameters. 

This function needs to be invoked each time a synapse population is added, after all primary parameters have been set, and before code for synapse evaluation is generated. It should be invoked only once per population and in order population by population.
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedSynapsePara(unsigned int i /**< index of the synapse population */)
{
    vector<float> tmpP;
    unsigned int synt= synapseType[i];
    for (int j= 0; j < weightUpdateModels[synt].dpNames.size(); ++j) {
	float retVal = weightUpdateModels[synt].dps->calculateDerivedParameter(j, synapsePara[i], DT);
	cerr << j << " " << retVal << endl;
	tmpP.push_back(retVal);
    }
    assert(dsp_w.size() == i);
    dsp_w.push_back(tmpP);	
}

//--------------------------------------------------------------------------
/*! \brief This function calculates the derived synaptic parameters in the employed post-synaptic model  based on the given underlying post-synapse parameters */
//--------------------------------------------------------------------------

void NNmodel::initDerivedPostSynapsePara(unsigned int i)
{
    vector<float> tmpP;
    unsigned int psynt= postSynapseType[i];
    for (int j=0; j < postSynModels[psynt].dpNames.size(); ++j) {
	float retVal = postSynModels[psynt].dps->calculateDerivedParameter(j, postSynapsePara[i], DT);
	tmpP.push_back(retVal);
    }	
    assert(dpsp.size() == i);
    dpsp.push_back(tmpP);
}

//--------------------------------------------------------------------------
/*! \brief This function generates the necessary entries so that a synapse population is known to source and target neuron groups.

This function needs to be invoked each time a synapse population is added, after all primary parameters have been set, and before code for synapse evaluation is generated. It should be invoked only once per population.
*/
//--------------------------------------------------------------------------

void NNmodel::registerSynapsePopulation(unsigned int i /**< index of the synapse population */)
{
    // figure out at what threshold we need to detect spiking events
    synapseInSynNo.push_back(inSyn[synapseTarget[i]].size());
    inSyn[synapseTarget[i]].push_back(i);
    synapseOutSynNo.push_back(outSyn[synapseSource[i]].size());
    outSyn[synapseSource[i]].push_back(i);

    // padnN is the lowest multiple of synapseBlkSz >= neuronN[synapseTarget[i]]
    // TODO: are these sums and padded sums used anywhere at all???
    unsigned int padnN = ceil((float) neuronN[synapseTarget[i]] / (float) synapseBlkSz) * (float) synapseBlkSz;
    if (i == 0) {
	sumSynapseTrgN.push_back(neuronN[synapseTarget[i]]);
	padSumSynapseTrgN.push_back(padnN);
	padSumSynapseKrnl.push_back(padnN);
    }
    else {
	sumSynapseTrgN.push_back(sumSynapseTrgN[i - 1] + neuronN[synapseTarget[i]]);
	padSumSynapseTrgN.push_back(padSumSynapseTrgN[i - 1] + padnN);
	padSumSynapseKrnl.push_back(padSumSynapseKrnl[i - 1] + padnN);
    }
    //fprintf(stderr, " sum of padded postsynaptic neurons for group %u is %u, krnl size is %u\n", i, padSumSynapseTrgN[i],padSumSynapseKrnl[i]);
}


//--------------------------------------------------------------------------
/*! \brief This function is a tool to find the numeric ID of a neuron population based on the name of the neuron population.
 */
//--------------------------------------------------------------------------

unsigned int NNmodel::findNeuronGrp(const string nName /**< Name of the neuron population */)
{
  for (int j= 0; j < neuronGrpN; j++) {
      if (nName == neuronName[j]) {
	  return j;
      }
  }
  fprintf(stderr, "neuron group %s not found, aborting ... \n", nName.c_str());
  exit(1);
}


//--------------------------------------------------------------------------
/*! \brief This function is for setting which host and which device a neuron group will be simulated on
 */
//--------------------------------------------------------------------------

void NNmodel::setNeuronClusterIndex(const string neuronGroup, /**< Name of the neuron population */
                                    int hostID, /**< ID of the host */
                                    int deviceID /**< ID of the device */)
{
    int groupNo = findNeuronGrp(neuronGroup);
    neuronHostID[groupNo] = hostID;
    neuronDeviceID[groupNo] = deviceID;
}

void NNmodel::initLearnGrps()
{
    usesTrueSpikes.resize(synapseGrpN);
    usesSpikeEvents.resize(synapseGrpN);
    usesPostLearning.resize(synapseGrpN);
    for (int i=0; i< synapseGrpN; i++){
	unsigned int padnN = ceil((float) neuronN[synapseSource[i]] / (float) learnBlkSz) * (float) learnBlkSz;
	weightUpdateModel wu= weightUpdateModels[synapseType[i]];
	if (wu.simCode != tS("")) {
	    usesTrueSpikes[i]= TRUE;
	}
	if (wu.simCodeEvnt != tS("")) {
	    assert(wu.evntThreshold != tS(""));
	    cerr << synapseName[i] << " uses events." << endl;
	    usesSpikeEvents[i]= TRUE;
            // find the necessary pre-synaptic variables contained in Threshold condition
	    vector<string> vars= nModels[neuronType[synapseSource[i]]].varNames;
	    for (int j= 0; j < vars.size(); j++) {
		size_t found= wu.evntThreshold.find(vars[j]+tS("_pre"));
		if (found != string::npos) {
		    synapseSpkEvntVars[i].push_back(vars[j]);
		    cerr << "synapsepop: " << i << ", neuronGrpNo: " << synapseSource[i] << ", added variable: " << vars[j] << endl;
		}
	    }
	    // add to the source population spike event condition
	    unsigned int src= synapseSource[i];
	    if (neuronNeedSpkEvnt[src]) {
		neuronSpkEvntCondition[src]+= tS(" || (")+wu.evntThreshold+tS(")");
	    }
	    else {
		neuronNeedSpkEvnt[src]= TRUE;
		needSpkEvnt= TRUE;
		neuronSpkEvntCondition[src]= tS("(")+wu.evntThreshold+tS(")");
	    }
	}
	if (weightUpdateModels[synapseType[i]].simLearnPost != tS("")){
	    usesPostLearning[i]=TRUE;
	    fprintf(stdout, "detected learning synapse at %d \n", i);
	    if (lrnGroups == 0) {
		padSumLearnN.push_back(padnN);
	    }
	    else {
		padSumLearnN.push_back(padSumLearnN[i - 1] + padnN); 
	    }
	    lrnSynGrp.push_back(i);
	    lrnGroups++;
	}
	else {
	    usesPostLearning[i]= FALSE;
	}
    }
}

//--------------------------------------------------------------------------
/*! \brief This function is a tool to find the numeric ID of a synapse population based on the name of the synapse population.
 */
//--------------------------------------------------------------------------

unsigned int NNmodel::findSynapseGrp(const string sName /**< Name of the synapse population */)
{
    for (int j= 0; j < synapseGrpN; j++) {
	if (sName == synapseName[j]) {
	    return j;
	}
    }
    fprintf(stderr, "synapse group %s not found, aborting ...\n", sName.c_str());
    exit(1);
}


//--------------------------------------------------------------------------
/*! \brief This function is for setting which host and which device a synapse group will be simulated on
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseClusterIndex(const string synapseGroup, /**< Name of the synapse population */
                                     int hostID, /**< ID of the host */
                                     int deviceID /**< ID of the device */)
{
    int groupNo = findSynapseGrp(synapseGroup);
    synapseHostID[groupNo] = hostID;
    synapseDeviceID[groupNo] = deviceID;  
}


//--------------------------------------------------------------------------
/*! \overload This function is an alternative method to the standard addNeuronPopulation that allows the use of constant character arrays instead of C++ strings
 */
//--------------------------------------------------------------------------

void NNmodel::addNeuronPopulation(const char *name, /**< Name of the neuron population */
                                  unsigned int nNo, /**< Number of neurons in the population  */
                                  unsigned int type, /**< Type of the neurons, refers to either a standard type or user-defined type */
                                  float *p, /**< Parameters of this neuron type */
                                  float *ini /**< Initial values for variables of this neuron type */)
{
    addNeuronPopulation(toString(name), nNo, type, p, ini);
}


//--------------------------------------------------------------------------
/*! \brief This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values.
 */
//--------------------------------------------------------------------------

void NNmodel::addNeuronPopulation(const string name, /**<  The name of the neuron population*/
                                  unsigned int nNo, /**<  Number of neurons in the population */
                                  unsigned int type, /**<  Type of the neurons, refers to either a standard type or user-defined type*/
                                  float *p, /**< Parameters of this neuron type */
                                  float *ini /**< Initial values for variables of this neuron type */)
{
    if (!GeNNReady) {
	cerr << "You need to call initGeNN first." << endl;
	exit(1);
    }
    
    unsigned int i= neuronGrpN++;
    
    neuronName.push_back(toString(name));
    neuronN.push_back(nNo);
    neuronType.push_back(type);
    vector<float> tmpP;
    for (int j= 0; j < nModels[neuronType[i]].pNames.size(); j++) {
    tmpP.push_back(p[j]);
    }
    neuronPara.push_back(tmpP);
    tmpP.clear();
    for (int j= 0; j < nModels[neuronType[i]].varNames.size(); j++) {
	tmpP.push_back(ini[j]);
    }
    neuronIni.push_back(tmpP);
    vector<unsigned int> tv;
    inSyn.push_back(tv);  // empty list of input synapse groups for neurons i 
    outSyn.push_back(tv);  // empty list of input synapse groups for neurons i 
    receivesInputCurrent.push_back(0);
    neuronNeedSt.push_back(FALSE);
    neuronNeedSpkEvnt.push_back(FALSE);
    string tmp= tS("");
    neuronSpkEvntCondition.push_back(tmp);
    neuronDelaySlots.push_back(1);
    initDerivedNeuronPara(i);
    initNeuronSpecs(i);
    
    // initially set neuron group indexing variables to device 0 host 0
    neuronDeviceID.push_back(0);
    neuronHostID.push_back(0);
}


//--------------------------------------------------------------------------
/*! \brief This function defines the type of the explicit input to the neuron model. Current options are common constant input to all neurons, input  from a file and input defines as a rule.
*/ 
//--------------------------------------------------------------------------
void NNmodel::activateDirectInput(const string name, /**< Name of the neuron population */
                                  unsigned int type /**< Type of input: 1 if common input, 2 if custom input from file, 3 if custom input as a rule*/)
{
    unsigned int i= findNeuronGrp(name);
    receivesInputCurrent[i]= type;	// (TODO) 4 if random input with Gaussian distribution.
}

//--------------------------------------------------------------------------
/*! \overload This deprecated function is provided for compatibility with the previous release of GeNN.
 * Default values are provide for new parameters, it is strongly recommended these be selected explicity via the new version othe function
 */
//--------------------------------------------------------------------------
void NNmodel::addSynapsePopulation(const string name, /**<  The name of the synapse population*/
                                   unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
                                   unsigned int conntype, /**< The type of synaptic connectivity*/ 
                                   unsigned int gtype, /**< The way how the synaptic conductivity g will be defined*/
                                   const string src, /**< Name of the (existing!) pre-synaptic neuron population*/
                                   const string target, /**< Name of the (existing!) post-synaptic neuron population*/
                                   float *params/**< A C-type array of floats that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/)
{
    fprintf(stderr,"WARNING. Use of deprecated version of fn. addSynapsePopulation(). Some parameters have been supplied with default-only values\n");

    float *postSynV = NULL;
    
    //Tries to borrow these values from the first set of synapse parameters supplied
    float postExpSynapsePopn[2] = {
	params[2], 	//tau_S: decay time constant [ms]
	params[0]	// Erev: Reversal potential
    };
    addSynapsePopulation(name, syntype, conntype, gtype, NO_DELAY, EXPDECAY, src, target, params, postSynV, postExpSynapsePopn);
}

//--------------------------------------------------------------------------
/*! \overload This function is an alternative method to the standard addSynapsePopulation that allows the use of constant character arrays instead of C++ strings.
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(const char *name, /**<  The name of the synapse population*/
                                   unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
                                   unsigned int conntype, /**< The type of synaptic connectivity*/
                                   unsigned int gtype, /**< The way how the synaptic conductivity g will be defined*/
                                   unsigned int delaySteps, /**< Number of delay slots*/
                                   unsigned int postsyn, /**< Postsynaptic integration method*/
                                   const char *src, /**< Name of the (existing!) pre-synaptic neuron population*/
                                   const char *trg, /**< Name of the (existing!) post-synaptic neuron population*/
                                   float *p, /**< A C-type array of floats that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float * PSVini, /**< A C-type array of floats that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float *ps/**< A C-type array of floats that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/) 
{
  addSynapsePopulation(toString(name), syntype, conntype, gtype, delaySteps, postsyn, toString(src), toString(trg), p, PSVini, ps);
}


//--------------------------------------------------------------------------
/*! \brief Overloaded old version 
*/
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(const string name, /**<  The name of the synapse population*/
                                   unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
                                   unsigned int conntype, /**< The type of synaptic connectivity*/
                                   unsigned int gtype, /**< The way how the synaptic conductivity g will be defined*/
                                   unsigned int delaySteps, /**< Number of delay slots*/
                                   unsigned int postsyn, /**< Postsynaptic integration method*/
                                   const string src, /**< Name of the (existing!) pre-synaptic neuron population*/
                                   const string trg, /**< Name of the (existing!) post-synaptic neuron population*/
                                   float *p, /**< A C-type array of floats that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float* PSVini, /**< A C-type array of floats that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float *ps /**< A C-type array of floats that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/ )
{
    cerr << "!!!!!!GeNN WARNING: You use the overloaded method which passes a null pointer for the initial values of weight update variables. If you use a method that uses synapse variables, please add a pointer to this vector in the function call, like:\n  	addSynapsePopulation(name, syntype, conntype, gtype, NO_DELAY, EXPDECAY, src, target, float * SYNVARINI, params, postSynV,postExpSynapsePopn);" << endl;
    float *iniv = NULL;
    addSynapsePopulation(name, syntype, conntype, gtype, delaySteps, postsyn, src, trg, iniv, p, PSVini, ps);
}

//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(const string name, /**<  The name of the synapse population*/
                                   unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
                                   unsigned int conntype, /**< The type of synaptic connectivity*/
                                   unsigned int gtype, /**< The way how the synaptic conductivity g will be defined*/
                                   unsigned int delaySteps, /**< Number of delay slots*/
                                   unsigned int postsyn, /**< Postsynaptic integration method*/
                                   const string src, /**< Name of the (existing!) pre-synaptic neuron population*/
                                   const string trg, /**< Name of the (existing!) post-synaptic neuron population*/
                                   float* synini, /**< A C-type array of floats that contains the initial values for synapse variables (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float *p, /**< A C-type array of floats that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float* PSVini, /**< A C-type array of floats that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float *ps /**< A C-type array of floats that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/ )
{
    if (!GeNNReady) {
	cerr << "You need to call initGeNN first." << endl;
	exit(1);
    }
    unsigned int i= synapseGrpN++;
    unsigned int srcNumber, trgNumber;
    vector<float> tmpP;
    vector<float> tmpV;
    vector<float> tmpPS;
    vector<float> tmpPV;
    vector<float> tmpDsp;
    preparePostSynModels();
       
    synapseName.push_back(name);
    synapseType.push_back(syntype);
    synapseConnType.push_back(conntype);
    synapseGType.push_back(gtype);
    srcNumber = findNeuronGrp(src);
    synapseSource.push_back(srcNumber);
    trgNumber = findNeuronGrp(trg);
    synapseTarget.push_back(trgNumber);
    synapseDelay.push_back(delaySteps);
    if (delaySteps > neuronDelaySlots[srcNumber]) {
	neuronDelaySlots[srcNumber] = delaySteps;
	needSynapseDelay = 1;
    }
    if (weightUpdateModels[syntype].needPreSt) {
	neuronNeedSt[srcNumber]= TRUE;
	needSt= TRUE;
    }
    if (weightUpdateModels[syntype].needPostSt) {
	neuronNeedSt[trgNumber]= TRUE;
	needSt= TRUE;
    }

    //TODO: We want to get rid of SYNPNO array for code generation flexibility. It would be useful to predefine synapse models as we do for neurons in utils.h. This would also help for checkSizes.
    
    // for (int j= 0; j < nModels[synapseType[i]].pNames.size(); j++) { 
    int maxp;
    maxp= weightUpdateModels[syntype].pNames.size(); 
    for (int j= 0; j < maxp; j++) {
	tmpP.push_back(p[j]);
    }
    synapsePara.push_back(tmpP);
    
    for (int j= 0; j < weightUpdateModels[syntype].varNames.size(); j++) {
	tmpV.push_back(synini[j]);
    }
    synapseIni.push_back(tmpV);
      
    postSynapseType.push_back(postsyn);
    for (int j= 0; j <  postSynModels[postSynapseType[i]].pNames.size(); j++) {
	tmpPS.push_back(ps[j]);
    }
  
    postSynapsePara.push_back(tmpPS);  
    tmpPV.clear();
    for (int j= 0; j < postSynModels[postSynapseType[i]].varNames.size(); j++) {
	tmpPV.push_back(PSVini[j]);
  }
    
    postSynIni.push_back(tmpPV);  
    
    vector<string> tmpS;
    synapseSpkEvntVars.push_back(tmpS); 

    registerSynapsePopulation(i);
    initDerivedSynapsePara(i);
    initDerivedPostSynapsePara(i);

    // initially set synapase group indexing variables to device 0 host 0
    synapseDeviceID.push_back(0);
    synapseHostID.push_back(0);

// TODO set uses*** variables for synaptic populations  
}

//--------------------------------------------------------------------------
/*! \brief This functions sets the global value of the maximal synaptic conductance for a synapse population that was idfentified as conductance specifcation method "GLOBALG" 
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseG(const string sName, /**<  */
                          float g /**<  */)
{
  cerr << "NOTE: This function has been deprecated. Please provide the correct initial values in \"addSynapsePopulation\" for all your variables and they will be the constant values in the GLOBALG mode - global \"G\" not set." << endl; 
  //  unsigned int found= findSynapseGrp(sName);
  //  if (g0.size() < found+1) g0.resize(found+1);
  //  g0[found]= g;
}


//--------------------------------------------------------------------------
/*! \brief This function sets a global input value to the specified neuron group.
 */
//--------------------------------------------------------------------------

void NNmodel::setConstInp(const string sName, /**<  */
                          float globalInp0 /**<  */)
{
  unsigned int found= findNeuronGrp(sName);
  if (globalInp.size() < found+1) globalInp.resize(found+1);
  globalInp[found]= globalInp0;

}


//--------------------------------------------------------------------------
/*! \brief This function re-calculates the block-size-padded sum of threads needed to compute the
  groups of neurons and synapses assigned to each device. Must be called after changing the
  hostID:deviceID of any neuron or synapse group.
 */
//--------------------------------------------------------------------------

void NNmodel::resetPaddedSums()
{
  // array for each host with arrays for each device goes here
  //vector<vector<int> > padSum = int[hostCount][deviceCount]

  for (int synapseGroup = 0; synapseGroup < synapseGrpN; synapseGroup++) {

    // CODE FOR RESETTING PADSUM* VARIABLES GOES HERE (use setMaxConn function)

  }
}


//--------------------------------------------------------------------------
/*! \brief This function sets the numerical precision of floating type variables. By default, it is float.
 */
//--------------------------------------------------------------------------

void NNmodel::setPrecision(unsigned int floattype /**<  */)
{
  switch (floattype) {
     case 0:
	ftype = toString("float");
	break;
     case 1:
	ftype = toString("double"); // not supported by compute capability < 1.3
	break;
     case 2:
	ftype = toString("long double"); // not supported by CUDA at the moment.
	break;
     default:
	ftype = toString("float");
  }
}


//--------------------------------------------------------------------------
/*! \brief This function sets the random seed. If the passed argument is > 0, automatic seeding is disabled. If the argument is 0, the underlying seed is obtained from the time() function.
 */
//--------------------------------------------------------------------------

void NNmodel::setSeed(unsigned int inseed /*!< the new seed  */)
{
    seed= inseed;
}

//--------------------------------------------------------------------------
/*! \brief This function defines the maximum number of connections for a neuron in the population
*/ 
//--------------------------------------------------------------------------

void NNmodel::setMaxConn(const string sname, /**<  */
                         unsigned int maxConnP /**<  */)
{
  unsigned int found= findSynapseGrp(sname);
  if (padSumSynapseKrnl.size() < found+1) padSumSynapseKrnl.resize(found+1);

  if (synapseConnType[found] == SPARSE){
    if (maxConn.size() < found+1) maxConn.resize(found+1);
    maxConn[found]= maxConnP;

    // set padnC is the lowest multiple of synapseBlkSz >= maxConn[found]
    unsigned int padnC = ceil((float)maxConn[found] / (float)synapseBlkSz) * (float)synapseBlkSz;

    if (found == 0) {
      padSumSynapseKrnl[found]=padnC;
      //fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", found, padSumSynapseKrnl[found]);
    }
    else {
      unsigned int toOmitK = padSumSynapseKrnl[found]-padSumSynapseKrnl[found-1];
      //fprintf(stderr, "old padSumSynapseKrnl[%d] is %u\n", found,padSumSynapseKrnl[found]);
      padSumSynapseKrnl[found]=padSumSynapseKrnl[found-1]+padnC;
      //fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", found,padSumSynapseKrnl[found]);
      for (int j=found+1;j<padSumSynapseKrnl.size();j++){    	
	//fprintf(stderr, "old padSumSynapseKrnl[%d] is %u\n",j,padSumSynapseKrnl[j]);
	padSumSynapseKrnl[j]=padSumSynapseKrnl[j]-toOmitK+padnC;
	//fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", j,padSumSynapseKrnl[j]);
      }
    }
  }
  else {
    fprintf(stderr,"WARNING: Synapse group %u is all-to-all connected. Maxconn variable is not needed in this case. Setting size to %u is not stable. Skipping...\n", found, maxConnP);

    /*unsigned int padnC = ceil((float)maxConnP / (float)synapseBlkSz) * (float)synapseBlkSz;
      if (found == 0) {
      padSumSynapseKrnl[found]=padnN;
      }
      else{
      unsigned int toOmitK = padSumSynapseKrnl[found]-padSumSynapseKrnl[found-1];
      padSumSynapseKrnl[found]=padSumSynapseKrnl[found-1]+padnC;
      for (int j=found+1,j<padSumSynapseKrnl.size(),j++){    	
      padSumSynapseKrnl[j]=padSumSynapseKrnl[j]-toOmitK+padnC;
      }
      }*/
  }
}

//--------------------------------------------------------------------------
/*! \brief This function defines the way how the GPU is chosen. If "AUTODEVICE" (-1) is given as the argument, GeNN will use internal heuristics to choose the device. Otherwise the argument is the device number and the indicated device will be used.
*/ 
//--------------------------------------------------------------------------

void NNmodel::setGPUDevice(int device)
{
  int deviceCount;
  CHECK_CUDA_ERRORS(cudaGetDeviceCount(&deviceCount));
  assert(device >= -1);
  assert(device < deviceCount);
  chooseGPUDevice= device;
}


#endif
