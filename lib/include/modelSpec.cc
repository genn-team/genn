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

#include "modelSpec.h"


// class NNmodel for specifying a neuronal network model
NNmodel::NNmodel() 
{
  dt = DT;
  valid = 0;
  neuronGrpN = 0;
  synapseGrpN = 0;
  lrnGroups = 0;
  needSt = 0;
  needSynapseDelay = 0;
  setPrecision(0);
}

NNmodel::~NNmodel() 
{
}

void NNmodel::setName(const string inname)
{
  name = toString(inname);
}

void NNmodel::setDT(float newDT)
{
  dt = newDT;
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
    float retVal = nModels[neuronType[i]].dps->calculateDerivedParameter(j, neuronPara[i], dt);
    tmpP.push_back(retVal);
  }
  /*
  if (neuronType[i] == MAPNEURON) {
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][0]*neuronPara[i][1]); // ip0
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][2]);                  // ip1
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][1]                   // ip2
    +neuronPara[i][0]*neuronPara[i][2]); 
  }*/
  dnp.push_back(tmpP);
}


void NNmodel::initNeuronSpecs(unsigned int i /**< index of the neuron population */)
{
  //default to a high but plausible value. This will usually get lowered by the configuration of any outgoing synapses
  //but note that if the nerons have no efferent synapses (e.g. the output neurons of a perceptron type setup)
  //then this value will be the one used.
  nSpkEvntThreshold.push_back(20.0);

  // padnN is the lowest multiple of neuronBlkSz >= neuronN[i]
  unsigned int padnN = ceil((float) neuronN[i] / (float) neuronBlkSz[nrnDevID[i]]) * (float) neuronBlkSz[nrnDevID[i]];
  if (i == 0) {
    sumNeuronN.push_back(neuronN[i]);
    padSumNeuronN.push_back(padnN);
  }
  else {
    sumNeuronN.push_back(sumNeuronN[i - 1] + neuronN[i]); 
    padSumNeuronN.push_back(padSumNeuronN[i - 1] + padnN); 
  }
  neuronNeedSt.push_back(0); // by default last spike times are not saved
}


void NNmodel::initDerivedPostSynapsePara(unsigned int i)
{
  vector<float> tmpP;
  for (int j=0; j < postSynModels[postSynapseType[i]].dpNames.size(); ++j) {
    float retVal = postSynModels[postSynapseType[i]].dps->calculateDerivedParameter(j, postSynapsePara[i], dt);
    tmpP.push_back(retVal);
  }	
  dpsp.push_back(tmpP);
}


//--------------------------------------------------------------------------
/*! \brief This function calculates derived synapse parameters from primary synapse parameters. 

This function needs to be invoked each time a synapse population is added, after all primary parameters have been set, and before code for synapse evaluation is generated. It should be invoked only once per population, as derived synapse parameters (dsp) are appended to mpdel parameter vectors indiscriminantly.
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedSynapsePara(unsigned int i /**< index of the synapse population */)
{
  vector<float> tmpP;
  if (synapseType[i] == LEARN1SYNAPSE) {
    tmpP.push_back(expf(-dt / synapsePara[i][2]));               // kdecay
    tmpP.push_back((1.0f / synapsePara[i][7] + 1.0f / synapsePara[i][4])
		   * synapsePara[i][3] / (2.0f / synapsePara[i][4]));      // lim0
    tmpP.push_back(-(1.0f / synapsePara[i][6] + 1.0f / synapsePara[i][4])
		   * synapsePara[i][3] / (2.0f / synapsePara[i][4]));      // lim1
    tmpP.push_back(-2.0f * synapsePara[i][8]
		   / (synapsePara[i][4] * synapsePara[i][3]));             // slope0
    tmpP.push_back(-tmpP[3]);                                    // slope1
    tmpP.push_back(synapsePara[i][8] / synapsePara[i][7]);       // off0
    tmpP.push_back(synapsePara[i][8] / synapsePara[i][4]);       // off1
    tmpP.push_back(synapsePara[i][8] / synapsePara[i][6]);       // off2
    // make sure spikes are detected at the post-synaptic neuron as well
    if (SPK_THRESH_STDP < nSpkEvntThreshold[synapseTarget[i]]) {
      nSpkEvntThreshold[synapseTarget[i]] = SPK_THRESH_STDP;
    }
    neuronNeedSt[synapseTarget[i]] = 1;
    neuronNeedSt[synapseSource[i]] = 1;
    needSt = 1;
    // padnN is the lowest multiple of learnBlkSz >= neuronN[synapseSource[i]]
    unsigned int padnN = ceil((float) neuronN[synapseSource[i]] / (float) learnBlkSz[synDevID[i]]) * (float) learnBlkSz[synDevID[i]];
    if (lrnGroups == 0) {
      padSumLearnN.push_back(padnN);
    }
    else {
      padSumLearnN.push_back(padSumLearnN[lrnGroups - 1] + padnN); 
    }
    lrnSynGrp.push_back(i);
    lrnGroups++;
  }
  
  if ((synapseType[i] == NSYNAPSE) || (synapseType[i] == NGRADSYNAPSE)) {
    tmpP.push_back(expf(-dt / synapsePara[i][2]));              // kdecay
  }
  dsp.push_back(tmpP);
  // figure out at what threshold we need to detect spiking events
  synapseInSynNo.push_back(inSyn[synapseTarget[i]].size());
  inSyn[synapseTarget[i]].push_back(i);
  if (nSpkEvntThreshold[synapseSource[i]] > synapsePara[i][1]) {
    nSpkEvntThreshold[synapseSource[i]] = synapsePara[i][1];
  }
  // padnN is the lowest multiple of synapseBlkSz >= neuronN[synapseTarget[i]]
  unsigned int padnN = ceil((float) neuronN[synapseTarget[i]] / (float) synapseBlkSz[synDevID[i]]) * (float) synapseBlkSz[synDevID[i]];
  if (i == 0) {
    sumSynapseTrgN.push_back(neuronN[synapseTarget[i]]);
    padSumSynapseKrnl.push_back(padnN);
  }
  else {
    sumSynapseTrgN.push_back(sumSynapseTrgN[i - 1] + neuronN[synapseTarget[i]]);
    padSumSynapseKrnl.push_back(padSumSynapseKrnl[i - 1] + padnN);
  }
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
  nrnHostID[groupNo] = hostID;
  nrnDevID[groupNo] = deviceID;
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
  synHostID[groupNo] = hostID;
  synDevID[groupNo] = deviceID;  
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
  if (nModels.size() < 1) prepareStandardModels();

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
  neuronDelaySlots.push_back(1);
  receivesInputCurrent.push_back(0);
  inSyn.push_back(tv);  // empty list of input synapse groups for neurons i 
  nrnDevID.push_back(0);
  nrnHostID.push_back(0);

  initDerivedNeuronPara(i);
  initNeuronSpecs(i);
}


//--------------------------------------------------------------------------
/*! \brief This function checks if the number of parameters and variables that are defined 
by the user are of correct size with respect to the selected neuron and synapse type.
*/ 
//--------------------------------------------------------------------------
void NNmodel::checkSizes(unsigned int * NeuronpSize, /**< Array containing the number of neuron population parameters for each neuron population added, in the adding order */
                         unsigned int * NeuronvSize, /**< Array containing the number of neuron population variables for each neuron population added, in the adding order */
                         unsigned int * SynpSize /**< Array containing the number of synapse population parameters for each neuron population added, in the adding order */)
{
  for (int j = 0; j < neuronGrpN; j++){
    if ((NeuronpSize[j]/sizeof(float)) != nModels[neuronType[j]].pNames.size()){
      cerr << "Error: Size of parameter values for "<< neuronName[j] <<" neuron group is " << (NeuronpSize[j]/sizeof(float)) << ", while it should be:" << nModels[neuronType[j]].pNames.size() << endl;
      exit(0);
    }
    if ((NeuronvSize[j]/sizeof(float)) != nModels[neuronType[j]].varNames.size()){
      cerr << "Error: Size of initial values for "<< neuronName[j] <<" neuron group is " << (NeuronvSize[j]/sizeof(float)) << ", while it should be:" << nModels[neuronType[j]].varNames.size() << endl;
      exit(0);
    }   	  	
  }
  for (int j = 0; j < synapseGrpN; j++){
    if ((SynpSize[j]/sizeof(float)) != SYNPNO[synapseType[j]]){
      cerr << "Error: Size of parameter values for "<< synapseName[j] <<" synapse group is " << (SynpSize[j]/sizeof(float)) << ", while it should be:" << nModels[synapseType[j]].pNames.size() << endl;
      exit(0);
    }
  }
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

  float postSynV[0]={};

  //Tries to borrow these values from the first set of synapse parameters supplied
  float postExpSynapsePopn[2] = {
    params[2], 	//tau_S: decay time constant [ms]
    params[0]	// Erev: Reversal potential
  };

  addSynapsePopulation(name, syntype, conntype, gtype, NO_DELAY, EXPDECAY, src, target, params, postSynV,postExpSynapsePopn);
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
                                   float *p, /**< A C-type array of floats that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float* PSVini, /**< A C-type array of floats that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/
                                   float *ps /**< A C-type array of floats that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses. The array must contain the right number of parameters in the right order for the chosen synapse type. If too few, segmentation faults will occur, if too many, excess will be ignored.*/ )
{
  unsigned int i= synapseGrpN++;
  unsigned int srcNumber, trgNumber;
  vector<float> tmpP;
  vector<float> tmpPS;
  vector<float> tmpPV;
  
  if (postSynModels.size() < 1) preparePostSynModels();

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
  
  //TODO: We want to get rid of SYNPNO array for code generation flexibility. It would be useful to predefine synapse models as we do for neurons in utils.h. This would also help for checkSizes.

  // for (int j= 0; j < nModels[synapseType[i]].pNames.size(); j++) { 
  for (int j= 0; j < SYNPNO[synapseType[i]]; j++) {
    tmpP.push_back(p[j]);
  }
  synapsePara.push_back(tmpP);
  postSynapseType.push_back(postsyn);
  for (int j= 0; j <  postSynModels[postSynapseType[i]].pNames.size(); j++) {
    tmpPS.push_back(ps[j]);
    //printf("%d th var in group %d is %f \n", j, i, ps[j]);
    //printf("%s\n", postSynModels[postSynapseType[i]].pNames[j].c_str());
  }
  postSynapsePara.push_back(tmpPS);  
  tmpPV.clear();
  for (int j= 0; j < postSynModels[postSynapseType[i]].varNames.size(); j++) {
    tmpPV.push_back(PSVini[j]);
  }
  postSynIni.push_back(tmpPV);
  synDevID.push_back(0);
  synHostID.push_back(0);

  initDerivedSynapsePara(i);
  initDerivedPostSynapsePara(i);
}


//--------------------------------------------------------------------------
/*! \brief This functions sets the global value of the maximal synaptic conductance for a synapse population that was idfentified as conductance specifcation method "GLOBALG" 
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseG(const string sName, /**<  */
                          float g /**<  */)
{
  unsigned int found= findSynapseGrp(sName);
  if (g0.size() < found+1) g0.resize(found+1);
  g0[found]= g;
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
/*! \brief This function defines the maximum number of connections for a neuron in the population
*/ 
//--------------------------------------------------------------------------

void NNmodel::setMaxConn(const string sname, /**<  */
                         unsigned int maxConnP /**<  */)
{
  unsigned int found = findSynapseGrp(sname);
  if (synapseConnType[found] == SPARSE) {
    if (maxConn.size() < found + 1) maxConn.resize(found + 1);
    maxConn[found] = maxConnP;

    // set padnC is the lowest multiple of synapseBlkSz >= maxConn[found]
    unsigned int padnC = ceil((float) maxConn[found] / (float) synapseBlkSz[synDevID[found]]) * (float) synapseBlkSz[synDevID[found]];

    unsigned int toOmitK;
    if (found == 0) {
      toOmitK = padSumSynapseKrnl[found];
      padSumSynapseKrnl[found] = padnC;
      //fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", found, padSumSynapseKrnl[found]);
    }
    else {
      toOmitK = padSumSynapseKrnl[found] - padSumSynapseKrnl[found - 1];
      //fprintf(stderr, "old padSumSynapseKrnl[%d] is %u\n", found,padSumSynapseKrnl[found]);
      padSumSynapseKrnl[found] = padSumSynapseKrnl[found - 1] + padnC;
      //fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", found,padSumSynapseKrnl[found]);
      for (int j = found + 1; j < padSumSynapseKrnl.size(); j++) {
	//fprintf(stderr, "old padSumSynapseKrnl[%d] is %u\n",j,padSumSynapseKrnl[j]);
	padSumSynapseKrnl[j] = padSumSynapseKrnl[j] - toOmitK + padnC;
	//fprintf(stderr, "padSumSynapseKrnl[%d] is %u\n", j,padSumSynapseKrnl[j]);
      }
    }
  }
  else {
    fprintf(stderr, "WARNING: Synapse group %u is all-to-all connected. Maxconn variable is not needed in this case. Setting size to %u is not stable. Skipping...\n", found, maxConnP);
  }
}


//--------------------------------------------------------------------------
/*! \brief This function re-calculates the block-size-padded sum of threads needed to compute the
  groups of neurons and synapses assigned to each device. Must be called AFTER setting the deviceID
  of the neuron and synapse groups.
 */
//--------------------------------------------------------------------------

void NNmodel::calcPaddedThreadSums()
{
  int padN;
  vector<int> padSum(synapseGrpN);
  vector<int> sumSoFar(deviceCount, 0);
  for (int i = 0; i < synapseGrpN; i++) {
    if (synapseConnType[i] == SPARSE) {
      // sparse synapse kernel thread sum
      padN = ceil((float) maxConn[i] / (float) synapseBlkSz[synDevID[i]]) * (float) synapseBlkSz[synDevID[i]];
    }
    else {
      // non-sparse synapse kernel thread sum
      padN = ceil((float) neuronN[synapseTarget[i]] / (float) synapseBlkSz[synDevID[i]]) * (float) synapseBlkSz[synDevID[i]];
    }
    sumSoFar[synDevID[i]] = sumSoFar[synDevID[i]] + padN;
    padSum[i] = sumSoFar[synDevID[i]];
  }

  padSum.resize(lrnGroups);
  sumSoFar.assign(deviceCount, 0);
  for (int i = 0; i < lrnGroups; i++) {
    // learning kernel thread sum
    padN = ceil((float) neuronN[synapseSource[i]] / (float) learnBlkSz[synDevID[i]]) * (float) learnBlkSz[synDevID[i]];
    sumSoFar[synDevID[i]] = sumSoFar[synDevID[i]] + padN;
    padSum[i] = sumSoFar[synDevID[i]];
  }

  padSum.resize(neuronGrpN);
  sumSoFar.assign(deviceCount, 0);
  for (int i = 0; i < neuronGrpN; i++) {
    // neuron kernel thread sum
    padN = ceil((float) neuronN[i] / (float) neuronBlkSz[nrnDevID[i]]) * (float) neuronBlkSz[nrnDevID[i]];
    sumSoFar[nrnDevID[i]] = sumSoFar[nrnDevID[i]] + padN;
    padSum[i] = sumSoFar[nrnDevID[i]];
  }
}


#endif
