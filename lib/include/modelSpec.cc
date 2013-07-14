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


// class NNmodel for specifying a neuronal network model

NNmodel::NNmodel() 
{
  valid= 0;
  neuronGrpN= 0;
  synapseGrpN= 0;
  lrnGroups= 0;
  needSt= 0;
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

void NNmodel::initDerivedNeuronPara(unsigned int i)
{
  vector<float> tmpP;
  if (neuronType[i] == MAPNEURON) {
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][0]*neuronPara[i][1]); // ip0
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][2]);                  // ip1
    tmpP.push_back(neuronPara[i][0]*neuronPara[i][1]                   // ip2
		   +neuronPara[i][0]*neuronPara[i][2]); 
  }
  dnp.push_back(tmpP);
  nThresh.push_back(200.0f);
  unsigned int padnN= (neuronN[i] >> logNeuronBlkSz) << logNeuronBlkSz;
  if (padnN < neuronN[i]) padnN+= neuronBlkSz;
  if (i == 0) {
    sumNeuronN.push_back(neuronN[i]);
    padSumNeuronN.push_back(padnN);
    fprintf(stderr, "%d\n", padSumNeuronN[i]);
  }
  else {
    sumNeuronN.push_back(sumNeuronN[i-1] + neuronN[i]); 
    padSumNeuronN.push_back(padSumNeuronN[i-1] + padnN); 
    fprintf(stderr, "%d\n", padSumNeuronN[i]);
  }
  neuronNeedSt.push_back(0);  // by default last spike times are not saved
}

//--------------------------------------------------------------------------
/*! \brief This function calculates derived synapse parameters from primary synapse parameters. 

This function needs to be invoked after all primary parameters have been set and before code for synapse evaluation is generated.
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedSynapsePara(unsigned int i)
{
  // to be called when all para have been set!
  // also, call this only once and right after a population has been added
  // as values are appended to vectors indiscriminantly
  // derived synapse parameters (dsp)
  vector<float> tmpP;
  if (synapseType[i] == LEARN1SYNAPSE) {
    tmpP.push_back(expf(-DT/synapsePara[i][2]));              // kdecay
    tmpP.push_back((1.0f/synapsePara[i][7] + 1.0f/synapsePara[i][4])
		   *synapsePara[i][3] / (2.0f/synapsePara[i][4]));      // lim0
    tmpP.push_back(-(1.0f/synapsePara[i][6] + 1.0f/synapsePara[i][4])
		   *synapsePara[i][3] / (2.0f/synapsePara[i][4]));      // lim1
    tmpP.push_back(-2.0f*synapsePara[i][8] / 
		   (synapsePara[i][4]*synapsePara[i][3]));              // slope0
    tmpP.push_back(-tmpP[3]);                              // slope1
    tmpP.push_back(synapsePara[i][8]/synapsePara[i][7]);       // off0
    tmpP.push_back(synapsePara[i][8]/synapsePara[i][4]);       // off1
    tmpP.push_back(synapsePara[i][8]/synapsePara[i][6]);       // off2
    // make sure spikes are detected at the post-synaptic neuron as well
    if (SPK_THRESH < nThresh[synapseTarget[i]]) {
      nThresh[synapseTarget[i]]= SPK_THRESH;
    }
    neuronNeedSt[synapseTarget[i]]= 1;
    neuronNeedSt[synapseSource[i]]= 1;
    needSt= 1;
    unsigned int padnN= (neuronN[synapseSource[i]] >> logLearnBlkSz) << logLearnBlkSz;
    if (padnN < neuronN[synapseSource[i]]) padnN+= learnBlkSz;
    if (lrnGroups == 0) {
      padSumLearnN.push_back(padnN);
    }
    else {
      padSumLearnN.push_back(padSumLearnN[i-1] + padnN); 
    }
    lrnSynGrp.push_back(i);
    lrnGroups++;
  }
 
  
  if ((synapseType[i] == NSYNAPSE) || (synapseType[i] == NGRADSYNAPSE)) {
    tmpP.push_back(expf(-DT/synapsePara[i][2]));              // kdecay
  }
  dsp.push_back(tmpP);
  // figure out at what threshold we need to detect spiking events
  synapseInSynNo.push_back(inSyn[synapseTarget[i]].size());
  inSyn[synapseTarget[i]].push_back(i);
  if (nThresh[synapseSource[i]] > synapsePara[i][1]) {
    nThresh[synapseSource[i]]= synapsePara[i][1];
  }
  // update synapse target neuron numbers etc
  unsigned int nN= neuronN[synapseTarget[i]];
  unsigned int padnN= (nN >> logSynapseBlkSz) << logSynapseBlkSz;
  if (padnN < nN) padnN+= synapseBlkSz;
  if (i == 0) {
    sumSynapseTrgN.push_back(nN);
    padSumSynapseTrgN.push_back(padnN);
    fprintf(stderr, "%d\n", padSumSynapseTrgN[i]);
  }
  else {
    sumSynapseTrgN.push_back(sumSynapseTrgN[i-1]+nN);
    padSumSynapseTrgN.push_back(padSumSynapseTrgN[i-1]+padnN);
    fprintf(stderr, "%d\n", padSumSynapseTrgN[i]);
  }
}

//--------------------------------------------------------------------------
/*! \brief This function is a tool to find the numeric ID of a neuron population based on the name of the neuron population.
 */
//--------------------------------------------------------------------------

unsigned int NNmodel::findNeuronGrp(const string nName)
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
/*! \brief This function is a tool to find the numeric ID of a synapse population based on the name of the synapse population.
 */
//--------------------------------------------------------------------------


unsigned int NNmodel::findSynapseGrp(const string sName)
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
/*! \brief This function is an alternative method to the standard addNeuronPopulation that allows the use of constant character arrays instead of C++ strings
 */
//--------------------------------------------------------------------------


void NNmodel::addNeuronPopulation(const char *name, unsigned int nNo, unsigned int type, float *p, float *ini)
{
  addNeuronPopulation(toString(name), nNo, type, p, ini);
}

//--------------------------------------------------------------------------
/*! \brief This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values.
 */
//--------------------------------------------------------------------------

void NNmodel::addNeuronPopulation(const string name, unsigned int nNo, unsigned int type, float *p, float *ini)
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
  inSyn.push_back(tv);  // empty list of input synapse groups for neurons i
  initDerivedNeuronPara(i);
}

//--------------------------------------------------------------------------
/*! \brief This function is an alternative method to the standard addSynapsePopulation that allows the use of constant character arrays instead of C++ strings
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(const char *name, unsigned int syntype, unsigned int conntype, unsigned int gtype, const char *src, const char *trg, float *p) 
{
  addSynapsePopulation(toString(name), syntype, conntype, gtype, toString(src), toString(trg), p);
}

//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(const string name, unsigned int syntype, unsigned int conntype, unsigned int gtype, const string src, const string trg, float *p)
{
  unsigned int i= synapseGrpN++;
  unsigned int found;

  synapseName.push_back(name);
  synapseType.push_back(syntype);
  synapseConnType.push_back(conntype);
  synapseGType.push_back(gtype);
  found= findNeuronGrp(src);
  synapseSource.push_back(found);
  found= findNeuronGrp(trg);
  synapseTarget.push_back(found);
  vector<float> tmpP;
  for (int j= 0; j < SYNPNO[synapseType[i]]; j++) {
    tmpP.push_back(p[j]);
  }
  synapsePara.push_back(tmpP);
  initDerivedSynapsePara(i);
}


//--------------------------------------------------------------------------
/*! \brief This functions sets the global value of the maximal synaptic conductance for a synapse population that was idfentified as conductance specifcation method "GLOBALG" 
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseG(const string sName, float g)
{
  unsigned int found= findSynapseGrp(sName);
  if (g0.size() < found+1) g0.resize(found+1);
  g0[found]= g;
}

#endif
