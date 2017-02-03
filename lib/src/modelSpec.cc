/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef MODELSPEC_CC
#define MODELSPEC_CC

#include "modelSpec.h"
#include "global.h"
#include "utils.h"
#include "stringUtils.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>


unsigned int GeNNReady = 0;

// ------------------------------------------------------------------------
//! \brief Method for GeNN initialisation (by preparing standard models)
    
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
    final= 0;
    neuronGrpN= 0;
    synapseGrpN= 0;
    lrnGroups= 0;
    synDynGroups= 0;
    needSt= 0;
    needSynapseDelay = 0;
    setDT(0.5);
    setPrecision(GENN_FLOAT);
    setTiming(false);
    RNtype= "uint64_t";
#ifndef CPU_ONLY
    setGPUDevice(AUTODEVICE);
#endif
    setSeed(0);
}

NNmodel::~NNmodel() 
{
}

void NNmodel::setName(const string inname)
{
    if (final) {
        gennError("Trying to set the name of a finalized model.");
    }
    name= inname;
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
}


//--------------------------------------------------------------------------
/*! \brief This function is a tool to find the numeric ID of a neuron population based on the name of the neuron population.
 */
//--------------------------------------------------------------------------

unsigned int NNmodel::findNeuronGrp(const string &nName /**< Name of the neuron population */) const
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

void NNmodel::setNeuronClusterIndex(const string &neuronGroup, /**< Name of the neuron population */
                                    int hostID, /**< ID of the host */
                                    int deviceID /**< ID of the device */)
{
    int groupNo = findNeuronGrp(neuronGroup);
    neuronHostID[groupNo] = hostID;
    neuronDeviceID[groupNo] = deviceID;
}


//--------------------------------------------------------------------------
/*! \brief 
 */
//--------------------------------------------------------------------------

void NNmodel::initLearnGrps()
{
    synapseUsesTrueSpikes.assign(synapseGrpN, false);
    synapseUsesSpikeEvents.assign(synapseGrpN, false);
    synapseUsesPostLearning.assign(synapseGrpN, false);
    synapseUsesSynapseDynamics.assign(synapseGrpN, false);

    neuronNeedTrueSpk.assign(neuronGrpN, false);
    neuronNeedSpkEvnt.assign(neuronGrpN, false);

    neuronVarNeedQueue.resize(neuronGrpN);
    for (int i = 0; i < neuronGrpN; i++) {
        neuronVarNeedQueue[i] = vector<bool>(nModels[neuronType[i]].varNames.size(), false);
    }
    neuronSpkEvntCondition.assign(neuronGrpN, "");

    for (int i = 0; i < synapseGrpN; i++) {
        const auto &wu = weightUpdateModels[synapseType[i]];
        unsigned int src = synapseSource[i];
        vector<string> vars = nModels[neuronType[src]].varNames;
        needEvntThresholdReTest.push_back(false);

        if (wu.simCode != "") {
            synapseUsesTrueSpikes[i] = true;
            neuronNeedTrueSpk[src] = true;

            // analyze which neuron variables need queues
            for (int j = 0; j < vars.size(); j++) {
                if (wu.simCode.find(vars[j] + "_pre") != string::npos) {
                    neuronVarNeedQueue[src][j] = true;
                }
            }
        }

        if (wu.simLearnPost != "") {
            synapseUsesPostLearning[i] = true;
            lrnSynGrp.push_back(i);
            lrnGroups++;
            for (int j = 0; j < vars.size(); j++) {
                if (wu.simLearnPost.find(vars[j] + "_pre") != string::npos) {
                    neuronVarNeedQueue[src][j] = true;
                }
            }
        }

        if (wu.synapseDynamics != "") {
            synapseUsesSynapseDynamics[i]= true;
            synDynGrp.push_back(i);
            synDynGroups++;
            for (int j = 0; j < vars.size(); j++) {
                if (wu.synapseDynamics.find(vars[j] + "_pre") != string::npos) {
                    neuronVarNeedQueue[src][j] = true;
                }
            }
        }
    }

    for (int i = 0; i < neuronGrpN; i++) {
        string eCode0;
        vector<string> vars = nModels[neuronType[i]].varNames;
        bool needReTest= false;
        for (int j= 0, l= outSyn[i].size(); j < l; j++) {
            int synPopID= outSyn[i][j];
            const auto &wu= weightUpdateModels[synapseType[synPopID]];
            if (wu.simCodeEvnt != "") {
                synapseUsesSpikeEvents[synPopID] = true;
                neuronNeedSpkEvnt[i] = true;
                assert(wu.evntThreshold != "");

                // do an early replacement of parameters, derived parameters and extraglobalsynapse parameters
                string eCode= wu.evntThreshold;
                value_substitutions(eCode, wu.pNames, synapsePara[synPopID]);
                value_substitutions(eCode, wu.dpNames, dsp_w[synPopID]);
                name_substitutions(eCode, "", wu.extraGlobalSynapseKernelParameters, synapseName[synPopID]);

                // add to the source population spike event condition
                if (neuronSpkEvntCondition[i] == "") {
                    neuronSpkEvntCondition[i] = "(" + eCode + ")";
                    eCode0= eCode; // remember the first condition
                }
                else {
                    if (eCode != eCode0) {
                        needReTest= true;
                    }
                    neuronSpkEvntCondition[i] += " || (" + eCode + ")";
                }

                // analyze which neuron variables need queues
                for (int j = 0; j < vars.size(); j++) {
                    if (wu.simCodeEvnt.find(vars[j] + "_pre") != string::npos) {
                        neuronVarNeedQueue[i][j] = true;
                    }
                }
            }
        }
        if (needReTest) {
            for (int j= 0, l= outSyn[i].size(); j < l; j++) {
                int synPopID= outSyn[i][j];
                weightUpdateModel wu= weightUpdateModels[synapseType[synPopID]];
                if (wu.simCodeEvnt != "") {
                    needEvntThresholdReTest[synPopID]= true;
                }
            }
        }
    }
    // related to kernel parameters: make kernel parameter lists
    // for neuron kernel
    for (int i = 0; i < neuronGrpN; i++) {
        unsigned int nt= neuronType[i];
        for (int j= 0, l= nModels[nt].extraGlobalNeuronKernelParameters.size(); j < l; j++) {
            string pname= nModels[nt].extraGlobalNeuronKernelParameters[j];
            string pnamefull= pname + neuronName[i];
            string ptype= nModels[nt].extraGlobalNeuronKernelParameterTypes[j];
            if (find(neuronKernelParameters.begin(), neuronKernelParameters.end(), pnamefull) == neuronKernelParameters.end()) {
                // parameter wasn't registered yet - is it used?
                bool used= 0;
                if (nModels[nt].simCode.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                if (nModels[nt].thresholdConditionCode.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                if (nModels[nt].resetCode.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                if (used) {
                    neuronKernelParameters.push_back(pnamefull);
                    neuronKernelParameterTypes.push_back(ptype);
                }
            }
        }
    }
    for (int i = 0; i < synapseGrpN; i++) {
        const auto &wu = weightUpdateModels[synapseType[i]];
        unsigned int src = synapseSource[i];
        for (int j= 0, l= wu.extraGlobalSynapseKernelParameters.size(); j < l; j++) {
            string pname= wu.extraGlobalSynapseKernelParameters[j];
            string pnamefull= pname + synapseName[i];
            string ptype= wu.extraGlobalSynapseKernelParameterTypes[j];
            if (find(neuronKernelParameters.begin(), neuronKernelParameters.end(), pnamefull) == neuronKernelParameters.end()) {
                // parameter wasn't registered yet - is it used?
                bool used= 0;
                if (neuronSpkEvntCondition[src].find(pnamefull) != string::npos) used= 1; // it's used
                 if (used) {
                    neuronKernelParameters.push_back(pnamefull);
                    neuronKernelParameterTypes.push_back(ptype);
                }
            }
        }
    }
    // for synapse kernel
    for (int i = 0; i < synapseGrpN; i++) {
        const auto &wu = weightUpdateModels[synapseType[i]];
        unsigned int src = synapseSource[i];
        unsigned int trg = synapseTarget[i];
        unsigned int nt[2];
        nt[0]= neuronType[src]; // pre
        nt[1]= neuronType[trg]; // post
        string suffix[2];
        suffix[0]= "_pre";
        suffix[1]= "_post";
        for (int k= 0; k < 2; k++) {
            for (int j= 0, l= nModels[nt[k]].extraGlobalNeuronKernelParameters.size(); j < l; j++) {
                string pname= nModels[nt[k]].extraGlobalNeuronKernelParameters[j];
                string pnamefull= pname + neuronName[src];
                string ptype= nModels[nt[k]].extraGlobalNeuronKernelParameterTypes[j];
                if (find(synapseKernelParameters.begin(), synapseKernelParameters.end(), pnamefull) == synapseKernelParameters.end()) {
                    // parameter wasn't registered yet - is it used?
                    bool used= 0;
                    if (wu.simCode.find("$(" + pname + suffix[k] + ")") != string::npos) used= 1; // it's used
                    if (wu.simCodeEvnt.find("$(" + pname + suffix[k] + ")") != string::npos) used= 1; // it's used
                    if (wu.evntThreshold.find("$(" + pname + suffix[k] + ")") != string::npos) used= 1; // it's used
                    if (used) {
                        synapseKernelParameters.push_back(pnamefull);
                        synapseKernelParameterTypes.push_back(ptype);
                    }
                }
            }
        }
        for (int j= 0, l= wu.extraGlobalSynapseKernelParameters.size(); j < l; j++) {
            string pname= wu.extraGlobalSynapseKernelParameters[j];
            string pnamefull= pname + synapseName[i];
            string ptype= wu.extraGlobalSynapseKernelParameterTypes[j];
            if (find(synapseKernelParameters.begin(), synapseKernelParameters.end(), pnamefull) == synapseKernelParameters.end()) {
                // parameter wasn't registered yet - is it used?
                bool used= 0;
                if (wu.simCode.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                if (wu.simCodeEvnt.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                if (wu.evntThreshold.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                 if (used) {
                    synapseKernelParameters.push_back(pnamefull);
                    synapseKernelParameterTypes.push_back(ptype);
                }
            }
        }
    }
    
    // for simLearnPost
    for (int i = 0; i < synapseGrpN; i++) {
        const auto &wu = weightUpdateModels[synapseType[i]];
        unsigned int src = synapseSource[i];
        unsigned int trg = synapseTarget[i];
        unsigned int nt[2];
        nt[0]= neuronType[src]; // pre
        nt[1]= neuronType[trg]; // post
        string suffix[2];
        suffix[0]= "_pre";
        suffix[1]= "_post";
        for (int k= 0; k < 2; k++) {
            for (int j= 0, l= nModels[nt[k]].extraGlobalNeuronKernelParameters.size(); j < l; j++) {
                string pname= nModels[nt[k]].extraGlobalNeuronKernelParameters[j];
                string pnamefull= pname + neuronName[src];
                string ptype= nModels[nt[k]].extraGlobalNeuronKernelParameterTypes[j];
                if (find(simLearnPostKernelParameters.begin(), simLearnPostKernelParameters.end(), pnamefull) == simLearnPostKernelParameters.end()) {
                    // parameter wasn't registered yet - is it used?
                    bool used= 0;
                    if (wu.simLearnPost.find("$(" + pname + suffix[k]) != string::npos) used= 1; // it's used
                    if (used) {
                        simLearnPostKernelParameters.push_back(pnamefull);
                        simLearnPostKernelParameterTypes.push_back(ptype);
                    }
                }
            }
        }
        for (int j= 0, l= wu.extraGlobalSynapseKernelParameters.size(); j < l; j++) {
            string pname= wu.extraGlobalSynapseKernelParameters[j];
            string pnamefull= pname + synapseName[i];
            string ptype= wu.extraGlobalSynapseKernelParameterTypes[j];
            if (find(simLearnPostKernelParameters.begin(), simLearnPostKernelParameters.end(), pnamefull) == simLearnPostKernelParameters.end()) {
                // parameter wasn't registered yet - is it used?
                bool used= 0;
                if (wu.simLearnPost.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                 if (used) {
                    simLearnPostKernelParameters.push_back(pnamefull);
                    simLearnPostKernelParameterTypes.push_back(ptype);
                }
            }
        }
    }
   
    // for synapse Dynamics
    for (int i = 0; i < synapseGrpN; i++) {
        const auto &wu = weightUpdateModels[synapseType[i]];
        unsigned int src = synapseSource[i];
        unsigned int trg = synapseTarget[i];
        unsigned int nt[2];
        nt[0]= neuronType[src]; // pre
        nt[1]= neuronType[trg]; // post
        string suffix[2];
        suffix[0]= "_pre";
        suffix[1]= "_post";
        for (int k= 0; k < 2; k++) {
            for (int j= 0, l= nModels[nt[k]].extraGlobalNeuronKernelParameters.size(); j < l; j++) {
                string pname= nModels[nt[k]].extraGlobalNeuronKernelParameters[j];
                string pnamefull= pname + neuronName[src];
                string ptype= nModels[nt[k]].extraGlobalNeuronKernelParameterTypes[j];
                if (find(synapseDynamicsKernelParameters.begin(), synapseDynamicsKernelParameters.end(), pnamefull) == synapseDynamicsKernelParameters.end()) {
                    // parameter wasn't registered yet - is it used?
                    bool used= 0;
                    if (wu.synapseDynamics.find("$(" + pname + suffix[k]) != string::npos) used= 1; // it's used
                    if (used) {
                        synapseDynamicsKernelParameters.push_back(pnamefull);
                        synapseDynamicsKernelParameterTypes.push_back(ptype);
                    }
                }
            }
        }
        for (int j= 0, l= wu.extraGlobalSynapseKernelParameters.size(); j < l; j++) {
            string pname= wu.extraGlobalSynapseKernelParameters[j];
            string pnamefull= pname + synapseName[i];
            string ptype= wu.extraGlobalSynapseKernelParameterTypes[j];
            if (find(synapseDynamicsKernelParameters.begin(), synapseDynamicsKernelParameters.end(), pnamefull) == synapseDynamicsKernelParameters.end()) {
                // parameter wasn't registered yet - is it used?
                bool used= 0;
                if (wu.synapseDynamics.find("$(" + pname + ")") != string::npos) used= 1; // it's used
                 if (used) {
                    synapseDynamicsKernelParameters.push_back(pnamefull);
                    synapseDynamicsKernelParameterTypes.push_back(ptype);
                }
            }
        }
    }

#ifndef CPU_ONLY
    // figure out where to reset the spike counters
    if (synapseGrpN == 0) { // no synapses -> reset in neuron kernel
        resetKernel= GENN_FLAGS::calcNeurons;
    }
    else { // there are synapses
        if (lrnGroups > 0) {
            resetKernel= GENN_FLAGS::learnSynapsesPost;
        }
        else {
            resetKernel= GENN_FLAGS::calcSynapses;
        }
    }
#endif
}


//--------------------------------------------------------------------------
/*! \brief This function is a tool to find the numeric ID of a synapse population based on the name of the synapse population.
 */
//--------------------------------------------------------------------------

unsigned int NNmodel::findSynapseGrp(const string &sName /**< Name of the synapse population */) const
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

void NNmodel::setSynapseClusterIndex(const string &synapseGroup, /**< Name of the synapse population */
                                     int hostID, /**< ID of the host */
                                     int deviceID /**< ID of the device */)
{
    int groupNo = findSynapseGrp(synapseGroup);
    synapseHostID[groupNo] = hostID;
    synapseDeviceID[groupNo] = deviceID;  
}


//--------------------------------------------------------------------------
/*! \overload

  This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values, the latter two defined as double *
 */
//--------------------------------------------------------------------------

void NNmodel::addNeuronPopulation(
  const string &name, /**<  The name of the neuron population*/
  unsigned int nNo, /**<  Number of neurons in the population */
  unsigned int type, /**<  Type of the neurons, refers to either a standard type or user-defined type*/
  const double *p, /**< Parameters of this neuron type */
  const double *ini /**< Initial values for variables of this neuron type */)
{
  vector<double> vp;
  vector<double> vini;
  for (int i= 0; i < nModels[type].pNames.size(); i++) {
    vp.push_back(p[i]);
  }
  for (int i= 0; i < nModels[type].varNames.size(); i++) {
    vini.push_back(ini[i]);
  }
  addNeuronPopulation(name, nNo, type, vp, vini);
}
  

//--------------------------------------------------------------------------
/*! \brief This function adds a neuron population to a neuronal network models, assigning the name, the number of neurons in the group, the neuron type, parameters and initial values. The latter two defined as STL vectors of double.
 */
//--------------------------------------------------------------------------

void NNmodel::addNeuronPopulation(
  const string &name, /**<  The name of the neuron population*/
  unsigned int nNo, /**<  Number of neurons in the population */
  unsigned int type, /**<  Type of the neurons, refers to either a standard type or user-defined type*/
  const vector<double> &p, /**< Parameters of this neuron type */
  const vector<double> &ini /**< Initial values for variables of this neuron type */)
{
    if (!GeNNReady) {
        gennError("You need to call initGeNN first.");
    }
    if (final) {
        gennError("Trying to add a neuron population to a finalized model.");
    }
    if (p.size() != nModels[type].pNames.size()) {
        gennError("The number of parameter values for neuron group " + name + " does not match that of their neuron type, " + tS(p.size()) + " != " + tS(nModels[type].pNames.size()));
    }
    if (ini.size() != nModels[type].varNames.size()) {
        gennError("The number of variable initial values for neuron group " + name + " does not match that of their neuron type, " + tS(ini.size()) + " != " + tS(nModels[type].varNames.size()));
    }   

    unsigned int i= neuronGrpN++;
    neuronName.push_back(name);
    neuronN.push_back(nNo);
    neuronType.push_back(type);
    neuronPara.push_back(p);
    neuronIni.push_back(ini);
    inSyn.push_back(vector<unsigned int>());
    outSyn.push_back(vector<unsigned int>());
    neuronNeedSt.push_back(false);
    neuronNeedSpkEvnt.push_back(false);
    neuronSpkEvntCondition.push_back("");
    neuronDelaySlots.push_back(1);

    // initially set neuron group indexing variables to device 0 host 0
    neuronDeviceID.push_back(0);
    neuronHostID.push_back(0);
}


//--------------------------------------------------------------------------
/*! \brief This function defines the type of the explicit input to the neuron model. Current options are common constant input to all neurons, input  from a file and input defines as a rule.
*/ 
//--------------------------------------------------------------------------
void NNmodel::activateDirectInput(
  const string &name, /**< Name of the neuron population */
  unsigned int type /**< Type of input: 1 if common input, 2 if custom input from file, 3 if custom input as a rule*/)
{
    gennError("This function has been deprecated since GeNN 2.2. Use neuron variables, extraGlobalNeuronKernelParameters, or parameters instead.");
}


//--------------------------------------------------------------------------
/*! \overload

  This deprecated function is provided for compatibility with the previous release of GeNN.
 * Default values are provide for new parameters, it is strongly recommended these be selected explicity via the new version othe function
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &target, /**< Name of the (existing!) post-synaptic neuron population*/
  const double *params/**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
  gennError("This version of addSynapsePopulation() has been deprecated since GeNN 2.2. Please use the newer addSynapsePopulation functions instead.");
}


//--------------------------------------------------------------------------
/*! \brief Overloaded old version (deprecated)
*/
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const double *p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const double* PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
    cerr << "!!!!!!GeNN WARNING: This function has been deprecated since GeNN 2.2, and will be removed in a future release. You use the overloaded method which passes a null pointer for the initial values of weight update variables. If you use a method that uses synapse variables, please add a pointer to this vector in the function call, like:\n          addSynapsePopulation(name, syntype, conntype, gtype, NO_DELAY, EXPDECAY, src, target, double * SYNVARINI, params, postSynV,postExpSynapsePopn);" << endl;
    const double *iniv = NULL;
    addSynapsePopulation(name, syntype, conntype, gtype, delaySteps, postsyn, src, trg, iniv, p, PSVini, ps);
}


//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const double* synini, /**< A C-type array of doubles that contains the initial values for synapse variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const double* PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const double *ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
  vector<double> vsynini;
  for (int j= 0; j < weightUpdateModels[syntype].varNames.size(); j++) {
    vsynini.push_back(synini[j]);
  }
  vector<double> vp;
  for (int j= 0; j < weightUpdateModels[syntype].pNames.size(); j++) {
    vp.push_back(p[j]);
  }
  vector<double> vpsini;
  for (int j= 0; j < postSynModels[postsyn].varNames.size(); j++) {
    vpsini.push_back(PSVini[j]);
  }
  vector<double> vps;
  for (int j= 0; j <  postSynModels[postsyn].pNames.size(); j++) {
    vps.push_back(ps[j]);
  }
  addSynapsePopulation(name, syntype, conntype, gtype, delaySteps, postsyn, src, trg, vsynini, vp, vpsini, vps);
}


//--------------------------------------------------------------------------
/*! \brief This function adds a synapse population to a neuronal network model, assigning the name, the synapse type, the connectivity type, the type of conductance specification, the source and destination neuron populations, and the synaptic parameters.
 */
//--------------------------------------------------------------------------

void NNmodel::addSynapsePopulation(
  const string &name, /**<  The name of the synapse population*/
  unsigned int syntype, /**< The type of synapse to be added (i.e. learning mode) */
  SynapseConnType conntype, /**< The type of synaptic connectivity*/
  SynapseGType gtype, /**< The way how the synaptic conductivity g will be defined*/
  unsigned int delaySteps, /**< Number of delay slots*/
  unsigned int postsyn, /**< Postsynaptic integration method*/
  const string &src, /**< Name of the (existing!) pre-synaptic neuron population*/
  const string &trg, /**< Name of the (existing!) post-synaptic neuron population*/
  const vector<double> &synini, /**< A C-type array of doubles that contains the initial values for synapse variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &p, /**< A C-type array of doubles that contains synapse parameter values (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &PSVini, /**< A C-type array of doubles that contains the initial values for postsynaptic mechanism variables (common to all synapses of the population) which will be used for the defined synapses.*/
  const vector<double> &ps /**< A C-type array of doubles that contains postsynaptic mechanism parameter values (common to all synapses of the population) which will be used for the defined synapses.*/)
{
    if (!GeNNReady) {
        gennError("You need to call initGeNN first.");
    }
    if (final) {
        gennError("Trying to add a synapse population to a finalized model.");
    }
    if (p.size() != weightUpdateModels[syntype].pNames.size()) {
        gennError("The number of presynaptic parameter values for synapse group " + name + " does not match that of their synapse type, " + tS(p.size()) + " != " + tS(weightUpdateModels[syntype].pNames.size()));
    }
    if (synini.size() != weightUpdateModels[syntype].varNames.size()) {
        gennError("The number of presynaptic variable initial values for synapse group " + name + " does not match that of their synapse type, " + tS(synini.size()) + " != " + tS(weightUpdateModels[syntype].varNames.size()));
    }
    if (ps.size() != postSynModels[postsyn].pNames.size()) {
        gennError("The number of presynaptic parameter values for synapse group " + name + " does not match that of their synapse type, " + tS(ps.size()) + " != " + tS(postSynModels[postsyn].pNames.size()));
    }
    if (PSVini.size() != postSynModels[postsyn].varNames.size()) {
        gennError("The number of presynaptic variable initial values for synapse group " + name + " does not match that of their synapse type, " + tS(PSVini.size()) + " != " + tS(postSynModels[postsyn].varNames.size()));
    }

    unsigned int i= synapseGrpN++;
    unsigned int srcNumber = findNeuronGrp(src);
    unsigned int trgNumber = findNeuronGrp(trg);
    synapseName.push_back(name);
    synapseType.push_back(syntype);
    synapseConnType.push_back(conntype);
    synapseGType.push_back(gtype);
    synapseSource.push_back(srcNumber);
    synapseTarget.push_back(trgNumber);
    synapseDelay.push_back(delaySteps);
    if (delaySteps >= neuronDelaySlots[srcNumber]) {
        neuronDelaySlots[srcNumber] = delaySteps + 1;
        needSynapseDelay = 1;
    }
    if (weightUpdateModels[syntype].needPreSt) {
        neuronNeedSt[srcNumber]= true;
        needSt= true;
    }
    if (weightUpdateModels[syntype].needPostSt) {
        neuronNeedSt[trgNumber]= true;
        needSt= true;
    }
    synapseIni.push_back(synini);
    synapsePara.push_back(p);
    postSynapseType.push_back(postsyn);
    postSynIni.push_back(PSVini);  
    postSynapsePara.push_back(ps);  
    registerSynapsePopulation(i);
    maxConn.push_back(neuronN[trgNumber]);
    synapseSpanType.push_back(0);

    // initially set synapase group indexing variables to device 0 host 0
    synapseDeviceID.push_back(0);
    synapseHostID.push_back(0);

    // TODO set uses*** variables for synaptic populations
}


//--------------------------------------------------------------------------
/*! \brief This function defines the maximum number of connections for a neuron in the population
*/ 
//--------------------------------------------------------------------------

void NNmodel::setMaxConn(const string &sname, /**<  */
                         unsigned int maxConnP /**<  */)
{
    if (final) {
        gennError("Trying to set MaxConn in a finalized model.");
    }
    unsigned int found = findSynapseGrp(sname);
    if (synapseConnType[found] == SPARSE) {
        maxConn[found] = maxConnP;
    }
    else {
        gennError("setMaxConn: Synapse group %u is all-to-all connected. Maxconn variable is not needed in this case. Setting size to %u is not stable.");
    }
}


//--------------------------------------------------------------------------
/*! \brief This function defines the execution order of the synapses in the kernels
  (0 : execute for every postsynaptic neuron 1: execute for every presynaptic neuron)
 */ 
//--------------------------------------------------------------------------

void NNmodel::setSpanTypeToPre(const string &sname /**< name of the synapse group to which to apply the pre-synaptic span type */)
{
    if (final) {
        gennError("Trying to set spanType in a finalized model.");
    }
    unsigned int found = findSynapseGrp(sname);
    if (synapseConnType[found] == SPARSE) {
        synapseSpanType[found] = 1;
    }
    else {
        gennError("setSpanTypeToPre: This function is not enabled for dense connectivity type.");
    }
}


//--------------------------------------------------------------------------
/*! \brief This functions sets the global value of the maximal synaptic conductance for a synapse population that was idfentified as conductance specifcation method "GLOBALG" 
 */
//--------------------------------------------------------------------------

void NNmodel::setSynapseG(const string &sName, /**<  */
                          double g /**<  */)
{
    gennError("NOTE: This function has been deprecated as of GeNN 2.2. Please provide the correct initial values in \"addSynapsePopulation\" for all your variables and they will be the constant values in the GLOBALG mode.");
}


//--------------------------------------------------------------------------
/*! \brief This function sets a global input value to the specified neuron group.
 */
//--------------------------------------------------------------------------

void NNmodel::setConstInp(const string &sName, /**<  */
                          double globalInp0 /**<  */)
{
    gennError("This function has been deprecated as of GeNN 2.2. Use parameters in the neuron model instead.");
}


//--------------------------------------------------------------------------
/*! \brief This function sets the integration time step DT of the model
 */
//--------------------------------------------------------------------------

void NNmodel::setDT(double newDT /**<  */)
{
    if (final) {
        gennError("Trying to set DT in a finalized model.");
    }
    dt = newDT;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the numerical precision of floating type variables. By default, it is GENN_GENN_FLOAT.
 */
//--------------------------------------------------------------------------

void NNmodel::setPrecision(FloatType floattype /**<  */)
{
    if (final) {
        gennError("Trying to set the precision of a finalized model.");
    }
    switch (floattype) {
    case GENN_FLOAT:
        ftype = "float";
        break;
    case GENN_DOUBLE:
        ftype = "double"; // not supported by compute capability < 1.3
        break;
    case GENN_LONG_DOUBLE:
        ftype = "long double"; // not supported by CUDA at the moment.
        break;
    default:
        gennError("Unrecognised floating-point type.");
    }
}


//--------------------------------------------------------------------------
/*! \brief This function sets a flag to determine whether timers and timing commands are to be included in generated code.
 */
//--------------------------------------------------------------------------

void NNmodel::setTiming(bool theTiming /**<  */)
{
    if (final) {
        gennError("Trying to set timing flag in a finalized model.");
    }
    timing= theTiming;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the random seed. If the passed argument is > 0, automatic seeding is disabled. If the argument is 0, the underlying seed is obtained from the time() function.
 */
//--------------------------------------------------------------------------

void NNmodel::setSeed(unsigned int inseed /*!< the new seed  */)
{
    if (final) {
        gennError("Trying to set the random seed in a finalized model.");
    }
    seed= inseed;
}


#ifndef CPU_ONLY
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
  if (device == -1) GENN_PREFERENCES::autoChooseDevice= 1;
  else {
      GENN_PREFERENCES::autoChooseDevice= 0;
      GENN_PREFERENCES::defaultDevice= device;
  }
}
#endif


string NNmodel::scalarExpr(const double val) const
{
    string tmp;
    float fval= (float) val;
    if (ftype == "float") {
        tmp= tS(fval) + "f";
    }
    if (ftype == "double") {
        tmp= tS(val);
    }
    return tmp;
}


//--------------------------------------------------------------------------
/*! \brief Accumulate the sums and block-size-padded sums of all simulation groups.

  This method saves the neuron numbers of the populations rounded to the next multiple of the block size as well as the sums s(i) = sum_{1...i} n_i of the rounded population sizes. These are later used to determine the branching structure for the generated neuron kernel code. 
*/
//--------------------------------------------------------------------------

void NNmodel::setPopulationSums()
{
    unsigned int paddedSize;
    if (!final) {
        gennError("Your model must be finalized before we can calculate population sums. Aborting.");
    }

    // NEURON GROUPS
    sumNeuronN.resize(neuronGrpN);
    padSumNeuronN.resize(neuronGrpN);
    for (int i = 0; i < neuronGrpN; i++) {
        // paddedSize is the lowest multiple of neuronBlkSz >= neuronN[i]
        paddedSize = ceil((double) neuronN[i] / (double) neuronBlkSz) * (double) neuronBlkSz;
        if (i == 0) {
            sumNeuronN[i] = neuronN[i];
            padSumNeuronN[i] = paddedSize;
        }
        else {
            sumNeuronN[i] = sumNeuronN[i - 1] + neuronN[i];
            padSumNeuronN[i] = padSumNeuronN[i - 1] + paddedSize;
        }
    }

    // SYNAPSE GROUPS
    padSumSynapseKrnl.resize(synapseGrpN);
    for (int i = 0; i < synapseGrpN; i++) {
        if (synapseConnType[i] == SPARSE) {
            if (synapseSpanType[i] == 1) {
                // paddedSize is the lowest multiple of synapseBlkSz >= neuronN[synapseSource[i]
                paddedSize = ceil((double) neuronN[synapseSource[i]] / (double) synapseBlkSz) * (double) synapseBlkSz;
            }
            else {
                // paddedSize is the lowest multiple of synapseBlkSz >= maxConn[i]
                paddedSize = ceil((double) maxConn[i] / (double) synapseBlkSz) * (double) synapseBlkSz;
            }
        }
        else {
            // paddedSize is the lowest multiple of synapseBlkSz >= neuronN[synapseTarget[i]]
            paddedSize = ceil((double) neuronN[synapseTarget[i]] / (double) synapseBlkSz) * (double) synapseBlkSz;
        }
        if (i == 0) {
            padSumSynapseKrnl[i] = paddedSize;
        }
        else {
            padSumSynapseKrnl[i] = padSumSynapseKrnl[i - 1] + paddedSize;
        }
    }

    // SYNAPSE DYNAMICS GROUPS
    padSumSynDynN.resize(synDynGroups);
    for (int i = 0; i < synDynGroups; i++) {
        if (synapseConnType[i] == SPARSE) {
            // paddedSize is the lowest multiple of synDynBlkSz >= neuronN[synapseSource[i]] * maxConn[i]
            paddedSize = ceil((double) neuronN[synapseSource[i]] * maxConn[i] / (double) synDynBlkSz) * (double) synDynBlkSz;
        }
        else {
            // paddedSize is the lowest multiple of synDynBlkSz >= neuronN[synapseSource[i]] * neuronN[synapseTarget[i]]
            paddedSize = ceil((double) neuronN[synapseSource[i]] * neuronN[synapseTarget[i]] / (double) synDynBlkSz) * (double) synDynBlkSz;
        }
        if (i == 0) {
            padSumSynDynN[i] = paddedSize;
        }
        else {
            padSumSynDynN[i] = padSumSynDynN[i - 1] + paddedSize;
        }
    }

    // LEARN GROUPS
    padSumLearnN.resize(lrnGroups);
    for (int i = 0; i < lrnGroups; i++) {
        // paddedSize is the lowest multiple of learnBlkSz >= neuronN[synapseTarget[i]]
        paddedSize = ceil((double) neuronN[synapseSource[i]] / (double) learnBlkSz) * (double) learnBlkSz;
        if (i == 0) {
            padSumLearnN[i] = paddedSize;
        }
        else {
            padSumLearnN[i] = padSumLearnN[i - 1] + paddedSize;
        }
    }
}


//--------------------------------------------------------------------------
/*! \brief Method for calculating dependent parameter values from independent parameters.

This method is to be invoked when all independent parameters have been set.
It appends the derived values of dependent parameters to the corresponding vector (dnp) without checking for multiple calls. If called repeatedly, multiple copies of dependent parameters would be added leading to potential errors in the model execution.
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedNeuronPara()
{
    for (int i = 0; i < neuronGrpN; i++) {
        vector<double> tmpP;
        int numDpNames = nModels[neuronType[i]].dpNames.size();
        for (int j=0; j < nModels[neuronType[i]].dpNames.size(); ++j) {
            double retVal = nModels[neuronType[i]].dps->calculateDerivedParameter(j, neuronPara[i], dt);
            tmpP.push_back(retVal);
        }
        dnp.push_back(tmpP);
    }
}


//--------------------------------------------------------------------------
/*! \brief This function calculates dependent synapse parameters from independent synapse parameters.

  This method is to be invoked when all independent parameters have been set.
*/
//--------------------------------------------------------------------------

void NNmodel::initDerivedSynapsePara()
{
    for (int i = 0; i < synapseGrpN; i++) {
        vector<double> tmpP;
        unsigned int synt= synapseType[i];
        for (int j= 0; j < weightUpdateModels[synt].dpNames.size(); ++j) {
            double retVal = weightUpdateModels[synt].dps->calculateDerivedParameter(j, synapsePara[i], dt);
            tmpP.push_back(retVal);
        }
        assert(dsp_w.size() == i);
        dsp_w.push_back(tmpP);
    }
}


//--------------------------------------------------------------------------
/*! \brief This function calculates dependent synaptic parameters in the employed post-synaptic model based on the independent post-synapse parameters.

  This method is to be invoked when all independent parameters have been set.
 */
//--------------------------------------------------------------------------

void NNmodel::initDerivedPostSynapsePara()
{
    for (int i = 0; i < synapseGrpN; i++) {
        vector<double> tmpP;
        unsigned int psynt= postSynapseType[i];
        for (int j=0; j < postSynModels[psynt].dpNames.size(); ++j) {
            double retVal = postSynModels[psynt].dps->calculateDerivedParameter(j, postSynapsePara[i], dt);
            tmpP.push_back(retVal);
        }
        assert(dpsp.size() == i);
        dpsp.push_back(tmpP);
    }
}


void NNmodel::finalize()
{
    //initializing learning parameters to start
    if (final) {
        gennError("Your model has already been finalized");
    }
    final= 1;
    initDerivedNeuronPara();
    initDerivedSynapsePara();
    initDerivedPostSynapsePara();
    initLearnGrps();
    setPopulationSums();
}

#endif // MODELSPEC_CC
