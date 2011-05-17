/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

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



void NNmodel::initDerivedNeuronPara(unsigned int i)
{
  // to be called when all para have been set!
  // also, call this only once and right after a population has been added
  // as values are appended to vectors indiscriminantly
  // derived neuron parameters (dnp)
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
    cerr <<  padSumNeuronN[i] << endl;
  }
  else {
    sumNeuronN.push_back(sumNeuronN[i-1] + neuronN[i]); 
    padSumNeuronN.push_back(padSumNeuronN[i-1] + padnN); 
    cerr <<  padSumNeuronN[i] << endl;
  }
  neuronNeedSt.push_back(0);  // by default last spike times are not saved
}

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
    cerr <<  padSumSynapseTrgN[i] << endl;
  }
  else {
    sumSynapseTrgN.push_back(sumSynapseTrgN[i-1]+nN);
    padSumSynapseTrgN.push_back(padSumSynapseTrgN[i-1]+padnN);
    cerr <<  padSumSynapseTrgN[i] << endl;
  }
}

unsigned int NNmodel::findNeuronGrp(const string nName)
{
  for (int j= 0; j < neuronGrpN; j++) {
    if (nName == neuronName[j]) {
      return j;
    }
  }
  cerr << "neuron group " << nName << " not found, aborting ..." << endl;
  exit(1);
}

unsigned int NNmodel::findSynapseGrp(const string sName)
{
  for (int j= 0; j < synapseGrpN; j++) {
    if (sName == synapseName[j]) {
      return j;
    }
  }
  cerr << "synapse group " << sName << " not found, aborting ..." << endl;
  exit(1);
}


void NNmodel::addNeuronPopulation(const char *name, unsigned int nNo, unsigned int type, float *p, float *ini)
{
  addNeuronPopulation(toString(name), nNo, type, p, ini);
}


void NNmodel::addNeuronPopulation(const string name, unsigned int nNo, unsigned int type, float *p, float *ini)
{
  unsigned int i= neuronGrpN++;

  neuronName.push_back(toString(name));
  neuronN.push_back(nNo);
  neuronType.push_back(type);
  vector<float> tmpP;
  for (int j= 0; j < NPNO[neuronType[i]]; j++) {
    tmpP.push_back(p[j]);
  }
  neuronPara.push_back(tmpP);
  tmpP.clear();
  for (int j= 0; j < NININO[neuronType[i]]; j++) {
    tmpP.push_back(ini[j]);
  }
  neuronIni.push_back(tmpP);
  vector<unsigned int> tv;
  inSyn.push_back(tv);  // empty list of input synapse groups for neurons i
  initDerivedNeuronPara(i);
}

void NNmodel::addSynapsePopulation(const char *name, unsigned int syntype, unsigned int conntype, unsigned int gtype, const char *src, const char *trg, float *p) 
{
  addSynapsePopulation(toString(name), syntype, conntype, gtype, toString(src), toString(trg), p);
}

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

void NNmodel::setSynapseG(const string sName, float g)
{
  unsigned int found= findSynapseGrp(sName);
  if (g0.size() < found+1) g0.resize(found+1);
  g0[found]= g;
}
