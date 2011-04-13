/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#ifndef _MODELSPEC_H_
#define _MODELSPEC_H_

#include <vector>
#include "toString.h"

#define NTYPENO 3
#define MAPNEURON 0
#define POISSONNEURON 1
#define TRAUBMILES 2

unsigned int NPNO[NTYPENO]= {
  4,        // MAPNEURON_PNO 
  4,         // POISSONNEURON_PNO 
  7
};

unsigned int NININO[NTYPENO]= {
  2,        // MAPNEURON_ININO
  3,        // POISSONNEURON_ININO
  4
};

#define SYNTYPENO 3
#define NSYNAPSE 0
#define NGRADSYNAPSE 1
#define LEARN1SYNAPSE 2

unsigned int SYNPNO[SYNTYPENO]= {
  3,        // NSYNAPSE_PNO 
  4,        // NGRADSYNAPSE_PNO 
  13        // LEARN1SYNAPSE_PNO 
};

#define ALLTOALL 0
#define DENSE 1
//#define SPARSE 2

#define INDIVIDUALG 0
#define GLOBALG 1
#define INDIVIDUALID 2

#define NOLEARNING 0
#define LEARNING 1

#define EXITSYN 0
#define INHIBSYN 1

#define TRUE 1
#define FALSE 0

#define CPU 0
#define GPU 1

// for purposes of STDP
#define SPK_THRESH 0.0f
//#define MAXSPKCNT 50000


// class for specifying a neuron model
struct neuronModel
{
  string simCode; // needs to contain $(ISYN) if to receive input
  vector<string> varNames;
  vector<string> tmpVarNames;
  vector<string> varTypes;
  vector<string> tmpVarTypes;
  vector<string> pNames;
  vector<string> dpNames;
};


// class NNmodel for specifying a neuronal network model

class NNmodel
{
 public:
  string name;
  int valid;
  unsigned int needSt;
  unsigned int neuronGrpN; // number of neuron groups
  vector<string> neuronName; // names of neuron groups
  vector<unsigned int>neuronN; // number of neurons in group
  vector<unsigned int>sumNeuronN; // summed neuron numbers
  vector<unsigned int>padSumNeuronN; // padded summed neuron numbers
  vector<unsigned int>neuronType; // type of neurons
  vector<vector<float> >neuronPara; // parameters of neurons
  vector<unsigned int>neuronNeedSt; // whether last spike time needs to be saved
  vector<vector<float> >dnp; // derived neuron parameters
  vector<vector<float> >neuronIni; // parameters of neurons
  vector<vector<unsigned int> >inSyn; // the ids of the incoming synapse groups
  vector<float>nThresh; // neuron threshold for spiking
  unsigned int synapseGrpN; // number of synapse groups
  unsigned int lrnGroups; // number of synapse groups with learning
  vector<string> synapseName; // names of synapse groups
  vector<unsigned int>sumSynapseTrgN; // summed target neuron numbers
  vector<unsigned int>padSumSynapseTrgN; // padded summed target neuron numbers
  vector<unsigned int>padSumLearnN; // padded saummed neurons numbers of learn group srcs
  vector<unsigned int>lrnSynGrp; // enumeration of syn groups that learn
  vector<unsigned int>synapseType; // type of synapses
  vector<unsigned int>synapseSource; // presynaptic neuron group
  vector<unsigned int>synapseTarget; // postsynaptic neuron group
  vector<vector<float> >synapsePara; // parameters of neurons
  vector<unsigned int>synapseConnType; // connectivity type of synapses
  vector<unsigned int>synapseGType; // connectivity type of synapses
  vector<float>g0; // synapse conductance if GLOBALG
  vector<unsigned int>synapseInSynNo; // id of the target neurons incoming synapse variables for each synapse group
  vector<vector<float> >dsp;  // derived synapse parameters

 private:
  void setNeuronName(unsigned int, const string);
  void setSynapseName(unsigned int, const string);
  void setNeuronN(unsigned int, unsigned int);
  void setNeuronType(unsigned int, unsigned int);
  void setNeuronPara(unsigned int, float*);
  void setNeuronIni(unsigned int, float*);
  void setSynapseType(unsigned int, unsigned int);
  void setSynapseSource(unsigned int, unsigned int);
  void setSynapseTarget(unsigned int, unsigned int);
  void setSynapsePara(unsigned int, float*);
  void setSynapseConnType(unsigned int, unsigned int);
  void setSynapseGType(unsigned int, unsigned int);
  void initDerivedNeuronPara(unsigned int);
  void initDerivedSynapsePara(unsigned int);
  unsigned int findNeuronGrp(const string);
  unsigned int findSynapseGrp(const string);

 public:
  NNmodel();
  ~NNmodel();
  void setName(const string);
  void addNeuronPopulation(const string, unsigned int, unsigned int, float *, float *);
  void addSynapsePopulation(const string, unsigned int, unsigned int, unsigned int, const string, const string, float *);
  void setSynapseG(const string, float);
};


// global variables
unsigned int neuronBlkSz;
unsigned int logNeuronBlkSz;
unsigned int synapseBlkSz;
unsigned int logSynapseBlkSz;
unsigned int learnBlkSz;
unsigned int logLearnBlkSz;
unsigned int UIntSz;
unsigned int logUIntSz;
unsigned int theDev;

#endif
