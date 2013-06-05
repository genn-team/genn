/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model declarations.
  
--------------------------------------------------------------------------*/

#ifndef _MODELSPEC_H_
#define _MODELSPEC_H_

#include <vector>
#include "toString.h"

#define NTYPENO 3

//neuronType
#define MAPNEURON 0
#define POISSONNEURON 1
#define TRAUBMILES 2
#define IZHIKEVICH 3

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

//synapseType
#define NSYNAPSE 0 // no learning
#define NGRADSYNAPSE 1 // graded synapse wrt the presynaptic voltage
#define LEARN1SYNAPSE 2 // learning by spike timing: a primitive STDP rule

unsigned int SYNPNO[SYNTYPENO]= {
  3,        // NSYNAPSE_PNO 
  4,        // NGRADSYNAPSE_PNO 
  13        // LEARN1SYNAPSE_PNO 
};

//connectivity of the network (synapseConnType)
#define ALLTOALL 0
#define DENSE 1
//#define SPARSE 2

//conductance type (synapseGType)
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
  string simCode; // needs to contain $(ISYN) if to receive input. This is where the neuron model code is defined.
  vector<string> varNames;
  vector<string> tmpVarNames; //never used
  vector<string> varTypes;
  vector<string> tmpVarTypes; //never used
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
  void setNeuronName(unsigned int, const string); //never used
  void setSynapseName(unsigned int, const string);//never used
  void setNeuronN(unsigned int, unsigned int);//never used
  void setNeuronType(unsigned int, unsigned int);//never used
  void setNeuronPara(unsigned int, float*);//never used
  void setNeuronIni(unsigned int, float*);//never used
  void setSynapseType(unsigned int, unsigned int);//never used
  void setSynapseSource(unsigned int, unsigned int);//never used
  void setSynapseTarget(unsigned int, unsigned int);//never used
  void setSynapsePara(unsigned int, float*);//never used
  void setSynapseConnType(unsigned int, unsigned int);//never used
  void setSynapseGType(unsigned int, unsigned int);//never used
  void initDerivedNeuronPara(unsigned int);
  void initDerivedSynapsePara(unsigned int);
  unsigned int findNeuronGrp(const string);
  unsigned int findSynapseGrp(const string);

 public:
  NNmodel();
  ~NNmodel();
  void setName(const string);
  void addNeuronPopulation(const char *, unsigned int, unsigned int, float *, float *);
  void addNeuronPopulation(const string, unsigned int, unsigned int, float *, float *);
  void addSynapsePopulation(const char *, unsigned int, unsigned int, unsigned int, const char *, const char *, float *);
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
