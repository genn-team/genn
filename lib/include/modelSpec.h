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
#define _MODELSPEC_H_ //!< macro for avoiding multiple inclusion during compilation

//--------------------------------------------------------------------------
/*! \file modelSpec.h

\brief Header file that contains the class (struct) definition of neuronModel for defining a neuron model and the class definition of NNmodel for defining a neuronal network model. Part of the code generation and generated code sections.
*/
//--------------------------------------------------------------------------

#include <vector>
#include "toString.h"

//neuronType
#define MAPNEURON 0 //!< Macro attaching the name "MAPNEURON" to neuron type 0
#define POISSONNEURON 1 //!< Macro attaching the name "POISSONNEURON" to neuron type 1
#define TRAUBMILES 2 //!< Macro attaching the name "TRAUBMILES" to neuron type 2
#define IZHIKEVICH 3 //!< Macro attaching the name "IZHIKEVICH" to neuron type 3

#define SYNTYPENO 4

//synapseType
#define NSYNAPSE 0 //!< Macro attaching  the name NSYNAPSE to predefined synapse type 0, which is a non-learning synapse

#define NGRADSYNAPSE 1 //!< Macro attaching  the name NGRADSYNAPSE to predefined synapse type 1 which is a graded synapse wrt the presynaptic voltage

#define LEARN1SYNAPSE 2 //!< Macro attaching  the name LEARN1SYNAPSE to the predefined synapse type 2 which is a learning using spike timing; uses a primitive STDP rule for learning

#define USERDEFSYNAPSE 3 //!< Macro attaching  the name USERDEFSYNAPSE to the predefined synapse type 3 which is a user-defined synapse

//input type
#define NOINP 0 //!< Macro attaching  the name NOINP (no input) to 0
#define CONSTINP 1 //!< Macro attaching  the name CONSTINP (constant input) to 1
#define MATINP 2 //!< Macro attaching  the name MATINP (explicit input defined as a matrix) to 2
#define INPRULE 3 //!< Macro attaching  the name INPRULE (explicit dynamic input defined as a rule) to 3

unsigned int SYNPNO[SYNTYPENO]= {
  3,        // NSYNAPSE_PNO 
  4,        // NGRADSYNAPSE_PNO 
  13,       // LEARN1SYNAPSE_PNO 
  1			// USERDEFSYNAPSE_PNO 
}; //!< Global constant integer array containing the number of parameters of each of the predefined synapse types

//connectivity of the network (synapseConnType)
#define ALLTOALL 0  //!< Macro attaching the label "ALLTOALL" to connectivity type 0 
#define DENSE 1 //!< Macro attaching the label "DENSE" to connectivity type 1
#define SPARSE 2//!< Macro attaching the label "SPARSE" to connectivity type 2

//conductance type (synapseGType)
#define INDIVIDUALG 0  //!< Macro attaching the label "INDIVIDUALG" to method 0 for the definition of synaptic conductances

#define GLOBALG 1 //!< Macro attaching the label "GLOBALG" to method 1 for the definition of synaptic conductances

#define INDIVIDUALID 2 //!< Macro attaching the label "INDIVIDUALID" to method 2 for the definition of synaptic conductances

#define NOLEARNING 0 //!< Macro attaching the label "NOLEARNING" to flag 0 
#define LEARNING 1 //!< Macro attaching the label "LEARNING" to flag 1 

#define EXITSYN 0 //!< Macro attaching the label "EXITSYN" to flag 0 (excitatory synapse)
#define INHIBSYN 1 //!< Macro attaching the label "INHIBSYN" to flag 1 (inhibitory synapse)

#define TRUE 1 //!< Macro attaching the label "TRUE" to value 1
#define FALSE 0 //!< Macro attaching the label "FALSE" to value 1

#define CPU 0 //!< Macro attaching the label "CPU" to flag 0
#define GPU 1 //!< Macro attaching the label "GPU" to flag 1

// for purposes of STDP
#define SPK_THRESH 0.0f //!< Macro defining the spiking threshold for the purposes of STDP 
//#define MAXSPKCNT 50000


//! \brief class (struct) for specifying a neuron model.
struct neuronModel
{
  string simCode; /*!< \brief Code that defines the execution of one timestep of integration of the neuron model
		    
		    The code will refer to $(NN) for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain $(ISYN), if it is to receive input. */
  string thresholdCode; /*!< \brief Code that defines the threshold condition for spikes in the described neuron model */
  string resetCode; /*!< \brief Code that defines the reset action taken after a spike occurred. This can be empty */
  vector<string> varNames; //!< Names of the variables in the neuron model
  vector<string> tmpVarNames; //!< never used
  vector<string> varTypes; //!< Types of the variable named above, e.g. "float". Names and types are matched by their order of occurrence in the vector.
  vector<string> tmpVarTypes; //!< never used
  vector<string> pNames; //!< Names of (independent) parameters of the model. These are assumed to be always of type "float"
  vector<string> dpNames; /*!< \brief Names of dependent parameters of the model. These are assumed to be always of type "float"
			    
			    The dependent parameters are functions of independent parameters that enter into the neuron model. To avoid unecessary computational overhead, these parameters are calculated at compile time and inserted as explicit values into the generated code. See method NNmodel::initDerivedNeuronPara for how this is done.
			  */ 
};


//! \brief class NNmodel for specifying a neuronal network model.

class NNmodel
{
 public:
  string name; //!< Name of the neuronal newtwork model
  int valid; //!< Flag for whether the model has been validated (unused?)
  unsigned int needSt; //!< Whether last spike times are needed at all in this network model (related to STDP)
  unsigned int neuronGrpN; //!< Number of neuron groups
  vector<string> neuronName; //!< Names of neuron groups
  vector<unsigned int>neuronN; //!< Number of neurons in group
  vector<unsigned int>sumNeuronN; //!< Summed neuron numbers
  vector<unsigned int>padSumNeuronN; //!< Padded summed neuron numbers
  vector<unsigned int>neuronType; //!< Types of neurons
  vector<vector<float> >neuronPara; //!< Parameters of neurons
  vector<unsigned int>neuronNeedSt; //!< Whether last spike time needs to be saved for each indivual neuron type
  vector<vector<float> >dnp; //!< Derived neuron parameters
  vector<vector<float> >neuronIni; //!< Initial values of neurons
  vector<int> receivesInputCurrent; //!< flags whether neurons of a population receive explicit input currents
  vector<vector<unsigned int> >inSyn; //!< The ids of the incoming synapse groups
  vector<float>nThresh; //!< Threshold for spiking for each neuron type
  unsigned int synapseGrpN; //!< Number of synapse groups
  unsigned int lrnGroups; //!< Number of synapse groups with learning
  vector<string> synapseName; //!< Names of synapse groups
  vector<unsigned int>sumSynapseTrgN; //!< Summed naumber of target neurons
  vector<unsigned int>padSumSynapseTrgN; //!< "Padded" summed target neuron numbers
  vector<unsigned int>padSumLearnN; //!< Padded summed neuron numbers of learn group source populations
  vector<unsigned int>lrnSynGrp; //!< Enumeration of the IDs of synapse groups that learn
  vector<unsigned int>synapseType; //!< Types of synapses
  vector<unsigned int>synapseSource; //!< Presynaptic neuron groups
  vector<unsigned int>synapseTarget; //!< Postsynaptic neuron groups
  vector<vector<float> >synapsePara; //!< parameters of synapses
  vector<unsigned int>synapseConnType; //!< Connectivity type of synapses
  vector<unsigned int>synapseGType; //!< Type of specification method for synaptic conductance
  vector<float>g0; //!< Global synapse conductance if GLOBALG is chosen.
  vector<float>globalInp; //!< Global explicit input if CONSTINP is chosen.
  vector<unsigned int>synapseInSynNo; //!< IDs of the target neurons' incoming synapse variables for each synapse group
  vector<vector<float> >dsp;  //!> Derived synapse parameters

 private:
  void setNeuronName(unsigned int, const string); //!< Never used
  void setSynapseName(unsigned int, const string); //!< Never used
  void setNeuronN(unsigned int, unsigned int); //!< Never used
  void setNeuronType(unsigned int, unsigned int); //!< Never used
  void setNeuronPara(unsigned int, float*); //!< Never used
  void setNeuronIni(unsigned int, float*); //!< Never used
  void setSynapseType(unsigned int, unsigned int); //!< Never used
  void setSynapseSource(unsigned int, unsigned int); //!< Never used
  void setSynapseTarget(unsigned int, unsigned int); //!< Never used
  void setSynapsePara(unsigned int, float*); //!< Never used
  void setSynapseConnType(unsigned int, unsigned int); //!< Never used
  void setSynapseGType(unsigned int, unsigned int); //!< Never used
  void initDerivedNeuronPara(unsigned int); //!< Method for calculating the values of derived neuron parameters.
  void initNeuronSpecs(unsigned int); //!< Method for calculating neuron IDs, taking into account the blocksize padding between neuron populations; also initializes nThresh and neuronNeedSt for a population of neurons.
  void initDerivedSynapsePara(unsigned int); //!< Method for calculating the values of derived synapse parameters.
  unsigned int findNeuronGrp(const string); //!< Find the the ID number of a neuron group by its name 
  unsigned int findSynapseGrp(const string); //< Find the the ID number of a synapse group by its name  

 public:
  NNmodel();
  ~NNmodel();
  void setName(const string); //!< Method to set the neuronal network model name
  void addNeuronPopulation(const char *, unsigned int, unsigned int, float *, float *); //!< Method for adding a neuron population to a neuronal network model, using C style character array for the name of the population
  void addNeuronPopulation(const string, unsigned int, unsigned int, float *, float *); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population
  //void activateDirectInput(const char *, unsigned int);  
  void activateDirectInput(const string, unsigned int);  
  void addSynapsePopulation(const char *, unsigned int, unsigned int, unsigned int, const char *, const char *, float *); //!< Method for adding a synapse population to a neuronal network model, using C style character array for the name of the population
  void addSynapsePopulation(const string, unsigned int, unsigned int, unsigned int, const string, const string, float *); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population
  void setSynapseG(const string, float); //!< Method for setting the conductance (g) value for a synapse population with "GLOBALG" charactertistic
  void setConstInp(const string, float); //!< Method for setting the global input value for a neuron population if CONSTINP
};


// global variables
unsigned int neuronBlkSz; //!< Global variable that contains the block size for the neuron kernel function
unsigned int synapseBlkSz; //!< Global variable that contains the block size for the synapse kernel function
unsigned int learnBlkSz; //!< Global variable that contains the block size for the learning updates kernel for plastic synapses
unsigned int UIntSz; //!< size of the unsigned int variable type on the local architecture
unsigned int logUIntSz; //!< logarithm of the size of the unsigned int variable type on the local architecture

#endif
