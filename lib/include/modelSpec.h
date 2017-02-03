/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model declarations.
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file modelSpec.h

\brief Header file that contains the class (struct) definition of neuronModel for 
defining a neuron model and the class definition of NNmodel for defining a neuronal network model. 
Part of the code generation and generated code sections.
*/
//--------------------------------------------------------------------------

#ifndef _MODELSPEC_H_
#define _MODELSPEC_H_ //!< macro for avoiding multiple inclusion during compilation

#include "neuronModels.h"
#include "synapseModels.h"
#include "postSynapseModels.h"

#include <string>
#include <vector>

using namespace std;


void initGeNN();
extern unsigned int GeNNReady;

// connectivity of the network (synapseConnType)
#define ALLTOALL 0  //!< Macro attaching the label "ALLTOALL" to connectivity type 0 
#define DENSE 1 //!< Macro attaching the label "DENSE" to connectivity type 1
#define SPARSE 2//!< Macro attaching the label "SPARSE" to connectivity type 2

// conductance type (synapseGType)
#define INDIVIDUALG 0  //!< Macro attaching the label "INDIVIDUALG" to method 0 for the definition of synaptic conductances
#define GLOBALG 1 //!< Macro attaching the label "GLOBALG" to method 1 for the definition of synaptic conductances
#define INDIVIDUALID 2 //!< Macro attaching the label "INDIVIDUALID" to method 2 for the definition of synaptic conductances

#define NO_DELAY 0 //!< Macro used to indicate no synapse delay for the group (only one queue slot will be generated)

#define NOLEARNING 0 //!< Macro attaching the label "NOLEARNING" to flag 0 
#define LEARNING 1 //!< Macro attaching the label "LEARNING" to flag 1 

#define EXITSYN 0 //!< Macro attaching the label "EXITSYN" to flag 0 (excitatory synapse)
#define INHIBSYN 1 //!< Macro attaching the label "INHIBSYN" to flag 1 (inhibitory synapse)

#define CPU 0 //!< Macro attaching the label "CPU" to flag 0
#define GPU 1 //!< Macro attaching the label "GPU" to flag 1

#define GENN_FLOAT 0  //!< Macro attaching the label "GENN_FLOAT" to flag 0. Used by NNModel::setPrecision()
#define GENN_DOUBLE 1  //!< Macro attaching the label "GENN_DOUBLE" to flag 1. Used by NNModel::setPrecision()

#define AUTODEVICE -1  //!< Macro attaching the label AUTODEVICE to flag -1. Used by setGPUDevice


/*===============================================================
//! \brief class NNmodel for specifying a neuronal network model.
//
================================================================*/

class NNmodel
{
public:

  // PUBLIC MODEL VARIABLES
  //========================

  string name; //!< Name of the neuronal newtwork model
  string ftype; //!< Type of floating point variables (float, double, ...; default: float)
  string RNtype; //!< Underlying type for random number generation (default: long)
  double dt; //!< The integration time step of the model
  int final; //!< Flag for whether the model has been finalized
  unsigned int needSt; //!< Whether last spike times are needed at all in this network model (related to STDP)
  unsigned int needSynapseDelay; //!< Whether delayed synapse conductance is required in the network
  bool timing;
  unsigned int seed;
  unsigned int resetKernel;  //!< The identity of the kernel in which the spike counters will be reset.


  // PUBLIC NEURON VARIABLES
  //========================

  unsigned int neuronGrpN; //!< Number of neuron groups
  vector<string> neuronName; //!< Names of neuron groups
  vector<unsigned int> neuronN; //!< Number of neurons in group
  vector<unsigned int> sumNeuronN; //!< Summed neuron numbers
  vector<unsigned int> padSumNeuronN; //!< Padded summed neuron numbers
  vector<unsigned int> neuronPostSyn; //! Postsynaptic methods to the neuron
  vector<unsigned int> neuronType; //!< Types of neurons
  vector<vector<double> > neuronPara; //!< Parameters of neurons
  vector<vector<double> > dnp; //!< Derived neuron parameters
  vector<vector<double> > neuronIni; //!< Initial values of neurons
  vector<vector<unsigned int> > inSyn; //!< The ids of the incoming synapse groups
  vector<vector<unsigned int> > outSyn; //!< The ids of the outgoing synapse groups
  vector<bool> neuronNeedSt; //!< Whether last spike time needs to be saved for a group
  vector<bool> neuronNeedTrueSpk; //!< Whether spike-like events from a group are required
  vector<bool> neuronNeedSpkEvnt; //!< Whether spike-like events from a group are required
  vector<vector<bool> > neuronVarNeedQueue; //!< Whether a neuron variable needs queueing for syn code
  vector<string> neuronSpkEvntCondition; //!< Will contain the spike event condition code when spike events are used
  vector<unsigned int> neuronDelaySlots; //!< The number of slots needed in the synapse delay queues of a neuron group
  vector<int> neuronHostID; //!< The ID of the cluster node which the neuron groups are computed on
  vector<int> neuronDeviceID; //!< The ID of the CUDA device which the neuron groups are comnputed on


  // PUBLIC SYNAPSE VARIABLES
  //=========================

  unsigned int synapseGrpN; //!< Number of synapse groups
  vector<string> synapseName; //!< Names of synapse groups
  //vector<unsigned int>synapseNo; // !<numnber of synapses in a synapse group
  vector<unsigned int> maxConn; //!< Padded summed maximum number of connections for a neuron in the neuron groups
  vector<unsigned int> padSumSynapseKrnl; //Combination of padSumSynapseTrgN and padSumMaxConn to support both sparse and all-to-all connectivity in a model
  vector<unsigned int> synapseType; //!< Types of synapses
  vector<unsigned int> synapseConnType; //!< Connectivity type of synapses
  vector<unsigned int> synapseGType; //!< Type of specification method for synaptic conductance
  vector<unsigned int> synapseSpanType; //!< Execution order of synapses in the kernel. It determines whether synapses are executed in parallel for every postsynaptic neuron (0, default), or for every presynaptic neuron (1). 
  vector<unsigned int> synapseSource; //!< Presynaptic neuron groups
  vector<unsigned int> synapseTarget; //!< Postsynaptic neuron groups
  vector<unsigned int> synapseInSynNo; //!< IDs of the target neurons' incoming synapse variables for each synapse group
  vector<unsigned int> synapseOutSynNo; //!< The target neurons' outgoing synapse for each synapse group
  vector<bool> synapseUsesTrueSpikes; //!< Defines if synapse update is done after detection of real spikes (only one point after threshold)
  vector<bool> synapseUsesSpikeEvents; //!< Defines if synapse update is done after detection of spike events (every point above threshold)
  vector<bool> synapseUsesPostLearning; //!< Defines if anything is done in case of postsynaptic neuron spiking before presynaptic neuron (punishment in STDP etc.) 
  vector<bool> synapseUsesSynapseDynamics; //!< Defines if there is any continuos synapse dynamics defined
  vector<bool> needEvntThresholdReTest; //!< Defines whether the Evnt Threshold needs to be retested in the synapse kernel due to multiple non-identical events in the pre-synaptic neuron population
  vector<vector<double> > synapsePara; //!< parameters of synapses
  vector<vector<double> > synapseIni; //!< Initial values of synapse variables
  vector<vector<double> > dsp_w;  //!< Derived synapse parameters (weightUpdateModel only)
  vector<unsigned int> postSynapseType; //!< Types of post-synaptic model
  vector<vector<double> > postSynapsePara; //!< parameters of postsynapses
  vector<vector<double> > postSynIni; //!< Initial values of postsynaptic variables
  vector<vector<double> > dpsp;  //!< Derived postsynapse parameters
  unsigned int lrnGroups; //!< Number of synapse groups with learning
  vector<unsigned int> padSumLearnN; //!< Padded summed neuron numbers of learn group source populations
  vector<unsigned int> lrnSynGrp; //!< Enumeration of the IDs of synapse groups that learn
  vector<unsigned int> synapseDelay; //!< Global synaptic conductance delay for the group (in time steps)
  unsigned int synDynGroups; //!< Number of synapse groups that define continuous synapse dynamics
  vector<unsigned int> synDynGrp; //!< Enumeration of the IDs of synapse groups that have synapse Dynamics
  vector<unsigned int> padSumSynDynN; //!< Padded summed neuron numbers of synapse dynamics group source populations
  vector<int> synapseHostID; //!< The ID of the cluster node which the synapse groups are computed on
  vector<int> synapseDeviceID; //!< The ID of the CUDA device which the synapse groups are comnputed on


  // PUBLIC KERNEL PARAMETER VARIABLES
  //==================================

  vector<string> neuronKernelParameters;
  vector<string> neuronKernelParameterTypes;
  vector<string> synapseKernelParameters;
  vector<string> synapseKernelParameterTypes;
  vector<string> simLearnPostKernelParameters;
  vector<string> simLearnPostKernelParameterTypes;
  vector<string> synapseDynamicsKernelParameters;
  vector<string> synapseDynamicsKernelParameterTypes;
    
private:


  // PRIVATE NEURON FUNCTIONS
  //=========================

  void setNeuronName(unsigned int, const string); //!< Never used
  void setNeuronN(unsigned int, unsigned int); //!< Never used
  void setNeuronType(unsigned int, unsigned int); //!< Never used
  void setNeuronPara(unsigned int, double*); //!< Never used
  void setNeuronIni(unsigned int, double*); //!< Never used
  void initDerivedNeuronPara(); //!< Method for calculating the values of derived neuron parameters.


  // PRIVATE SYNAPSE FUNCTIONS
  //==========================

  void setSynapseName(unsigned int, const string); //!< Never used
  void setSynapseType(unsigned int, unsigned int); //!< Never used
  void setSynapseSource(unsigned int, unsigned int); //!< Never used
  void setSynapseTarget(unsigned int, unsigned int); //!< Never used
  void setSynapsePara(unsigned int, double*); //!< Never used
  void setSynapseConnType(unsigned int, unsigned int); //!< Never used
  void setSynapseGType(unsigned int, unsigned int); //!< Never used
  void initDerivedSynapsePara(); //!< Method for calculating the values of derived synapse parameters.
  void initDerivedPostSynapsePara(); //!< Method for calculating the values of derived postsynapse parameters.
  void registerSynapsePopulation(unsigned int); //!< Method to register a new synapse population with the inSyn list of the target neuron population

public:


  // PUBLIC MODEL FUNCTIONS
  //=======================

  NNmodel();
  ~NNmodel();
  void setName(const string); //!< Method to set the neuronal network model name
  void setPrecision(unsigned int); //!< Set numerical precision for floating point
  void setDT(double); //!< Set the integration step size of the model
  void setTiming(bool); //!< Set whether timers and timing commands are to be included
  void setSeed(unsigned int); //!< Set the random seed (disables automatic seeding if argument not 0).
  void checkSizes(unsigned int *, unsigned int *, unsigned int *); //< Check if the sizes of the initialized neuron and synapse groups are correct.
#ifndef CPU_ONLY
  void setGPUDevice(int); //!< Method to choose the GPU to be used for the model. If "AUTODEVICE' (-1), GeNN will choose the device based on a heuristic rule.
#endif
  string scalarExpr(const double) const;
  void setPopulationSums(); //!< Set the accumulated sums of lowest multiple of kernel block size >= group sizes for all simulated groups.
  void finalize(); //!< Declare that the model specification is finalised in modelDefinition().


  // PUBLIC NEURON FUNCTIONS
  //========================

  void addNeuronPopulation(const string&, unsigned int, unsigned int, const double *, const double *); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population
  void addNeuronPopulation(const string&, unsigned int, unsigned int, const vector<double>&, const vector<double>&); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population
  void setNeuronClusterIndex(const string &neuronGroup, int hostID, int deviceID); //!< Function for setting which host and which device a neuron group will be simulated on
  void activateDirectInput(const string&, unsigned int type); //! This function has been deprecated in GeNN 2.2
  void setConstInp(const string&, double);
  unsigned int findNeuronGrp(const string&) const; //!< Find the the ID number of a neuron group by its name
  

  // PUBLIC SYNAPSE FUNCTIONS
  //=========================

  void addSynapsePopulation(const string &name, unsigned int syntype, unsigned int conntype, unsigned int gtype, const string& src, const string& trg, const double *p); //!< This function has been depreciated as of GeNN 2.2.
  void addSynapsePopulation(const string&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *); //!< Overloaded version without initial variables for synapses
  void addSynapsePopulation(const string&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *, const double *); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population
  void addSynapsePopulation(const string&, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const string&, const string&,
                            const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population
  void setSynapseG(const string&, double); //!< This function has been depreciated as of GeNN 2.2.
  void setMaxConn(const string&, unsigned int); //< Set maximum connections per neuron for the given group (needed for optimization by sparse connectivity)
  void setSpanTypeToPre(const string&); //!< Method for switching the execution order of synapses to pre-to-post
  void setSynapseClusterIndex(const string &synapseGroup, int hostID, int deviceID); //!< Function for setting which host and which device a synapse group will be simulated on
  void initLearnGrps();
  unsigned int findSynapseGrp(const string&) const; //< Find the the ID number of a synapse group by its name
 
};

#endif
