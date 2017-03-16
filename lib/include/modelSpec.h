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

#include "neuronGroup.h"
#include "neuronModels.h"
#include "newNeuronModels.h"
#include "newPostsynapticModels.h"
#include "newWeightUpdateModels.h"
#include "synapseModels.h"
#include "synapseMatrixType.h"
#include "postSynapseModels.h"
#include "utils.h"

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;


void initGeNN();
extern unsigned int GeNNReady;

// connectivity of the network (synapseConnType)
enum SynapseConnType
{
    ALLTOALL,
    DENSE,
    SPARSE,
};

// conductance type (synapseGType)
enum SynapseGType
{
    INDIVIDUALG,
    GLOBALG,
    INDIVIDUALID,
};


#define NO_DELAY 0 //!< Macro used to indicate no synapse delay for the group (only one queue slot will be generated)

#define NOLEARNING 0 //!< Macro attaching the label "NOLEARNING" to flag 0 
#define LEARNING 1 //!< Macro attaching the label "LEARNING" to flag 1 

#define EXITSYN 0 //!< Macro attaching the label "EXITSYN" to flag 0 (excitatory synapse)
#define INHIBSYN 1 //!< Macro attaching the label "INHIBSYN" to flag 1 (inhibitory synapse)

#define CPU 0 //!< Macro attaching the label "CPU" to flag 0
#define GPU 1 //!< Macro attaching the label "GPU" to flag 1

// Floating point precision to use for models
enum FloatType
{
    GENN_FLOAT,
    GENN_DOUBLE,
    GENN_LONG_DOUBLE,
};

#define AUTODEVICE -1  //!< Macro attaching the label AUTODEVICE to flag -1. Used by setGPUDevice


/*===============================================================
//! \brief class NNmodel for specifying a neuronal network model.
//
================================================================*/

class NNmodel
{
public:
    // Model members
    string name; //!< Name of the neuronal newtwork model
    string ftype; //!< Type of floating point variables (float, double, ...; default: float)
    string RNtype; //!< Underlying type for random number generation (default: long)
    double dt; //!< The integration time step of the model
    bool final; //!< Flag for whether the model has been finalized
    bool needSt; //!< Whether last spike times are needed at all in this network model (related to STDP)
    bool needSynapseDelay; //!< Whether delayed synapse conductance is required in the network
    bool timing;
    unsigned int seed;
    unsigned int resetKernel;  //!< The identity of the kernel in which the spike counters will be reset.

    // PUBLIC SYNAPSE VARIABLES
    //=========================
    unsigned int synapseGrpN; //!< Number of synapse groups
    vector<string> synapseName; //!< Names of synapse groups
    //vector<unsigned int>synapseNo; // !<numnber of synapses in a synapse group
    vector<unsigned int> maxConn; //!< Padded summed maximum number of connections for a neuron in the neuron groups
    vector<unsigned int> padSumSynapseKrnl; //Combination of padSumSynapseTrgN and padSumMaxConn to support both sparse and all-to-all connectivity in a model
    vector<const WeightUpdateModels::Base*> synapseModel; //!< Types of synapses
    vector<SynapseMatrixType> synapseMatrixType; //!< Connectivity type of synapses
    vector<unsigned int> synapseSpanType; //!< Execution order of synapses in the kernel. It determines whether synapses are executed in parallel for every postsynaptic neuron (0, default), or for every presynaptic neuron (1).
    vector<string> synapseSource; //!< Presynaptic neuron groups
    vector<string> synapseTarget; //!< Postsynaptic neuron groups
    vector<unsigned int> synapseInSynNo; //!< IDs of the target neurons' incoming synapse variables for each synapse group
    vector<unsigned int> synapseOutSynNo; //!< The target neurons' outgoing synapse for each synapse group
    vector<bool> synapseUsesTrueSpikes; //!< Defines if synapse update is done after detection of real spikes (only one point after threshold)
    vector<bool> synapseUsesSpikeEvents; //!< Defines if synapse update is done after detection of spike events (every point above threshold)
    vector<bool> synapseUsesPostLearning; //!< Defines if anything is done in case of postsynaptic neuron spiking before presynaptic neuron (punishment in STDP etc.)
    vector<bool> synapseUsesSynapseDynamics; //!< Defines if there is any continuos synapse dynamics defined
    vector<bool> needEvntThresholdReTest; //!< Defines whether the Evnt Threshold needs to be retested in the synapse kernel due to multiple non-identical events in the pre-synaptic neuron population
    vector<vector<double>> synapsePara; //!< parameters of synapses
    vector<vector<double>> synapseIni; //!< Initial values of synapse variables
    vector<vector<double>> dsp_w;  //!< Derived synapse parameters (weightUpdateModel only)
    vector<const PostsynapticModels::Base*> postSynapseModel; //!< Types of post-synaptic model
    vector<vector<double>> postSynapsePara; //!< parameters of postsynapses
    vector<vector<double>> postSynIni; //!< Initial values of postsynaptic variables
    vector<vector<double>> dpsp;  //!< Derived postsynapse parameters
    unsigned int lrnGroups; //!< Number of synapse groups with learning
    vector<unsigned int> padSumLearnN; //!< Padded summed neuron numbers of learn group source populations
    vector<unsigned int> lrnSynGrp; //!< Enumeration of the IDs of synapse groups that learn
    vector<unsigned int> synapseDelay; //!< Global synaptic conductance delay for the group (in time steps)
    unsigned int synDynGroups; //!< Number of synapse groups that define continuous synapse dynamics
    vector<unsigned int> synDynGrp; //!< Enumeration of the IDs of synapse groups that have synapse Dynamics
    vector<unsigned int> padSumSynDynN; //!< Padded summed neuron numbers of synapse dynamics group source populations
    vector<int> synapseHostID; //!< The ID of the cluster node which the synapse groups are computed on
    vector<int> synapseDeviceID; //!< The ID of the CUDA device which the synapse groups are comnputed on
    vector<set<string>> synapseVarZeroCopy; //!< Whether indidividual weight update model state variables of a synapse group should use zero-copied memory
    vector<set<string>> postSynapseVarZeroCopy; //!< Whether indidividual post synapse model state variables of a synapse group should use zero-copied memory

    // PUBLIC KERNEL PARAMETER VARIABLES
    //==================================



public:
    // PUBLIC MODEL FUNCTIONS
    //=======================
    NNmodel();
    ~NNmodel();
    void setName(const std::string&); //!< Method to set the neuronal network model name
    void setPrecision(FloatType); //!< Set numerical precision for floating point
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

    bool zeroCopyInUse() const;

    // PUBLIC NEURON FUNCTIONS
    //========================
    const map<string, NeuronGroup> &getNeuronGroups() const{ return m_NeuronGroups; }

    void addNeuronPopulation(const string&, unsigned int, unsigned int, const double *, const double *); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population
    void addNeuronPopulation(const string&, unsigned int, unsigned int, const vector<double>&, const vector<double>&); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population

    template<typename NeuronModel>
    void addNeuronPopulation(const string &name, unsigned int size,
                            const typename NeuronModel::ParamValues &paramValues, const typename NeuronModel::VarValues &varValues)
    {
        if (!GeNNReady) {
            gennError("You need to call initGeNN first.");
        }
        if (final) {
            gennError("Trying to add a neuron population to a finalized model.");
        }

        // Add neuron group
        auto result = m_NeuronGroups.insert(
            pair<string, NeuronGroup>(
                name, NeuronGroup(size, NeuronModel::GetInstance(),
                                  paramValues.GetValues(), varValues.GetValues())));

        if(!result.second)
        {
            gennError("Cannot add a neuron population with duplicate name:" + name);
        }

    }

    void setNeuronClusterIndex(const string &neuronGroup, int hostID, int deviceID); //!< Function for setting which host and which device a neuron group will be simulated on
    void setNeuronSpikeZeroCopy(const string &neuronGroup);   //!< Function to specify that neuron group should use zero-copied memory for its spikes - May improve IO performance at the expense of kernel performance
    void setNeuronSpikeEventZeroCopy(const string &neuronGroup);   //!< Function to specify that neuron group should use zero-copied memory for its spike-like events - May improve IO performance at the expense of kernel performance
    void setNeuronSpikeTimeZeroCopy(const string &neuronGroup);   //!< Function to specify that neuron group should use zero-copied memory for its spike times - May improve IO performance at the expense of kernel performance
    void setNeuronVarZeroCopy(const string &neuronGroup, const string &var);   //!< Function to specify that neuron group should use zero-copied memory for a particular state variable - May improve IO performance at the expense of kernel performance

    void activateDirectInput(const string&, unsigned int type); //! This function has been deprecated in GeNN 2.2
    void setConstInp(const string&, double);

    const NeuronGroup *findNeuronGroup(const std::string &name) const;


    // PUBLIC SYNAPSE FUNCTIONS
    //=========================

    void addSynapsePopulation(const string &name, unsigned int syntype, SynapseConnType conntype, SynapseGType gtype, const string& src, const string& trg, const double *p); //!< This function has been depreciated as of GeNN 2.2.
    void addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *); //!< Overloaded version without initial variables for synapses
    void addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *, const double *); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population
    void addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&,
                              const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population

    template<typename WeightUpdateModel, typename PostsynapticModel>
    void addSynapsePopulation(const string &name, SynapseMatrixType mtype, unsigned int delaySteps, const string& src, const string& trg,
                              const typename WeightUpdateModel::ParamValues &weightParamValues, const typename WeightUpdateModel::VarValues &weightVarValues,
                              const typename PostsynapticModel::ParamValues &postsynapticParamValues, const typename PostsynapticModel::VarValues &postsynapticVarValues)
    {
        if (!GeNNReady) {
            gennError("You need to call initGeNN first.");
        }
        if (final) {
            gennError("Trying to add a synapse population to a finalized model.");
        }

        // Increase synapse group count
        synapseGrpN++;

        auto srcNeuronGrp = findNeuronGroup(src);
        auto trgNeuronGrp = findNeuronGroup(trg);

        synapseName.push_back(name);
        synapseModel.push_back(WeightUpdateModel::GetInstance());
        synapseMatrixType.push_back(mtype);
        synapseSource.push_back(src);
        synapseTarget.push_back(trg);
        synapseDelay.push_back(delaySteps);

        srcNeuronGrp->checkNumDelaySlots(delaySteps);
        if (delaySteps != NO_DELAY)
        {
            needSynapseDelay = true;
        }

        if (WeightUpdateModel::NeedsPreSpikeTime) {
            srcNeuronGrp->setNeedSpikeTiming();
            needSt = true;
        }
        if (WeightUpdateModel::NeedsPostSpikeTime) {
            trgNeuronGrp->setNeedSpikeTiming();
            needSt = true;
        }

        synapseIni.push_back(weightVarValues.GetValues());
        synapsePara.push_back(weightParamValues.GetValues());
        postSynapseModel.push_back(PostsynapticModel::GetInstance());
        postSynIni.push_back(postsynapticVarValues.GetValues());
        postSynapsePara.push_back(postsynapticParamValues.GetValues());

        synapseInSynNo.push_back(trgNeuronGrp->addInSyn(name));
        synapseOutSynNo.push_back(srcNeuronGrp->addOutSyn(name));

        maxConn.push_back(trgNeuronGrp->getNumNeurons());
        synapseSpanType.push_back(0);

        // By default zero-copy should be disabled
        synapseVarZeroCopy.push_back(set<string>());
        postSynapseVarZeroCopy.push_back(set<string>());

        // initially set synapase group indexing variables to device 0 host 0
        synapseDeviceID.push_back(0);
        synapseHostID.push_back(0);
    }

    void setSynapseG(const string&, double); //!< This function has been depreciated as of GeNN 2.2.
    void setMaxConn(const string&, unsigned int); //< Set maximum connections per neuron for the given group (needed for optimization by sparse connectivity)
    void setSpanTypeToPre(const string&); //!< Method for switching the execution order of synapses to pre-to-post
    void setSynapseClusterIndex(const string &synapseGroup, int hostID, int deviceID); //!< Function for setting which host and which device a synapse group will be simulated on
    void setSynapseWeightUpdateVarZeroCopy(const string &synapseGroup, const string &var);   //!< Function to specify that synapse group should use zero-copied memory for a particular weight update model state variable - May improve IO performance at the expense of kernel performance
    void setSynapsePostsynapticVarZeroCopy(const string &synapseGroup, const string &var);   //!< Function to specify that synapse group should use zero-copied memory for a particular postsynaptic model state variable - May improve IO performance at the expense of kernel performance

    void initLearnGrps();
    unsigned int findSynapseGrp(const string&) const; //< Find the the ID number of a synapse group by its name

private:
    //--------------------------------------------------------------------------
    // Private neuron methods
    //--------------------------------------------------------------------------
    void initDerivedNeuronParams(); //!< Method for calculating the values of derived neuron parameters.

    //--------------------------------------------------------------------------
    // Private synapse methods
    //--------------------------------------------------------------------------
    void initDerivedSynapsePara(); //!< Method for calculating the values of derived synapse parameters.
    void initDerivedPostSynapsePara(); //!< Method for calculating the values of derived postsynapse parameters.

    NeuronGroup *findNeuronGroup(const std::string &name);

    //--------------------------------------------------------------------------
    // Private members
    //--------------------------------------------------------------------------
    //!< Named neuron groups
    map<string, NeuronGroup> m_NeuronGroups;

    // Kernel members
    map<string, string> neuronKernelParameters;

    vector<string> synapseKernelParameters;
    vector<string> synapseKernelParameterTypes;
    vector<string> simLearnPostKernelParameters;
    vector<string> simLearnPostKernelParameterTypes;
    vector<string> synapseDynamicsKernelParameters;
    vector<string> synapseDynamicsKernelParameterTypes;
};

#endif
