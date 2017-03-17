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
#include "synapseGroup.h"
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
    const NeuronGroup *findNeuronGroup(const std::string &name) const;

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

    // PUBLIC SYNAPSE FUNCTIONS
    //=========================
    const map<string, SynapseGroup> &getSynapseGroups() const{ return m_SynapseGroups; }
    const SynapseGroup *findSynapseGroup(const std::string &name) const;

    const map<string, unsigned int> &getSynapsePostLearnGroups() const{ return m_SynapsePostLearnGroups; }
    const map<string, unsigned int> &getSynapseDynamicsGroups() const{ return m_SynapseDynamicsGroups; }


    bool isSynapseGroupDynamicsRequired(const std::string &name) const;
    bool isSynapseGroupPostLearningRequired(const std::string &name) const;

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


        auto srcNeuronGrp = findNeuronGroup(src);
        auto trgNeuronGrp = findNeuronGroup(trg);

        srcNeuronGrp->checkNumDelaySlots(delaySteps);
        if (delaySteps != NO_DELAY)
        {
            needSynapseDelay = true;
        }

        if (WeightUpdateModel::NeedsPreSpikeTime) {
            srcNeuronGrp->setSpikeTimeRequired();
            needSt = true;
        }
        if (WeightUpdateModel::NeedsPostSpikeTime) {
            trgNeuronGrp->setSpikeTimeRequired();
            needSt = true;
        }

        // Add synapse group
        auto result = m_SynapseGroups.insert(
            pair<string, SynapseGroup>(
                name, SynapseGroup(mtype, delaySteps,
                                   WeightUpdateModel::GetInstance(), weightParamValues.GetValues(), weightVarValues.GetValues(),
                                   PostsynapticModel::GetInstance(), postsynapticParamValues.GetValues(), postsynapticVarValues.GetValues(),
                                   src, srcNeuronGrp, trg, trgNeuronGrp)));

        if(!result.second)
        {
            gennError("Cannot add a synapse population with duplicate name:" + name);
        }

        trgNeuronGrp->addInSyn(name);
        srcNeuronGrp->addOutSyn(name);
    }

    void setSynapseG(const string&, double); //!< This function has been depreciated as of GeNN 2.2.
    void setMaxConn(const string&, unsigned int); //< Set maximum connections per neuron for the given group (needed for optimization by sparse connectivity)
    void setSpanTypeToPre(const string&); //!< Method for switching the execution order of synapses to pre-to-post
    void setSynapseClusterIndex(const string &synapseGroup, int hostID, int deviceID); //!< Function for setting which host and which device a synapse group will be simulated on
    void setSynapseWeightUpdateVarZeroCopy(const string &synapseGroup, const string &var);   //!< Function to specify that synapse group should use zero-copied memory for a particular weight update model state variable - May improve IO performance at the expense of kernel performance
    void setSynapsePostsynapticVarZeroCopy(const string &synapseGroup, const string &var);   //!< Function to specify that synapse group should use zero-copied memory for a particular postsynaptic model state variable - May improve IO performance at the expense of kernel performance

private:
    //--------------------------------------------------------------------------
    // Private neuron methods
    //--------------------------------------------------------------------------
    NeuronGroup *findNeuronGroup(const std::string &name);

    //--------------------------------------------------------------------------
    // Private synapse methods
    //--------------------------------------------------------------------------
    SynapseGroup *findSynapseGroup(const std::string &name);

    //--------------------------------------------------------------------------
    // Private members
    //--------------------------------------------------------------------------
    //!< Named neuron groups
    map<string, NeuronGroup> m_NeuronGroups;

    //!< Named synapse groups
    map<string, SynapseGroup> m_SynapseGroups;

    //!< Mapping  of synapse group names which have postsynaptic learning to their padded size
    //!< **THINK** is this the right container?
    map<string, unsigned int> m_SynapsePostLearnGroups;

    //!< Mapping of synapse group names which have synapse dynamics to their padded size
    //!< **THINK** is this the right container?
    map<string, unsigned int> m_SynapseDynamicsGroups;

    // Kernel members
    map<string, string> neuronKernelParameters;
    map<string, string> synapseKernelParameters;
    map<string, string> simLearnPostKernelParameters;
    map<string, string> synapseDynamicsKernelParameters;

};

#endif
