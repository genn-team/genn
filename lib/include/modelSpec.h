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

// Wrappers to save typing when declaring VarInitialisers structures
template<typename Snippet>
inline NewModels::VarInit initVar(const typename Snippet::ParamValues &params)
{
    return NewModels::VarInit(Snippet::getInstance(), params.getValues());
}

inline NewModels::VarInit uninitialisedVar()
{
    return NewModels::VarInit(InitVarSnippet::Uninitialised::getInstance(), {});
}

/*===============================================================
//! \brief class NNmodel for specifying a neuronal network model.
//
================================================================*/

class NNmodel
{
public:
    // Typedefines
    //=======================
    typedef map<string, NeuronGroup>::value_type NeuronGroupValueType;
    typedef map<string, SynapseGroup>::value_type SynapseGroupValueType;


    NNmodel();
    ~NNmodel();

    // PUBLIC MODEL FUNCTIONS
    //=======================
    void setName(const std::string&); //!< Method to set the neuronal network model name

    void setPrecision(FloatType); //!< Set numerical precision for floating point
    void setDT(double); //!< Set the integration step size of the model
    void setTiming(bool); //!< Set whether timers and timing commands are to be included
    void setSeed(unsigned int); //!< Set the random seed (disables automatic seeding if argument not 0).
    void setRNType(const std::string &type); //! Sets the underlying type for random number generation (default: uint64_t)

#ifndef CPU_ONLY
    void setGPUDevice(int); //!< Method to choose the GPU to be used for the model. If "AUTODEVICE' (-1), GeNN will choose the device based on a heuristic rule.
#endif
    //! Get the string literal that should be used to represent a value in the model's floating-point type
    string scalarExpr(const double) const;

    void setPopulationSums(); //!< Set the accumulated sums of lowest multiple of kernel block size >= group sizes for all simulated groups.
    void finalize(); //!< Declare that the model specification is finalised in modelDefinition().

    //! Are any variables in any populations in this model using zero-copy memory?
    bool zeroCopyInUse() const;

    //! Do any populations or initialisation code in this model require an RNG?
    bool isRNGRequired() const;

    //! Gets the name of the neuronal network model
    const std::string &getName() const{ return name; }

    //! Gets the floating point numerical precision
    const std::string &getPrecision() const{ return ftype; }

    //! Which kernel should contain the reset logic? Specified in terms of GENN_FLAGS
    unsigned int getResetKernel() const{ return resetKernel; }

    //! Gets the model integration step size
    double getDT() const { return dt; }

    //! Get the random seed
    unsigned int getSeed() const { return seed; }

    //! Gets the underlying type for random number generation (default: uint64_t)
    const std::string &getRNType() const{ return RNtype; }

    //! Is the model specification finalized
    bool isFinalized() const{ return final; }

    //! Are timers and timing commands enabled
    bool isTimingEnabled() const{ return timing; }

    // PUBLIC NEURON FUNCTIONS
    //========================
    //! Get std::map containing all named NeuronGroup objects in model
    const map<string, NeuronGroup> &getNeuronGroups() const{ return m_NeuronGroups; }

    //! Gets std::map containing names and types of each parameter that should be passed through to the neuron kernel
    const map<string, string> &getNeuronKernelParameters() const{ return neuronKernelParameters; }

    //! Gets the size of the neuron kernel thread grid
    /*! This is calculated by adding together the number of threads required by
        each neuron population, padded to be a multiple of GPU's thread block size.*/
    unsigned int getNeuronGridSize() const;

    //! How many neurons make up the entire model
    unsigned int getNumNeurons() const;

    //! Find a neuron group by name
    const NeuronGroup *findNeuronGroup(const std::string &name) const;

    //! Find a neuron group by name
    NeuronGroup *findNeuronGroup(const std::string &name);

    NeuronGroup *addNeuronPopulation(const string&, unsigned int, unsigned int, const double *, const double *); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population
    NeuronGroup *addNeuronPopulation(const string&, unsigned int, unsigned int, const vector<double>&, const vector<double>&); //!< Method for adding a neuron population to a neuronal network model, using C++ string for the name of the population

     //! Adds a new neuron group to the model
    /*! \tparam NeuronModel type of neuron model (derived from NeuronModels::Base).
        \param name string containing unique name of neuron population.
        \param size integer specifying how many neurons are in the population.
        \param paramValues parameters for model wrapped in NeuronModel::ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.
        \return pointer to newly created NeuronGroup */
    template<typename NeuronModel>
    NeuronGroup *addNeuronPopulation(const string &name, unsigned int size, const NeuronModel *model,
                                     const typename NeuronModel::ParamValues &paramValues,
                                     const typename NeuronModel::VarValues &varInitialisers)
    {
        if (!GeNNReady) {
            gennError("You need to call initGeNN first.");
        }
        if (final) {
            gennError("Trying to add a neuron population to a finalized model.");
        }

        // Add neuron group
        auto result = m_NeuronGroups.emplace(std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(name, size, model,
                                  paramValues.getValues(), varInitialisers.getInitialisers()));

        if(!result.second)
        {
            gennError("Cannot add a neuron population with duplicate name:" + name);
            return NULL;
        }
        else
        {
            return &result.first->second;
        }
    }

    template<typename NeuronModel>
    NeuronGroup *addNeuronPopulation(const string &name, unsigned int size,
                                     const typename NeuronModel::ParamValues &paramValues, const typename NeuronModel::VarValues &varInitialisers)
    {
        return addNeuronPopulation<NeuronModel>(name, size, NeuronModel::getInstance(), paramValues, varInitialisers);
    }

    void setNeuronClusterIndex(const string &neuronGroup, int hostID, int deviceID); //!< Function for setting which host and which device a neuron group will be simulated on

    void activateDirectInput(const string&, unsigned int type); //! This function has been deprecated in GeNN 2.2
    void setConstInp(const string&, double);

    // PUBLIC SYNAPSE FUNCTIONS
    //=========================
    //! Get std::map containing all named SynapseGroup objects in model
    const map<string, SynapseGroup> &getSynapseGroups() const{ return m_SynapseGroups; }

    //! Get std::map containing names of synapse groups which require postsynaptic learning and their thread IDs within
    //! the postsynaptic learning kernel (padded to multiples of the GPU thread block size)
    const map<string, std::pair<unsigned int, unsigned int>> &getSynapsePostLearnGroups() const{ return m_SynapsePostLearnGroups; }

    //! Get std::map containing names of synapse groups which require synapse dynamics and their thread IDs within
    //! the synapse dynamics kernel (padded to multiples of the GPU thread block size)
    const map<string, std::pair<unsigned int, unsigned int>> &getSynapseDynamicsGroups() const{ return m_SynapseDynamicsGroups; }

    //! Gets std::map containing names and types of each parameter that should be passed through to the synapse kernel
    const map<string, string> &getSynapseKernelParameters() const{ return synapseKernelParameters; }

    //! Gets std::map containing names and types of each parameter that should be passed through to the postsynaptic learning kernel
    const map<string, string> &getSimLearnPostKernelParameters() const{ return simLearnPostKernelParameters; }

    //! Gets std::map containing names and types of each parameter that should be passed through to the synapse dynamics kernel
    const map<string, string> &getSynapseDynamicsKernelParameters() const{ return synapseDynamicsKernelParameters; }

    //! Gets the size of the synapse kernel thread grid
    /*! This is calculated by adding together the number of threads required by each
        synapse population's synapse kernel, padded to be a multiple of GPU's thread block size.*/
    unsigned int getSynapseKernelGridSize() const;

    //! Gets the size of the post-synaptic learning kernel thread grid
    /*! This is calculated by adding together the number of threads required by each
        synapse population's postsynaptic learning kernel, padded to be a multiple of GPU's thread block size.*/
    unsigned int getSynapsePostLearnGridSize() const;

    //! Gets the size of the synapse dynamics kernel thread grid
    /*! This is calculated by adding together the number of threads required by each
        synapse population's synapse dynamics kernel, padded to be a multiple of GPU's thread block size.*/
    unsigned int getSynapseDynamicsGridSize() const;

    //! Find a synapse group by name
    const SynapseGroup *findSynapseGroup(const std::string &name) const;

    //! Find a synapse group by name
    SynapseGroup *findSynapseGroup(const std::string &name);    

    //! Does named synapse group have synapse dynamics
    bool isSynapseGroupDynamicsRequired(const std::string &name) const;

    //! Does named synapse group have post-synaptic learning
    bool isSynapseGroupPostLearningRequired(const std::string &name) const;

    SynapseGroup *addSynapsePopulation(const string &name, unsigned int syntype, SynapseConnType conntype, SynapseGType gtype, const string& src, const string& trg, const double *p); //!< This function has been depreciated as of GeNN 2.2.
    SynapseGroup *addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *); //!< Overloaded version without initial variables for synapses
    SynapseGroup *addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&, const double *, const double *, const double *, const double *); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population
    SynapseGroup *addSynapsePopulation(const string&, unsigned int, SynapseConnType, SynapseGType, unsigned int, unsigned int, const string&, const string&,
                              const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&); //!< Method for adding a synapse population to a neuronal network model, using C++ string for the name of the population

    /*! \tparam WeightUpdateModel type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name string containing unique name of neuron population.
        \param mtype how the synaptic matrix associated with this synapse population should be represented.
        \param delayStep integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)
        \param src string specifying name of presynaptic (source) population
        \param trg string specifying name of postsynaptic (target) population
        \param weightParamValues parameters for weight update model wrapped in WeightUpdateModel::ParamValues object.
        \param weightVarInitialisers weight update model state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.
        \param postsynapticParamValues parameters for postsynaptic model wrapped in PostsynapticModel::ParamValues object.
        \param postsynapticVarInitialisers postsynaptic model state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.
        \return pointer to newly created SynapseGroup */
    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const string &name, SynapseMatrixType mtype, unsigned int delaySteps, const string& src, const string& trg,
                                       const WeightUpdateModel *wum, const typename WeightUpdateModel::ParamValues &weightParamValues, const typename WeightUpdateModel::VarValues &weightVarInitialisers,
                                       const PostsynapticModel *psm, const typename PostsynapticModel::ParamValues &postsynapticParamValues, const typename PostsynapticModel::VarValues &postsynapticVarInitialisers)
    {
        if (!GeNNReady) {
            gennError("You need to call initGeNN first.");
        }
        if (final) {
            gennError("Trying to add a synapse population to a finalized model.");
        }

        auto srcNeuronGrp = findNeuronGroup(src);
        auto trgNeuronGrp = findNeuronGroup(trg);

        // Add synapse group
        auto result = m_SynapseGroups.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(name, mtype, delaySteps,
                                  wum, weightParamValues.getValues(), weightVarInitialisers.getInitialisers(),
                                  psm, postsynapticParamValues.getValues(), postsynapticVarInitialisers.getInitialisers(),
                                  srcNeuronGrp, trgNeuronGrp));

        if(!result.second)
        {
            gennError("Cannot add a synapse population with duplicate name:" + name);
            return NULL;
        }
        else
        {
            return &result.first->second;
        }
    }

    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const string &name, SynapseMatrixType mtype, unsigned int delaySteps, const string& src, const string& trg,
                                       const typename WeightUpdateModel::ParamValues &weightParamValues, const typename WeightUpdateModel::VarValues &weightVarInitialisers,
                                       const typename PostsynapticModel::ParamValues &postsynapticParamValues, const typename PostsynapticModel::VarValues &postsynapticVarInitialisers)
    {
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    WeightUpdateModel::getInstance(), weightParamValues, weightVarInitialisers,
                                    PostsynapticModel::getInstance(), postsynapticParamValues, postsynapticVarInitialisers);

    }

    void setSynapseG(const string&, double); //!< This function has been depreciated as of GeNN 2.2.
    void setMaxConn(const string&, unsigned int); //< Set maximum connections per neuron for the given group (needed for optimization by sparse connectivity)
    void setSpanTypeToPre(const string&); //!< Method for switching the execution order of synapses to pre-to-post
    void setSynapseClusterIndex(const string &synapseGroup, int hostID, int deviceID); //!< Function for setting which host and which device a synapse group will be simulated on

private:
    //--------------------------------------------------------------------------
    // Private members
    //--------------------------------------------------------------------------
    //!< Named neuron groups
    map<string, NeuronGroup> m_NeuronGroups;

    //!< Named synapse groups
    map<string, SynapseGroup> m_SynapseGroups;

    //!< Mapping  of synapse group names which have postsynaptic learning to their start and end padded indices
    //!< **THINK** is this the right container?
    map<string, std::pair<unsigned int, unsigned int>> m_SynapsePostLearnGroups;

    //!< Mapping of synapse group names which have synapse dynamics to their start and end padded indices
    //!< **THINK** is this the right container?
    map<string, std::pair<unsigned int, unsigned int>> m_SynapseDynamicsGroups;

    // Kernel members
    map<string, string> neuronKernelParameters;
    map<string, string> synapseKernelParameters;
    map<string, string> simLearnPostKernelParameters;
    map<string, string> synapseDynamicsKernelParameters;

     // Model members
    string name;                //!< Name of the neuronal newtwork model
    string ftype;               //!< Type of floating point variables (float, double, ...; default: float)
    string RNtype;              //!< Underlying type for random number generation (default: uint64_t)
    double dt;                  //!< The integration time step of the model
    bool final;                 //!< Flag for whether the model has been finalized
    bool timing;
    unsigned int seed;
    unsigned int resetKernel;   //!< The identity of the kernel in which the spike counters will be reset.
};

#endif
