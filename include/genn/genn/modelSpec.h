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
defining a neuron model and the class definition of ModelSpec for defining a neuronal network model. 
Part of the code generation and generated code sections.
*/
//--------------------------------------------------------------------------
#pragma once

// Standard C++ includes
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// GeNN includes
#include "currentSourceInternal.h"
#include "customUpdateInternal.h"
#include "gennExport.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

#define NO_DELAY 0 //!< Macro used to indicate no synapse delay for the group (only one queue slot will be generated)

using ParamValues = std::unordered_map<std::string, double>;
using VarValues = std::unordered_map<std::string, Models::VarInit>;

//! Floating point precision to use for models
enum FloatType
{
    GENN_FLOAT,
    GENN_DOUBLE,
    GENN_LONG_DOUBLE,
};

//! Precision to use for variables which store time
enum class TimePrecision
{
    DEFAULT,    //!< Time uses default model precision
    FLOAT,      //!< Time uses single precision - not suitable for long simulations
    DOUBLE,     //!< Time uses double precision - may reduce performance
};

//! Initialise a variable using an initialisation snippet
/*! \tparam S       type of variable initialisation snippet (derived from InitVarSnippet::Base).
    \param params   parameters for snippet wrapped in ParamValues object.
    \return         Models::VarInit object for use within model's VarValues*/
template<typename S>
inline Models::VarInit initVar(const ParamValues &params)
{
    return Models::VarInit(S::getInstance(), params);
}

//! Initialise a variable using an initialisation snippet
/*! \tparam S       type of variable initialisation snippet (derived from InitVarSnippet::Base).
    \return         Models::VarInit object for use within model's VarValues*/
template<typename S>
inline Models::VarInit initVar()
{
    return Models::VarInit(S::getInstance(), {});
}

//! Mark a variable as uninitialised
/*! This means that the backend will not generate any automatic initialization code, but will instead
    copy the variable from host to device during ``initializeSparse`` function */
inline Models::VarInit uninitialisedVar()
{
    return Models::VarInit(InitVarSnippet::Uninitialised::getInstance(), {});
}

//! Initialise connectivity using a sparse connectivity snippet
/*! \tparam S       type of sparse connectivitiy initialisation snippet (derived from InitSparseConnectivitySnippet::Base).
    \param params   parameters for snippet wrapped in ParamValues object.
    \return         InitSparseConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline InitSparseConnectivitySnippet::Init initConnectivity(const ParamValues &params)
{
    return InitSparseConnectivitySnippet::Init(S::getInstance(), params);
}

//! Initialise connectivity using a sparse connectivity snippet
/*! \tparam S       type of sparse connectivitiy initialisation snippet (derived from InitSparseConnectivitySnippet::Base).
    \return         InitSparseConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline InitSparseConnectivitySnippet::Init initConnectivity()
{
    return InitSparseConnectivitySnippet::Init(S::getInstance(), {});
}

//! Mark a synapse group's sparse connectivity as uninitialised
/*! This means that the backend will not generate any automatic initialization code, but will instead
    copy the connectivity from host to device during ``initializeSparse`` function
    (and, if necessary generate any additional data structures it requires)*/
inline InitSparseConnectivitySnippet::Init uninitialisedConnectivity()
{
    return InitSparseConnectivitySnippet::Init(InitSparseConnectivitySnippet::Uninitialised::getInstance(), {});
}

//! Initialise toeplitz connectivity using a toeplitz connectivity snippet
/*! \tparam S       type of toeplitz connectivitiy initialisation snippet (derived from InitToeplitzConnectivitySnippet::Base).
    \param params   parameters for snippet wrapped in ParamValues object.
    \return         InitToeplitzConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline InitToeplitzConnectivitySnippet::Init initToeplitzConnectivity(const ParamValues &params)
{
    return InitToeplitzConnectivitySnippet::Init(S::getInstance(), params);
}

//! Initialise toeplitz connectivity using a toeplitz connectivity snippet
/*! \tparam S       type of toeplitz connectivitiy initialisation snippet (derived from InitToeplitzConnectivitySnippet::Base).
    \return         InitToeplitzConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline InitToeplitzConnectivitySnippet::Init initToeplitzConnectivity()
{
    return InitToeplitzConnectivitySnippet::Init(S::getInstance(), {});
}

//! Creates a reference to a neuron group variable
inline Models::VarReference createVarRef(const NeuronGroup *ng, const std::string &varName)
{
    return Models::VarReference::createVarRef(ng, varName);
}

//! Creates a reference to a current source variable
inline Models::VarReference createVarRef(const CurrentSource *cs, const std::string &varName)
{
    return Models::VarReference::createVarRef(cs, varName);
}

//! Creates a reference to a custom update variable
inline Models::VarReference createVarRef(const CustomUpdate *cu, const std::string &varName)
{
    return Models::VarReference::createVarRef(cu, varName);
}

//! Creates a reference to a postsynaptic model variable
inline Models::VarReference createPSMVarRef(const SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createPSMVarRef(sg, varName);
}

//! Creates a reference to a weight update model presynaptic variable
inline Models::VarReference createWUPreVarRef(const SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createWUPreVarRef(sg, varName);
}

//! Creates a reference to a weight update model postsynapticvariable
inline Models::VarReference createWUPostVarRef(const SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createWUPostVarRef(sg, varName);
}

//! Creates a reference to a weight update model variable
inline Models::WUVarReference createWUVarRef(const SynapseGroup *sg, const std::string &varName,
                                             const SynapseGroup *transposeSG = nullptr, const std::string &transposeVarName = "")
{
    return Models::WUVarReference(sg, varName, transposeSG, transposeVarName);
}

//! Creates a reference to a custom weight update variable
inline Models::WUVarReference createWUVarRef(const CustomUpdateWU *cu, const std::string &varName)
{
    return Models::WUVarReference(cu, varName);
}

//----------------------------------------------------------------------------
// ModelSpec
//----------------------------------------------------------------------------
//! Object used for specifying a neuronal network model
class GENN_EXPORT ModelSpec
{
public:
    // Typedefines
    //=======================
    typedef std::map<std::string, NeuronGroupInternal>::value_type NeuronGroupValueType;
    typedef std::map<std::string, SynapseGroupInternal>::value_type SynapseGroupValueType;
    typedef std::map<std::string, CurrentSourceInternal>::value_type CurrentSourceValueType;
    typedef std::map<std::string, CustomUpdateInternal>::value_type CustomUpdateValueType;
    typedef std::map<std::string, CustomUpdateWUInternal>::value_type CustomUpdateWUValueType;

    //typedef std::map<std::string, CustomUpdateInternal>::value_type CustomUpdateValueType;

    ModelSpec();
    ModelSpec(const ModelSpec&) = delete;
    ModelSpec &operator=(const ModelSpec &) = delete;
    ~ModelSpec();

    // PUBLIC MODEL FUNCTIONS
    //=======================
    //! Method to set the neuronal network model name
    void setName(const std::string &name){ m_Name = name; }

    //! Set numerical precision for floating point
    void setPrecision(FloatType);

    //! Set numerical precision for time
    void setTimePrecision(TimePrecision timePrecision){ m_TimePrecision = timePrecision; }

    //! Set the integration step size of the model
    void setDT(double dt){ m_DT = dt; }

    //! Set whether timers and timing commands are to be included
    void setTiming(bool timingEnabled){ m_TimingEnabled = timingEnabled; }

    //! Set the random seed (disables automatic seeding if argument not 0).
    void setSeed(unsigned int rngSeed){ m_Seed = rngSeed; }

    //! What is the default location for model state variables?
    /*! Historically, everything was allocated on both the host AND device */
    void setDefaultVarLocation(VarLocation loc){ m_DefaultVarLocation = loc; }

    //! What is the default location for model extra global parameters?
    /*! Historically, this was just left up to the user to handle*/
    void setDefaultExtraGlobalParamLocation(VarLocation loc){ m_DefaultExtraGlobalParamLocation = loc; }

    //! What is the default location for sparse synaptic connectivity? 
    /*! Historically, everything was allocated on both the host AND device */
    void setDefaultSparseConnectivityLocation(VarLocation loc){ m_DefaultSparseConnectivityLocation = loc; }

    //! Sets default for whether narrow i.e. less than 32-bit types are used for sparse matrix indices
    void setDefaultNarrowSparseIndEnabled(bool enabled){ m_DefaultNarrowSparseIndEnabled = enabled; }

    //! Should compatible postsynaptic models and dendritic delay buffers be fused?
    /*! This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved */
    void setMergePostsynapticModels(bool merge){ m_ShouldFusePostsynapticModels = merge; }

    //! Should compatible postsynaptic models and dendritic delay buffers be fused?
    /*! This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved */
    void setFusePostsynapticModels(bool fuse) { m_ShouldFusePostsynapticModels = fuse; }
    
    //! Should compatible pre and postsynaptic weight update model variables and updates be fused?
    /*! This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved */
    void setFusePrePostWeightUpdateModels(bool fuse){ m_ShouldFusePrePostWeightUpdateModels = fuse; }

    void setBatchSize(unsigned int batchSize) { m_BatchSize = batchSize;  }

    //! Gets the name of the neuronal network model
    const std::string &getName() const{ return m_Name; }

    //! Gets the floating point numerical precision
    const std::string &getPrecision() const{ return m_Precision; }

    //! Gets the floating point numerical precision used to represent time
    std::string getTimePrecision() const;

    //! Gets the model integration step size
    double getDT() const { return m_DT; }

    //! Get the random seed
    unsigned int getSeed() const { return m_Seed; }

    //! Are timers and timing commands enabled
    bool isTimingEnabled() const{ return m_TimingEnabled; }

    unsigned int getBatchSize() const { return m_BatchSize;  }

    // PUBLIC NEURON FUNCTIONS
    //========================
    //! How many neurons make up the entire model
    unsigned int getNumNeurons() const;

    //! Find a neuron group by name
    NeuronGroup *findNeuronGroup(const std::string &name){ return static_cast<NeuronGroup*>(findNeuronGroupInternal(name)); }

    //! Adds a new neuron group to the model using a neuron model managed by the user
    /*! \param name string containing unique name of neuron population.
        \param size integer specifying how many neurons are in the population.
        \param model neuron model to use for neuron group.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created NeuronGroup */
    NeuronGroup *addNeuronPopulation(const std::string &name, unsigned int size, const NeuronModels::Base *model,
                                     const ParamValues &paramValues,
                                     const VarValues &varInitialisers);

    //! Adds a new neuron group to the model using a singleton neuron model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam NeuronModel type of neuron model (derived from NeuronModels::Base).
        \param name string containing unique name of neuron population.
        \param size integer specifying how many neurons are in the population.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created NeuronGroup */
    template<typename NeuronModel>
    NeuronGroup *addNeuronPopulation(const std::string &name, unsigned int size,
                                     const ParamValues &paramValues, const VarValues &varInitialisers)
    {
        return addNeuronPopulation(name, size, NeuronModel::getInstance(), paramValues, varInitialisers);
    }

    // PUBLIC SYNAPSE FUNCTIONS
    //=========================
    //! Find a synapse group by name
    SynapseGroup *findSynapseGroup(const std::string &name);    

    //! Adds a synapse population to the model using weight update and postsynaptic models managed by the user
    /*! \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur
                                            (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param wum                          weight update model to use for synapse group.
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPreVarInitialisers     weight update model presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPostVarInitialisers    weight update model postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param psm                          postsynaptic model to use for synapse group.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      sparse connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::SPARSE or SynapseMatrixConnectivity::BITMASK.
                                            Typically wrapped with it's parameters using ``initConnectivity`` function
        \return pointer to newly created SynapseGroup */
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const WeightUpdateModels::Base *wum, const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                       const PostsynapticModels::Base *psm, const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitSparseConnectivitySnippet::Init &connectivityInitialiser = uninitialisedConnectivity())
    {
        auto uninitialisedToeplitz = InitToeplitzConnectivitySnippet::Init(InitToeplitzConnectivitySnippet::Uninitialised::getInstance(), {});
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    wum, weightParamValues, weightVarInitialisers, weightPreVarInitialisers, weightPostVarInitialisers,
                                    psm, postsynapticParamValues, postsynapticVarInitialisers,
                                    connectivityInitialiser, uninitialisedToeplitz);
    }

    //! Adds a synapse population to the model using weight update and postsynaptic models managed by the user
    /*! \tparam WeightUpdateModel           type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel           type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur
                                            (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param wum                          weight update model to use for synapse group.
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPreVarInitialisers     weight update model presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPostVarInitialisers    weight update model postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param psm                          postsynaptic model to use for synapse group.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      toeplitz connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::TOEPLITZ. Typically wrapped with it's parameters using ``initToeplitzConnectivity`` function
        \return pointer to newly created SynapseGroup */
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const WeightUpdateModels::Base *wum, const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                       const PostsynapticModels::Base *psm, const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitToeplitzConnectivitySnippet::Init &connectivityInitialiser)
    {
        auto uninitialisedToeplitz = InitToeplitzConnectivitySnippet::Init(InitToeplitzConnectivitySnippet::Uninitialised::getInstance(), {});
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    wum, weightParamValues, weightVarInitialisers, weightPreVarInitialisers, weightPostVarInitialisers,
                                    psm, postsynapticParamValues, postsynapticVarInitialisers,
                                    uninitialisedConnectivity(), connectivityInitialiser);
    }

    //! Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam WeightUpdateModel           type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel           type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      sparse connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::SPARSE or SynapseMatrixConnectivity::BITMASK.
                                            Typically wrapped with it's parameters using ``initConnectivity`` function
        \return pointer to newly created SynapseGroup */
    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const ParamValues &weightParamValues, const VarValues &weightVarInitialisers,
                                       const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitSparseConnectivitySnippet::Init &connectivityInitialiser = uninitialisedConnectivity())
    {
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    WeightUpdateModel::getInstance(), weightParamValues, weightVarInitialisers, {}, {},
                                    PostsynapticModel::getInstance(), postsynapticParamValues, postsynapticVarInitialisers,
                                    connectivityInitialiser);
    }

    //! Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam WeightUpdateModel           type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel           type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      toeplitz connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::TOEPLITZ. Typically wrapped with it's parameters using ``initToeplitzConnectivity`` function
        \return pointer to newly created SynapseGroup */
    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const ParamValues &weightParamValues, const VarValues &weightVarInitialisers,
                                       const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitToeplitzConnectivitySnippet::Init &connectivityInitialiser)
    {
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    WeightUpdateModel::getInstance(), weightParamValues, weightVarInitialisers, {}, {},
                                    PostsynapticModel::getInstance(), postsynapticParamValues, postsynapticVarInitialisers,
                                    connectivityInitialiser);
    }

    //! Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam WeightUpdateModel           type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel           type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model per-synapse state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPreVarInitialisers     weight update model presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPostVarInitialisers    weight update model postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      sparse connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::SPARSE or SynapseMatrixConnectivity::BITMASK.
                                            Typically wrapped with it's parameters using ``initConnectivity`` function
        \return pointer to newly created SynapseGroup */
    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                       const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitSparseConnectivitySnippet::Init &connectivityInitialiser = uninitialisedConnectivity())
    {
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    WeightUpdateModel::getInstance(), weightParamValues, weightVarInitialisers, weightPreVarInitialisers, weightPostVarInitialisers,
                                    PostsynapticModel::getInstance(), postsynapticParamValues, postsynapticVarInitialisers,
                                    connectivityInitialiser);

    }

    //! Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam WeightUpdateModel           type of weight update model (derived from WeightUpdateModels::Base).
        \tparam PostsynapticModel           type of postsynaptic model (derived from PostsynapticModels::Base).
        \param name                         string containing unique name of synapse population.
        \param mtype                        how the synaptic matrix associated with this synapse population should be represented.
        \param delaySteps                   integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)
        \param src                          string specifying name of presynaptic (source) population
        \param trg                          string specifying name of postsynaptic (target) population
        \param weightParamValues            parameters for weight update model wrapped in ParamValues object.
        \param weightVarInitialisers        weight update model per-synapse state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPreVarInitialisers     weight update model presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param weightPostVarInitialisers    weight update model postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postsynapticParamValues      parameters for postsynaptic model wrapped in ParamValues object.
        \param postsynapticVarInitialisers  postsynaptic model state variable initialiser snippets and parameters wrapped in VarValues object.
        \param connectivityInitialiser      toeplitz connectivity initialisation snippet used to initialise connectivity for
                                            SynapseMatrixConnectivity::TOEPLITZ. Typically wrapped with it's parameters using ``initToeplitzConnectivity`` function
        \return pointer to newly created SynapseGroup */
    template<typename WeightUpdateModel, typename PostsynapticModel>
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                       const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                       const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitToeplitzConnectivitySnippet::Init &connectivityInitialiser)
    {
        return addSynapsePopulation(name, mtype, delaySteps, src, trg,
                                    WeightUpdateModel::getInstance(), weightParamValues, weightVarInitialisers, weightPreVarInitialisers, weightPostVarInitialisers,
                                    PostsynapticModel::getInstance(), postsynapticParamValues, postsynapticVarInitialisers,
                                    connectivityInitialiser);

    }

    // PUBLIC CURRENT SOURCE FUNCTIONS
    //================================
    //! Find a current source by name
    CurrentSource *findCurrentSource(const std::string &name);

    //! Adds a new current source to the model using a current source model managed by the user
    /*! \param currentSourceName string containing unique name of current source.
        \param model current source model to use for current source.
        \param targetNeuronGroupName string name of the target neuron group
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created CurrentSource */
    CurrentSource *addCurrentSource(const std::string &currentSourceName, const CurrentSourceModels::Base *model, const std::string &targetNeuronGroupName,
                                    const ParamValues &paramValues, const VarValues &varInitialisers);

    //! Adds a new current source to the model using a singleton current source model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CurrentSourceModel type of neuron model (derived from CurrentSourceModel::Base).
        \param currentSourceName string containing unique name of current source.
        \param targetNeuronGroupName string name of the target neuron group
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created CurrentSource */
    template<typename CurrentSourceModel>
    CurrentSource *addCurrentSource(const std::string &currentSourceName, const std::string &targetNeuronGroupName,
                                    const ParamValues &paramValues, const VarValues &varInitialisers)
    {
        return addCurrentSource(currentSourceName, CurrentSourceModel::getInstance(),
                                targetNeuronGroupName, paramValues, varInitialisers);
    }

    //! Adds a new custom update with references to the model using a custom update model managed by the user
    /*! \tparam CustomUpdateModel type of custom update model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in CustomUpdateModel::VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdate *addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModel *model,
                                  const ParamValues &paramValues, const VarValues &varInitialisers,
                                  const typename CustomUpdateModel::VarReferences &varReferences)
    {
         // Add neuron group to map
        auto result = m_CustomUpdates.emplace(std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(name, updateGroupName, model,
                                  paramValues, varInitialisers, varReferences.getInitialisers(),
                                  m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

        if(!result.second) {
            throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
        }
        else {
            return &result.first->second;
        }
    }

    //! Adds a new custom update with references to weight update model variable to the 
    //! model using a custom update model managed by the user
    /*! \tparam CustomUpdateModel type of custom update model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param operation CustomUpdate::Operation specifying operation update should be performed within
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in CustomUpdateModel::VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdateWU *addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModel *model, 
                                    const ParamValues &paramValues, const VarValues &varInitialisers,
                                    const typename CustomUpdateModel::WUVarReferences &varReferences)
    {
        // Add neuron group to map
        auto result = m_CustomWUUpdates.emplace(std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(name, updateGroupName, model,
                                  paramValues, varInitialisers, varReferences.getInitialisers(),
                                  m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

        if(!result.second) {
            throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
        }
        else {
            return &result.first->second;
        }
    }

    //! Adds a new custom update to the model using a singleton custom update model 
    //! created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CustomUpdateModel type of custom update model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varInitialisers variable references wrapped in CustomUpdateModel::VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdate *addCustomUpdate(const std::string &name, const std::string &updateGroupName,
                                  const ParamValues &paramValues, const VarValues &varInitialisers,
                                  const typename CustomUpdateModel::VarReferences &varReferences)
    {
        return addCustomUpdate<CustomUpdateModel>(name, updateGroupName, CustomUpdateModel::getInstance(),
                                                  paramValues, varInitialisers, varReferences);
    }


    //! Adds a new custom update with references to weight update model variables to the model using a singleton 
    //! custom update model created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CustomUpdateModel type of neuron model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param operation CustomUpdate::Operation specifying operation update should be performed within
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varInitialisers variable references wrapped in CustomUpdateModel::VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdateWU *addCustomUpdate(const std::string &name, const std::string &updateGroupName,
                                    const ParamValues &paramValues, const VarValues &varInitialisers,
                                    const typename CustomUpdateModel::WUVarReferences &varReferences)
    {
        return addCustomUpdate<CustomUpdateModel>(name, updateGroupName, CustomUpdateModel::getInstance(),
                                                  paramValues, varInitialisers, varReferences);
    }

protected:
    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    //! Finalise model
    void finalize();

    //--------------------------------------------------------------------------
    // Protected const methods
    //--------------------------------------------------------------------------
    //! Get the string literal that should be used to represent a value in the model's floating-point type
    std::string scalarExpr(double) const;

    //! Are any variables in any populations in this model using zero-copy memory?
    bool zeroCopyInUse() const;

    //! Is recording enabled on any population in this model?
    bool isRecordingInUse() const;

    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Get std::map containing local named NeuronGroup objects in model
    const std::map<std::string, NeuronGroupInternal> &getNeuronGroups() const{ return m_LocalNeuronGroups; }

    //! Get std::map containing local named SynapseGroup objects in model
    const std::map<std::string, SynapseGroupInternal> &getSynapseGroups() const{ return m_LocalSynapseGroups; }

    //! Get std::map containing local named CurrentSource objects in model
    const std::map<std::string, CurrentSourceInternal> &getLocalCurrentSources() const{ return m_LocalCurrentSources; }

    //! Get std::map containing named CustomUpdate objects in model
    const std::map<std::string, CustomUpdateInternal> &getCustomUpdates() const { return m_CustomUpdates; }
    const std::map<std::string, CustomUpdateWUInternal> &getCustomWUUpdates() const { return m_CustomWUUpdates; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Find a neuron group by name
    NeuronGroupInternal *findNeuronGroupInternal(const std::string &name);
    
    //! Find a synapse group by name
    SynapseGroupInternal *findSynapseGroupInternal(const std::string &name);

    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string &src, const std::string &trg,
                                       const WeightUpdateModels::Base *wum, const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                       const PostsynapticModels::Base *psm, const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                       const InitSparseConnectivitySnippet::Init &connectivityInitialiser, const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser);

    //--------------------------------------------------------------------------
    // Private members
    //--------------------------------------------------------------------------
    //! Named local neuron groups
    std::map<std::string, NeuronGroupInternal> m_LocalNeuronGroups;

    //! Named local synapse groups
    std::map<std::string, SynapseGroupInternal> m_LocalSynapseGroups;

    //! Named local current sources
    std::map<std::string, CurrentSourceInternal> m_LocalCurrentSources;

    //! Grouped named custom updates
    std::map<std::string, CustomUpdateInternal> m_CustomUpdates;
    std::map<std::string, CustomUpdateWUInternal> m_CustomWUUpdates;

    //! Name of the neuronal newtwork model
    std::string m_Name;

    //! Type of floating point variables (float, double, ...; default: float)
    std::string m_Precision;

    //! Type of floating point variables used to store time
    TimePrecision m_TimePrecision;

    //! The integration time step of the model
    double m_DT;

    //! Whether timing code should be inserted into model
    bool m_TimingEnabled;

    //! RNG seed
    unsigned int m_Seed;

    //! What is the default location for model state variables? Historically, everything was allocated on both host AND device
    VarLocation m_DefaultVarLocation;

    //! What is the default location for model extra global parameters? Historically, this was just left up to the user to handle
    VarLocation m_DefaultExtraGlobalParamLocation;

    //! What is the default location for sparse synaptic connectivity? Historically, everything was allocated on both the host AND device
    VarLocation m_DefaultSparseConnectivityLocation; 

    //! The default for whether narrow i.e. less than 32-bit types are used for sparse matrix indices
    bool m_DefaultNarrowSparseIndEnabled;

    //! Should compatible postsynaptic models and dendritic delay buffers be fused?
    /*! This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved */
    bool m_ShouldFusePostsynapticModels; 
    
    //! Should compatible pre and postsynaptic weight update model variables and updates be fused?
    /*! This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved */
    bool m_ShouldFusePrePostWeightUpdateModels;

    //! Batch size of this model - efficiently duplicates model
    unsigned int m_BatchSize;
};

// Typedefine NNmodel for backward compatibility
typedef ModelSpec NNmodel;
