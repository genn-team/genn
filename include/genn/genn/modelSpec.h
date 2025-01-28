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
#include <optional>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceInternal.h"
#include "customUpdateInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "gennExport.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

namespace GeNN
{
using VarValues = std::map<std::string, InitVarSnippet::Init>;
using VarReferences = std::map<std::string, Models::VarReference>;
using LocalVarReferences = std::map<std::string, std::variant<std::string, Models::VarReference>>;
using WUVarReferences = std::map<std::string, Models::WUVarReference>;
using EGPReferences = std::map<std::string, Models::EGPReference>;

//! Initialise a variable using an initialisation snippet
/*! \tparam S       type of variable initialisation snippet (derived from InitVarSnippet::Base).
    \param params   parameters for snippet wrapped in ParamValues object.
    \return         InitVarSnippet::Init object for use within model's VarValues*/
template<typename S>
inline InitVarSnippet::Init initVar(const ParamValues &params = {})
{
    return InitVarSnippet::Init(S::getInstance(), params);
}

//! Mark a variable as uninitialised
/*! This means that the backend will not generate any automatic initialization code, but will instead
    copy the variable from host to device during ``initializeSparse`` function */
inline InitVarSnippet::Init uninitialisedVar()
{
    return InitVarSnippet::Init(InitVarSnippet::Uninitialised::getInstance(), {});
}

//! Initialise connectivity using a sparse connectivity snippet
/*! \tparam S       type of sparse connectivitiy initialisation snippet (derived from InitSparseConnectivitySnippet::Base).
    \param params   parameters for snippet wrapped in ParamValues object.
    \return         InitSparseConnectivitySnippet::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline InitSparseConnectivitySnippet::Init initConnectivity(const ParamValues &params = {})
{
    return InitSparseConnectivitySnippet::Init(S::getInstance(), params);
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
inline InitToeplitzConnectivitySnippet::Init initToeplitzConnectivity(const ParamValues &params = {})
{
    return InitToeplitzConnectivitySnippet::Init(S::getInstance(), params);
}

//! Initialise postsynaptic update model
/*! \tparam S               type of postsynaptic model initialisation snippet (derived from PostSynapticModels::Base).
    \param params           parameters for snippet wrapped in ParamValues object.
    \param vars             variables for snippet wrapped in VarValues object.
    \param neuronVarRefs    neuron variable references for snippet wrapped in VarReferences object.
    \return                 PostsynapticModels::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline PostsynapticModels::Init initPostsynaptic(const ParamValues &params = {}, const VarValues &vars = {}, 
                                                 const LocalVarReferences &neuronVarRefs = {})
{
    return PostsynapticModels::Init(S::getInstance(), params, vars, neuronVarRefs);
}

//! Initialise weight update model
/*! \tparam S                   type of postsynaptic model initialisation snippet (derived from PostSynapticModels::Base).
    \param params               parameters for snippet wrapped in ParamValues object.
    \param vars                 variables for snippet wrapped in VarValues object.
    \param preVars              presynaptic variables for snippet wrapped in VarValues object.
    \param postVars             postsynaptic variables for snippet wrapped in VarValues object.
    \param preNeuronVarRefs     presynaptic neuron variable references for snippet wrapped in VarReferences object.
    \param postNeuronVarRefs    postsynaptic neuron variable references for snippet wrapped in VarReferences object.
    \param psmVarRefs           postsynaptic modelvariable references for snippet wrapped in VarReferences object.
    \return                     PostsynapticModels::Init object for passing to ``ModelSpec::addSynapsePopulation``*/
template<typename S>
inline WeightUpdateModels::Init initWeightUpdate(const ParamValues &params = {}, const VarValues &vars = {}, 
                                                 const VarValues &preVars = {}, const VarValues &postVars = {}, 
                                                 const LocalVarReferences &preNeuronVarRefs = {}, 
                                                 const LocalVarReferences &postNeuronVarRefs = {},
                                                 const LocalVarReferences &psmVarRefs = {})
{
    return WeightUpdateModels::Init(S::getInstance(), params, vars, preVars, postVars, 
                                    preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
}

//! Creates a reference to a neuron group variable.
inline Models::VarReference createVarRef(NeuronGroup *ng, const std::string &varName)
{
    return Models::VarReference::createVarRef(ng, varName);
}

//! Creates a reference to a current source variable.
inline Models::VarReference createVarRef(CurrentSource *cs, const std::string &varName)
{
    return Models::VarReference::createVarRef(cs, varName);
}

//! Creates a reference to a custom update variable.
inline Models::VarReference createVarRef(CustomUpdate *cu, const std::string &varName)
{
    return Models::VarReference::createVarRef(cu, varName);
}

//! Creates a reference to a presynaptic custom connectivity update variable.
inline Models::VarReference createPreVarRef(CustomConnectivityUpdate *cu, const std::string &varName)
{
    return Models::VarReference::createPreVarRef(cu, varName);
}

//! Creates a reference to a postsynaptic custom connectivity update variable.
inline Models::VarReference createPostVarRef(CustomConnectivityUpdate *cu, const std::string &varName)
{
    return Models::VarReference::createPostVarRef(cu, varName);
}

//! Creates a reference to a postsynaptic model variable.
inline Models::VarReference createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createPSMVarRef(sg, varName);
}

//! Creates a reference to a weight update model presynaptic variable.
inline Models::VarReference createWUPreVarRef(SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createWUPreVarRef(sg, varName);
}

//! Creates a reference to a weight update model postsynapticvariable.
inline Models::VarReference createWUPostVarRef(SynapseGroup *sg, const std::string &varName)
{
    return Models::VarReference::createWUPostVarRef(sg, varName);
}

//! Creates a reference to a synapse group's postsynaptic output buffer
inline Models::VarReference createOutPostVarRef(SynapseGroup *sg)
{
    return Models::VarReference::createOutPostVarRef(sg);
}

//! Creates a reference to a synapse group's dendritic delay buffer
inline Models::VarReference createDenDelayVarRef(SynapseGroup *sg)
{
    return Models::VarReference::createDenDelayVarRef(sg);
}

//! Creates a reference to a neuron group's spike times
inline Models::VarReference createSpikeTimeVarRef(NeuronGroup *ng)
{
    return Models::VarReference::createSpikeTimeVarRef(ng);
}

//! Creates a reference to a neuron group's previous spike times
inline Models::VarReference createPrevSpikeTimeVarRef(NeuronGroup *ng)
{
    return Models::VarReference::createPrevSpikeTimeVarRef(ng);
}

//! Creates a reference to a weight update model variable.
inline Models::WUVarReference createWUVarRef(SynapseGroup *sg, const std::string &varName,
                                             SynapseGroup *transposeSG = nullptr, const std::string &transposeVarName = "")
{
    return Models::WUVarReference::createWUVarReference(sg, varName, transposeSG, transposeVarName);
}

//! Creates a reference to a custom weight update variable.
inline Models::WUVarReference createWUVarRef(CustomUpdateWU *cu, const std::string &varName)
{
    return Models::WUVarReference::createWUVarReference(cu, varName);
}

//! Creates a reference to a custom connectivity update update variable.
inline Models::WUVarReference createWUVarRef(CustomConnectivityUpdate *cu, const std::string &varName)
{
    return Models::WUVarReference::createWUVarReference(cu, varName);
}

//! Creates a reference to a neuron group extra global parameter.
inline Models::EGPReference createEGPRef(NeuronGroup *ng, const std::string &egpName)
{
    return Models::EGPReference::createEGPRef(ng, egpName);
}

//! Creates a reference to a current source extra global parameter.
inline Models::EGPReference createEGPRef(CurrentSource *cs, const std::string &egpName)
{
    return Models::EGPReference::createEGPRef(cs, egpName);
}

//! Creates a reference to a custom update extra global parameter.
inline Models::EGPReference createEGPRef(CustomUpdate *cu, const std::string &egpName)
{
    return Models::EGPReference::createEGPRef(cu, egpName);
}

//! Creates a reference to a custom weight update extra global parameter.
inline Models::EGPReference createEGPRef(CustomUpdateWU *cu, const std::string &egpName)
{
    return Models::EGPReference::createEGPRef(cu, egpName);
}

//! Creates a reference to a custom connectivity update extra global parameter.
inline Models::EGPReference createEGPRef(CustomConnectivityUpdate *ccu, const std::string &egpName)
{
    return Models::EGPReference::createEGPRef(ccu, egpName);
}

//! Creates a reference to a postsynaptic model extra global parameter.
inline Models::EGPReference createPSMEGPRef(SynapseGroup *sg, const std::string &egpName)
{
    return Models::EGPReference::createPSMEGPRef(sg, egpName);
}

//! Creates a reference to a weight update model extra global parameter.
inline Models::EGPReference createWUEGPRef(SynapseGroup *sg, const std::string &egpName)
{
    return Models::EGPReference::createWUEGPRef(sg, egpName);
}

//----------------------------------------------------------------------------
// GeNN::ModelSpec
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
    typedef std::map<std::string, CustomConnectivityUpdateInternal>::value_type CustomConnectivityUpdateValueType;

    ModelSpec();
    ModelSpec(const ModelSpec&) = delete;
    ModelSpec &operator=(const ModelSpec &) = delete;
    ~ModelSpec();

    // PUBLIC MODEL FUNCTIONS
    //=======================
    //! Method to set the neuronal network model name
    void setName(const std::string &name){ m_Name = name; }

    //! Set numerical precision for scalar type
    void setPrecision(const Type::UnresolvedType &precision);

    //! Set numerical precision for time type
    void setTimePrecision(const Type::UnresolvedType &timePrecision);

    //! Set the integration step size of the model
    void setDT(double dt){ m_DT = dt; }

    //! Set whether timers and timing commands are to be included
    void setTimingEnabled(bool timingEnabled){ m_TimingEnabled = timingEnabled; }

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
    void setFusePostsynapticModels(bool fuse) { m_FusePostsynapticModels = fuse; }
    
    //! Should compatible pre and postsynaptic weight update model variables and updates be fused?
    /*! This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved */
    void setFusePrePostWeightUpdateModels(bool fuse){ m_FusePrePostWeightUpdateModels = fuse; }

    void setBatchSize(unsigned int batchSize) { m_BatchSize = batchSize;  }

    //! Gets the name of the neuronal network model
    const std::string &getName() const{ return m_Name; }

    //! Gets the floating point numerical precision
    const Type::ResolvedType &getPrecision() const{ return m_Precision; }

    //! Gets the floating point numerical precision used to represent time
    const Type::ResolvedType &getTimePrecision() const{ return m_TimePrecision ? m_TimePrecision.value() : m_Precision; }

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
    NeuronGroup *findNeuronGroup(const std::string &name);

    //! Find a neuron group by name
    const NeuronGroup *findNeuronGroup(const std::string &name) const;

    //! Adds a new neuron group to the model using a neuron model managed by the user
    /*! \param name string containing unique name of neuron population.
        \param size integer specifying how many neurons are in the population.
        \param model neuron model to use for neuron group.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created NeuronGroup */
    NeuronGroup *addNeuronPopulation(const std::string &name, unsigned int size, const NeuronModels::Base *model,
                                     const ParamValues &paramValues = {}, const VarValues &varInitialisers = {});

    //! Adds a new neuron group to the model using a singleton neuron model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam NeuronModel type of neuron model (derived from NeuronModels::Base).
        \param name string containing unique name of neuron population.
        \param size integer specifying how many neurons are in the population.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created NeuronGroup */
    template<typename NeuronModel>
    NeuronGroup *addNeuronPopulation(const std::string &name, unsigned int size,
                                     const ParamValues &paramValues = {}, const VarValues &varInitialisers = {})
    {
        return addNeuronPopulation(name, size, NeuronModel::getInstance(), paramValues, varInitialisers);
    }

    // PUBLIC SYNAPSE FUNCTIONS
    //=========================
    //! Find a synapse group by name
    SynapseGroup *findSynapseGroup(const std::string &name);
    
    //! Find a synapse group by name
    const SynapseGroup *findSynapseGroup(const std::string &name) const;

    //! Adds a synapse population to the model using weight update and postsynaptic models managed by the user
    /*! \param name                             string containing unique name of synapse population.
        \param mtype                            how the synaptic matrix associated with this synapse population should be represented.
        \param src                              pointer to presynaptic neuron group
        \param trg                              pointer to postsynaptic neuron group
        \param wum                              weight update model to use for synapse group.
        \param wumInitialiser                   WeightUpdateModels::Init object used to initialiser weight update model
        \param psmInitialiser                   PostsynapticModels::Init object used to initialiser postsynaptic model
        \param connectivityInitialiser          sparse connectivity initialisation snippet used to initialise connectivity for
                                                SynapseMatrixConnectivity::SPARSE or SynapseMatrixConnectivity::BITMASK.
                                                Typically wrapped with it's parameters using ``initConnectivity`` function
        \return pointer to newly created SynapseGroup */
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, NeuronGroup *src, NeuronGroup *trg,
                                       const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                                       const InitSparseConnectivitySnippet::Init &connectivityInitialiser = uninitialisedConnectivity())
    {
        auto uninitialisedToeplitz = InitToeplitzConnectivitySnippet::Init(InitToeplitzConnectivitySnippet::Uninitialised::getInstance(), {});
        return addSynapsePopulation(name, mtype, src, trg,
                                    wumInitialiser, psmInitialiser, 
                                    connectivityInitialiser, uninitialisedToeplitz);
    }

     //! Adds a synapse population to the model using weight update and postsynaptic models managed by the user
    /*! \param name                             string containing unique name of synapse population.
        \param mtype                            how the synaptic matrix associated with this synapse population should be represented.
        \param src                              pointer to presynaptic neuron group
        \param trg                              pointer to postsynaptic neuron group
        \param wum                              weight update model to use for synapse group.
        \param wumInitialiser                   WeightUpdateModels::Init object used to initialiser weight update model
        \param psmInitialiser                   PostsynapticModels::Init object used to initialiser postsynaptic model
         \param connectivityInitialiser          toeplitz connectivity initialisation snippet used to initialise connectivity for
                                                SynapseMatrixConnectivity::TOEPLITZ. Typically wrapped with it's parameters using ``initToeplitzConnectivity`` function
        \return pointer to newly created SynapseGroup */
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, NeuronGroup *src, NeuronGroup *trg,
                                       const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                                       const InitToeplitzConnectivitySnippet::Init &connectivityInitialiser)
    {
        return addSynapsePopulation(name, mtype, src, trg,
                                    wumInitialiser, psmInitialiser, 
                                    uninitialisedConnectivity(), connectivityInitialiser);
    }

    // PUBLIC CURRENT SOURCE FUNCTIONS
    //================================
    //! Find a current source by name
    CurrentSource *findCurrentSource(const std::string &name);

    //! Adds a new current source to the model using a current source model managed by the user
    /*! \param currentSourceName string containing unique name of current source.
        \param model current source model to use for current source.
        \param neuronGroup pointer to target neuron group
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created CurrentSource */
    CurrentSource *addCurrentSource(const std::string &currentSourceName, const CurrentSourceModels::Base *model, NeuronGroup *neuronGroup,
                                    const ParamValues &paramValues = {}, const VarValues &varInitialisers = {},
                                    const LocalVarReferences &neuronVarReferences = {});

    //! Adds a new current source to the model using a singleton current source model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CurrentSourceModel type of neuron model (derived from CurrentSourceModel::Base).
        \param currentSourceName string containing unique name of current source.
        \param neuronGroup pointer to target neuron group
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \return pointer to newly created CurrentSource */
    template<typename CurrentSourceModel>
    CurrentSource *addCurrentSource(const std::string &currentSourceName, NeuronGroup *neuronGroup,
                                    const ParamValues &paramValues = {}, const VarValues &varInitialisers = {}, 
                                    const LocalVarReferences &neuronVarReferences = {})
    {
        return addCurrentSource(currentSourceName, CurrentSourceModel::getInstance(),
                                neuronGroup, paramValues, varInitialisers, neuronVarReferences);
    }

    //! Adds a new custom update with references to the model using a custom update model managed by the user
    /*! \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    CustomUpdate *addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model,
                                  const ParamValues &paramValues = {}, const VarValues &varInitialisers = {},
                                  const VarReferences &varReferences = {}, const EGPReferences &egpReferences = {});

    //! Adds a new custom update with references to weight update model variable to the 
    //! model using a custom update model managed by the user
    /*! \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    CustomUpdateWU *addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model, 
                                    const ParamValues &paramValues, const VarValues &varInitialisers,
                                    const WUVarReferences &varReferences, const EGPReferences &egpReferences = {});

    //! Adds a new custom update to the model using a singleton custom update model 
    //! created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CustomUpdateModel type of custom update model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varInitialisers variable references wrapped in VarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdate *addCustomUpdate(const std::string &name, const std::string &updateGroupName,
                                  const ParamValues &paramValues, const VarValues &varInitialisers,
                                  const VarReferences &varReferences, const EGPReferences &egpReferences = {})
    {
        return addCustomUpdate(name, updateGroupName, CustomUpdateModel::getInstance(),
                               paramValues, varInitialisers, varReferences, egpReferences);
    }


    //! Adds a new custom update with references to weight update model variables to the model using a singleton 
    //! custom update model created using standard DECLARE_CUSTOM_UPDATE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CustomUpdateModel type of neuron model (derived from CustomUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varInitialisers variable references wrapped in WUVarReferences object.
        \return pointer to newly created CustomUpdateBase */
    template<typename CustomUpdateModel>
    CustomUpdateWU *addCustomUpdate(const std::string &name, const std::string &updateGroupName,
                                    const ParamValues &paramValues, const VarValues &varInitialisers,
                                    const WUVarReferences &varReferences, const EGPReferences &egpReferences = {})
    {
        return addCustomUpdate(name, updateGroupName, CustomUpdateModel::getInstance(),
                               paramValues, varInitialisers, varReferences, egpReferences);
    }

    //! Adds a new custom connectivity update attached to synapse group and potentially with synaptic, presynaptic and 
    //! postsynaptic state variables and variable references using a custom connectivity update model managed by the user
    /*! \tparam CustomConnectivityUpdateModel type of custom connectivity update model (derived from CustomConnectivityUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param synapseGroup pointer to the synapse group whose connectivity this group will update
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers synaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param preVarInitialisers presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postVarInitialisers postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in WUVarReferences object.
        \param varReferences variable references wrapped in VarReferences object.
        \param varReferences variable references wrapped in VarReferences object.
        \return pointer to newly created CustomConnectivityUpdate */
    CustomConnectivityUpdate *addCustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, 
                                                          SynapseGroup *synapseGroup, const CustomConnectivityUpdateModels::Base *model, 
                                                          const ParamValues &paramValues = {}, const VarValues &varInitialisers = {},
                                                          const VarValues &preVarInitialisers = {}, const VarValues &postVarInitialisers = {},
                                                          const WUVarReferences &varReferences = {}, const VarReferences &preVarReferences = {},
                                                          const VarReferences &postVarReferences = {}, const EGPReferences &egpReferences = {});

    //! Adds a new custom connectivity update attached to synapse group and potentially with synaptic, presynaptic and 
    //! postsynaptic state variables and variable references using a singleton custom connectivity update model created 
    //! using standard DECLARE_CUSTOM_CONNECTIVITY_UPDATE_MODEL and IMPLEMENT_MODEL macros
    /*! \tparam CustomConnectivityUpdateModel type of custom connectivity update model (derived from CustomConnectivityUpdateModels::Base).
        \param name string containing unique name of custom update
        \param updateGroupName string containing name of group to add this custom update to
        \param synapseGroup pointer to the synapse group whose connectivity this group will update
        \param model custom update model to use for custom update.
        \param paramValues parameters for model wrapped in ParamValues object.
        \param varInitialisers synaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param preVarInitialisers presynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param postVarInitialisers postsynaptic state variable initialiser snippets and parameters wrapped in VarValues object.
        \param varReferences variable references wrapped in WUVarReferences object.
        \param varReferences variable references wrapped in VarReferences object.
        \param varReferences variable references wrapped in VarReferences object.
        \return pointer to newly created CustomConnectivityUpdate */
    template<typename CustomConnectivityUpdateModel>
    CustomConnectivityUpdate *addCustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, SynapseGroup *synapseGroup,
                                                          const ParamValues &paramValues = {}, const VarValues &varInitialisers = {},
                                                          const VarValues &preVarInitialisers = {}, const VarValues &postVarInitialisers = {},
                                                          const WUVarReferences &varReferences = {}, const VarReferences &preVarReferences = {},
                                                          const VarReferences &postVarReferences = {}, const EGPReferences &egpReferences = {})
    {
        return addCustomConnectivityUpdate(name, updateGroupName, synapseGroup, 
                                           CustomConnectivityUpdateModel::getInstance(), paramValues,
                                           varInitialisers, preVarInitialisers, postVarInitialisers, 
                                           varReferences, preVarReferences, postVarReferences, egpReferences);
    }

protected:
    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    //! Finalise model
    void finalise();

    //--------------------------------------------------------------------------
    // Protected const methods
    //--------------------------------------------------------------------------
    //! Are any variables in any populations in this model using zero-copy memory?
    bool zeroCopyInUse() const;

    //! Is recording enabled on any population in this model?
    bool isRecordingInUse() const;

    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    const Type::TypeContext &getTypeContext() const{ return m_TypeContext; }

    //! Get std::map containing local named NeuronGroup objects in model
    const auto &getNeuronGroups() const{ return m_LocalNeuronGroups; }

    //! Get std::map containing local named SynapseGroup objects in model
    const auto &getSynapseGroups() const{ return m_LocalSynapseGroups; }

    //! Get std::map containing local named CurrentSource objects in model
    const auto &getLocalCurrentSources() const{ return m_LocalCurrentSources; }

    //! Get std::map containing named CustomUpdate objects in model
    const auto &getCustomUpdates() const { return m_CustomUpdates; }
    const auto &getCustomWUUpdates() const { return m_CustomWUUpdates; }

    //! Get std::map containing named CustomConnectivity objects in model
    const auto &getCustomConnectivityUpdates() const { return m_CustomConnectivityUpdates; }

    //! Build set of custom update group names
    std::set<std::string> getCustomUpdateGroupNames(bool includeTranspose = true, bool includeNonTranspose = true) const;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    SynapseGroup *addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, NeuronGroup *src, NeuronGroup *trg,
                                       const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
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

    //! Named custom updates
    std::map<std::string, CustomUpdateInternal> m_CustomUpdates;
    std::map<std::string, CustomUpdateWUInternal> m_CustomWUUpdates;

    //! Named custom connectivity updates
    std::map<std::string, CustomConnectivityUpdateInternal> m_CustomConnectivityUpdates;

    //! Name of the network model
    std::string m_Name;

    //! Type of floating point variables used for 'scalar' types
    Type::ResolvedType m_Precision;

    Type::TypeContext m_TypeContext;

    //! Type of floating point variables used for 'timepoint' types
    std::optional<Type::ResolvedType> m_TimePrecision;

    //! The integration time step of the model
    double m_DT;

    //! Whether timing code should be inserted into model
    bool m_TimingEnabled;

    //! RNG seed
    unsigned int m_Seed;

    //! The default location for model state variables?
    VarLocation m_DefaultVarLocation;

    //! The default location for model extra global parameters
    VarLocation m_DefaultExtraGlobalParamLocation;

    //! The default location for sparse synaptic connectivity
    VarLocation m_DefaultSparseConnectivityLocation; 

    //! Should 'narrow' i.e. less than 32-bit types be used to store postsyanptic neuron indices in SynapseMatrixConnectivity::SPARSE connectivity?
    /*! If this is true and postsynaptic population has < 256 neurons, 8-bit indices will be used and, 
        if it has < 65536 neurons, 16-bit indices will be used. */
    bool m_DefaultNarrowSparseIndEnabled;

    //! Should compatible postsynaptic models and dendritic delay buffers be fused?
    /*! This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved */
    bool m_FusePostsynapticModels; 
    
    //! Should compatible pre and postsynaptic weight update model variables and updates be fused?
    /*! This can significantly reduce the cost of updating neuron populations but means that per-synaptic group per and postsynaptic variables cannot be retrieved */
    bool m_FusePrePostWeightUpdateModels;

    //! Batch size of this model - efficiently duplicates model
    unsigned int m_BatchSize;
};
}   // namespace GeNN
