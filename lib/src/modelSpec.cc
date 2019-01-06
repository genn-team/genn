/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/
// Standard C++ includes
#include <algorithm>
#include <numeric>
#include <typeinfo>

// Standard C includes
#include <cstdio>
#include <cmath>
#include <cassert>

// GeNN includes
#include "codeGenUtils.h"
#include "global.h"
#include "modelSpec.h"

// ------------------------------------------------------------------------
// NNmodel
// ------------------------------------------------------------------------
// class NNmodel for specifying a neuronal network model

NNmodel::NNmodel() : m_TimePrecision(TimePrecision::DEFAULT)
{
    setDT(0.5);
    setPrecision(GENN_FLOAT);
    setTiming(false);
    setSeed(0);
}

NNmodel::~NNmodel() 
{
}

void NNmodel::setName(const std::string &inname)
{
    name= inname;
}

bool NNmodel::zeroCopyInUse() const
{
    // If any neuron groups use zero copy return true
    if(any_of(begin(m_LocalNeuronGroups), end(m_LocalNeuronGroups),
        [](const NeuronGroupValueType &n){ return n.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any synapse groups use zero copy return true
    if(any_of(begin(m_LocalSynapseGroups), end(m_LocalSynapseGroups),
        [](const SynapseGroupValueType &s){ return s.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    return false;
}

size_t NNmodel::getNumPreSynapseResetRequiredGroups() const
{
    return std::count_if(getLocalSynapseGroups().cbegin(), getLocalSynapseGroups().cend(),
                         [](const SynapseGroupValueType &s){ return s.second.isDendriticDelayRequired(); });
}

std::string NNmodel::getTimePrecision() const
{
    // If time precision is set to match model precision
    if(m_TimePrecision == TimePrecision::DEFAULT) {
        return getPrecision();
    }
    // Otherwise return appropriate type
    else if(m_TimePrecision == TimePrecision::FLOAT) {
        return "float";
    }
    else {
        return "double";
    }
}

std::string NNmodel::getGeneratedCodePath(const std::string &path, const std::string &filename) const{
#ifdef MPI_ENABLE
    int localHostID = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &localHostID);
    return path + "/" + getName() + "_" + std::to_string(localHostID) + "_CODE/" + filename;
#else
    return path + "/" + getName() + "_CODE/" + filename;
#endif
    }
/*
bool NNmodel::isDeviceInitRequired(int localHostID) const
{
    // If any local neuron groups require device initialisation, return true
    if(std::any_of(std::begin(m_LocalNeuronGroups), std::end(m_LocalNeuronGroups),
        [](const NNmodel::NeuronGroupValueType &n){ return n.second.isDeviceInitRequired(); }))
    {
        return true;
    }

    // If any remote neuron groups with local outputs require their spike variables to be initialised on device
    if(std::any_of(std::begin(m_RemoteNeuronGroups), std::end(m_RemoteNeuronGroups),
        [localHostID](const NNmodel::NeuronGroupValueType &n)
        {
            return (n.second.hasOutputToHost(localHostID) && (n.second.getSpikeVarMode() & VarInit::DEVICE));
        }))
    {
        return true;
    }


    // If any local synapse groups require device initialisation, return true
    if(std::any_of(std::begin(m_LocalSynapseGroups), std::end(m_LocalSynapseGroups),
        [](const NNmodel::SynapseGroupValueType &s){ return s.second.isDeviceInitRequired(); }))
    {
        return true;
    }

    return false;
}

bool NNmodel::isDeviceSparseInitRequired() const
{
    // Return true if any of the synapse groups require device sparse initialisation
    return std::any_of(std::begin(m_LocalSynapseGroups), std::end(m_LocalSynapseGroups),
        [](const NNmodel::SynapseGroupValueType &s) { return s.second.isDeviceSparseInitRequired(); });
}*/

unsigned int NNmodel::getNumLocalNeurons() const
{
    // Return sum of local neuron group sizes
    return std::accumulate(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}

unsigned int NNmodel::getNumRemoteNeurons() const
{
    // Return sum of local remote neuron group sizes
    return std::accumulate(m_RemoteNeuronGroups.cbegin(), m_RemoteNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}

const NeuronGroup *NNmodel::findNeuronGroup(const std::string &name) const
{
    // If a matching local neuron group is found, return it
    auto localNeuronGroup = m_LocalNeuronGroups.find(name);
    if(localNeuronGroup != m_LocalNeuronGroups.cend()) {
        return &localNeuronGroup->second;
    }

    // Otherwise, if a matching remote neuron group is found, return it
    auto remoteNeuronGroup = m_RemoteNeuronGroups.find(name);
    if(remoteNeuronGroup != m_RemoteNeuronGroups.cend()) {
        return &remoteNeuronGroup->second;

    }
    // Otherwise, error
    else {
        throw std::runtime_error("neuron group " + name + " not found, aborting ...");
    }
}

NeuronGroup *NNmodel::findNeuronGroup(const std::string &name)
{
    // If a matching local neuron group is found, return it
    auto localNeuronGroup = m_LocalNeuronGroups.find(name);
    if(localNeuronGroup != m_LocalNeuronGroups.cend()) {
        return &localNeuronGroup->second;
    }

    // Otherwise, if a matching remote neuron group is found, return it
    auto remoteNeuronGroup = m_RemoteNeuronGroups.find(name);
    if(remoteNeuronGroup != m_RemoteNeuronGroups.cend()) {
        return &remoteNeuronGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("neuron group " + name + " not found, aborting ...");
    }
}

const SynapseGroup *NNmodel::findSynapseGroup(const std::string &name) const
{
    // If a matching local synapse group is found, return it
    auto localSynapseGroup = m_LocalSynapseGroups.find(name);
    if(localSynapseGroup != m_LocalSynapseGroups.cend()) {
        return &localSynapseGroup->second;
    }

    // Otherwise, if a matching remote synapse group is found, return it
    auto remoteSynapseGroup = m_RemoteSynapseGroups.find(name);
    if(remoteSynapseGroup != m_RemoteSynapseGroups.cend()) {
        return &remoteSynapseGroup->second;

    }
    // Otherwise, error
    else {
        throw std::runtime_error("synapse group " + name + " not found, aborting ...");
    }
}

SynapseGroup *NNmodel::findSynapseGroup(const std::string &name)
{
    // If a matching local synapse group is found, return it
    auto localSynapseGroup = m_LocalSynapseGroups.find(name);
    if(localSynapseGroup != m_LocalSynapseGroups.cend()) {
        return &localSynapseGroup->second;
    }

    // Otherwise, if a matching remote synapse group is found, return it
    auto remoteSynapseGroup = m_RemoteSynapseGroups.find(name);
    if(remoteSynapseGroup != m_RemoteSynapseGroups.cend()) {
        return &remoteSynapseGroup->second;

    }
    // Otherwise, error
    else {
        throw std::runtime_error("synapse group " + name + " not found, aborting ...");
    }
}

//--------------------------------------------------------------------------
/*! \brief This function attempts to find an existing current source */
//--------------------------------------------------------------------------

const CurrentSource *NNmodel::findCurrentSource(const std::string &name) const
{
    // If a matching local current source is found, return it
    auto localCurrentSource = m_LocalCurrentSources.find(name);
    if(localCurrentSource != m_LocalCurrentSources.cend()) {
        return &localCurrentSource->second;
    }

    // Otherwise, if a matching remote current source is found, return it
    auto remoteCurrentSource = m_RemoteCurrentSources.find(name);
    if(remoteCurrentSource != m_RemoteCurrentSources.cend()) {
        return &remoteCurrentSource->second;

    }
    // Otherwise, error
    else {
        throw std::runtime_error("current source " + name + " not found, aborting ...");
    }
}

CurrentSource *NNmodel::findCurrentSource(const std::string &name)
{
    // If a matching local current source is found, return it
    auto localCurrentSource = m_LocalCurrentSources.find(name);
    if(localCurrentSource != m_LocalCurrentSources.cend()) {
        return &localCurrentSource->second;
    }

    // Otherwise, if a matching remote current source is found, return it
    auto remoteCurrentSource = m_RemoteCurrentSources.find(name);
    if(remoteCurrentSource != m_RemoteCurrentSources.cend()) {
        return &remoteCurrentSource->second;

    }
    // Otherwise, error
    else {
        throw std::runtime_error("current source " + name + " not found, aborting ...");
    }
}
//--------------------------------------------------------------------------
/*! \brief This function sets the integration time step DT of the model
 */
//--------------------------------------------------------------------------

void NNmodel::setDT(double newDT /**<  */)
{
    dt = newDT;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the numerical precision of floating type variables. By default, it is GENN_GENN_FLOAT.
 */
//--------------------------------------------------------------------------

void NNmodel::setPrecision(FloatType floattype /**<  */)
{
    switch (floattype) {
    case GENN_FLOAT:
        ftype = "float";
        break;
    case GENN_DOUBLE:
        ftype = "double"; // not supported by compute capability < 1.3
        break;
    case GENN_LONG_DOUBLE:
        ftype = "long double"; // not supported by CUDA at the moment.
        break;
    default:
        throw std::runtime_error("Unrecognised floating-point type.");
    }
}

void NNmodel::setTimePrecision(TimePrecision timePrecision)
{
    m_TimePrecision = timePrecision;
}

//--------------------------------------------------------------------------
/*! \brief This function sets a flag to determine whether timers and timing commands are to be included in generated code.
 */
//--------------------------------------------------------------------------

void NNmodel::setTiming(bool theTiming /**<  */)
{
    timing= theTiming;
}


//--------------------------------------------------------------------------
/*! \brief This function sets the random seed. If the passed argument is > 0, automatic seeding is disabled. If the argument is 0, the underlying seed is obtained from the time() function.
 */
//--------------------------------------------------------------------------

void NNmodel::setSeed(unsigned int inseed /*!< the new seed  */)
{
    seed= inseed;
}

std::string NNmodel::scalarExpr(const double val) const
{
    std::string tmp;
    float fval= (float) val;
    if (ftype == "float") {
        tmp= std::to_string(fval) + "f";
    }
    if (ftype == "double") {
        tmp= std::to_string(val);
    }
    return tmp;
}

/*void NNmodel::finalize()
{
    //initializing learning parameters to start
    if (final) {
        gennError("Your model has already been finalized");
    }
    final = true;

    // Loop through neuron populations and their outgoing synapse populations
    for(auto &n : m_LocalNeuronGroups) {
        for(auto *sg : n.second.getOutSyn()) {
            const auto *wu = sg->getWUModel();

            if (!wu->getEventCode().empty()) {
                sg->setSpikeEventRequired(true);
                n.second.setSpikeEventRequired(true);
                assert(!wu->getEventThresholdConditionCode().empty());

                 // Create iteration context to iterate over derived and extra global parameters
                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());

                // do an early replacement of parameters, derived parameters and extraglobalsynapse parameters
                string eCode = wu->getEventThresholdConditionCode();
                value_substitutions(eCode, wu->getParamNames(), sg->getWUParams());
                value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
                name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

                // Add code and name of
                string supportCodeNamespaceName = wu->getSimSupportCode().empty() ?
                    "" : sg->getName() + "_weightupdate_simCode";

                // Add code and name of support code namespace to set
                n.second.addSpkEventCondition(eCode, supportCodeNamespaceName);

                // analyze which neuron variables need queues
                n.second.updatePreVarQueues(wu->getEventCode());
            }
        }
        if (n.second.getSpikeEventCondition().size() > 1) {
            for(auto *sg : n.second.getOutSyn()) {
                if (!sg->getWUModel()->getEventCode().empty()) {
                    sg->setEventThresholdReTestRequired(true);
                }
            }
        }
    }

    // NEURON GROUPS
    for(auto &n : m_LocalNeuronGroups) {
        // Initialize derived parameters
        n.second.initDerivedParams(dt);

        // Make extra global parameter lists
        n.second.addExtraGlobalParams(neuronKernelParameters);
    }

    // SYNAPSE groups
    for(auto &s : m_LocalSynapseGroups) {
        const auto *wu = s.second.getWUModel();

        // Initialize derived parameters
        s.second.initDerivedParams(dt);

        if (!wu->getSimCode().empty()) {
            s.second.setTrueSpikeRequired(true);
            s.second.getSrcNeuronGroup()->setTrueSpikeRequired(true);

            // analyze which neuron variables need queues
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSimCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSimCode());
        }

        if (!wu->getLearnPostCode().empty()) {
            s.second.getTrgNeuronGroup()->setTrueSpikeRequired(true);

            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getLearnPostCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getLearnPostCode());
        }

        if (!wu->getSynapseDynamicsCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSynapseDynamicsCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSynapseDynamicsCode());
        }

        // Make extra global parameter lists
        s.second.addExtraGlobalConnectivityInitialiserParams(m_InitKernelParameters);
        s.second.addExtraGlobalNeuronParams(neuronKernelParameters);
        s.second.addExtraGlobalSynapseParams(synapseKernelParameters);
        s.second.addExtraGlobalPostLearnParams(simLearnPostKernelParameters);
        s.second.addExtraGlobalSynapseDynamicsParams(synapseDynamicsKernelParameters);

        // If this synapse group has either ragged or bitmask connectivity which is initialised
        // using a connectivity snippet AND has individual synaptic variables
        if(((s.second.getMatrixType() & SynapseMatrixConnectivity::RAGGED)
            || (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK))
            && !s.second.getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty()
            && s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)
        {
            // Loop through variables and check that they are initialised in the same place as the sparse connectivity
            auto wuVars = s.second.getWUModel()->getVars();
            for (size_t k= 0, l= wuVars.size(); k < l; k++) {
                if((s.second.getSparseConnectivityVarMode() & VarInit::HOST) != (s.second.getWUVarMode(k) & VarInit::HOST)) {
                    gennError("Weight update mode variables must be initialised in same place as sparse connectivity variable '" + wuVars[k].first + "' in population '" + s.first + "' is not");
                }
            }
        }
    }

    // CURRENT SOURCES
    for(auto &cs : m_LocalCurrentSources) {
        // Initialize derived parameters
        cs.second.initDerivedParams(dt);

        // Make extra global parameter lists
        cs.second.addExtraGlobalParams(currentSourceKernelParameters);
    }

    // Merge incoming postsynaptic models
    for(auto &n : m_LocalNeuronGroups) {
        if(!n.second.getInSyn().empty()) {
            n.second.mergeIncomingPSM();
        }
    }

    // CURRENT SOURCES
    for(auto &cs : m_LocalCurrentSources) {
        // Initialize derived parameters
        cs.second.initDerivedParams(dt);

        // Make extra global parameter lists
        cs.second.addExtraGlobalParams(currentSourceKernelParameters);
    }
}*/