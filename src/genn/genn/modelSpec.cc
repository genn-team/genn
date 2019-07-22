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
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"

// ------------------------------------------------------------------------
// ModelSpec
// ------------------------------------------------------------------------
// class ModelSpec for specifying a neuronal network model

ModelSpec::ModelSpec()
:   m_TimePrecision(TimePrecision::DEFAULT), m_DT(0.5), m_TimingEnabled(false), m_Seed(0),
    m_DefaultVarLocation(VarLocation::HOST_DEVICE), m_DefaultExtraGlobalParamLocation(VarLocation::HOST_DEVICE),
    m_DefaultSparseConnectivityLocation(VarLocation::HOST_DEVICE), m_ShouldMergePostsynapticModels(false)
{
    setPrecision(GENN_FLOAT);
}

ModelSpec::~ModelSpec() 
{
}

std::string ModelSpec::getTimePrecision() const
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

unsigned int ModelSpec::getNumLocalNeurons() const
{
    // Return sum of local neuron group sizes
    return std::accumulate(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}

unsigned int ModelSpec::getNumRemoteNeurons() const
{
    // Return sum of local remote neuron group sizes
    return std::accumulate(m_RemoteNeuronGroups.cbegin(), m_RemoteNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}

SynapseGroup *ModelSpec::findSynapseGroup(const std::string &name)
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
CurrentSource *ModelSpec::findCurrentSource(const std::string &name)
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
/*! \brief This function sets the numerical precision of floating type variables. By default, it is GENN_GENN_FLOAT.
 */
//--------------------------------------------------------------------------
void ModelSpec::setPrecision(FloatType floattype /**<  */)
{
    switch (floattype) {
    case GENN_FLOAT:
        m_Precision = "float";
        break;
    case GENN_DOUBLE:
        m_Precision = "double"; // not supported by compute capability < 1.3
        break;
    case GENN_LONG_DOUBLE:
        m_Precision = "long double"; // not supported by CUDA at the moment.
        break;
    default:
        throw std::runtime_error("Unrecognised floating-point type.");
    }
}


void ModelSpec::finalize()
{
    // Loop through neuron populations and their outgoing synapse populations
    for(auto &n : m_LocalNeuronGroups) {
        for(auto *sg : n.second.getOutSyn()) {
            const auto *wu = sg->getWUModel();

            if (!wu->getEventCode().empty()) {
                using namespace CodeGenerator;
                assert(!wu->getEventThresholdConditionCode().empty());

                 // Create iteration context to iterate over derived and extra global parameters
                EGPNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());

                // do an early replacement of parameters, derived parameters and extraglobalsynapse parameters
                // **NOTE** this is really gross but I can't really see an alternative - backend logic changes based on whether event threshold retesting is required
                std::string eCode = wu->getEventThresholdConditionCode();
                value_substitutions(eCode, wu->getParamNames(), sg->getWUParams());
                value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
                name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sg->getName());

                // Add code and name of
                std::string supportCodeNamespaceName = wu->getSimSupportCode().empty() ?
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
        n.second.initDerivedParams(m_DT);
    }

    // SYNAPSE groups
    for(auto &s : m_LocalSynapseGroups) {
        const auto *wu = s.second.getWUModel();

        // Initialize derived parameters
        s.second.initDerivedParams(m_DT);

        if (!wu->getSimCode().empty()) {
            // analyze which neuron variables need queues
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSimCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSimCode());
        }

        if (!wu->getLearnPostCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getLearnPostCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getLearnPostCode());
        }

        if (!wu->getSynapseDynamicsCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSynapseDynamicsCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSynapseDynamicsCode());
        }
    }

    // CURRENT SOURCES
    for(auto &cs : m_LocalCurrentSources) {
        // Initialize derived parameters
        cs.second.initDerivedParams(m_DT);
    }

    // Merge incoming postsynaptic models
    for(auto &n : m_LocalNeuronGroups) {
        if(!n.second.getInSyn().empty()) {
            n.second.mergeIncomingPSM(m_ShouldMergePostsynapticModels);
        }
    }
}

std::string ModelSpec::scalarExpr(double val) const
{
    if (m_Precision == "float") {
        return std::to_string((float)val) + "f";
    }
    else if (m_Precision == "double") {
        return std::to_string(val);
    }
    else {
        throw std::runtime_error("Unrecognised floating-point type.");
    }
}


bool ModelSpec::zeroCopyInUse() const
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

NeuronGroupInternal *ModelSpec::findNeuronGroupInternal(const std::string &name)
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
