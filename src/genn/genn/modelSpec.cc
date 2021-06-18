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
#include "code_generator/substitutions.h"

// ------------------------------------------------------------------------
// ModelSpec
// ------------------------------------------------------------------------
// class ModelSpec for specifying a neuronal network model
ModelSpec::ModelSpec()
:   m_TimePrecision(TimePrecision::DEFAULT), m_DT(0.5), m_TimingEnabled(false), m_Seed(0),
    m_DefaultVarLocation(VarLocation::HOST_DEVICE), m_DefaultExtraGlobalParamLocation(VarLocation::HOST_DEVICE),
    m_DefaultSparseConnectivityLocation(VarLocation::HOST_DEVICE), m_DefaultNarrowSparseIndEnabled(false),
    m_ShouldMergePostsynapticModels(false), m_BatchSize(1)
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

unsigned int ModelSpec::getNumNeurons() const
{
    // Return sum of local neuron group sizes
    return std::accumulate(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(), 0,
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

        // Mark any pre or postsyaptic neuron variables referenced in sim code as requiring queues
        if (!wu->getSimCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSimCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSimCode());
        }

        // Mark any pre or postsyaptic neuron variables referenced in event code as requiring queues
        if (!wu->getEventCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getEventCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getEventCode());
        }

        // Mark any pre or postsyaptic neuron variables referenced in postsynaptic update code as requiring queues
        if (!wu->getLearnPostCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getLearnPostCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getLearnPostCode());
        }

        // Mark any pre or postsyaptic neuron variables referenced in synapse dynamics code as requiring queues
        if (!wu->getSynapseDynamicsCode().empty()) {
            s.second.getSrcNeuronGroup()->updatePreVarQueues(wu->getSynapseDynamicsCode());
            s.second.getTrgNeuronGroup()->updatePostVarQueues(wu->getSynapseDynamicsCode());
        }

        // Set flag specifying whether any of this synapse groups variables are referenced by a custom update
        s.second.setWUVarReferencedByCustomUpdate(std::any_of(getCustomWUUpdates().cbegin(), getCustomWUUpdates().cend(),
                                                              [&s](const CustomUpdateWUValueType &cg) { return (cg.second.getSynapseGroup() == &s.second); }));
    }

    // CURRENT SOURCES
    for(auto &cs : m_LocalCurrentSources) {
        // Initialize derived parameters
        cs.second.initDerivedParams(m_DT);
    }

    // Custom update groups
    for(auto &c : m_CustomUpdates) {
        c.second.finalize(getBatchSize());
        c.second.initDerivedParams(m_DT);
    }

    // Custom WUM update groups
    for(auto &c : m_CustomWUUpdates) {
        c.second.finalize(getBatchSize());
        c.second.initDerivedParams(m_DT);
    }

    // Merge incoming postsynaptic models
    for(auto &n : m_LocalNeuronGroups) {
        if(!n.second.getInSyn().empty()) {
            n.second.mergeIncomingPSM(m_ShouldMergePostsynapticModels);
        }
    }

    // Loop through neuron populations and their outgoing synapse populations
    for(auto &n : m_LocalNeuronGroups) {
        for(auto *sg : n.second.getOutSyn()) {
            const auto *wu = sg->getWUModel();

            if(!wu->getEventThresholdConditionCode().empty()) {
                using namespace CodeGenerator;

                // do an early replacement of weight update model parameters and derived parameters
                // **NOTE** this is really gross but I can't really see an alternative - merging decisions are based on the spike event conditions set
                // **NOTE** we do not substitute EGP names here as they aren't known and don't effect merging
                // **NOTE** this prevents heterogeneous parameters being allowed in event threshold conditions but I can't see any way around this
                Substitutions thresholdSubs;
                thresholdSubs.addParamValueSubstitution(wu->getParamNames(), sg->getWUParams());
                thresholdSubs.addVarValueSubstitution(wu->getDerivedParams(), sg->getWUDerivedParams());
                
                std::string eCode = wu->getEventThresholdConditionCode();
                thresholdSubs.apply(eCode);

                // Add code and name of support code namespace to set	
                n.second.addSpkEventCondition(eCode, sg);
            }
        }
        if (n.second.getSpikeEventCondition().size() > 1) {
            if(n.second.isSpikeEventTimeRequired() || n.second.isPrevSpikeEventTimeRequired()) {
                LOGW << "Neuron group '" << n.first << "' records spike-like-event times but, it has outgoing synapse groups with multiple spike-like-event conditions so the recorded times may be ambiguous.";
            }
            for(auto *sg : n.second.getOutSyn()) {
                if (!sg->getWUModel()->getEventCode().empty()) {
                    sg->setEventThresholdReTestRequired(true);
                }
            }
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
    if(std::any_of(std::begin(m_LocalNeuronGroups), std::end(m_LocalNeuronGroups),
                   [](const NeuronGroupValueType &n){ return n.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any current sources use zero copy return true
    if(std::any_of(std::begin(m_LocalCurrentSources), std::end(m_LocalCurrentSources),
                   [](const CurrentSourceValueType &c){ return c.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any synapse groups use zero copy return true
    if(std::any_of(std::begin(m_LocalSynapseGroups), std::end(m_LocalSynapseGroups),
                   [](const SynapseGroupValueType &s){ return s.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

     // If any custom updates use zero copy return true
     /*if(std::any_of(std::begin(m_CustomUpdates), std::end(m_CustomUpdates),
                   [](const CustomUpdateValueType &c){ return c.second.isZeroCopyEnabled(); }))
    {
        return true;
    }*/

    return false;
}

bool ModelSpec::isRecordingInUse() const
{
    return std::any_of(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(),
                       [](const NeuronGroupValueType &n) { return n.second.isRecordingEnabled(); });
}

NeuronGroupInternal *ModelSpec::findNeuronGroupInternal(const std::string &name)
{
    // If a matching local neuron group is found, return it
    auto localNeuronGroup = m_LocalNeuronGroups.find(name);
    if(localNeuronGroup != m_LocalNeuronGroups.cend()) {
        return &localNeuronGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("neuron group " + name + " not found, aborting ...");
    }
}

SynapseGroupInternal *ModelSpec::findSynapseGroupInternal(const std::string &name)
{
    // If a matching local synapse group is found, return it
    auto synapseGroup = m_LocalSynapseGroups.find(name);
    if(synapseGroup != m_LocalSynapseGroups.cend()) {
        return &synapseGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("synapse group " + name + " not found, aborting ...");
    }
}