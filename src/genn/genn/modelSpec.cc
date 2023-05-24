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
#include "gennUtils.h"
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/substitutions.h"

// ---------------------------------------------------------------------------
// GeNN::ModelSpec
// ---------------------------------------------------------------------------
namespace GeNN
{
ModelSpec::ModelSpec()
:   m_Precision(Type::Float), m_TimePrecision(std::nullopt), m_DT(0.5), m_TimingEnabled(false), m_Seed(0),
    m_DefaultVarLocation(VarLocation::HOST_DEVICE), m_DefaultExtraGlobalParamLocation(VarLocation::HOST_DEVICE),
    m_DefaultSparseConnectivityLocation(VarLocation::HOST_DEVICE), m_DefaultNarrowSparseIndEnabled(false),
    m_ShouldFusePostsynapticModels(false), m_ShouldFusePrePostWeightUpdateModels(false), m_BatchSize(1)
{
}
// ---------------------------------------------------------------------------
ModelSpec::~ModelSpec() 
{
}
// ---------------------------------------------------------------------------
void ModelSpec::setPrecision(const Type::ResolvedType &precision)
{
    if (!precision.isNumeric()) {
        throw std::runtime_error("Only numeric types can be used for precision");
    }
    else {
        if (precision.getNumeric().isIntegral) {
            throw std::runtime_error("Only floating point types can be used for precision");
        }
        m_Precision = precision;
    }
}
// ---------------------------------------------------------------------------
void ModelSpec::setTimePrecision(const Type::ResolvedType &timePrecision)
{ 
    if (!timePrecision.isNumeric()) {
        throw std::runtime_error("Only numeric types can be used for timeprecision");
    }
    else {
        if (timePrecision.getNumeric().isIntegral) {
            throw std::runtime_error("Only floating point types can be used for time precision");
        }
        m_TimePrecision = timePrecision; 
    }
}
// ---------------------------------------------------------------------------
unsigned int ModelSpec::getNumNeurons() const
{
    // Return sum of local neuron group sizes
    return std::accumulate(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}
// ---------------------------------------------------------------------------
NeuronGroup *ModelSpec::addNeuronPopulation(const std::string &name, unsigned int size, const NeuronModels::Base *model,
                                            const ParamValues &paramValues, const VarValues &varInitialisers)
{
    // Add neuron group to map
    auto result = m_LocalNeuronGroups.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, size, model,
                              paramValues, varInitialisers, 
                              m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

    if(!result.second) {
        throw std::runtime_error("Cannot add a neuron population with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
CurrentSource *ModelSpec::addCurrentSource(const std::string &currentSourceName, const CurrentSourceModels::Base *model, const std::string &targetNeuronGroupName, 
                                           const ParamValues &paramValues, const VarValues &varInitialisers)
{
    auto targetGroup = findNeuronGroupInternal(targetNeuronGroupName);

    // Add current source to map
    auto result = m_LocalCurrentSources.emplace(std::piecewise_construct,
        std::forward_as_tuple(currentSourceName),
        std::forward_as_tuple(currentSourceName, model, paramValues,
                              varInitialisers, targetGroup, 
                              m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

    if(!result.second) {
        throw std::runtime_error("Cannot add a current source with duplicate name:" + currentSourceName);
    }
    else {
        targetGroup->injectCurrent(&result.first->second);
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomUpdate *ModelSpec::addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model,
                                         const ParamValues &paramValues, const VarValues &varInitialisers,
                                         const VarReferences &varReferences)
{
    // Add neuron group to map
    auto result = m_CustomUpdates.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, updateGroupName, model,
                              paramValues, varInitialisers, varReferences,
                              m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomConnectivityUpdate *ModelSpec::addCustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, 
                                                                 const std::string &targetSynapseGroupName, const CustomConnectivityUpdateModels::Base *model, 
                                                                 const ParamValues &paramValues, const VarValues &varInitialisers,
                                                                 const VarValues &preVarInitialisers, const VarValues &postVarInitialisers,
                                                                 const WUVarReferences &varReferences, const VarReferences &preVarReferences,
                                                                 const VarReferences &postVarReferences)
{
    // Find target synapse group
    auto targetSynapseGroup = findSynapseGroupInternal(targetSynapseGroupName);

    // Add neuron group to map
    auto result = m_CustomConnectivityUpdates.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, updateGroupName, targetSynapseGroup, model,
                              paramValues, varInitialisers, preVarInitialisers, postVarInitialisers, 
                              varReferences, preVarReferences, postVarReferences, 
                              m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom connectivity update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomUpdateWU *ModelSpec::addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model, 
                                           const ParamValues &paramValues, const VarValues &varInitialisers,
                                           const WUVarReferences &varReferences)
{
    // Add neuron group to map
    auto result = m_CustomWUUpdates.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, updateGroupName, model,
                              paramValues, varInitialisers, varReferences,
                              m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation));

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
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
    }

    // CURRENT SOURCES
    for(auto &cs : m_LocalCurrentSources) {
        // Initialize derived parameters
        cs.second.initDerivedParams(m_DT);
    }

    // Custom update groups
    for(auto &c : m_CustomUpdates) {
        c.second.finalize(m_BatchSize);
        c.second.initDerivedParams(m_DT);
    }

    // Custom WUM update groups
    for(auto &c : m_CustomWUUpdates) {
        c.second.finalize(m_BatchSize);
        c.second.initDerivedParams(m_DT);
    }

    // Custom connectivity update groups
    for (auto &c : m_CustomConnectivityUpdates) {
        c.second.finalize(m_BatchSize);
        c.second.initDerivedParams(m_DT);
    }

    // Merge incoming postsynaptic models
    for(auto &n : m_LocalNeuronGroups) {
        n.second.fusePrePostSynapses(m_ShouldFusePostsynapticModels, m_ShouldFusePrePostWeightUpdateModels);
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
// ---------------------------------------------------------------------------
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
     if(std::any_of(std::begin(m_CustomUpdates), std::end(m_CustomUpdates),
                   [](const CustomUpdateValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

     // If any custom WU updates use zero copy return true
     if(std::any_of(std::begin(m_CustomWUUpdates), std::end(m_CustomWUUpdates),
                   [](const CustomUpdateWUValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

     // If any custom connectivity updates use zero copy return true
     if(std::any_of(std::begin(m_CustomConnectivityUpdates), std::end(m_CustomConnectivityUpdates),
                   [](const CustomConnectivityUpdateValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

    return false;
}
// ---------------------------------------------------------------------------
bool ModelSpec::isRecordingInUse() const
{
    return std::any_of(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(),
                       [](const NeuronGroupValueType &n) { return n.second.isRecordingEnabled(); });
}
// ---------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpec::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    Utils::updateHash(getName(), hash);
    Type::updateHash(getPrecision(), hash);
    Type::updateHash(getTimePrecision(), hash);
    Utils::updateHash(getDT(), hash);
    Utils::updateHash(isTimingEnabled(), hash);
    Utils::updateHash(getBatchSize(), hash);
    Utils::updateHash(getSeed(), hash);

    return hash.get_digest();
}
// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
SynapseGroup *ModelSpec::addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, unsigned int delaySteps, const std::string& src, const std::string& trg,
                                              const WeightUpdateModels::Base *wum, const ParamValues &weightParamValues, const VarValues &weightVarInitialisers, const VarValues &weightPreVarInitialisers, const VarValues &weightPostVarInitialisers,
                                              const PostsynapticModels::Base *psm, const ParamValues &postsynapticParamValues, const VarValues &postsynapticVarInitialisers,
                                              const InitSparseConnectivitySnippet::Init &connectivityInitialiser, const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser)
{
    // Get source and target neuron groups
    auto srcNeuronGrp = findNeuronGroupInternal(src);
    auto trgNeuronGrp = findNeuronGroupInternal(trg);

    // Add synapse group to map
    auto result = m_LocalSynapseGroups.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(name, mtype, delaySteps,
                                wum, weightParamValues, weightVarInitialisers, weightPreVarInitialisers, weightPostVarInitialisers,
                                psm, postsynapticParamValues, postsynapticVarInitialisers,
                                srcNeuronGrp, trgNeuronGrp,
                                connectivityInitialiser, toeplitzConnectivityInitialiser, 
                                m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation,
                                m_DefaultSparseConnectivityLocation, m_DefaultNarrowSparseIndEnabled));

    if(!result.second) {
        throw std::runtime_error("Cannot add a synapse population with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
}   // namespace GeNN