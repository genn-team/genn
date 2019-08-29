#include "neuronGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "gennUtils.h"

// ------------------------------------------------------------------------
// NeuronGroup
// ------------------------------------------------------------------------
void NeuronGroup::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getNeuronModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void NeuronGroup::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getNeuronModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getNeuronModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
VarLocation NeuronGroup::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getNeuronModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation NeuronGroup::getExtraGlobalParamLocation(const std::string &paramName) const
{
    return m_ExtraGlobalParamLocation[getNeuronModel()->getExtraGlobalParamIndex(paramName)];
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeTimeRequired() const
{
    // If any INCOMING synapse groups require POSTSYNAPTIC spike times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
        [](SynapseGroup *sg){ return sg->getWUModel()->isPostSpikeTimeRequired(); }))
    {
        return true;
    }

    // If any OUTGOING synapse groups require PRESYNAPTIC spike times, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
        [](SynapseGroup *sg){ return sg->getWUModel()->isPreSpikeTimeRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isTrueSpikeRequired() const
{
    // If any OUTGOING synapse groups require true spikes, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
        [](SynapseGroupInternal *sg){ return sg->isTrueSpikeRequired(); }))
    {
        return true;
    }

    // If any INCOMING synapse groups require postsynaptic learning, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
        [](SynapseGroupInternal *sg){ return !sg->getWUModel()->getLearnPostCode().empty(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeEventRequired() const
{
     // Spike like events are required if any OUTGOING synapse groups require spike like events
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroupInternal *sg){ return sg->isSpikeEventRequired(); });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isZeroCopyEnabled() const
{
    // If any bits of spikes require zero-copy return true
    if((m_SpikeLocation & VarLocation::ZERO_COPY) || (m_SpikeEventLocation & VarLocation::ZERO_COPY) || (m_SpikeTimeLocation & VarLocation::ZERO_COPY)) {
        return true;
    }

    // If there are any variables implemented in zero-copy mode return true
    if(std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSimRNGRequired() const
{
    // Returns true if any parts of the neuron code require an RNG
    if(Utils::isRNGRequired(getNeuronModel()->getSimCode())
        || Utils::isRNGRequired(getNeuronModel()->getThresholdConditionCode())
        || Utils::isRNGRequired(getNeuronModel()->getResetCode()))
    {
        return true;
    }

    // Return true if any current sources require an RNG for simulation
    if(std::any_of(m_CurrentSources.cbegin(), m_CurrentSources.cend(),
        [](const CurrentSourceInternal *cs){ return cs->isSimRNGRequired(); }))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups require an RNG in their postsynaptic model
    // **NOTE** these are included as they are simulated in the neuron kernel/function
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroupInternal *sg)
                       {
                           return (Utils::isRNGRequired(sg->getPSModel()->getApplyInputCode()) ||
                                   Utils::isRNGRequired(sg->getPSModel()->getDecayCode()));
                       });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isInitRNGRequired(m_VarInitialisers)) {
        return true;
    }

    // Return true if any current sources require an RNG for initialisation
    if(std::any_of(m_CurrentSources.cbegin(), m_CurrentSources.cend(),
        [](const CurrentSourceInternal *cs){ return cs->isInitRNGRequired(); }))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups have state variables which require an RNG to initialise
    // **NOTE** these are included here as they are initialised in neuron initialisation threads
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroupInternal *sg){ return sg->isPSInitRNGRequired(); });
}
//----------------------------------------------------------------------------
bool NeuronGroup::hasOutputToHost(int targetHostID) const
{
    // Return true if any of the outgoing synapse groups have target populations on specified host ID
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [targetHostID](SynapseGroupInternal *sg)
                       {
                           return (sg->getTrgNeuronGroup()->getClusterHostID() == targetHostID);
                       });

}
//----------------------------------------------------------------------------
void NeuronGroup::injectCurrent(CurrentSourceInternal *src)
{
    m_CurrentSources.push_back(src);
}
//----------------------------------------------------------------------------
void NeuronGroup::checkNumDelaySlots(unsigned int requiredDelay)
{
    if (requiredDelay >= getNumDelaySlots())
    {
        m_NumDelaySlots = requiredDelay + 1;
    }
}
//----------------------------------------------------------------------------
void NeuronGroup::updatePreVarQueues(const std::string &code)
{
    updateVarQueues(code, "_pre");
}
//----------------------------------------------------------------------------
void NeuronGroup::updatePostVarQueues(const std::string &code)
{
    updateVarQueues(code, "_post");
}
//----------------------------------------------------------------------------
void NeuronGroup::initDerivedParams(double dt)
{
    auto derivedParams = getNeuronModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(m_Params, dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//----------------------------------------------------------------------------
void NeuronGroup::mergeIncomingPSM(bool merge)
{
    // Create a copy of this neuron groups incoming synapse populations
    std::vector<SynapseGroupInternal*> inSyn = getInSyn();

    // Loop through un-merged incoming synapse populations
    for(unsigned int i = 0; !inSyn.empty(); i++) {
        // Remove last element from vector
        SynapseGroupInternal *a = inSyn.back();
        inSyn.pop_back();

        // Add A to vector of merged incoming synape populations - initially only merged with itself
        m_MergedInSyn.emplace_back(a, std::vector<SynapseGroupInternal*>{a});

        // Continue if merging of postsynaptic models is disabled
        if(!merge) {
            continue;
        }

        // Continue if postsynaptic model has any variables
        // **NOTE** many models with variables would work fine, but nothing stops
        // initialisers being used to configure PS models to behave totally different
        if(!a->getPSVarInitialisers().empty()) {
            continue;
        }

        // Create a name for mmerged
        const std::string mergedPSMName = "Merged" + std::to_string(i) + "_" + getName();

        // Cache useful bits from A
        const auto &aParamsBegin = a->getPSParams().cbegin();
        const auto &aParamsEnd = a->getPSParams().cend();
        const auto &aDerivedParamsBegin = a->getPSDerivedParams().cbegin();
        const auto &aDerivedParamsEnd = a->getPSDerivedParams().cend();
        const auto aModelTypeHash = typeid(a->getPSModel()).hash_code();


        // Loop through remainder of incoming synapse populations
        for(auto b = inSyn.begin(); b != inSyn.end();) {
            // If synapse population b has the same model type as a and; their varmodes, parameters and derived parameters match
            if(typeid((*b)->getPSModel()).hash_code() == aModelTypeHash
                && a->getInSynLocation() == (*b)->getInSynLocation()
                && a->getMaxDendriticDelayTimesteps() == (*b)->getMaxDendriticDelayTimesteps()
                && std::equal(aParamsBegin, aParamsEnd, (*b)->getPSParams().cbegin())
                && std::equal(aDerivedParamsBegin, aDerivedParamsEnd, (*b)->getPSDerivedParams().cbegin()))
            {
                LOGD << "Merging '" << (*b)->getName() << "' with '" << a->getName() << "' into '" << mergedPSMName << "'";

                // Add to list of merged synapses
                m_MergedInSyn.back().second.push_back(*b);

                // Set b's merge target to our unique name
                (*b)->setPSModelMergeTarget(mergedPSMName);

                // Remove from temporary vector
                b = inSyn.erase(b);
            }
            // Otherwise, advance to next synapse group
            else {
                LOGD << "Unable to merge '" << (*b)->getName() << "' with '" << a->getName() << "'";
                ++b;
            }
        }

        // If synapse group A was successfully merged with anything, set it's merge target to the unique name
        if(m_MergedInSyn.back().second.size() > 1) {
            a->setPSModelMergeTarget(mergedPSMName);
        }
    }
}
//----------------------------------------------------------------------------
void NeuronGroup::addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace)
{
    m_SpikeEventCondition.insert(std::pair<std::string, std::string>(code, supportCodeNamespace));
}
//----------------------------------------------------------------------------
bool NeuronGroup::isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const
{
    // Loop through event conditions
    for(const auto &spkEventCond : m_SpikeEventCondition) {
        // If the event threshold code contains this parameter
        // (in it's non-uniquified form), set flag and stop searching
        if(spkEventCond.first.find(pnamefull) != std::string::npos) {
            return true;
        }
    }

    return false;
}
//----------------------------------------------------------------------------
std::string NeuronGroup::getCurrentQueueOffset(const std::string &devPrefix) const
{
    assert(isDelayRequired());

    return "(" + devPrefix + "spkQuePtr" + getName() + " * " + std::to_string(getNumNeurons()) + ")";
}
//----------------------------------------------------------------------------
std::string NeuronGroup::getPrevQueueOffset(const std::string &devPrefix) const
{
    assert(isDelayRequired());

    return "(((" + devPrefix + "spkQuePtr" + getName() + " + " + std::to_string(getNumDelaySlots() - 1) + ") % " + std::to_string(getNumDelaySlots()) + ") * " + std::to_string(getNumNeurons()) + ")";
}
//----------------------------------------------------------------------------
bool NeuronGroup::isVarQueueRequired(const std::string &var) const
{
    // Return flag corresponding to variable
    return m_VarQueueRequired[getNeuronModel()->getVarIndex(var)];
}
//----------------------------------------------------------------------------
void NeuronGroup::updateVarQueues(const std::string &code, const std::string &suffix)
{
    // Loop through variables
    const auto vars = getNeuronModel()->getVars();
    for(size_t i = 0; i < vars.size(); i++) {
        // If the code contains a reference to this variable, set corresponding flag
        if (code.find(vars[i].name + suffix) != std::string::npos) {
            m_VarQueueRequired[i] = true;
        }
    }
}

