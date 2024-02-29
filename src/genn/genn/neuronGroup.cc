#include "neuronGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <vector>

// GeNN includes
#include "currentSourceInternal.h"
#include "logging.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "gennUtils.h"

using namespace GeNN;

// ------------------------------------------------------------------------
// Anonymous namespace
// ------------------------------------------------------------------------
namespace
{
template<typename T, typename D>
void updateHashList(const NeuronGroup *ng, const std::vector<T*> &objects, boost::uuids::detail::sha1 &hash, D getHashDigestFunc)
{
    // Build vector to hold digests
    std::vector<boost::uuids::detail::sha1::digest_type> digests;
    digests.reserve(objects.size());

    // Loop through objects and add their digests to vector
    for(auto *o : objects) {
        digests.push_back(std::invoke(getHashDigestFunc, o, ng));
    }
    // Sort digests
    std::sort(digests.begin(), digests.end());

    // Concatenate the digests to the hash
    Utils::updateHash(digests, hash);
}
// ------------------------------------------------------------------------
template<typename M, typename H, typename T>
void fuseSynapseGroups(const NeuronGroup *ng, const std::vector<SynapseGroupInternal*> &unfusedSyn, bool fuse, std::vector<SynapseGroupInternal*> &fusedSyn,
                       const std::string &logDescription, M isSynFusableFunc, H getSynFusedHashFunc, T setSynFuseTargetFunc)
{
    // Create a copy of list of synapse groups
    std::vector<SynapseGroupInternal*> syn = unfusedSyn;

    // Loop through un-merged synapse groups
    for(unsigned int i = 0; !syn.empty(); i++) {
        // Remove last element from vector
        auto *a = syn.back();
        syn.pop_back();

        // Add A to vector of fused groups
        fusedSyn.push_back(a);

        // Continue if fusing is disabled
        if(!fuse) {
            continue;
        }

        // If this synapse group can be fused at all
        if(!std::invoke(isSynFusableFunc, a, ng)) {
            continue;
        }

        // Get hash digest used for checking compatibility
        const auto aHashDigest = std::invoke(getSynFusedHashFunc, a, ng);

        // Loop through remainder of synapse groups
        bool anyMerged = false;
        for(auto b = syn.begin(); b != syn.end();) {
            // If synapse group b can be fused with others and it's compatible with a
            if(std::invoke(isSynFusableFunc, *b, ng) && (aHashDigest == std::invoke(getSynFusedHashFunc, *b, ng))) {
                LOGD_GENN << "Fusing " << logDescription << " of '" << (*b)->getName() << "' with '" << a->getName() << "'";

                // Set b's merge target to ourselves
                std::invoke(setSynFuseTargetFunc, *b, ng, *a);

                // Remove from temporary vector
                b = syn.erase(b);

                // Set flag
                anyMerged = true;
            }
            // Otherwise, advance to next synapse group
            else {
                LOGD_GENN << "Unable to merge " << logDescription << " of '" << (*b)->getName() << "' with '" << a->getName() << "'";
                ++b;
            }
        }

        // If synapse group A was successfully merged with anything, set it's merge target to the unique name
        if(anyMerged) {
            std::invoke(setSynFuseTargetFunc, a, ng, *a);
        }
    }
}
}   // Anonymous namespace

// ------------------------------------------------------------------------
// GeNN::NeuronGroup
// ------------------------------------------------------------------------
namespace GeNN
{
void NeuronGroup::setVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getModel()->getVar(varName)) {
        throw std::runtime_error("Unknown neuron model variable '" + varName + "'");
    }
    m_VarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void NeuronGroup::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getModel()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown neuron model extra global parameter '" + paramName + "'");
    }
    m_ExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void NeuronGroup::setParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getModel()->getParam(paramName)) {
        throw std::runtime_error("Unknown neuron model parameter '" + paramName + "'");
    }
    m_DynamicParams.set(paramName, dynamic); 
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeTimeRequired() const
{
    // If any INCOMING synapse groups require POSTSYNAPTIC spike times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPostSpikeTimeRequired(); }))
    {
        return true;
    }

    // If any OUTGOING synapse groups require PRESYNAPTIC spike times, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPreSpikeTimeRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isPrevSpikeTimeRequired() const
{
    // If any INCOMING synapse groups require previous POSTSYNAPTIC spike times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPrevPostSpikeTimeRequired(); }))
    {
        return true;
    }

    // If any OUTGOING synapse groups require previous PRESYNAPTIC spike times, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPrevPreSpikeTimeRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeEventTimeRequired() const
{
    // If any INCOMING synapse groups require POSTSYNAPTIC spike-like event times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPostSpikeEventTimeRequired(); }))
    {
        return true;
    }
    
    // If any OUTGOING synapse groups require PRESYNAPTIC spike-like event times, return true
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroup *sg) { return sg->isPreSpikeEventTimeRequired(); });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isPrevSpikeEventTimeRequired() const
{
    // If any INCOMING synapse groups require previous POSTSYNAPTIC spike-like event times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroup *sg){ return sg->isPrevPostSpikeEventTimeRequired(); }))
    {
        return true;
    }

    // If any OUTGOING synapse groups require previous PRESYNAPTIC spike-like event times, return true
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroup *sg) { return sg->isPrevPreSpikeEventTimeRequired(); });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isTrueSpikeRequired() const
{
    // If any OUTGOING synapse groups require presynaptic spikes, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](SynapseGroupInternal *sg){ return sg->isPreSpikeRequired(); }))
    {
        return true;
    }

    // If any INCOMING synapse groups require postsynaptic spikes, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroupInternal *sg){ return sg->isPostSpikeRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeEventRequired() const
{
    // If any OUTGOING synapse groups require presynaptic spike events, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](SynapseGroupInternal *sg){ return sg->isPreSpikeEventRequired(); }))
    {
        return true;
    }

    // If any INCOMING synapse groups require postsynaptic spike events, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](SynapseGroupInternal *sg){ return sg->isPostSpikeEventRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isZeroCopyEnabled() const
{
    // If any bits of spikes require zero-copy return true
    if(m_RecordingZeroCopyEnabled || (m_SpikeLocation & VarLocation::ZERO_COPY) 
       || (m_SpikeEventLocation & VarLocation::ZERO_COPY) || (m_SpikeTimeLocation & VarLocation::ZERO_COPY) 
       || (m_PrevSpikeTimeLocation & VarLocation::ZERO_COPY) || (m_SpikeEventTimeLocation& VarLocation::ZERO_COPY) 
       || (m_PrevSpikeEventTimeLocation& VarLocation::ZERO_COPY)) 
    {
        return true;
    }

    // If there are any variables implemented in zero-copy mode return true
    return (m_VarLocation.anyZeroCopy() || m_ExtraGlobalParamLocation.anyZeroCopy());
}

//----------------------------------------------------------------------------
bool NeuronGroup::isRecordingEnabled() const
{
    // Return true if spike recording is enabled
    if(m_SpikeRecordingEnabled) {
        return true;
    }

    // Return true if spike event recording is enabled
    if(m_SpikeEventRecordingEnabled) {
        return true;
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool NeuronGroup::isVarInitRequired() const
{
    // Returns true if any neuron variables require initialisation
    if(std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                   [](const auto &init)
                   { 
                       return !Utils::areTokensEmpty(init.second.getCodeTokens());
                   }))
    {
        return true;
    }

    // Return true if any current sources variables require initialisation
    if(std::any_of(getCurrentSources().cbegin(), getCurrentSources().cend(),
                   [](const auto *cs){ return cs->isVarInitRequired(); }))
    {
        return true;
    }
    
    // Return true if any incoming synapse groups have 
    // postsynaptic model variables which require initialisation
    if(std::any_of(getFusedPSMInSyn().cbegin(), getFusedPSMInSyn().cend(),
                   [](const auto *sg){ return sg->isPSVarInitRequired(); }))
    {
        return true;
    }

    // Return true if any incoming synapse groups have postsynaptic
    // weight update model variables which require initialisation
    const auto fusedInSynWithPostVars = getFusedInSynWithPostVars();
    if(std::any_of(fusedInSynWithPostVars.cbegin(), fusedInSynWithPostVars.cend(),
                   [](const auto *sg){ return sg->isWUPostVarInitRequired(); }))
    {
        return true;
    }

    // Return true if any outgoing synapse groups have presynaptic
    // weight update model variables which require initialisation
    const auto fusedOutSynWithPreVars = getFusedOutSynWithPreVars();
    if(std::any_of(fusedOutSynWithPreVars.cbegin(), fusedOutSynWithPreVars.cend(),
                   [](const auto *sg){ return sg->isWUPreVarInitRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
void NeuronGroup::injectCurrent(CurrentSourceInternal *src)
{
    m_CurrentSourceGroups.push_back(src);
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSimRNGRequired() const
{
    // Returns true if any parts of the neuron code require an RNG
    if(Utils::isRNGRequired(getSimCodeTokens())
        || Utils::isRNGRequired(getThresholdConditionCodeTokens())
        || Utils::isRNGRequired(getResetCodeTokens()))
    {
        return true;
    }

    // Return true if any current sources require an RNG for simulation
    if(std::any_of(m_CurrentSourceGroups.cbegin(), m_CurrentSourceGroups.cend(),
        [](const CurrentSourceInternal *cs){ return Utils::isRNGRequired(cs->getInjectionCodeTokens()); }))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups require an RNG in their postsynaptic model
    // **NOTE** these are included as they are simulated in the neuron kernel/function
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroupInternal *sg)
                       {
                           return Utils::isRNGRequired(sg->getPSInitialiser().getSimCodeTokens());
                       });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(m_VarInitialisers)) {
        return true;
    }

    // Return true if any current sources require an RNG for initialisation
    if(std::any_of(m_CurrentSourceGroups.cbegin(), m_CurrentSourceGroups.cend(),
        [](const CurrentSourceInternal *cs){ return Utils::isRNGRequired(cs->getVarInitialisers()); }))
    {
        return true;
    }

    // Return true if any incoming synapse groups require and RNG to initialize their postsynaptic variables
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](const SynapseGroupInternal *sg) { return Utils::isRNGRequired(sg->getWUInitialiser().getPostVarInitialisers()); }))
    {
        return true;
    }

    // Return true if any outgoing synapse groups require and RNG to initialize their presynaptic variables
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](const SynapseGroupInternal *sg) { return Utils::isRNGRequired(sg->getWUInitialiser().getPreVarInitialisers()); }))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups have state variables which require an RNG to initialise
    // **NOTE** these are included here as they are initialised in neuron initialisation threads
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroupInternal *sg){ return Utils::isRNGRequired(sg->getPSInitialiser().getVarInitialisers()); });
}
//----------------------------------------------------------------------------
NeuronGroup::NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                         const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_NumNeurons(numNeurons), m_Model(neuronModel), m_Params(params), m_VarInitialisers(varInitialisers),
    m_NumDelaySlots(1), m_RecordingZeroCopyEnabled(false), m_SpikeLocation(defaultVarLocation), m_SpikeEventLocation(defaultVarLocation),
    m_SpikeTimeLocation(defaultVarLocation), m_PrevSpikeTimeLocation(defaultVarLocation), m_SpikeEventTimeLocation(defaultVarLocation), 
    m_PrevSpikeEventTimeLocation(defaultVarLocation), m_VarLocation(defaultVarLocation), m_ExtraGlobalParamLocation(defaultExtraGlobalParamLocation),
    m_SpikeRecordingEnabled(false), m_SpikeEventRecordingEnabled(false)
{
    // Validate names
    Utils::validatePopName(name, "Neuron group");
    getModel()->validate(getParams(), getVarInitialisers(), "Neuron group " + getName());

     // Scan neuron model code strings
    m_SimCodeTokens = Utils::scanCode(getModel()->getSimCode(), 
                                      "Neuron group '" + getName() + "' sim code");
    m_ThresholdConditionCodeTokens = Utils::scanCode(getModel()->getThresholdConditionCode(),
                                                     "Neuron group '" + getName() + "' threshold condition code");
    m_ResetCodeTokens = Utils::scanCode(getModel()->getResetCode(),
                                        "Neuron group '" + getName() + "' reset code");
}
//----------------------------------------------------------------------------
void NeuronGroup::checkNumDelaySlots(unsigned int requiredDelay)
{
    if (requiredDelay >= getNumDelaySlots()) {
        m_NumDelaySlots = requiredDelay + 1;
    }
}
//----------------------------------------------------------------------------
void NeuronGroup::finalise(double dt)
{
    auto derivedParams = getModel()->getDerivedParams();

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.emplace(d.name, d.func(m_Params, dt));
    }

    // Finalise variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }
}
//----------------------------------------------------------------------------
void NeuronGroup::fusePrePostSynapses(bool fusePSM, bool fusePrePostWUM)
{
    // If there are any incoming synapse groups
    if(!getInSyn().empty()) {
        // All incoming synapse groups will have postsynaptic models so attempt to merge these directly
        fuseSynapseGroups(this, getInSyn(), fusePSM, m_FusedPSMInSyn, "postsynaptic update",
                          &SynapseGroupInternal::canPSBeFused, &SynapseGroupInternal::getPSFuseHashDigest,
                          &SynapseGroupInternal::setFusedPSTarget);

        // Copy groups with some form of postsynaptic update into new vector
        std::vector<SynapseGroupInternal *> inSynWithPostUpdate;
        std::copy_if(getInSyn().cbegin(), getInSyn().cend(), std::back_inserter(inSynWithPostUpdate),
                     [](SynapseGroupInternal *sg)
                     {
                         return (!Utils::areTokensEmpty(sg->getWUInitialiser().getPostSpikeCodeTokens())
                                 || !Utils::areTokensEmpty(sg->getWUInitialiser().getPostDynamicsCodeTokens())
                                 || !sg->getWUInitialiser().getSnippet()->getPostVars().empty());
                     });

        // If there are any, merge
        if(!inSynWithPostUpdate.empty()) {
            fuseSynapseGroups(this, inSynWithPostUpdate, fusePrePostWUM, m_FusedWUPostInSyn, "postsynaptic weight update",
                              &SynapseGroupInternal::canWUMPrePostUpdateBeFused, &SynapseGroupInternal::getWUPrePostFuseHashDigest,
                              &SynapseGroupInternal::setFusedWUPrePostTarget);
        }
    }

    // Copy groups with some form of presynaptic update into new vector
    std::vector<SynapseGroupInternal *> outSynWithPreUpdate;
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(outSynWithPreUpdate),
                 [](SynapseGroupInternal *sg)
                 {
                     return (!Utils::areTokensEmpty(sg->getWUInitialiser().getPreSpikeCodeTokens())
                             || !Utils::areTokensEmpty(sg->getWUInitialiser().getPreDynamicsCodeTokens())
                             || !sg->getWUInitialiser().getSnippet()->getPreVars().empty());
                 });

     // If there are any
    if(!outSynWithPreUpdate.empty()) {
        fuseSynapseGroups(this, outSynWithPreUpdate, fusePrePostWUM, m_FusedWUPreOutSyn, "presynaptic weight update",
                          &SynapseGroupInternal::canWUMPrePostUpdateBeFused, &SynapseGroupInternal::getWUPrePostFuseHashDigest,
                          &SynapseGroupInternal::setFusedWUPrePostTarget);
    }

    // Copy groups with output onto the presynaptic neuron into new vector
    std::vector<SynapseGroupInternal *> outSynWithPreOutput;
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(outSynWithPreOutput),
                 [](SynapseGroupInternal *sg)
                 {
                     return (sg->isPresynapticOutputRequired());
                 });

    // If there are any
    if(!outSynWithPreOutput.empty()) {
        fuseSynapseGroups(this, outSynWithPreOutput, fusePSM, m_FusedPreOutputOutSyn, "presynaptic synapse output",
                          &SynapseGroupInternal::canPreOutputBeFused, &SynapseGroupInternal::getPreOutputHashDigest,
                          &SynapseGroupInternal::setFusedPreOutputTarget);
    }

    // Copy incoming synapse groups which require back-projected 
    // spikes and outgoing groups which require spikes
    std::vector<SynapseGroupInternal*> synWithSpike;
    std::copy_if(getInSyn().cbegin(), getInSyn().cend(), std::back_inserter(synWithSpike),
                 [](SynapseGroupInternal *sg) { return sg->isPostSpikeRequired(); });
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(synWithSpike),
                 [](SynapseGroupInternal *sg) { return sg->isPreSpikeRequired(); });
    
    // If there are any, fuse together
    if(!synWithSpike.empty()) {
        fuseSynapseGroups(this, synWithSpike, true, m_FusedSpike, "spike",
                          &SynapseGroupInternal::canSpikeBeFused, &SynapseGroupInternal::getSpikeHashDigest,
                          &SynapseGroupInternal::setFusedSpikeTarget);
    }
   
    // Copy incoming synapse groups which require back-projected 
    // spike-events and outgoing groups which require spike-events
    std::vector<SynapseGroupInternal*> synWithSpikeEvent;
    std::copy_if(getInSyn().cbegin(), getInSyn().cend(), std::back_inserter(synWithSpikeEvent),
                 [](SynapseGroupInternal *sg) { return sg->isPostSpikeEventRequired(); });
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(synWithSpikeEvent),
                 [](SynapseGroupInternal *sg) { return sg->isPreSpikeEventRequired(); });
    
    // If there are any, fuse together
    if(!synWithSpikeEvent.empty()) {
        fuseSynapseGroups(this, synWithSpikeEvent, true, m_FusedSpikeEvent, "spike event",
                          &SynapseGroupInternal::canWUSpikeEventBeFused, &SynapseGroupInternal::getWUSpikeEventFuseHashDigest,
                          &SynapseGroupInternal::setFusedSpikeEventTarget);
    }
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getFusedInSynWithPostCode() const
{
    std::vector<SynapseGroupInternal*> vec;
    std::copy_if(getFusedWUPostInSyn().cbegin(), getFusedWUPostInSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg)
                 {
                     return (!Utils::areTokensEmpty(sg->getWUInitialiser().getPostSpikeCodeTokens())
                             || !Utils::areTokensEmpty(sg->getWUInitialiser().getPostDynamicsCodeTokens()));
                 });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getFusedOutSynWithPreCode() const
{
    std::vector<SynapseGroupInternal*> vec;
    std::copy_if(getFusedWUPreOutSyn().cbegin(), getFusedWUPreOutSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg)
                 {
                     return (!Utils::areTokensEmpty(sg->getWUInitialiser().getPreSpikeCodeTokens())
                             || !Utils::areTokensEmpty(sg->getWUInitialiser().getPreDynamicsCodeTokens()));
                });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getFusedInSynWithPostVars() const
{
    std::vector<SynapseGroupInternal *> vec;
    std::copy_if(getFusedWUPostInSyn().cbegin(), getFusedWUPostInSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg) { return !sg->getWUInitialiser().getSnippet()->getPostVars().empty(); });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getFusedOutSynWithPreVars() const
{
    std::vector<SynapseGroupInternal *> vec;
    std::copy_if(getFusedWUPreOutSyn().cbegin(), getFusedWUPreOutSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg) { return !sg->getWUInitialiser().getSnippet()->getPreVars().empty(); });
    return vec;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isVarQueueRequired(const std::string &var) const
{
    return (m_VarQueueRequired.count(var) == 0) ? false : true;
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getModel()->getHashDigest(), hash);
    Utils::updateHash(isSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeTimeRequired(), hash);
    Utils::updateHash(isSpikeEventTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeEventTimeRequired(), hash);
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isTrueSpikeRequired(), hash);
    Utils::updateHash(isSpikeRecordingEnabled(), hash);
    Utils::updateHash(isSpikeEventRecordingEnabled(), hash);
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(m_VarQueueRequired, hash);
    m_DynamicParams.updateHash(hash);

    // Update hash with number of fused spike conditions
    // **NOTE** nothing else is required as logic of each update only depends on number of delay slots
    Utils::updateHash(getFusedSpike().size(), hash);

    // Update hash with hash list built from fused spike event thresholds
    updateHashList(this, getFusedSpikeEvent(), hash, &SynapseGroupInternal::getWUSpikeEventHashDigest);

    // Update hash with hash list built from current sources
    updateHashList(this, getCurrentSources(), hash, &CurrentSourceInternal::getHashDigest);

    // Update hash with hash list built from fused incoming synapse groups with post code
    updateHashList(this, getFusedInSynWithPostCode(), hash, &SynapseGroupInternal::getWUPrePostHashDigest);

    // Update hash with hash list built from fused outgoing synapse groups with pre code
    updateHashList(this, getFusedOutSynWithPreCode(), hash, &SynapseGroupInternal::getWUPrePostHashDigest);

    // Update hash with hash list built from fused incoming synapses
    updateHashList(this, getFusedPSMInSyn(), hash, &SynapseGroupInternal::getPSHashDigest);

    // Update hash with hash list built from fused outgoing synapses with presynaptic output
    updateHashList(this, getFusedPreOutputOutSyn(), hash, &SynapseGroupInternal::getPreOutputHashDigest);
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(isSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeTimeRequired(), hash);
    Utils::updateHash(isSpikeEventTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeEventTimeRequired(), hash);
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isTrueSpikeRequired(), hash);
    Utils::updateHash(isSimRNGRequired(), hash);
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(m_VarQueueRequired, hash);
    Utils::updateHash(getModel()->getVars(), hash);

    // Include variable initialiser hashes
    for(const auto &n : getVarInitialisers()) {
        Utils::updateHash(n.first, hash);
        Utils::updateHash(n.second.getHashDigest(), hash);
    }

    // Update hash with number of fused spike conditions
    // **NOTE** nothing else is required as logic of initialisation only depends on number of delay slots
    Utils::updateHash(getFusedSpike().size(), hash);

    // Update hash with hash list built from current sources
    updateHashList(this, getCurrentSources(), hash, &CurrentSourceInternal::getInitHashDigest);

    // Update hash with hash list built from fused incoming synapse groups with post vars
    updateHashList(this, getFusedInSynWithPostVars(), hash, &SynapseGroupInternal::getWUPrePostInitHashDigest);

    // Update hash with hash list built from fusedoutgoing synapse groups with pre vars
    updateHashList(this, getFusedOutSynWithPreVars(), hash, &SynapseGroupInternal::getWUPrePostInitHashDigest);

    // Update hash with hash list built from fused incoming synapses
    updateHashList(this, getFusedPSMInSyn(), hash, &SynapseGroupInternal::getPSInitHashDigest);

    // Update hash with hash list built from fused outgoing synapses with presynaptic output
    updateHashList(this, getFusedPreOutputOutSyn(), hash, &SynapseGroupInternal::getPreOutputInitHashDigest);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getSpikeQueueUpdateHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getNumDelaySlots(), hash);
    
    // Update hash with number of fused spike and spike event conditions
    // **NOTE** nothing else is required as logic of each update only depends on number of delay slots
    Utils::updateHash(getFusedSpike().size(), hash);
    Utils::updateHash(getFusedSpikeEvent().size(), hash);
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getPrevSpikeTimeUpdateHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getNumDelaySlots(), hash);
    
     // Update hash with number of fused spike and spike event conditions
    // **NOTE** nothing else is required as logic of each update only depends on number of delay slots
    Utils::updateHash(getFusedSpike().size(), hash);
    Utils::updateHash(getFusedSpikeEvent().size(), hash);

    Utils::updateHash(isPrevSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeEventTimeRequired(), hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getSpikeLocation(), hash);
    Utils::updateHash(getSpikeEventLocation(), hash);
    Utils::updateHash(getSpikeTimeLocation(), hash);
    Utils::updateHash(getPrevSpikeTimeLocation(), hash);
    Utils::updateHash(getSpikeEventTimeLocation(), hash);
    Utils::updateHash(getPrevSpikeEventTimeLocation(), hash);
    m_VarLocation.updateHash(hash);
    m_ExtraGlobalParamLocation.updateHash(hash);
    return hash.get_digest();
}
}   // namespace GeNN
