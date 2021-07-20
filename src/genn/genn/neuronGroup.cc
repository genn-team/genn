#include "neuronGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "currentSourceInternal.h"
#include "logging.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "gennUtils.h"

// ------------------------------------------------------------------------
// Anonymous namespace
// ------------------------------------------------------------------------
namespace
{
template<typename T, typename M>
bool checkCompatibleUnordered(const std::vector<T> &ours, std::vector<T> &others, M canMerge)
{
     // If both groups have the same number
    if(ours.size() == others.size()) {
        // Loop through our groups
        for(const auto a : ours) {
            // If a compatible group can be found amongst the other vector, remove it
            const auto b = std::find_if(others.cbegin(), others.cend(),
                                        [a, canMerge](T b)
                                        {
                                            return canMerge(a, b);
                                        });
            if(b != others.cend()) {
                others.erase(b);
            }
            // Otherwise, these can't be merged - return false
            else {
                return false;
            }
        }

        return true;
    }
    else {
        return false;
    }
}
// ------------------------------------------------------------------------
template<typename T, typename D>
void updateHashList(const std::vector<T*> &objects, boost::uuids::detail::sha1 &hash, D getHashDigestFunc)
{
    // Build vector to hold digests
    std::vector<boost::uuids::detail::sha1::digest_type> digests;
    digests.reserve(objects.size());

    // Loop through objects and add their digests to vector
    for(auto *o : objects) {
        digests.push_back((o->*getHashDigestFunc)());
    }
    // Sort digests
    std::sort(digests.begin(), digests.end());

    // Concatenate the digests to the hash
    Utils::updateHash(digests, hash);
}
}   // Anonymous namespace

// ------------------------------------------------------------------------
// NeuronGroup
// ------------------------------------------------------------------------
void NeuronGroup::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation.at(getNeuronModel()->getVarIndex(varName)) = loc;
}
//----------------------------------------------------------------------------
void NeuronGroup::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getNeuronModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getNeuronModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ExtraGlobalParamLocation.at(extraGlobalParamIndex) = loc;
}
//----------------------------------------------------------------------------
VarLocation NeuronGroup::getVarLocation(const std::string &varName) const
{
    return m_VarLocation.at(getNeuronModel()->getVarIndex(varName));
}
//----------------------------------------------------------------------------
VarLocation NeuronGroup::getExtraGlobalParamLocation(const std::string &paramName) const
{
    return m_ExtraGlobalParamLocation.at(getNeuronModel()->getExtraGlobalParamIndex(paramName));
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
bool NeuronGroup::isPrevSpikeTimeRequired() const
{
    // If any INCOMING synapse groups require previous POSTSYNAPTIC spike times, return true
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
        [](SynapseGroup *sg){ return sg->getWUModel()->isPrevPostSpikeTimeRequired(); }))
    {
        return true;
    }

    // If any OUTGOING synapse groups require previous PRESYNAPTIC spike times, return true
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
        [](SynapseGroup *sg){ return sg->getWUModel()->isPrevPreSpikeTimeRequired(); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool NeuronGroup::isSpikeEventTimeRequired() const
{
    // If any OUTGOING synapse groups require PRESYNAPTIC spike-like event times, return true
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroup *sg) { return sg->getWUModel()->isPreSpikeEventTimeRequired(); });
}
//----------------------------------------------------------------------------
bool NeuronGroup::isPrevSpikeEventTimeRequired() const
{
    // If any OUTGOING synapse groups require previous PRESYNAPTIC spike-like event times, return true
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroup *sg) { return sg->getWUModel()->isPrevPreSpikeEventTimeRequired(); });
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
    // Spike like events are required if any OUTGOING synapse groups has a spike like event threshold
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [](SynapseGroupInternal *sg){ return !sg->getWUModel()->getEventThresholdConditionCode().empty(); });
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
    if(Utils::isRNGRequired(m_VarInitialisers)) {
        return true;
    }

    // Return true if any current sources require an RNG for initialisation
    if(std::any_of(m_CurrentSources.cbegin(), m_CurrentSources.cend(),
        [](const CurrentSourceInternal *cs){ return cs->isInitRNGRequired(); }))
    {
        return true;
    }

    // Return true if any incoming synapse groups require and RNG to initialize their postsynaptic variables
    if(std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                   [](const SynapseGroupInternal *sg) { return sg->isWUPostInitRNGRequired(); }))
    {
        return true;
    }

    // Return true if any outgoing synapse groups require and RNG to initialize their presynaptic variables
    if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                   [](const SynapseGroupInternal *sg) { return sg->isWUPreInitRNGRequired(); }))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups have state variables which require an RNG to initialise
    // **NOTE** these are included here as they are initialised in neuron initialisation threads
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroupInternal *sg){ return sg->isPSInitRNGRequired(); });
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
void NeuronGroup::injectCurrent(CurrentSourceInternal *src)
{
    m_CurrentSources.push_back(src);
}
//----------------------------------------------------------------------------
NeuronGroup::NeuronGroup(const std::string &name, int numNeurons, const NeuronModels::Base *neuronModel,
                         const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                         VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_NumNeurons(numNeurons), m_NeuronModel(neuronModel), m_Params(params), m_VarInitialisers(varInitialisers),
    m_NumDelaySlots(1), m_VarQueueRequired(varInitialisers.size(), false), m_SpikeLocation(defaultVarLocation), m_SpikeEventLocation(defaultVarLocation),
    m_SpikeTimeLocation(defaultVarLocation), m_PrevSpikeTimeLocation(defaultVarLocation), m_SpikeEventTimeLocation(defaultVarLocation), m_PrevSpikeEventTimeLocation(defaultVarLocation),
    m_VarLocation(varInitialisers.size(), defaultVarLocation), m_ExtraGlobalParamLocation(neuronModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
    m_SpikeRecordingEnabled(false), m_SpikeEventRecordingEnabled(false)
{
    // Validate names
    Utils::validateVarPopName(name, "Neuron group");
    getNeuronModel()->validate();

    // If any variables have a reduction access mode, give an error
    const auto vars = getNeuronModel()->getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Neuron models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
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

        // Add A to vector of merged incoming synape populations
        m_MergedInSyn.push_back(a);

        // Continue if merging of postsynaptic models is disabled
        if(!merge) {
            continue;
        }

        // If this synapse group's postsynaptic model can be linearly combined with others
        if(!a->canPSBeLinearlyCombined()) {
            continue;
        }

        // Get hash digest used for checking compatibility
        const auto aHashDigest = a->getPSLinearCombineHashDigest();

        // Create a name for mmerged
        const std::string mergedPSMName = "Merged" + std::to_string(i) + "_" + getName();

        // Loop through remainder of incoming synapse populations
        bool anyMerged = false;
        for(auto b = inSyn.begin(); b != inSyn.end();) {
            // If synapse group b's postsynaptic model can be linearly combined with others and it's compatible with a
            if((*b)->canPSBeLinearlyCombined() && (aHashDigest == (*b)->getPSLinearCombineHashDigest())) {
                LOGD_GENN << "Merging '" << (*b)->getName() << "' with '" << a->getName() << "' into '" << mergedPSMName << "'";

                // Set b's merge target to our unique name
                (*b)->setPSModelMergeTarget(mergedPSMName);

                // Remove from temporary vector
                b = inSyn.erase(b);

                // Set flag
                anyMerged = true;
            }
            // Otherwise, advance to next synapse group
            else {
                LOGD_GENN << "Unable to merge '" << (*b)->getName() << "' with '" << a->getName() << "'";
                ++b;
            }
        }

        // If synapse group A was successfully merged with anything, set it's merge target to the unique name
        if(anyMerged) {
            a->setPSModelMergeTarget(mergedPSMName);
        }
    }
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getInSynWithPostCode() const
{
    std::vector<SynapseGroupInternal*> vec;
    std::copy_if(getInSyn().cbegin(), getInSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg)
                 {
                     return (!sg->getWUModel()->getPostSpikeCode().empty()
                             || !sg->getWUModel()->getPostDynamicsCode().empty());
                 });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getOutSynWithPreCode() const
{
    std::vector<SynapseGroupInternal*> vec;
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg)
                 {
                     return (!sg->getWUModel()->getPreSpikeCode().empty()
                             || !sg->getWUModel()->getPreDynamicsCode().empty());
                });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getInSynWithPostVars() const
{
    std::vector<SynapseGroupInternal *> vec;
    std::copy_if(getInSyn().cbegin(), getInSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg) { return !sg->getWUModel()->getPostVars().empty(); });
    return vec;
}
//----------------------------------------------------------------------------
std::vector<SynapseGroupInternal*> NeuronGroup::getOutSynWithPreVars() const
{
    std::vector<SynapseGroupInternal *> vec;
    std::copy_if(getOutSyn().cbegin(), getOutSyn().cend(), std::back_inserter(vec),
                 [](SynapseGroupInternal *sg) { return !sg->getWUModel()->getPreVars().empty(); });
    return vec;
}
//----------------------------------------------------------------------------
void NeuronGroup::addSpkEventCondition(const std::string &code, SynapseGroupInternal *synapseGroup)
{
    const auto *wu = synapseGroup->getWUModel();

    // Determine if any EGPs are required by threshold code
    const auto wuEGPs = wu->getExtraGlobalParams();
    const bool egpInThresholdCode = std::any_of(wuEGPs.cbegin(), wuEGPs.cend(),
                                                [&code](const Snippet::Base::EGP &egp)
                                                {
                                                    return (code.find("$(" + egp.name + ")") != std::string::npos);
                                                });

    // Determine if any presynaptic variables are required by threshold code
    const auto wuPreVars = wu->getPreVars();
    const bool preVarInThresholdCode = std::any_of(wuPreVars.cbegin(), wuPreVars.cend(),
                                                   [&code](const Models::Base::Var &var)
                                                   {
                                                       return (code.find("$(" + var.name + ")") != std::string::npos);
                                                   });

    // Add threshold, support code, synapse group and whether egps are required to set
    m_SpikeEventCondition.emplace(code, wu->getSimSupportCode(), egpInThresholdCode || preVarInThresholdCode, synapseGroup);
}
//----------------------------------------------------------------------------
bool NeuronGroup::isVarQueueRequired(const std::string &var) const
{
    // Return flag corresponding to variable
    return m_VarQueueRequired[getNeuronModel()->getVarIndex(var)];
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getNeuronModel()->getHashDigest(), hash);
    Utils::updateHash(isSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeTimeRequired(), hash);
    //Utils::updateHash(getSpikeEventCondition(), hash); **FIXME**
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isSpikeRecordingEnabled(), hash);
    Utils::updateHash(isSpikeEventRecordingEnabled(), hash);
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(m_VarQueueRequired, hash);

    // Update hash with hash list built from current sources
    updateHashList(getCurrentSources(), hash, &CurrentSourceInternal::getHashDigest);

    // Update hash with hash list built from incoming synapse groups with post code
    updateHashList(getInSynWithPostCode(), hash, &SynapseGroupInternal::getWUPostHashDigest);

    // Update hash with hash list built from outgoing synapse groups with pre code
    updateHashList(getOutSynWithPreCode(), hash, &SynapseGroupInternal::getWUPreHashDigest);

    // Update hash with hash list built from merged incoming synapses
    updateHashList(getMergedInSyn(), hash, &SynapseGroupInternal::getPSHashDigest);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(isSpikeTimeRequired(), hash);
    Utils::updateHash(isPrevSpikeTimeRequired(), hash);
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isSimRNGRequired(), hash);
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(m_VarQueueRequired, hash);
    Utils::updateHash(getNeuronModel()->getVars(), hash);

    // Include variable initialiser hashes
    for(const auto &n : getVarInitialisers()) {
        Utils::updateHash(n.getHashDigest(), hash);
    }

    // Update hash with hash list built from current sources
    updateHashList(getCurrentSources(), hash, &CurrentSourceInternal::getInitHashDigest);

    // Update hash with hash list built from incoming synapse groups with post vars
    updateHashList(getInSynWithPostVars(), hash, &SynapseGroupInternal::getWUPostInitHashDigest);

    // Update hash with hash list built from outgoing synapse groups with pre vars
    updateHashList(getOutSynWithPreVars(), hash, &SynapseGroupInternal::getWUPreInitHashDigest);

    // Update hash with hash list built from merged incoming synapses
    updateHashList(getMergedInSyn(), hash, &SynapseGroupInternal::getPSInitHashDigest);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getSpikeQueueUpdateHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isTrueSpikeRequired(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronGroup::getPrevSpikeTimeUpdateHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getNumDelaySlots(), hash);
    Utils::updateHash(isSpikeEventRequired(), hash);
    Utils::updateHash(isTrueSpikeRequired(), hash);
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
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
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

