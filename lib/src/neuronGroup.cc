#include "neuronGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
#include "standardSubstitutions.h"
#include "synapseGroup.h"
#include "utils.h"

// ------------------------------------------------------------------------
// NeuronGroup
// ------------------------------------------------------------------------
void NeuronGroup::checkNumDelaySlots(unsigned int requiredDelay)
{
    if (requiredDelay >= getNumDelaySlots())
    {
        m_NumDelaySlots = requiredDelay + 1;
    }
}

void NeuronGroup::updateVarQueues(const string &code)
{
    // Loop through variables
    const auto vars = getNeuronModel()->getVars();
    for(size_t i = 0; i < vars.size(); i++) {
        const std::string &varName = vars[i].first;

        // If the code contains a reference to this variable, set corresponding flag
        if (code.find(varName + "_pre") != string::npos) {
            m_AnyVarQueuesRequired = true;
            m_VarQueueRequired[i] = true;
        }
    }
}

void NeuronGroup::setVarMode(const std::string &varName, VarMode mode)
{
    m_VarMode[getNeuronModel()->getVarIndex(varName)] = mode;
}

VarMode NeuronGroup::getVarMode(const std::string &varName) const
{
    return m_VarMode[getNeuronModel()->getVarIndex(varName)];
}

void NeuronGroup::addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace)
{
    m_SpikeEventCondition.insert(std::pair<std::string, std::string>(code, supportCodeNamespace));
}

void NeuronGroup::initDerivedParams(double dt)
{
    auto derivedParams = getNeuronModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.second(m_Params, dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}

void NeuronGroup::calcSizes(unsigned int blockSize,  unsigned int &idStart, unsigned int &paddedIDStart)
{
    // paddedSize is the lowest multiple of neuronBlkSz >= neuronN[i]
    const unsigned int paddedSize = ceil((double)getNumNeurons() / (double) blockSize) * (double) blockSize;

    // Store current cummulative sum in first
    m_IDRange.first = idStart;
    m_PaddedIDRange.first = paddedIDStart;

    // Update global cummulative sums of neurons
    idStart += getNumNeurons();
    paddedIDStart +=  paddedSize;

    // Store cummulative sums of point after this neuron group
    m_IDRange.second = idStart;
    m_PaddedIDRange.second = paddedIDStart;
}

bool NeuronGroup::isVarQueueRequired(const std::string &var) const
{
    // Return flag corresponding to variable
    return m_VarQueueRequired[getNeuronModel()->getVarIndex(var)];
}

bool NeuronGroup::isZeroCopyEnabled() const
{
    // If any bits of spikes require zero-copy return true
    if(isSpikeZeroCopyEnabled() || isSpikeEventZeroCopyEnabled() || isSpikeTimeZeroCopyEnabled()) {
        return true;
    }

    // If there are any variables implemented in zero-copy mode return true
    if(any_of(m_VarMode.begin(), m_VarMode.end(),
        [](VarMode mode){ return (mode & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}

bool NeuronGroup::isParamRequiredBySpikeEventCondition(const std::string &pnamefull) const
{
    // Loop through event conditions
    for(const auto &spkEventCond : m_SpikeEventCondition) {
        // If the event threshold code contains this parameter
        // (in it's non-uniquified form), set flag and stop searching
        if(spkEventCond.first.find(pnamefull) != string::npos) {
            return true;
        }
    }

    return false;
}

void NeuronGroup::addExtraGlobalParams(std::map<string, string> &kernelParameters) const
{
    for(auto const &p : getNeuronModel()->getExtraGlobalParams()) {
        std::string pnamefull = p.first + getName();
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getNeuronModel()->getSimCode().find("$(" + p.first + ")") != string::npos
                || getNeuronModel()->getThresholdConditionCode().find("$(" + p.first + ")") != string::npos
                || getNeuronModel()->getResetCode().find("$(" + p.first + ")") != string::npos) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

bool NeuronGroup::isInitCodeRequired() const
{
    // Return true if any of the variables initialisers have any code
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const NewModels::VarInit &v)
                       {
                           return !v.getSnippet()->getCode().empty();
                       });
}

bool NeuronGroup::isSimRNGRequired() const
{
    // Returns true if any parts of the neuron code require an RNG
    if(::isRNGRequired(getNeuronModel()->getSimCode())
        || ::isRNGRequired(getNeuronModel()->getThresholdConditionCode())
        || ::isRNGRequired(getNeuronModel()->getResetCode()))
    {
        return true;
    }

    // Return true if any of the incoming synapse groups require an RNG in their postsynaptic model
    // **NOTE** these are included as they are simulated in the neuron kernel/function
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroup *sg)
                       {
                           return (::isRNGRequired(sg->getPSModel()->getApplyInputCode()) ||
                                   ::isRNGRequired(sg->getPSModel()->getDecayCode()));
                       });
}

bool NeuronGroup::isInitRNGRequired(VarInit varInitMode) const
{
    // If initialising the neuron variables require an RNG, return true
    if(::isInitRNGRequired(m_VarInitialisers, m_VarMode, varInitMode)) {
        return true;
    }

    // Return true if any of the incoming synapse groups have state variables which require an RNG to initialise
    // **NOTE** these are included here as they are initialised in neuron initialisation threads
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [varInitMode](const SynapseGroup *sg){ return sg->isPSInitRNGRequired(varInitMode); });
}

bool NeuronGroup::isDeviceVarInitRequired() const
{
    // If spike var is initialised on device, return true
    if(m_SpikeVarMode & VarInit::DEVICE) {
        return true;
    }

    // If spike event var is initialised on device, return true
    if(isSpikeEventRequired() && m_SpikeEventVarMode & VarInit::DEVICE) {
        return true;
    }

    // If spike time var is initialised on device, return true
    if(isSpikeTimeRequired() && m_SpikeTimeVarMode & VarInit::DEVICE) {
        return true;
    }

    // Return true if any of the variables are initialised on the device
    if(std::any_of(m_VarMode.cbegin(), m_VarMode.cend(),
                   [](const VarMode mode){ return (mode & VarInit::DEVICE); })) {
        return true;
    }

    // Return true if any of the incoming synapse groups have state variables or input variables which should be initialised on device
    // **NOTE** these are included here as they are initialised in neuron initialisation threads
    return std::any_of(getInSyn().cbegin(), getInSyn().cend(),
                       [](const SynapseGroup *sg)
                       {
                           return sg->isPSDeviceVarInitRequired() || (sg->getInSynVarMode() & VarInit::DEVICE);
                       });
}

bool NeuronGroup::canRunOnCPU() const
{
    // If spike var isn't present on host return false
    if(!(m_SpikeVarMode & VarLocation::HOST)) {
        return false;
    }

    // If spike event var isn't present on host return false
    if(isSpikeEventRequired() && !(m_SpikeEventVarMode & VarLocation::HOST)) {
        return false;
    }

    // If spike time var isn't present on host return false
    if(isSpikeTimeRequired() && !(m_SpikeTimeVarMode & VarLocation::HOST)) {
        return false;
    }

    // Return true if all of the variables are present on the host
    return std::all_of(m_VarMode.cbegin(), m_VarMode.cend(),
                       [](const VarMode mode){ return (mode & VarLocation::HOST); });
}

bool NeuronGroup::hasOutputToHost(int targetHostID) const
{
    // Return true if any of the outgoing synapse groups have target populations on specified host ID
    return std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                       [targetHostID](SynapseGroup *sg)
                       {
                           return (sg->getTrgNeuronGroup()->getClusterHostID() == targetHostID);
                       });

}

std::string NeuronGroup::getQueueOffset(const std::string &devPrefix) const
{
    return isDelayRequired()
        ? "(" + devPrefix + "spkQuePtr" + getName() + " * " + to_string(getNumNeurons()) + ") + "
        : "";
}

void NeuronGroup::injectCurrent(CurrentSource *src)
{
    m_CurrentSources.push_back(src);
}
