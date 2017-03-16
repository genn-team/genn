#include "neuronGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
#include "utils.h"

// ------------------------------------------------------------------------
// NNmodel::NeuronGroup
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
    // Loop through neuron variables
    auto vars = getNeuronModel()->GetVars();
    for (size_t v = 0; v < vars.size(); v++) {
        // If the code contains a reference to this variable, set queued flag
        if (code.find(vars[v].first + "_pre") != string::npos) {
            m_VarNeedQueue[v] = true;
        }
    }
}

void NeuronGroup::setVarZeroCopy(const std::string &var)
{
    // If named variable doesn't exist give error
    auto nmVars = getNeuronModel()->GetVars();
    auto nmVarNameBegin = GetPairKeyConstIter(begin(nmVars));
    auto nmVarNameEnd = GetPairKeyConstIter(end(nmVars));
    if(find(nmVarNameBegin, nmVarNameEnd, var) == nmVarNameEnd)
    {
        gennError("Cannot find variable " + var);
    }
    // Otherwise add name of variable to set
    else
    {
        m_VarZeroCopy.insert(var);
    }
}

bool NeuronGroup::varZeroCopyEnabled(const std::string &var) const
{
    return (m_VarZeroCopy.find(var) != std::end(m_VarZeroCopy));
}

void NeuronGroup::addSpkEventCondition(const std::string &code, const std::string &supportCodeNamespace)
{
    m_SpikeEventCondition.insert(std::pair<std::string, std::string>(code, supportCodeNamespace));
}

void NeuronGroup::initDerivedParams(double dt)
{
    auto derivedParams = getNeuronModel()->GetDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.second(m_Params, dt));
    }
}

void NeuronGroup::calcSizes(unsigned int blockSize, unsigned int &cumSum, unsigned int &paddedCumSum)
{
    // paddedSize is the lowest multiple of neuronBlkSz >= neuronN[i]
    const unsigned int paddedSize = ceil((double)getNumNeurons() / (double) blockSize) * (double) blockSize;

    // Update global cummulative sums of neurons
    cumSum += getNumNeurons();
    paddedCumSum +=  paddedSize;

    // Store cummulative sums of point after this neuron group
    m_CumSumNeurons = cumSum;
    m_PaddedCumSumNeurons = paddedCumSum;
}

size_t NeuronGroup::addInSyn(const string &synapseName)
{
    m_InSyn.push_back(synapseName);
    return (m_InSyn.size() - 1);
}

size_t NeuronGroup::addOutSyn(const string &synapseName)
{
    m_OutSyn.push_back(synapseName);
    return (m_OutSyn.size() - 1);
}

bool NeuronGroup::anyVarNeedQueue() const
{
    return (find(begin(m_VarNeedQueue), end(m_VarNeedQueue), true) != end(m_VarNeedQueue));
}

bool NeuronGroup::usesZeroCopy() const
{
    // If any bits of spikes require zero-copy return true
    if(usesSpikeZeroCopy() || usesSpikeEventZeroCopy() || usesSpikeTimeZeroCopy())
    {
        return true;
    }

    // If there are any variables return true
    if(!m_VarZeroCopy.empty())
    {
        return true;
    }

    return false;
}

void NeuronGroup::addExtraGlobalParams(const std::string &groupName, std::map<string, string> &kernelParameters) const
{
    for(auto const &p : getNeuronModel()->GetExtraGlobalParams()) {
        std::string pnamefull = p.first + groupName;
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getNeuronModel()->GetSimCode().find("$(" + p.first + ")") != string::npos
                || getNeuronModel()->GetThresholdConditionCode().find("$(" + p.first + ")") != string::npos
                || getNeuronModel()->GetResetCode().find("$(" + p.first + ")") != string::npos) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

void NeuronGroup::addSpikeEventConditionParams(const std::pair<std::string, std::string> &param, const std::string &groupName,
                                               std::map<string, string> &kernelParameters) const
{
    std::string pnamefull = param.first + groupName;
    if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
        // parameter wasn't registered yet - is it used?
        bool used = false;

        // Loop through event conditions
        for(const auto &spkEventCond : m_SpikeEventCondition) {
            // If the event threshold code contains this parameter
            // (in it's non-uniquified form), set flag and stop searching
            if(spkEventCond.first.find(pnamefull) != string::npos) {
                used = true;
                break;
            }
        }

        if (used) {
            kernelParameters.insert(pair<string, string>(pnamefull, param.second));
        }
    }
}
