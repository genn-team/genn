#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
#include "standardSubstitutions.h"
#include "utils.h"

// ------------------------------------------------------------------------
// SynapseGroup
// ------------------------------------------------------------------------
void SynapseGroup::setWUVarZeroCopyEnabled(const std::string &var, bool enabled)
{
    // If named variable doesn't exist give error
    VarNameIterCtx wuVars(getWUModel()->getVars());
    if(find(wuVars.nameBegin, wuVars.nameEnd, var) == wuVars.nameEnd) {
        gennError("Cannot find variable " + var);
    }
    // Otherwise add name of variable to set
    else {
        // If enabled, add variable to set
        if(enabled) {
            m_WUVarZeroCopyEnabled.insert(var);
        }
        // Otherwise, remove it
        else {
            m_WUVarZeroCopyEnabled.erase(var);
        }
    }
}

void SynapseGroup::setPSVarZeroCopyEnabled(const std::string &var, bool enabled)
{
    // If named variable doesn't exist give error
    VarNameIterCtx psVars(getPSModel()->getVars());
    if(find(psVars.nameBegin, psVars.nameEnd, var) == psVars.nameEnd) {
        gennError("Cannot find variable " + var);
    }
    // Otherwise
    else  {
        // If enabled, add variable to set
        if(enabled) {
            m_PSVarZeroCopyEnabled.insert(var);
        }
        // Otherwise, remove it
        else {
            m_PSVarZeroCopyEnabled.erase(var);
        }
    }
}

void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
     if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        m_MaxConnections = maxConnections;
    }
    else {
        gennError("setMaxConn: Synapse group is densely connected. Maxconn variable is not needed in this case.");
    }
}

void SynapseGroup::initDerivedParams(double dt)
{
    auto wuDerivedParams = getWUModel()->getDerivedParams();
    auto psDerivedParams = getPSModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_WUDerivedParams.reserve(wuDerivedParams.size());
    m_PSDerivedParams.reserve(psDerivedParams.size());

    // Loop through derived parameters
    for(const auto &d : wuDerivedParams) {
        m_WUDerivedParams.push_back(d.second(m_WUParams, dt));
    }

    // Loop through derived parameters
    for(const auto &d : psDerivedParams) {
        m_PSDerivedParams.push_back(d.second(m_PSParams, dt));
    }
}

void SynapseGroup::calcKernelSizes(unsigned int blockSize, unsigned int &paddedKernelIDStart)
{
    m_PaddedKernelIDRange.first = paddedKernelIDStart;

    if (getSpanType() == SpanType::PRESYNAPTIC) {
        // paddedSize is the lowest multiple of blockSize >= neuronN[synapseSource[i]
        paddedKernelIDStart += ceil((double) getSrcNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
    }
    else {
        if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            // paddedSize is the lowest multiple of blockSize >= maxConn[i]
            paddedKernelIDStart += ceil((double) getMaxConnections() / (double) blockSize) * (double) blockSize;
        }
        else {
            // paddedSize is the lowest multiple of blockSize >= neuronN[synapseTarget[i]]
            paddedKernelIDStart += ceil((double) getTrgNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
        }
    }

    // Store padded cumulative sum
    m_PaddedKernelIDRange.second = paddedKernelIDStart;
}

unsigned int SynapseGroup::getPaddedDynKernelSize(unsigned int blockSize) const
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        // paddedSize is the lowest multiple of synDynBlkSz >= neuronN[synapseSource[i]] * maxConn[i]
        return ceil((double) getSrcNeuronGroup()->getNumNeurons() * getMaxConnections() / (double) blockSize) * (double) blockSize;
    }
    else {
        // paddedSize is the lowest multiple of synDynBlkSz >= neuronN[synapseSource[i]] * neuronN[synapseTarget[i]]
        return ceil((double) getSrcNeuronGroup()->getNumNeurons() * getTrgNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
    }
}

unsigned int SynapseGroup::getPaddedPostLearnKernelSize(unsigned int blockSize) const
{
    return ceil((double) getSrcNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
}

bool SynapseGroup::isZeroCopyEnabled() const
{
    // If there are any variables return true
    if(!m_WUVarZeroCopyEnabled.empty() || !m_PSVarZeroCopyEnabled.empty())
    {
        return true;
    }

    return false;
}

bool SynapseGroup::isWUVarZeroCopyEnabled(const std::string &var) const
{
    return (m_WUVarZeroCopyEnabled.find(var) != std::end(m_WUVarZeroCopyEnabled));
}

bool SynapseGroup::isPSVarZeroCopyEnabled(const std::string &var) const
{
    return (m_PSVarZeroCopyEnabled.find(var) != std::end(m_PSVarZeroCopyEnabled));
}

bool SynapseGroup::arePreVarsRequiredForSynapse(const std::string &code) const
{
    // Get presynaptic neuron model
    const auto *preNeuronModel = getSrcNeuronGroup()->getNeuronModel();

    // If presynaptic neuron is poisson and code references it's voltage - return true
    if (preNeuronModel->isPoisson() && code.find("$(V_pre)") != std::string::npos) {
        return true;
    }

    // If code references presynaptic spike time - return true
    if(code.find("$(sT_pre)") != std::string::npos)
    {
        return true;
    }

    // Loop through presynaptic neuron model variables
    for(const auto &v : preNeuronModel->getVars()) {
        // Get name this variable would be referred to in synapse code
        const std::string preVarName = "$(" + v.first + "_pre)";

        // If code references variable - return true
        if(code.find(preVarName) != std::string::npos)
        {
            return true;
        }
    }

    return false;
}

void SynapseGroup::addExtraGlobalNeuronParams(std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of extra global weight update parameters
    for(auto const &p : getWUModel()->getExtraGlobalParams()) {
        // If it's not already in set
        std::string pnamefull = p.first + getName();
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // If the presynaptic neuron requires this parameter in it's spike event conditions, add it
            if (getSrcNeuronGroup()->isParamRequiredBySpikeEventCondition(pnamefull)) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

void SynapseGroup::addExtraGlobalSynapseParams(std::map<std::string, std::string> &kernelParameters) const
{
    // Synapse kernel
    // --------------
    // Add any of the pre or postsynaptic neuron group's extra global
    // parameters referenced in the sim code to the map of kernel parameters
    addExtraGlobalSimParams(getSrcNeuronGroup()->getName(), "_pre", getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                             kernelParameters);
    addExtraGlobalSimParams(getTrgNeuronGroup()->getName(), "_post", getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                             kernelParameters);

    // Finally add any weight update model extra global
    // parameters referenced in the sim to the map of kernel paramters
    addExtraGlobalSimParams(getName(), "", getWUModel()->getExtraGlobalParams(), kernelParameters);
}


void SynapseGroup::addExtraGlobalPostLearnParams(std::map<string, string> &kernelParameters) const
{
    // Add any of the pre or postsynaptic neuron group's extra global
    // parameters referenced in the sim code to the map of kernel parameters
    addExtraGlobalPostLearnParams(getSrcNeuronGroup()->getName(), "_pre", getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                                  kernelParameters);
    addExtraGlobalPostLearnParams(getTrgNeuronGroup()->getName(), "_post", getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                                  kernelParameters);

    // Finally add any weight update model extra global
    // parameters referenced in the sim to the map of kernel paramters
    addExtraGlobalPostLearnParams(getName(), "", getWUModel()->getExtraGlobalParams(), kernelParameters);

}

void SynapseGroup::addExtraGlobalSynapseDynamicsParams(std::map<string, string> &kernelParameters) const
{
    // Add any of the pre or postsynaptic neuron group's extra global
    // parameters referenced in the sim code to the map of kernel parameters
    addExtraGlobalSynapseDynamicsParams(getSrcNeuronGroup()->getName(), "_pre", getSrcNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                                        kernelParameters);
    addExtraGlobalSynapseDynamicsParams(getTrgNeuronGroup()->getName(), "_post", getTrgNeuronGroup()->getNeuronModel()->getExtraGlobalParams(),
                                        kernelParameters);

    // Finally add any weight update model extra global
    // parameters referenced in the sim to the map of kernel paramters
    addExtraGlobalSynapseDynamicsParams(getName(), "", getWUModel()->getExtraGlobalParams(), kernelParameters);
}

void SynapseGroup::addExtraGlobalSimParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                           std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of global parameters
    for(auto const &p : extraGlobalParameters) {
        std::string pnamefull = p.first + prefix;
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getWUModel()->getSimCode().find("$(" + p.first + suffix + ")") != string::npos
                || getWUModel()->getEventCode().find("$(" + p.first + suffix + ")") != string::npos
                || getWUModel()->getEventThresholdConditionCode().find("$(" + p.first + suffix + ")") != string::npos) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

void SynapseGroup::addExtraGlobalPostLearnParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                                 std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of global parameters
    for(auto const &p : extraGlobalParameters) {
        std::string pnamefull = p.first + prefix;
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getWUModel()->getLearnPostCode().find("$(" + p.first + suffix) != string::npos) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

void SynapseGroup::addExtraGlobalSynapseDynamicsParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                                       std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of global parameters
    for(auto const &p : extraGlobalParameters) {
        std::string pnamefull = p.first + prefix;
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getWUModel()->getSynapseDynamicsCode().find("$(" + p.first + suffix) != string::npos) {
                kernelParameters.insert(pair<string, string>(pnamefull, p.second));
            }
        }
    }
}

std::string SynapseGroup::getOffsetPre() const
{
    return getSrcNeuronGroup()->isDelayRequired()
        ? "(delaySlot * " + to_string(getSrcNeuronGroup()->getNumNeurons()) + ") + "
        : "";
}

std::string SynapseGroup::getOffsetPost(const std::string &devPrefix) const
{
    return getTrgNeuronGroup()->getQueueOffset(devPrefix);
}