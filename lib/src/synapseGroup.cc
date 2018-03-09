#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "codeGenUtils.h"
#include "global.h"
#include "standardSubstitutions.h"
#include "utils.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
std::vector<double> getConstInitVals(const std::vector<NewModels::VarInit> &varInitialisers)
{
    // Reserve initial values to match initialisers
    std::vector<double> initVals;
    initVals.reserve(varInitialisers.size());

    // Transform variable initialisers into a vector of doubles
    std::transform(varInitialisers.cbegin(), varInitialisers.cend(), std::back_inserter(initVals),
                   [](const NewModels::VarInit &v)
                   {
                       // Check
                       if(dynamic_cast<const InitVarSnippet::Constant*>(v.getSnippet()) == nullptr) {
                           throw std::runtime_error("Only 'Constant' variable initialisation snippets can be used to initialise state variables of synapse groups using GLOBALG");
                       }

                       // Return the first parameter (the value)
                       return v.getParams()[0];
                   });

    return initVals;
}
}   // Anonymous namespace

// ------------------------------------------------------------------------
// SynapseGroup
// ------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<NewModels::VarInit> &wuVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<NewModels::VarInit> &psVarInitialisers,
                           NeuronGroup *srcNeuronGroup, NeuronGroup *trgNeuronGroup)
    :   m_PaddedKernelIDRange(0, 0), m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_DelaySteps(delaySteps), m_MaxConnections(trgNeuronGroup->getNumNeurons()), m_MatrixType(matrixType),
        m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup),
        m_TrueSpikeRequired(false), m_SpikeEventRequired(false), m_EventThresholdReTestRequired(false), m_InSynVarMode(GENN_PREFERENCES::defaultVarMode),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarMode(wuVarInitialisers.size(), GENN_PREFERENCES::defaultVarMode), m_PSVarMode(psVarInitialisers.size(), GENN_PREFERENCES::defaultVarMode)
{
    // Check that the source neuron group supports the desired number of delay steps
    srcNeuronGroup->checkNumDelaySlots(delaySteps);

    // If the weight update model requires presynaptic
    // spike times, set flag in source neuron group
    if (getWUModel()->isPreSpikeTimeRequired()) {
        srcNeuronGroup->setSpikeTimeRequired(true);
    }

    // If the weight update model requires postsynaptic
    // spike times, set flag in target neuron group
    if (getWUModel()->isPostSpikeTimeRequired()) {
        trgNeuronGroup->setSpikeTimeRequired(true);
    }

    // Add references to target and source neuron groups
    trgNeuronGroup->addInSyn(this);
    srcNeuronGroup->addOutSyn(this);
}

void SynapseGroup::setWUVarMode(const std::string &varName, VarMode mode)
{
    m_WUVarMode[getWUModel()->getVarIndex(varName)] = mode;
}

void SynapseGroup::setPSVarMode(const std::string &varName, VarMode mode)
{
    m_PSVarMode[getPSModel()->getVarIndex(varName)] = mode;
}

void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
     if((getMatrixType() & SynapseMatrixConnectivity::SPARSE) 
         || (getMatrixType() & SynapseMatrixConnectivity::RAGGED)) 
    {
        m_MaxConnections = maxConnections;
    }
    else {
        gennError("setMaxConn: Synapse group is densely connected. Maxconn variable is not needed in this case.");
    }
}

void SynapseGroup::setSpanType(SpanType spanType)
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        m_SpanType = spanType;
    }
    else {
        gennError("setSpanType: This function is not enabled for dense connectivity type.");
    }
}

void SynapseGroup::initDerivedParams(double dt)
{
    auto wuDerivedParams = getWUModel()->getDerivedParams();
    auto psDerivedParams = getPSModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_WUDerivedParams.reserve(wuDerivedParams.size());
    m_PSDerivedParams.reserve(psDerivedParams.size());

    // Loop through WU derived parameters
    for(const auto &d : wuDerivedParams) {
        m_WUDerivedParams.push_back(d.second(m_WUParams, dt));
    }

    // Loop through PSM derived parameters
    for(const auto &d : psDerivedParams) {
        m_PSDerivedParams.push_back(d.second(m_PSParams, dt));
    }

    // Initialise derived parameters for WU variable initialisers
    for(auto &v : m_WUVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for PSM variable initialisers
    for(auto &v : m_PSVarInitialisers) {
        v.initDerivedParams(dt);
    }
}

void SynapseGroup::calcKernelSizes(unsigned int blockSize, unsigned int &paddedKernelIDStart)
{
    m_PaddedKernelIDRange.first = paddedKernelIDStart;

    if((getMatrixType() & SynapseMatrixConnectivity::SPARSE) 
        || (getMatrixType() & SynapseMatrixConnectivity::RAGGED)) 
    {
        if (getSpanType() == SpanType::PRESYNAPTIC) {
            // paddedSize is the lowest multiple of blockSize >= neuronN[synapseSource[i]
            paddedKernelIDStart += ceil((double) getSrcNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
        }
        else {
            // paddedSize is the lowest multiple of blockSize >= maxConn[i]
            paddedKernelIDStart += ceil((double) getMaxConnections() / (double) blockSize) * (double) blockSize;
        }
    }
    else {
        // paddedSize is the lowest multiple of blockSize >= neuronN[synapseTarget[i]]
        paddedKernelIDStart += ceil((double) getTrgNeuronGroup()->getNumNeurons() / (double) blockSize) * (double) blockSize;
    }

    // Store padded cumulative sum
    m_PaddedKernelIDRange.second = paddedKernelIDStart;
}

unsigned int SynapseGroup::getPaddedDynKernelSize(unsigned int blockSize) const
{
    if((getMatrixType() & SynapseMatrixConnectivity::SPARSE) 
        || (getMatrixType() & SynapseMatrixConnectivity::RAGGED)) 
    {
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

const std::vector<double> SynapseGroup::getWUConstInitVals() const
{
    return getConstInitVals(m_WUVarInitialisers);
}

const std::vector<double> SynapseGroup::getPSConstInitVals() const
{
    return getConstInitVals(m_PSVarInitialisers);
}

bool SynapseGroup::isZeroCopyEnabled() const
{
    // If there are any postsynaptic variables implemented in zero-copy mode return true
    if(any_of(m_PSVarMode.begin(), m_PSVarMode.end(),
        [](VarMode mode){ return (mode & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any weight update variables implemented in zero-copy mode return true
    if(any_of(m_WUVarMode.begin(), m_WUVarMode.end(),
        [](VarMode mode){ return (mode & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}

VarMode SynapseGroup::getWUVarMode(const std::string &var) const
{
    return m_WUVarMode[getWUModel()->getVarIndex(var)];
}

VarMode SynapseGroup::getPSVarMode(const std::string &var) const
{
    return m_PSVarMode[getPSModel()->getVarIndex(var)];
}

bool SynapseGroup::isPSAtomicAddRequired(unsigned int blockSize) const
{
    if ((getMatrixType() & SynapseMatrixConnectivity::SPARSE)
        || (getMatrixType() & SynapseMatrixConnectivity::RAGGED))
    {
        if (getSpanType() == SpanType::POSTSYNAPTIC && getTrgNeuronGroup()->getNumNeurons() > blockSize) {
            return true;
        }
        if (getSpanType()  == SpanType::PRESYNAPTIC && getSrcNeuronGroup()->getNumNeurons() > blockSize) {
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

bool SynapseGroup::isPSInitRNGRequired(VarInit varInitMode) const
{
    // If initialising the postsynaptic variables require an RNG, return true
    return isInitRNGRequired(m_PSVarInitialisers, m_PSVarMode, varInitMode);
}

bool SynapseGroup::isWUInitRNGRequired(VarInit varInitMode) const
{
    // If initialising the weight update variables require an RNG, return true
    return isInitRNGRequired(m_WUVarInitialisers, m_WUVarMode, varInitMode);
}

bool SynapseGroup::isPSDeviceVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of the postsynapse variables are initialised on the device
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        return std::any_of(m_PSVarMode.cbegin(), m_PSVarMode.cend(),
                        [](const VarMode mode){ return (mode & VarInit::DEVICE); });
    }
    else {
        return false;
    }
}

bool SynapseGroup::isWUDeviceVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of the weight update variables are initialised on the device
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        return std::any_of(m_WUVarMode.cbegin(), m_WUVarMode.cend(),
                        [](const VarMode mode){ return (mode & VarInit::DEVICE); });
    }
    else {
        return false;
    }
}

bool SynapseGroup::canRunOnCPU() const
{
#ifndef CPU_ONLY
    // Return false if insyn variable isn't present on the host
    if(!(getInSynVarMode() & VarLocation::HOST)) {
        return false;
    }

    // Return false if any of the weight update variables aren't present on the host
    if(std::any_of(m_WUVarMode.cbegin(), m_WUVarMode.cend(),
                   [](const VarMode mode){ return !(mode & VarLocation::HOST); }))
    {
        return false;
    }

    // Return false if any of the postsynaptic variables aren't present on the host
    if(std::any_of(m_PSVarMode.cbegin(), m_PSVarMode.cend(),
                   [](const VarMode mode){ return !(mode & VarLocation::HOST); }))
    {
        return false;
    }
#endif

    return true;
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
