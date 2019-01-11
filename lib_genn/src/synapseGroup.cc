#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>

// GeNN includes
#include "codeGenUtils.h"

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
                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<NewModels::VarInit> &wuVarInitialisers, const std::vector<NewModels::VarInit> &wuPreVarInitialisers, const std::vector<NewModels::VarInit> &wuPostVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<NewModels::VarInit> &psVarInitialisers,
                           NeuronGroup *srcNeuronGroup, NeuronGroup *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser, 
                           VarLocation defaultVarLocation, VarLocation defaultSparseConnectivityLocation)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup),
        m_EventThresholdReTestRequired(false),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_WUPreVarInitialisers(wuPreVarInitialisers), m_WUPostVarInitialisers(wuPostVarInitialisers),
        m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarLocation(wuVarInitialisers.size(), defaultVarLocation), m_WUPreVarLocation(wuPreVarInitialisers.size(), defaultVarLocation),
        m_WUPostVarLocation(wuPostVarInitialisers.size(), defaultVarLocation), m_PSVarLocation(psVarInitialisers.size(), defaultVarLocation),
        m_ConnectivityInitialiser(connectivityInitialiser), m_SparseConnectivityLocation(defaultSparseConnectivityLocation),
        m_PSModelTargetName(name)
{
    // If connectivitity initialisation snippet provides a function to calculate row length, call it
    // **NOTE** only do this for sparse connectivity as this should not be set for bitmasks
    auto calcMaxRowLengthFunc = m_ConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc();
    if(calcMaxRowLengthFunc && (m_MatrixType & SynapseMatrixConnectivity::SPARSE)) {
        m_MaxConnections = calcMaxRowLengthFunc(srcNeuronGroup->getNumNeurons(), trgNeuronGroup->getNumNeurons(),
                                                m_ConnectivityInitialiser.getParams());
    }
    // Otherwise, default to the size of the target population
    else {
        m_MaxConnections = trgNeuronGroup->getNumNeurons();
    }

    // If connectivitity initialisation snippet provides a function to calculate row length, call it
    // **NOTE** only do this for sparse connectivity as this should not be set for bitmasks
    auto calcMaxColLengthFunc = m_ConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc();
    if(calcMaxColLengthFunc && (m_MatrixType & SynapseMatrixConnectivity::SPARSE)) {
        m_MaxSourceConnections = calcMaxColLengthFunc(srcNeuronGroup->getNumNeurons(), trgNeuronGroup->getNumNeurons(),
                                                      m_ConnectivityInitialiser.getParams());
    }
    // Otherwise, default to the size of the source population
    else {
        m_MaxSourceConnections = srcNeuronGroup->getNumNeurons();
    }

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

void SynapseGroup::setWUVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUVarLocation[getWUModel()->getVarIndex(varName)] = loc;
}

void SynapseGroup::setWUPreVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUPreVarLocation[getWUModel()->getPreVarIndex(varName)] = loc;
}

void SynapseGroup::setWUPostVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUPostVarLocation[getWUModel()->getPostVarIndex(varName)] = loc;
}

void SynapseGroup::setPSVarLocation(const std::string &varName, VarLocation loc)
{
    m_PSVarLocation[getPSModel()->getVarIndex(varName)] = loc;
}

void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if(m_ConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc()) {
            throw std::runtime_error("setMaxConnections: Synapse group already has max connections defined by connectivity initialisation snippet.");
        }
        
        m_MaxConnections = maxConnections;
    }
    else {
        throw std::runtime_error("setMaxConnections: Synapse group is densely connected. Setting max connections is not required in this case.");
    }
}

void SynapseGroup::setMaxSourceConnections(unsigned int maxConnections)
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if(m_ConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc()) {
            throw std::runtime_error("setMaxSourceConnections: Synapse group already has max source connections defined by connectivity initialisation snippet.");
        }

        m_MaxSourceConnections = maxConnections;
    }
    else {
        throw std::runtime_error("setMaxSourceConnections: Synapse group is densely connected. Setting max connections is not required in this case.");
    }
}

void SynapseGroup::setMaxDendriticDelayTimesteps(unsigned int maxDendriticDelayTimesteps)
{
    // **TODO** constraints on this
    m_MaxDendriticDelayTimesteps = maxDendriticDelayTimesteps;
}

void SynapseGroup::setSpanType(SpanType spanType)
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        m_SpanType = spanType;
    }
    else {
        throw std::runtime_error("setSpanType: This function is not enabled for dense connectivity type.");
    }
}

void SynapseGroup::setBackPropDelaySteps(unsigned int timesteps)
{
    m_BackPropDelaySteps = timesteps;

    m_TrgNeuronGroup->checkNumDelaySlots(m_BackPropDelaySteps);
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

    // Initialise any derived connectivity initialiser parameters
    m_ConnectivityInitialiser.initDerivedParams(dt);
}

bool SynapseGroup::isTrueSpikeRequired() const
{
    return !getWUModel()->getSimCode().empty();
}

bool SynapseGroup::isSpikeEventRequired() const
{
     return !getWUModel()->getEventCode().empty();
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
    if(std::any_of(m_PSVarLocation.begin(), m_PSVarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any weight update variables implemented in zero-copy mode return true
    if(std::any_of(m_WUVarLocation.begin(), m_WUVarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any weight update variables implemented in zero-copy mode return true
    if(std::any_of(m_WUPreVarLocation.begin(), m_WUPreVarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any weight update variables implemented in zero-copy mode return true
    if(std::any_of(m_WUPostVarLocation.begin(), m_WUPostVarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}

VarLocation SynapseGroup::getWUVarLocation(const std::string &var) const
{
    return m_WUVarLocation[getWUModel()->getVarIndex(var)];
}

VarLocation SynapseGroup::getWUPreVarLocation(const std::string &var) const
{
    return m_WUVarLocation[getWUModel()->getPreVarIndex(var)];
}

VarLocation SynapseGroup::getWUPostVarLocation(const std::string &var) const
{
    return m_WUVarLocation[getWUModel()->getPostVarIndex(var)];
}

VarLocation SynapseGroup::getPSVarLocation(const std::string &var) const
{
    return m_WUVarLocation[getPSModel()->getVarIndex(var)];
}

void SynapseGroup::addExtraGlobalConnectivityInitialiserParams(std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of global parameters
    for(auto const &p : getConnectivityInitialiser().getSnippet()->getExtraGlobalParams()) {
        const std::string pnamefull = "initSparseConn" + p.first + getName();
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getConnectivityInitialiser().getSnippet()->getRowBuildCode().find("$(" + p.first + ")") != std::string::npos) {
                kernelParameters.emplace(pnamefull, p.second);
            }
        }
    }
}

void SynapseGroup::addExtraGlobalNeuronParams(std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of extra global weight update parameters
    for(auto const &p : getWUModel()->getExtraGlobalParams()) {
        // If it's not already in set
        const std::string pnamefull = p.first + getName();
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // If the presynaptic neuron requires this parameter in it's spike event conditions, add it
            if (getSrcNeuronGroup()->isParamRequiredBySpikeEventCondition(pnamefull)) {
                kernelParameters.emplace(pnamefull, p.second);
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


void SynapseGroup::addExtraGlobalPostLearnParams(std::map<std::string, std::string> &kernelParameters) const
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

void SynapseGroup::addExtraGlobalSynapseDynamicsParams(std::map<std::string, std::string> &kernelParameters) const
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

std::string SynapseGroup::getPresynapticAxonalDelaySlot(const std::string &devPrefix) const
{
    assert(getSrcNeuronGroup()->isDelayRequired());

    if(getDelaySteps() == 0) {
        return devPrefix + "spkQuePtr" + getSrcNeuronGroup()->getName();
    }
    else {
        return "((" + devPrefix + "spkQuePtr" + getSrcNeuronGroup()->getName() + " + " + std::to_string(getSrcNeuronGroup()->getNumDelaySlots() - getDelaySteps()) + ") % " + std::to_string(getSrcNeuronGroup()->getNumDelaySlots()) + ")";
    }
}

std::string SynapseGroup::getPostsynapticBackPropDelaySlot(const std::string &devPrefix) const
{
    assert(getTrgNeuronGroup()->isDelayRequired());

    if(getBackPropDelaySteps() == 0) {
        return devPrefix + "spkQuePtr" + getTrgNeuronGroup()->getName();
    }
    else {
        return "((" + devPrefix + "spkQuePtr" + getTrgNeuronGroup()->getName() + " + " + std::to_string(getTrgNeuronGroup()->getNumDelaySlots() - getBackPropDelaySteps()) + ") % " + std::to_string(getTrgNeuronGroup()->getNumDelaySlots()) + ")";
    }
}

std::string SynapseGroup::getDendriticDelayOffset(const std::string &devPrefix, const std::string &offset) const
{
    assert(isDendriticDelayRequired());

    if(offset.empty()) {
        return "(" + devPrefix + "denDelayPtr" + getPSModelTargetName() + " * " + std::to_string(getTrgNeuronGroup()->getNumNeurons()) + ") + ";
    }
    else {
        return "(((" + devPrefix + "denDelayPtr" + getPSModelTargetName() + " + " + offset + ") % " + std::to_string(getMaxDendriticDelayTimesteps()) + ") * " + std::to_string(getTrgNeuronGroup()->getNumNeurons()) + ") + ";
    }
}

bool SynapseGroup::isDendriticDelayRequired() const
{
    // If addToInSynDelay function is used in sim code, return true
    if(getWUModel()->getSimCode().find("$(addToInSynDelay") != std::string::npos) {
        return true;
    }

    // If addToInSynDelay function is used in synapse dynamics, return true
    if(getWUModel()->getSynapseDynamicsCode().find("$(addToInSynDelay") != std::string::npos) {
        return true;
    }

    return false;
}

bool SynapseGroup::isPSInitRNGRequired() const
{
    // If initialising the postsynaptic variables require an RNG, return true
    return isInitRNGRequired(m_PSVarInitialisers);
}

bool SynapseGroup::isWUInitRNGRequired() const
{
    // If initialising the weight update variables require an RNG, return true
    if(isInitRNGRequired(m_WUVarInitialisers)) {
        return true;
    }

    // Return true if the var init mode we're querying is the one used for sparse connectivity and the connectivity initialiser requires an RNG
    return isRNGRequired(m_ConnectivityInitialiser.getSnippet()->getRowBuildCode());
}

bool SynapseGroup::isPSVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of them have initialisation code
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        return std::any_of(m_PSVarInitialisers.cbegin(), m_PSVarInitialisers.cend(),
                           [](const NewModels::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
    }
    else {
        return false;
    }
}

bool SynapseGroup::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of them have initialisation code
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        return std::any_of(m_WUVarInitialisers.cbegin(), m_WUVarInitialisers.cend(),
                           [](const NewModels::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
    }
    else {
        return false;
    }
}

bool SynapseGroup::isWUPreVarInitRequired() const
{
    return std::any_of(m_WUPreVarInitialisers.cbegin(), m_WUPreVarInitialisers.cend(),
                       [](const NewModels::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}

bool SynapseGroup::isWUPostVarInitRequired() const
{
    return std::any_of(m_WUPostVarInitialisers.cbegin(), m_WUPostVarInitialisers.cend(),
                       [](const NewModels::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}

bool SynapseGroup::isSparseConnectivityInitRequired() const
{
    // Return true if there is code to initialise sparse connectivity on device
    return !getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty();
}

bool SynapseGroup::isInitRequired() const
{
    // If the synaptic matrix is dense and some synaptic variables are initialised on device, return true
    if((getMatrixType() & SynapseMatrixConnectivity::DENSE) && isWUVarInitRequired()) {
        return true;
    }
    // Otherwise return true if there is sparse connectivity to be initialised on device
    else {
        return isSparseConnectivityInitRequired();
    }
}

bool SynapseGroup::isSparseInitRequired() const
{
    // If the synaptic connectivity is sparse and some synaptic variables should be initialised on device, return true
    if((getMatrixType() & SynapseMatrixConnectivity::SPARSE) && isWUVarInitRequired()) {
        return true;
    }

    // If sparse connectivity is initialised on device and the synapse group required either synapse dynamics or postsynaptic learning, return true
    if(isSparseConnectivityInitRequired() &&
        (!getWUModel()->getSynapseDynamicsCode().empty() || !getWUModel()->getLearnPostCode().empty()))
    {
        return true;
    }

    return false;
}

void SynapseGroup::addExtraGlobalSimParams(const std::string &prefix, const std::string &suffix, const NewModels::Base::StringPairVec &extraGlobalParameters,
                                           std::map<std::string, std::string> &kernelParameters) const
{
    // Loop through list of global parameters
    for(auto const &p : extraGlobalParameters) {
        std::string pnamefull = p.first + prefix;
        if (kernelParameters.find(pnamefull) == kernelParameters.end()) {
            // parameter wasn't registered yet - is it used?
            if (getWUModel()->getSimCode().find("$(" + p.first + suffix + ")") != std::string::npos
                || getWUModel()->getEventCode().find("$(" + p.first + suffix + ")") != std::string::npos
                || getWUModel()->getEventThresholdConditionCode().find("$(" + p.first + suffix + ")") != std::string::npos) 
            {
                kernelParameters.emplace(pnamefull, p.second);
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
            if (getWUModel()->getLearnPostCode().find("$(" + p.first + suffix) != std::string::npos) {
                kernelParameters.emplace(pnamefull, p.second);
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
            if (getWUModel()->getSynapseDynamicsCode().find("$(" + p.first + suffix) != std::string::npos) {
                kernelParameters.emplace(pnamefull, p.second);
            }
        }
    }
}
