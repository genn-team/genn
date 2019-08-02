#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>

// GeNN includes
#include "neuronGroupInternal.h"
#include "gennUtils.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
std::vector<double> getConstInitVals(const std::vector<Models::VarInit> &varInitialisers)
{
    // Reserve initial values to match initialisers
    std::vector<double> initVals;
    initVals.reserve(varInitialisers.size());

    // Transform variable initialisers into a vector of doubles
    std::transform(varInitialisers.cbegin(), varInitialisers.cend(), std::back_inserter(initVals),
                   [](const Models::VarInit &v)
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
void SynapseGroup::setWUVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUVarLocation[getWUModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUPreVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUPreVarLocation[getWUModel()->getPreVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUPostVarLocation(const std::string &varName, VarLocation loc)
{
    m_WUPostVarLocation[getWUModel()->getPostVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getWUModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getWUModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_WUExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setPSVarLocation(const std::string &varName, VarLocation loc)
{
    m_PSVarLocation[getPSModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setPSExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = getPSModel()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(getPSModel()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_PSExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setSparseConnectivityExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    const size_t extraGlobalParamIndex = m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ConnectivityExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
void SynapseGroup::setMaxDendriticDelayTimesteps(unsigned int maxDendriticDelayTimesteps)
{
    // **TODO** constraints on this
    m_MaxDendriticDelayTimesteps = maxDendriticDelayTimesteps;
}
//----------------------------------------------------------------------------
void SynapseGroup::setSpanType(SpanType spanType)
{
    if (getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        m_SpanType = spanType;
    }
    else {
        throw std::runtime_error("setSpanType: This function is not enabled for dense connectivity type.");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setNumThreadsPerSpike(unsigned int numThreadsPerSpike)
{
    if (m_SpanType == SpanType::PRESYNAPTIC) {
        m_NumThreadsPerSpike = numThreadsPerSpike;
    }
    else {
        throw std::runtime_error("setSpanType: This function is not enabled for dense connectivity type.");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setBackPropDelaySteps(unsigned int timesteps)
{
    m_BackPropDelaySteps = timesteps;

    m_TrgNeuronGroup->checkNumDelaySlots(m_BackPropDelaySteps);
}
//----------------------------------------------------------------------------
int SynapseGroup::getClusterHostID() const
{
    return m_TrgNeuronGroup->getClusterHostID();
}
//----------------------------------------------------------------------------
bool SynapseGroup::isTrueSpikeRequired() const
{
    return !getWUModel()->getSimCode().empty();
}
//----------------------------------------------------------------------------
bool SynapseGroup::isSpikeEventRequired() const
{
     return !getWUModel()->getEventCode().empty();
}
//----------------------------------------------------------------------------
const std::vector<double> SynapseGroup::getWUConstInitVals() const
{
    return getConstInitVals(m_WUVarInitialisers);
}
//----------------------------------------------------------------------------
const std::vector<double> SynapseGroup::getPSConstInitVals() const
{
    return getConstInitVals(m_PSVarInitialisers);
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getWUVarLocation(const std::string &var) const
{
    return m_WUVarLocation[getWUModel()->getVarIndex(var)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getWUPreVarLocation(const std::string &var) const
{
    return m_WUPreVarLocation[getWUModel()->getPreVarIndex(var)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getWUPostVarLocation(const std::string &var) const
{
    return m_WUPostVarLocation[getWUModel()->getPostVarIndex(var)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getWUExtraGlobalParamLocation(const std::string &paramName) const
{
    return m_WUExtraGlobalParamLocation[getWUModel()->getExtraGlobalParamIndex(paramName)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getPSVarLocation(const std::string &var) const
{
    return m_PSVarLocation[getPSModel()->getVarIndex(var)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getPSExtraGlobalParamLocation(const std::string &paramName) const
{
    return m_PSExtraGlobalParamLocation[getPSModel()->getExtraGlobalParamIndex(paramName)];
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getSparseConnectivityExtraGlobalParamLocation(const std::string &paramName) const
{
    return m_ConnectivityExtraGlobalParamLocation[m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName)];
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
bool SynapseGroup::isPSInitRNGRequired() const
{
    // If initialising the postsynaptic variables require an RNG, return true
    return Utils::isInitRNGRequired(m_PSVarInitialisers);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUInitRNGRequired() const
{
    // If initialising the weight update variables require an RNG, return true
    if(Utils::isInitRNGRequired(m_WUVarInitialisers)) {
        return true;
    }

    // Return true if the var init mode we're querying is the one used for sparse connectivity and the connectivity initialiser requires an RNG
    return Utils::isRNGRequired(m_ConnectivityInitialiser.getSnippet()->getRowBuildCode());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of them have initialisation code
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        return std::any_of(m_WUVarInitialisers.cbegin(), m_WUVarInitialisers.cend(),
                           [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::isSparseConnectivityInitRequired() const
{
    // Return true if there is code to initialise sparse connectivity on device
    return !getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty();
}
//----------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation, VarLocation defaultSparseConnectivityLocation)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_NumThreadsPerSpike(1), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup),
        m_EventThresholdReTestRequired(false),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_WUPreVarInitialisers(wuPreVarInitialisers), m_WUPostVarInitialisers(wuPostVarInitialisers),
        m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarLocation(wuVarInitialisers.size(), defaultVarLocation), m_WUPreVarLocation(wuPreVarInitialisers.size(), defaultVarLocation),
        m_WUPostVarLocation(wuPostVarInitialisers.size(), defaultVarLocation), m_WUExtraGlobalParamLocation(wu->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_PSVarLocation(psVarInitialisers.size(), defaultVarLocation), m_PSExtraGlobalParamLocation(ps->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_ConnectivityInitialiser(connectivityInitialiser), m_SparseConnectivityLocation(defaultSparseConnectivityLocation),
        m_ConnectivityExtraGlobalParamLocation(connectivityInitialiser.getSnippet()->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation), m_PSModelTargetName(name)
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
}
//----------------------------------------------------------------------------
void SynapseGroup::initDerivedParams(double dt)
{
    auto wuDerivedParams = getWUModel()->getDerivedParams();
    auto psDerivedParams = getPSModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_WUDerivedParams.reserve(wuDerivedParams.size());
    m_PSDerivedParams.reserve(psDerivedParams.size());

    // Loop through WU derived parameters
    for(const auto &d : wuDerivedParams) {
        m_WUDerivedParams.push_back(d.func(m_WUParams, dt));
    }

    // Loop through PSM derived parameters
    for(const auto &d : psDerivedParams) {
        m_PSDerivedParams.push_back(d.func(m_PSParams, dt));
    }

    // Initialise derived parameters for WU variable initialisers
    for(auto &v : m_WUVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for PSM variable initialisers
    for(auto &v : m_PSVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for WU presynaptic variable initialisers
    for(auto &v : m_WUPreVarInitialisers) {
        v.initDerivedParams(dt);
    }
    
    // Initialise derived parameters for WU postsynaptic variable initialisers
    for(auto &v : m_WUPostVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise any derived connectivity initialiser parameters
    m_ConnectivityInitialiser.initDerivedParams(dt);
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------
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
