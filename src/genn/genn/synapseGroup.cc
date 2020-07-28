#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>

// GeNN includes
#include "gennUtils.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

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
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setWUVarLocation: Synapse group is a weight sharing slave. Weight update var location can only be set on the master.");
    }
    else {
        m_WUVarLocation[getWUModel()->getVarIndex(varName)] = loc;
    }
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
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setSparseConnectivityExtraGlobalParamLocation: Synapse group is a weight sharing slave. Sparse connectivity EGP location can only be set on the master.");
    }
    else {
        const size_t extraGlobalParamIndex = m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName);
        if(!Utils::isTypePointer(m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
            throw std::runtime_error("Only extra global parameters with a pointer type have a location");
        }
        m_ConnectivityExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setSparseConnectivityLocation(VarLocation loc)
{ 
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setSparseConnectivityLocation: Synapse group is a weight sharing slave. Sparse connectivity location can only be set on the master.");
    }
    else {
        m_SparseConnectivityLocation = loc;
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setMaxConnections: Synapse group is a weight sharing slave. Max connections can only be set on the master.");
    }
    else {
        if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            if(m_ConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc()) {
                throw std::runtime_error("setMaxConnections: Synapse group already has max connections defined by connectivity initialisation snippet.");
            }

            m_MaxConnections = maxConnections;
        }
        else {
            throw std::runtime_error("setMaxConnections: Synapse group is densely connected. Setting max connections is not required in this case.");
        }
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setMaxSourceConnections(unsigned int maxConnections)
{
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setMaxSourceConnections: Synapse group is a weight sharing slave. Max source connections can only be set on the master.");
    }
    else {
        if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            if(m_ConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc()) {
                throw std::runtime_error("setMaxSourceConnections: Synapse group already has max source connections defined by connectivity initialisation snippet.");
            }

            m_MaxSourceConnections = maxConnections;
        }
        else {
            throw std::runtime_error("setMaxSourceConnections: Synapse group is densely connected. Setting max connections is not required in this case.");
        }
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
    if ((getMatrixType() & SynapseMatrixConnectivity::SPARSE) || (getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL)) {
        m_SpanType = spanType;
    }
    else {
        throw std::runtime_error("setSpanType: This function can only be used on synapse groups with sparse or bitmask connectivity.");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setNumThreadsPerSpike(unsigned int numThreadsPerSpike)
{
    if (m_SpanType == SpanType::PRESYNAPTIC) {
        m_NumThreadsPerSpike = numThreadsPerSpike;
    }
    else {
        throw std::runtime_error("setNumThreadsPerSpike: This function can only be used on synapse groups with a presynaptic span type.");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setBackPropDelaySteps(unsigned int timesteps)
{
    m_BackPropDelaySteps = timesteps;

    m_TrgNeuronGroup->checkNumDelaySlots(m_BackPropDelaySteps);
}
//----------------------------------------------------------------------------
void SynapseGroup::setNarrowSparseIndEnabled(bool enabled)
{
    if(isWeightSharingSlave()) {
        throw std::runtime_error("setNarrowSparseIndEnabled: Synapse group is a weight sharing slave. Sparse index type can only be set on the master.");
    }
    else {
        if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            m_NarrowSparseIndEnabled = enabled;
        }
        else {
            throw std::runtime_error("setNarrowSparseIndEnabled: This function can only be used on synapse groups with sparse connectivity.");
        }
    }
}
//----------------------------------------------------------------------------
unsigned int SynapseGroup::getMaxConnections() const
{ 
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    return isWeightSharingSlave() ? getWeightSharingMaster()->getMaxConnections() : m_MaxConnections; 
}
//----------------------------------------------------------------------------
unsigned int SynapseGroup::getMaxSourceConnections() const
{ 
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    return isWeightSharingSlave() ? getWeightSharingMaster()->getMaxSourceConnections() : m_MaxSourceConnections;
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getSparseConnectivityLocation() const
{ 
    return isWeightSharingSlave() ? getWeightSharingMaster()->getSparseConnectivityLocation() : m_SparseConnectivityLocation;
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
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    if(isWeightSharingSlave()) {
        return getWeightSharingMaster()->getWUVarLocation(var);
    }
    else {
        return m_WUVarLocation[getWUModel()->getVarIndex(var)];
    }
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getWUVarLocation(size_t index) const
{ 
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    if(isWeightSharingSlave()) {
        return getWeightSharingMaster()->getWUVarLocation(index);
    }
    else {
        return m_WUVarLocation.at(index);
    }
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
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    if(isWeightSharingSlave()) {
        return getWeightSharingMaster()->getSparseConnectivityExtraGlobalParamLocation(paramName);
    }
    else {
        return m_ConnectivityExtraGlobalParamLocation[m_ConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName)];
    }
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getSparseConnectivityExtraGlobalParamLocation(size_t index) const
{ 
    // **NOTE** these get retrived from weight sharing master 
    // as they can be set AFTER creation of synapse group
    if(isWeightSharingSlave()) {
        return getWeightSharingMaster()->getSparseConnectivityExtraGlobalParamLocation(index);
    }
    else {
        return m_ConnectivityExtraGlobalParamLocation.at(index);
    }
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
bool SynapseGroup::isProceduralConnectivityRNGRequired() const
{
    return ((m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) &&
            Utils::isRNGRequired(m_ConnectivityInitialiser.getSnippet()->getRowBuildCode()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPSInitRNGRequired() const
{
    // If initialising the postsynaptic variables require an RNG, return true
    return Utils::isRNGRequired(m_PSVarInitialisers);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUInitRNGRequired() const
{
    // If initialising the weight update variables require an RNG, return true
    if(Utils::isRNGRequired(m_WUVarInitialisers)) {
        return true;
    }

    // Return true if matrix has sparse or bitmask connectivity and an RNG is required to initialise connectivity
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && Utils::isRNGRequired(m_ConnectivityInitialiser.getSnippet()->getRowBuildCode()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUPreInitRNGRequired() const
{
    return Utils::isRNGRequired(m_WUPreVarInitialisers);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUPostInitRNGRequired() const
{
    return Utils::isRNGRequired(m_WUPostVarInitialisers);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isHostInitRNGRequired() const
{
    return (m_ConnectivityInitialiser.getSnippet()->getHostInitCode().find("$(rng)") != std::string::npos);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse state variables and isn't a 
    // weight sharing slave, return true if any of them have initialisation code
    if (!isWeightSharingSlave() && (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
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
    // Return true if the matrix type is sparse or bitmask, there is code to  
    // initialise sparse connectivity and synapse group isn't a weight sharing slave,
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && !getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty() && !isWeightSharingSlave());
}
//----------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string &name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup, const SynapseGroupInternal *weightSharingMaster,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                           VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_NumThreadsPerSpike(1), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup), m_WeightSharingMaster(weightSharingMaster),
        m_EventThresholdReTestRequired(false), m_NarrowSparseIndEnabled(defaultNarrowSparseIndEnabled),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_WUPreVarInitialisers(wuPreVarInitialisers), m_WUPostVarInitialisers(wuPostVarInitialisers),
        m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarLocation(wuVarInitialisers.size(), defaultVarLocation), m_WUPreVarLocation(wuPreVarInitialisers.size(), defaultVarLocation),
        m_WUPostVarLocation(wuPostVarInitialisers.size(), defaultVarLocation), m_WUExtraGlobalParamLocation(wu->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_PSVarLocation(psVarInitialisers.size(), defaultVarLocation), m_PSExtraGlobalParamLocation(ps->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_ConnectivityInitialiser(connectivityInitialiser), m_SparseConnectivityLocation(defaultSparseConnectivityLocation),
        m_ConnectivityExtraGlobalParamLocation(connectivityInitialiser.getSnippet()->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation), m_PSModelTargetName(name)
{
    // If connectivity is procedural
    if(m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) {
        // If there's no row build code, give an error
        if(m_ConnectivityInitialiser.getSnippet()->getRowBuildCode().empty()) {
            throw std::runtime_error("Cannot use procedural connectivity without specifying connectivity initialisation snippet");
        }

        // If the weight update model has code for postsynaptic-spike triggered updating, give an error
        if(!m_WUModel->getLearnPostCode().empty()) {
            throw std::runtime_error("Procedural connectivity cannot be used for synapse groups with postsynaptic spike-triggered learning");
        }

        // If weight update model has code for continuous synapse dynamics, give error
        // **THINK** this would actually be pretty trivial to implement
        if (!m_WUModel->getSynapseDynamicsCode().empty()) {
            throw std::runtime_error("Procedural connectivity cannot be used for synapse groups with continuous synapse dynamics");
        }
    }
    // Otherwise, if WEIGHTS are procedural e.g. in the case of DENSE_PROCEDURALG, give error if RNG is required for weights
    else if(m_MatrixType & SynapseMatrixWeight::PROCEDURAL) {
        if(::Utils::isRNGRequired(m_WUVarInitialisers)) {
            throw std::runtime_error("Procedural weights used without procedural connectivity cannot currently access RNG.");
        }
    }

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
std::string SynapseGroup::getSparseIndType() const
{
    // If narrow sparse inds are enabled
    if(m_NarrowSparseIndEnabled) {
        // If number of target neurons can be represented using a uint8, use this type
        const unsigned int numTrgNeurons = getTrgNeuronGroup()->getNumNeurons();
        if(numTrgNeurons <= std::numeric_limits<uint8_t>::max()) {
            return "uint8_t";
        }
        // Otherwise, if they can be represented as a uint16, use this type
        else if(numTrgNeurons <= std::numeric_limits<uint16_t>::max()) {
            return "uint16_t";
        }
    }

    // Otherwise, use 32-bit int
    return "uint32_t";

}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUBeMerged(const SynapseGroup &other) const
{
    if(getWUModel()->canBeMerged(other.getWUModel())
       && (getDelaySteps() == other.getDelaySteps())
       && (getBackPropDelaySteps() == other.getBackPropDelaySteps())
       && (getMaxDendriticDelayTimesteps() == other.getMaxDendriticDelayTimesteps())
       && (getSparseIndType() == other.getSparseIndType())
       && (getNumThreadsPerSpike() == other.getNumThreadsPerSpike())
       && (isEventThresholdReTestRequired() == other.isEventThresholdReTestRequired())
       && (getSpanType() == other.getSpanType())
       && (isPSModelMerged() == other.isPSModelMerged())
       && (getSrcNeuronGroup()->getNumDelaySlots() == other.getSrcNeuronGroup()->getNumDelaySlots())
       && (getTrgNeuronGroup()->getNumDelaySlots() == other.getTrgNeuronGroup()->getNumDelaySlots())
       && (getMatrixType() == other.getMatrixType()))
    {
        // If weights are procedural and any of the variable's initialisers can't be merged, return false
        if(getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
            for(size_t i = 0; i < getWUVarInitialisers().size(); i++) {
                if(!getWUVarInitialisers()[i].canBeMerged(other.getWUVarInitialisers()[i])) {
                    return false;
                }
            }
        }

        // If connectivity is either non-procedural or connectivity initialisers can be merged
        if(!(getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL)
           || getConnectivityInitialiser().canBeMerged(other.getConnectivityInitialiser()))
        {
            return true;
        }
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUPreBeMerged(const SynapseGroup &other) const
{
    const bool delayed = (getDelaySteps() != 0);
    const bool otherDelayed = (other.getDelaySteps() != 0);
    return (getWUModel()->canBeMerged(other.getWUModel())
            && (delayed == otherDelayed));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUPostBeMerged(const SynapseGroup &other) const
{
    const bool delayed = (getDelaySteps() != 0);
    const bool otherDelayed = (other.getDelaySteps() != 0);
    return (getWUModel()->canBeMerged(other.getWUModel())
            && (delayed == otherDelayed));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSBeMerged(const SynapseGroup &other) const
{
    const bool individualPSM = (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM);
    const bool otherIndividualPSM = (other.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM);
    if(getPSModel()->canBeMerged(other.getPSModel())
       && (getMaxDendriticDelayTimesteps() == other.getMaxDendriticDelayTimesteps())
       && (individualPSM == otherIndividualPSM))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSBeLinearlyCombined(const SynapseGroup &other) const
{
    // Postsynaptic models can be linearly combined if they can be merged and either 
    // they DON'T have individual postsynaptic model variables or they have no variable at all
    // **NOTE** many models with variables would work fine, but nothing stops
    // initialisers being used to configure PS models to behave totally different
    // **NOTE** similarly with EGPs
    return (canPSBeMerged(other)
            && (getPSParams() == other.getPSParams())
            && (getPSDerivedParams() == other.getPSDerivedParams())
            && getPSModel()->getExtraGlobalParams().empty()
            && (!(getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) || getPSVarInitialisers().empty()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUInitBeMerged(const SynapseGroup &other) const
{
    if((getMatrixType() == other.getMatrixType())
       && (getSparseIndType() == other.getSparseIndType())
       && (getWUModel()->getVars() == other.getWUModel()->getVars()))
    {
        // if any of the variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getWUVarInitialisers().size(); i++) {
            if(!getWUVarInitialisers()[i].canBeMerged(other.getWUVarInitialisers()[i])) {
                return false;
            }
        }

        return true;
    }
    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUPreInitBeMerged(const SynapseGroup &other) const
{
    if(getWUModel()->getPreVars() == other.getWUModel()->getPreVars()) {
        // if any of the presynaptic variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getWUPreVarInitialisers().size(); i++) {
            if(!getWUPreVarInitialisers()[i].canBeMerged(other.getWUPreVarInitialisers()[i])) {
                return false;
            }
        }

        return true;
    }
    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUPostInitBeMerged(const SynapseGroup &other) const
{
    if(getWUModel()->getPostVars() == other.getWUModel()->getPostVars()) {
        // if any of the postsynaptic variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getWUPostVarInitialisers().size(); i++) {
            if(!getWUPostVarInitialisers()[i].canBeMerged(other.getWUPostVarInitialisers()[i])) {
                return false;
            }
        }

        return true;
    }
    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSInitBeMerged(const SynapseGroup &other) const
{
    if((getPSModel()->getVars() == other.getPSModel()->getVars())
       && (getMaxDendriticDelayTimesteps() == other.getMaxDendriticDelayTimesteps()))
    {
        // if any of the variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getPSVarInitialisers().size(); i++) {
            if(!getPSVarInitialisers()[i].canBeMerged(other.getPSVarInitialisers()[i])) {
                return false;
            }
        }

        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canConnectivityInitBeMerged(const SynapseGroup &other) const
{
    // Connectivity initialization can be merged if the type of connectivity is the same and the initialisers can be merged
    return (getConnectivityInitialiser().canBeMerged(other.getConnectivityInitialiser())
            && (getSynapseMatrixConnectivity(getMatrixType()) == getSynapseMatrixConnectivity(other.getMatrixType()))
            && (getSparseIndType() == other.getSparseIndType()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canConnectivityHostInitBeMerged(const SynapseGroup &other) const
{
    // Connectivity host initialization can be merged if the initialisers 
    return getConnectivityInitialiser().canBeMerged(other.getConnectivityInitialiser());
}
