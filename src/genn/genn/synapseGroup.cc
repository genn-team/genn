#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// GeNN includes
#include "gennUtils.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
std::unordered_map<std::string, double> getConstInitVals(const std::unordered_map<std::string, Models::VarInit> &varInitialisers)
{
    // Reserve initial values to match initialisers
    std::unordered_map<std::string, double> initVals;

    // Transform variable initialisers into a vector of doubles
    std::transform(varInitialisers.cbegin(), varInitialisers.cend(), std::inserter(initVals, initVals.end()),
                   [](const auto &v)
                   {
                       // Check
                       if(dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr) {
                           throw std::runtime_error("Only 'Constant' variable initialisation snippets can be used to initialise state variables of synapse groups using GLOBALG");
                       }

                       // Return the first parameter (the value)
                       return std::make_pair(v.first, v.second.getParams().at("constant"));
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
void SynapseGroup::setPSTargetVar(const std::string &varName)
{
    // If varname is either 'ISyn' or name of target neuron group additional input variable, store
    const auto additionalInputVars = getTrgNeuronGroup()->getNeuronModel()->getAdditionalInputVars();
    if(varName == "Isyn" || 
       std::find_if(additionalInputVars.cbegin(), additionalInputVars.cend(), 
                    [&varName](const Models::Base::ParamVal &v){ return (v.name == varName); }) != additionalInputVars.cend())
    {
        m_PSTargetVar = varName;
    }
    else {
        throw std::runtime_error("Target neuron group has no input variable '" + varName + "'");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setPreTargetVar(const std::string &varName)
{
    // If varname is either 'ISyn' or name of a presynaptic neuron group additional input variable, store
    const auto additionalInputVars = getSrcNeuronGroup()->getNeuronModel()->getAdditionalInputVars();
    if(varName == "Isyn" || 
       std::find_if(additionalInputVars.cbegin(), additionalInputVars.cend(), 
                    [&varName](const Models::Base::ParamVal &v){ return (v.name == varName); }) != additionalInputVars.cend())
    {
        m_PreTargetVar = varName;
    }
    else {
        throw std::runtime_error("Presynaptic neuron group has no input variable '" + varName + "'");
    }
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
    const size_t extraGlobalParamIndex = m_SparseConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName);
    if(!Utils::isTypePointer(m_SparseConnectivityInitialiser.getSnippet()->getExtraGlobalParams()[extraGlobalParamIndex].type)) {
        throw std::runtime_error("Only extra global parameters with a pointer type have a location");
    }
    m_ConnectivityExtraGlobalParamLocation[extraGlobalParamIndex] = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setSparseConnectivityLocation(VarLocation loc)
{ 
    m_SparseConnectivityLocation = loc;
}
//----------------------------------------------------------------------------
void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
    if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if(m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc()) {
            throw std::runtime_error("setMaxConnections: Synapse group already has max connections defined by sparse connectivity initialisation snippet.");
        }

        m_MaxConnections = maxConnections;
    }
    else if(getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        throw std::runtime_error("setMaxConnections: Synapse group already has max connections defined by toeplitz connectivity initialisation snippet.");
    }
    else {
        throw std::runtime_error("setMaxConnections: Synapse group is densely connected. Setting max connections is not required in this case.");
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setMaxSourceConnections(unsigned int maxConnections)
{
    if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        if(m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc()) {
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
    if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        m_NarrowSparseIndEnabled = enabled;
    }
    else {
        throw std::runtime_error("setNarrowSparseIndEnabled: This function can only be used on synapse groups with sparse connectivity.");
    }
}
//----------------------------------------------------------------------------
unsigned int SynapseGroup::getMaxConnections() const
{ 
    return m_MaxConnections; 
}
//----------------------------------------------------------------------------
unsigned int SynapseGroup::getMaxSourceConnections() const
{ 
    return m_MaxSourceConnections;
}
//----------------------------------------------------------------------------
size_t SynapseGroup::getKernelSizeFlattened() const
{
    return std::accumulate(getKernelSize().cbegin(), getKernelSize().cend(), 1, std::multiplies<unsigned int>());
}
//----------------------------------------------------------------------------
VarLocation SynapseGroup::getSparseConnectivityLocation() const
{ 
    return m_SparseConnectivityLocation;
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
const std::unordered_map<std::string, double> SynapseGroup::getWUConstInitVals() const
{
    return getConstInitVals(m_WUVarInitialisers);
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
    return m_ConnectivityExtraGlobalParamLocation[m_SparseConnectivityInitialiser.getSnippet()->getExtraGlobalParamIndex(paramName)];
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
bool SynapseGroup::isPresynapticOutputRequired() const
{
    // If addToPre function is used in sim_code, return true
    if(getWUModel()->getSimCode().find("$(addToPre") != std::string::npos) {
        return true;
    }

    // If addToPre function is used in learn_post_code, return true
    if(getWUModel()->getLearnPostCode().find("$(addToPre") != std::string::npos) {
        return true;
    }

    // If addToPre function is used in event_code, return true
    if(getWUModel()->getEventCode().find("$(addToPre") != std::string::npos) {
        return true;
    }

    // If addToPre function is used in synapse_dynamics, return true
    if(getWUModel()->getSynapseDynamicsCode().find("$(addToPre") != std::string::npos) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::isProceduralConnectivityRNGRequired() const
{
    if(m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) {
        return (Utils::isRNGRequired(m_SparseConnectivityInitialiser.getSnippet()->getRowBuildCode())
                || Utils::isRNGRequired(m_SparseConnectivityInitialiser.getSnippet()->getColBuildCode()));
    }
    else if(m_MatrixType & SynapseMatrixConnectivity::TOEPLITZ) {
        return (Utils::isRNGRequired(m_ToeplitzConnectivityInitialiser.getSnippet()->getDiagonalBuildCode()));
    }
    else {
        return false;
    }
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
    const auto *snippet = m_SparseConnectivityInitialiser.getSnippet();
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && (Utils::isRNGRequired(snippet->getRowBuildCode()) || Utils::isRNGRequired(snippet->getColBuildCode())));
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
    return (m_SparseConnectivityInitialiser.getSnippet()->getHostInitCode().find("$(rng)") != std::string::npos);
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse or kernel state variables, 
    // return true if any of them have initialisation code which doesn't require a kernel
    if ((getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) || (getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        return std::any_of(m_WUVarInitialisers.cbegin(), m_WUVarInitialisers.cend(),
                           [](const auto &init)
                           { 
                               return !init.second.getSnippet()->getCode().empty() && !init.second.getSnippet()->requiresKernel(); 
                           });
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::isSparseConnectivityInitRequired() const
{
    // Return true if the matrix type is sparse or bitmask 
    // and there is code to initialise sparse connectivity 
    const auto *snippet = getSparseConnectivityInitialiser().getSnippet();
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && (!snippet->getRowBuildCode().empty() || !snippet->getColBuildCode().empty()));
}
//----------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string &name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Base *wu, const std::unordered_map<std::string, double> &wuParams, const std::unordered_map<std::string, Models::VarInit> &wuVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &wuPreVarInitialisers, const std::unordered_map<std::string, Models::VarInit> &wuPostVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::unordered_map<std::string, double> &psParams, const std::unordered_map<std::string, Models::VarInit> &psVarInitialisers,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                           const InitToeplitzConnectivitySnippet::Init &toeplitzInitialiser,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                           VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_NumThreadsPerSpike(1), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup), 
        m_EventThresholdReTestRequired(false), m_NarrowSparseIndEnabled(defaultNarrowSparseIndEnabled),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_WUPreVarInitialisers(wuPreVarInitialisers), m_WUPostVarInitialisers(wuPostVarInitialisers),
        m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarLocation(wuVarInitialisers.size(), defaultVarLocation), m_WUPreVarLocation(wuPreVarInitialisers.size(), defaultVarLocation),
        m_WUPostVarLocation(wuPostVarInitialisers.size(), defaultVarLocation), m_WUExtraGlobalParamLocation(wu->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_PSVarLocation(psVarInitialisers.size(), defaultVarLocation), m_PSExtraGlobalParamLocation(ps->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
        m_SparseConnectivityInitialiser(connectivityInitialiser), m_ToeplitzConnectivityInitialiser(toeplitzInitialiser), m_SparseConnectivityLocation(defaultSparseConnectivityLocation), 
        m_ConnectivityExtraGlobalParamLocation(connectivityInitialiser.getSnippet()->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation), 
        m_FusedPostOutputSource(nullptr), m_FusedWUPreSource(nullptr), m_FusedWUPostSource(nullptr), m_FusedPreOutputSource(nullptr), m_PSTargetVar("Isyn"), m_PreTargetVar("Isyn")
{
    // Validate names
    Utils::validatePopName(name, "Synapse group");
    getWUModel()->validate(getWUParams(), getWUVarInitialisers(), getWUPreVarInitialisers(), getWUPostVarInitialisers(), 
                           "Synapse group " + getName() + " weight update model ");
    getPSModel()->validate(getPSParams(), getPSVarInitialisers(), "Synapse group " + getName() + " postsynaptic model ");

    // If connectivity is procedural
    if(m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) {
        // If there's a toeplitz initialiser, give an error
        if(!m_ToeplitzConnectivityInitialiser.getSnippet()->getDiagonalBuildCode().empty()) {
            throw std::runtime_error("Cannot use procedural connectivity with toeplitz initialisation snippet");
        }

        // If there's no row build code, give an error
        if(m_SparseConnectivityInitialiser.getSnippet()->getRowBuildCode().empty()) {
            throw std::runtime_error("Cannot use procedural connectivity without specifying a connectivity initialisation snippet with row building code");
        }

        // If there's column build code, give an error
        if(!m_SparseConnectivityInitialiser.getSnippet()->getColBuildCode().empty()) {
            throw std::runtime_error("Cannot use procedural connectivity with connectivity initialisation snippets with column building code");
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
    
    // If synapse group has Toeplitz connectivity
    if(m_MatrixType & SynapseMatrixConnectivity::TOEPLITZ) {
        // Give an error if there is sparse connectivity initialiser code
        if(!m_SparseConnectivityInitialiser.getSnippet()->getRowBuildCode().empty() || !m_SparseConnectivityInitialiser.getSnippet()->getColBuildCode().empty()) {
            throw std::runtime_error("Cannot use TOEPLITZ connectivity with sparse connectivity initialisation snippet.");
        }

        // Give an error if there isn't toeplitz connectivity initialiser code
        if(m_ToeplitzConnectivityInitialiser.getSnippet()->getDiagonalBuildCode().empty()) {
            throw std::runtime_error("TOEPLITZ connectivity requires toeplitz connectivity initialisation snippet.");
        }

        // Give an error if connectivity initialisation snippet uses RNG
        if(::Utils::isRNGRequired(m_ToeplitzConnectivityInitialiser.getSnippet()->getDiagonalBuildCode())) {
            throw std::runtime_error("TOEPLITZ connectivity cannot currently access RNG.");
        }

        // If the weight update model has code for postsynaptic-spike triggered updating, give an error
        if(!m_WUModel->getLearnPostCode().empty()) {
            throw std::runtime_error("TOEPLITZ connectivity cannot be used for synapse groups with postsynaptic spike-triggered learning");
        }

        // If toeplitz initialisation snippet provides a function to calculate kernel size, call it
        auto calcKernelSizeFunc = m_ToeplitzConnectivityInitialiser.getSnippet()->getCalcKernelSizeFunc();
        if(calcKernelSizeFunc) {
            m_KernelSize = calcKernelSizeFunc(m_ToeplitzConnectivityInitialiser.getParams());
        }
        else {
            throw std::runtime_error("TOEPLITZ connectivity requires a toeplitz connectivity initialisation snippet which specifies a kernel size.");
        }

        // If toeplitz initialisation snippet provides a function to calculate max row length, call it
        auto calcMaxRowLengthFunc = m_ToeplitzConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc();
        if(calcMaxRowLengthFunc) {
            m_MaxConnections = calcMaxRowLengthFunc(srcNeuronGroup->getNumNeurons(), trgNeuronGroup->getNumNeurons(),
                                                    m_ToeplitzConnectivityInitialiser.getParams());
        }
        else {
            throw std::runtime_error("TOEPLITZ connectivity requires a toeplitz connectivity initialisation snippet which specifies a max row length.");
        }

        // No postsynaptic update through toeplitz matrices for now
        m_MaxSourceConnections = 0;
    }
    // Otherwise
    else {
        // If sparse connectivitity initialisation snippet provides a function to calculate kernel size, call it
        auto calcKernelSizeFunc = m_SparseConnectivityInitialiser.getSnippet()->getCalcKernelSizeFunc();
        if(calcKernelSizeFunc) {
            m_KernelSize = calcKernelSizeFunc(m_SparseConnectivityInitialiser.getParams());
        }

        // If connectivitity initialisation snippet provides a function to calculate row length, call it
        // **NOTE** only do this for sparse connectivity as this should not be set for bitmasks
        auto calcMaxRowLengthFunc = m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc();
        if(calcMaxRowLengthFunc && (m_MatrixType & SynapseMatrixConnectivity::SPARSE)) {
            m_MaxConnections = calcMaxRowLengthFunc(srcNeuronGroup->getNumNeurons(), trgNeuronGroup->getNumNeurons(),
                                                    m_SparseConnectivityInitialiser.getParams());
        }
        // Otherwise, default to the size of the target population
        else {
            m_MaxConnections = trgNeuronGroup->getNumNeurons();
        }

        // If connectivitity initialisation snippet provides a function to calculate row length, call it
        // **NOTE** only do this for sparse connectivity as this should not be set for bitmasks
        auto calcMaxColLengthFunc = m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc();
        if(calcMaxColLengthFunc && (m_MatrixType & SynapseMatrixConnectivity::SPARSE)) {
            m_MaxSourceConnections = calcMaxColLengthFunc(srcNeuronGroup->getNumNeurons(), trgNeuronGroup->getNumNeurons(),
                                                          m_SparseConnectivityInitialiser.getParams());
        }
        // Otherwise, default to the size of the source population
        else {
            m_MaxSourceConnections = srcNeuronGroup->getNumNeurons();
        }
    }

    // If connectivity initialisation snippet defines a kernel and matrix type doesn't support it, give error
    if(!m_KernelSize.empty() && (m_MatrixType != SynapseMatrixType::PROCEDURAL_PROCEDURALG) && (m_MatrixType != SynapseMatrixType::TOEPLITZ_KERNELG)
       && (m_MatrixType != SynapseMatrixType::SPARSE_INDIVIDUALG) && (m_MatrixType != SynapseMatrixType::PROCEDURAL_KERNELG)) 
    {
        throw std::runtime_error("Connectivity initialisation snippet which use a kernel can only be used with PROCEDURAL_PROCEDURALG, PROCEDURAL_KERNELG, TOEPLITZ_KERNELG or SPARSE_INDIVIDUALG connectivity.");
    }

    // If connectivity is dense and there is connectivity initialiser code, give error
    if((m_MatrixType & SynapseMatrixConnectivity::DENSE) 
       && (!m_SparseConnectivityInitialiser.getSnippet()->getRowBuildCode().empty() || !m_SparseConnectivityInitialiser.getSnippet()->getColBuildCode().empty())) 
    {
        throw std::runtime_error("Cannot use DENSE connectivity with connectivity initialisation snippet.");
    }

    // If synapse group uses sparse or procedural connectivity but no kernel size is provided, 
    // check that no variable's initialisation snippets require a kernel
    if(((m_MatrixType == SynapseMatrixType::SPARSE_INDIVIDUALG) || (m_MatrixType == SynapseMatrixType::PROCEDURAL_PROCEDURALG)) &&
       m_KernelSize.empty() && std::any_of(getWUVarInitialisers().cbegin(), getWUVarInitialisers().cend(), 
                                           [](const auto &v) { return v.second.getSnippet()->requiresKernel(); }))
    {
        throw std::runtime_error("Variable initialisation snippets which use $(id_kernel) must be used with a connectivity initialisation snippet which specifies how kernel size is calculated.");
    }

    // Check that the source neuron group supports the desired number of delay steps
    srcNeuronGroup->checkNumDelaySlots(delaySteps);
}
//----------------------------------------------------------------------------
void SynapseGroup::initDerivedParams(double dt)
{
    auto wuDerivedParams = getWUModel()->getDerivedParams();
    auto psDerivedParams = getPSModel()->getDerivedParams();

    // Loop through WU derived parameters
    for(const auto &d : wuDerivedParams) {
        m_WUDerivedParams.emplace(d.name, d.func(m_WUParams, dt));
    }

    // Loop through PSM derived parameters
    for(const auto &d : psDerivedParams) {
        m_PSDerivedParams.emplace(d.name, d.func(m_PSParams, dt));
    }

    // Initialise derived parameters for WU variable initialisers
    for(auto &v : m_WUVarInitialisers) {
        v.second.initDerivedParams(dt);
    }

    // Initialise derived parameters for PSM variable initialisers
    for(auto &v : m_PSVarInitialisers) {
        v.second.initDerivedParams(dt);
    }

    // Initialise derived parameters for WU presynaptic variable initialisers
    for(auto &v : m_WUPreVarInitialisers) {
        v.second.initDerivedParams(dt);
    }
    
    // Initialise derived parameters for WU postsynaptic variable initialisers
    for(auto &v : m_WUPostVarInitialisers) {
        v.second.initDerivedParams(dt);
    }

    // Initialise any derived connectivity initialiser parameters
    m_SparseConnectivityInitialiser.initDerivedParams(dt);
    m_ToeplitzConnectivityInitialiser.initDerivedParams(dt);
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSBeFused() const
{
    // Return true if there are no variables or extra global parameters
    // **NOTE** many models with variables would work fine, but  
    // nothing stops initialisers being used to configure PS models 
    // to behave totally different, similarly with EGPs
    return (getPSVarInitialisers().empty() && getPSModel()->getExtraGlobalParams().empty());
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUMPreUpdateBeFused() const
{
    // If any presynaptic variables aren't initialised to constant values, this synapse group's presynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    if(std::any_of(getWUPreVarInitialisers().cbegin(), getWUPreVarInitialisers().cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    const auto wumEGPs = getWUModel()->getExtraGlobalParams();
    const std::string preSpikeCode = getWUModel()->getPreSpikeCode();
    const std::string preDynamicsCode = getWUModel()->getPreDynamicsCode();
    for(const auto &egp : wumEGPs) {
        // If this EGP is referenced in presynaptic spike code, return false
        const std::string egpName = "$(" + egp.name + ")";
        if(preSpikeCode.find(egpName) != std::string::npos) {
            return false;
        }
        
        // If this EGP is referenced in presynaptic dynamics code, return false
        if(preDynamicsCode.find(egpName) != std::string::npos) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUMPostUpdateBeFused() const
{
    // If any postsynaptic variables aren't initialised to constant values, this synapse group's postsynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    if(std::any_of(getWUPostVarInitialisers().cbegin(), getWUPostVarInitialisers().cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    const auto wumEGPs = getWUModel()->getExtraGlobalParams();
    const std::string postSpikeCode = getWUModel()->getPostSpikeCode();
    const std::string postDynamicsCode = getWUModel()->getPostDynamicsCode();
    for(const auto &egp : wumEGPs) {
        // If this EGP is referenced in postsynaptic spike code, return false
        const std::string egpName = "$(" + egp.name + ")";
        if(postSpikeCode.find(egpName) != std::string::npos) {
            return false;
        }
        
        // If this EGP is referenced in postsynaptic dynamics code, return false
        if(postDynamicsCode.find(egpName) != std::string::npos) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPreOutputBeFused() const
{
    // There are no variables or other non-constant objects, so these can presumably always be fused
    return true;
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
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getHashDigest(), hash);
    Utils::updateHash(getDelaySteps(), hash);
    Utils::updateHash(getBackPropDelaySteps(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getNumThreadsPerSpike(), hash);
    Utils::updateHash(isEventThresholdReTestRequired(), hash);
    Utils::updateHash(getSpanType(), hash);
    Utils::updateHash(isPostOutputModelFused(), hash);
    Utils::updateHash(getSrcNeuronGroup()->getNumDelaySlots(), hash);
    Utils::updateHash(getTrgNeuronGroup()->getNumDelaySlots(), hash);
    Utils::updateHash(getMatrixType(), hash);

    // If weights are procedural, include variable initialiser hashes
    if(getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        for(const auto &w : getWUVarInitialisers()) {
            Utils::updateHash(w.first, hash);
            Utils::updateHash(w.second.getHashDigest(), hash);
        }
    }

    // If connectivity is procedural, include connectivitiy initialiser hash
    if(getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
        Utils::updateHash(getSparseConnectivityInitialiser().getHashDigest(), hash);
    }

    // If connectivity is Toepltiz, include Toeplitz connectivitiy initialiser hash
    if(getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        Utils::updateHash(getToeplitzConnectivityInitialiser().getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPreHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getHashDigest(), hash);
    Utils::updateHash((getDelaySteps() != 0), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getHashDigest(), hash);
    Utils::updateHash((getBackPropDelaySteps() != 0), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPSModel()->getHashDigest(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getPSTargetVar(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSFuseHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPSModel()->getHashDigest(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getPSTargetVar(), hash);
    Utils::updateHash(getPSParams(), hash);
    Utils::updateHash(getPSDerivedParams(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPreOutputHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPreTargetVar(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPreFuseHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getHashDigest(), hash);
    Utils::updateHash(getDelaySteps(), hash);

    // Loop through presynaptic variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPreUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    for(const auto &w : getWUPreVarInitialisers()) {
        assert(w.second.getParams().size() == 1);
        Utils::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    const std::string preSpikeCode = getWUModel()->getPreSpikeCode();
    const std::string preDynamicsCode = getWUModel()->getPreDynamicsCode();
    for(const auto &p : getWUModel()->getParamNames()) {
        const std::string paramName = "$(" + p + ")";
        if((preSpikeCode.find(paramName) != std::string::npos)
           || (preDynamicsCode.find(paramName) != std::string::npos)) 
        {
            Utils::updateHash(getWUParams().at(p), hash);
        }
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &d : getWUModel()->getDerivedParams()) {
        const std::string derivedParamName = "$(" + d.name + ")";
        if((preSpikeCode.find(derivedParamName) != std::string::npos)
           || (preDynamicsCode.find(derivedParamName) != std::string::npos)) 
        {
            Utils::updateHash(getWUDerivedParams().at(d.name), hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostFuseHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getHashDigest(), hash);
    Utils::updateHash(getBackPropDelaySteps(), hash);

    // Loop through postsynaptic variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPostUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    for(const auto &w : getWUPostVarInitialisers()) {
        assert(w.second.getParams().size() == 1);
        Utils::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    const std::string postSpikeCode = getWUModel()->getPostSpikeCode();
    const std::string postDynamicsCode = getWUModel()->getPostDynamicsCode();
    for(const auto &p : getWUModel()->getParamNames()) {
        const std::string paramName = "$(" + p + ")";
        if((postSpikeCode.find(paramName) != std::string::npos)
           || (postDynamicsCode.find(paramName) != std::string::npos)) 
        {
            Utils::updateHash(getWUParams().at(p), hash);
        }
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &d : getWUModel()->getDerivedParams()) {
        const std::string derivedParamName = "$(" + d.name + ")";
        if((postSpikeCode.find(derivedParamName) != std::string::npos)
           || (postDynamicsCode.find(derivedParamName) != std::string::npos)) 
        {
            Utils::updateHash(getWUDerivedParams().at(d.name), hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getDendriticDelayUpdateHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getMatrixType(), hash);
    Utils::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getWUModel()->getVars(), hash);

    // Include variable initialiser hashes
    for(const auto &w : getWUVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPreInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getPreVars(), hash);

    // Include presynaptic variable initialiser hashes
    for(const auto &w : getWUPreVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUModel()->getPostVars(), hash);

    // Include postsynaptic variable initialiser hashes
    for(const auto &w : getWUPostVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getPSModel()->getVars(), hash);

    // Include postsynaptic model variable initialiser hashes
    for(const auto &p : getPSVarInitialisers()) {
        Utils::updateHash(p.first, hash);
        Utils::updateHash(p.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPreOutputInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getConnectivityInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getSparseConnectivityInitialiser().getHashDigest(), hash);
    Utils::updateHash(getMatrixType(), hash);
    Utils::updateHash(getSparseIndType(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getConnectivityHostInitHashDigest() const
{
    return getSparseConnectivityInitialiser().getHashDigest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getRunnerHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getMatrixType(), hash);
    Utils::updateHash(isDendriticDelayRequired(), hash);
    Utils::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getWUModel()->getSynapseDynamicsCode().empty(), hash);
    Utils::updateHash(getWUModel()->getLearnPostCode().empty(), hash);
    Utils::updateHash(isPreOutputModelFused(), hash);
    Utils::updateHash(isPostOutputModelFused(), hash);
    Utils::updateHash(isWUPostModelFused(), hash);
    Utils::updateHash(isWUPreModelFused(), hash);
    Utils::updateHash(isPreOutputModelFuseSource(), hash);
    Utils::updateHash(isPostOutputModelFuseSource(), hash);
    Utils::updateHash(isWUPostModelFuseSource(), hash);
    Utils::updateHash(isWUPreModelFuseSource(), hash);

    // Weight update model variabless
    Utils::updateHash(getWUModel()->getVars(), hash);
    Utils::updateHash(getWUModel()->getExtraGlobalParams(), hash);

    // Weight update model variable initialisation EGPs
    for(const auto &var : getWUVarInitialisers()) {
        Utils::updateHash(var.first, hash);
        Utils::updateHash(var.second.getSnippet()->getExtraGlobalParams(), hash);
    }

    // Weight update model presynaptic variable initialisation EGPs
    for(const auto &var : getWUPreVarInitialisers()) {
        Utils::updateHash(var.first, hash);
        Utils::updateHash(var.second.getSnippet()->getExtraGlobalParams(), hash);
    }

    // Weight update model postsynaptic variable initialisation EGPs
    for(const auto &var : getWUPostVarInitialisers()) {
        Utils::updateHash(var.first, hash);
        Utils::updateHash(var.second.getSnippet()->getExtraGlobalParams(), hash);
    }

    // Postsynaptic model variables and EGPs
    Utils::updateHash(getPSModel()->getVars(), hash);
    Utils::updateHash(getPSModel()->getExtraGlobalParams(), hash);
    
    // Postsynaptic model variable initialisation EGPs
    for(const auto &var : getPSVarInitialisers()) {
        Utils::updateHash(var.first, hash);
        Utils::updateHash(var.second.getSnippet()->getExtraGlobalParams(), hash);
    }

    // **TODO** this will replace getVarLocationHashDigest entirely
    Utils::updateHash(getVarLocationHashDigest(), hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getInSynLocation(), hash);
    Utils::updateHash(getDendriticDelayLocation(), hash);
    Utils::updateHash(getSparseConnectivityLocation(), hash);
    Utils::updateHash(m_WUVarLocation, hash);
    Utils::updateHash(m_WUPreVarLocation, hash);
    Utils::updateHash(m_WUPostVarLocation, hash);
    Utils::updateHash(m_PSVarLocation, hash);
    Utils::updateHash(m_WUExtraGlobalParamLocation, hash);
    Utils::updateHash(m_PSExtraGlobalParamLocation, hash);
    return hash.get_digest();
}
