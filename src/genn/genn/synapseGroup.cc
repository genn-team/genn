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
#include "type.h"

// ------------------------------------------------------------------------
// GeNN::SynapseGroup
// ------------------------------------------------------------------------
namespace GeNN
{
void SynapseGroup::setWUVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getWUInitialiser().getSnippet()->getVar(varName)) {
        throw std::runtime_error("Unknown weight update model variable '" + varName + "'");
    }
    m_WUVarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUPreVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getWUInitialiser().getSnippet()->getPreVar(varName)) {
        throw std::runtime_error("Unknown weight update model presynaptic variable '" + varName + "'");
    }
    m_WUPreVarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUPostVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getWUInitialiser().getSnippet()->getPostVar(varName)) {
        throw std::runtime_error("Unknown weight update model postsynaptic variable '" + varName + "'");
    }
    m_WUPostVarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getWUInitialiser().getSnippet()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown weight update model extra global parameter '" + paramName + "'");
    }
    m_WUExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setPSVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getPSInitialiser().getSnippet()->getVar(varName)) {
        throw std::runtime_error("Unknown postsynaptic model variable '" + varName + "'");
    }
    m_PSVarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setPSExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getPSInitialiser().getSnippet()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown postsynaptic model extra global parameter '" + paramName + "'");
    }
    m_PSExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setPSParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getPSInitialiser().getSnippet()->getParam(paramName)) {
        throw std::runtime_error("Unknown postsynaptic model parameter '" + paramName + "'");
    }
    m_PSDynamicParams.set(paramName, dynamic); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setWUParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getWUInitialiser().getSnippet()->getParam(paramName)) {
        throw std::runtime_error("Unknown weight update model parameter '" + paramName + "'");
    }
    m_WUDynamicParams.set(paramName, dynamic); 
}
//----------------------------------------------------------------------------
void SynapseGroup::setPostTargetVar(const std::string &varName)
{
    // If varname is either 'ISyn' or name of target neuron group additional input variable, store
    const auto additionalInputVars = getTrgNeuronGroup()->getNeuronModel()->getAdditionalInputVars();
    if(varName == "Isyn" || 
       std::find_if(additionalInputVars.cbegin(), additionalInputVars.cend(), 
                    [&varName](const Models::Base::ParamVal &v){ return (v.name == varName); }) != additionalInputVars.cend())
    {
        m_PostTargetVar = varName;
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
void SynapseGroup::setMaxConnections(unsigned int maxConnections)
{
    if(getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        // If sparse connectivity initialiser provides a function to calculate max row length
        auto calcMaxRowLengthFunc = m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxRowLengthFunc();
        if(calcMaxRowLengthFunc) {
            // Call function and if max connections we specify is less than the bound imposed by the snippet, give error
            auto connectivityMaxRowLength = calcMaxRowLengthFunc(getSrcNeuronGroup()->getNumNeurons(), getTrgNeuronGroup()->getNumNeurons(),
                                                                 m_SparseConnectivityInitialiser.getParams());
            if (maxConnections < connectivityMaxRowLength) {
                throw std::runtime_error("setMaxConnections: max connections must be higher than that already specified by sparse connectivity initialisation snippet.");
            }
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
        // If sparse connectivity initialiser provides a function to calculate max col length
        auto calcMaxColLengthFunc = m_SparseConnectivityInitialiser.getSnippet()->getCalcMaxColLengthFunc();
        if (calcMaxColLengthFunc) {
            // Call function and if max connections we specify is less than the bound imposed by the snippet, give error
            auto connectivityMaxColLength = calcMaxColLengthFunc(getSrcNeuronGroup()->getNumNeurons(), getTrgNeuronGroup()->getNumNeurons(),
                                                                 m_SparseConnectivityInitialiser.getParams());
            if (maxConnections < connectivityMaxColLength) {
                throw std::runtime_error("setMaxSourceConnections: max source connections must be higher than that already specified by sparse connectivity initialisation snippet.");
            }
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
size_t SynapseGroup::getKernelSizeFlattened() const
{
    return std::accumulate(getKernelSize().cbegin(), getKernelSize().cend(), 1, std::multiplies<unsigned int>());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isTrueSpikeRequired() const
{
    return !Utils::areTokensEmpty(getWUInitialiser().getSimCodeTokens());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isSpikeEventRequired() const
{
     return !Utils::areTokensEmpty(getWUInitialiser().getEventCodeTokens());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPreSpikeTimeRequired() const
{
    return isPreTimeReferenced("st_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPreSpikeEventTimeRequired() const
{
    return isPreTimeReferenced("set_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPrevPreSpikeTimeRequired() const
{
    return isPreTimeReferenced("prev_st_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPrevPreSpikeEventTimeRequired() const
{
    return isPreTimeReferenced("prev_set_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostSpikeTimeRequired() const
{
    return isPostTimeReferenced("st_post");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPrevPostSpikeTimeRequired() const
{
    return isPostTimeReferenced("prev_st_post");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isZeroCopyEnabled() const
{
    if(m_InSynLocation & VarLocation::ZERO_COPY) {
        return true;
    }

    if(m_DendriticDelayLocation & VarLocation::ZERO_COPY) {
        return true;
    }
    
    if(m_SparseConnectivityLocation & VarLocation::ZERO_COPY) {
        return true;
    }

    // If there are any variables or EGPs implemented in zero-copy mode return true
    return (m_PSVarLocation.anyZeroCopy() || m_PSExtraGlobalParamLocation.anyZeroCopy()
            || m_WUVarLocation.anyZeroCopy() || m_WUPreVarLocation.anyZeroCopy() 
            || m_WUPostVarLocation.anyZeroCopy() || m_WUExtraGlobalParamLocation.anyZeroCopy());
}
//----------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string &name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                           const InitToeplitzConnectivitySnippet::Init &toeplitzInitialiser,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                           VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_NumThreadsPerSpike(1), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup), 
        m_EventThresholdReTestRequired(false), m_NarrowSparseIndEnabled(defaultNarrowSparseIndEnabled),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUInitialiser(wumInitialiser), m_PSInitialiser(psmInitialiser), m_SparseConnectivityInitialiser(connectivityInitialiser),  m_ToeplitzConnectivityInitialiser(toeplitzInitialiser), 
        m_WUVarLocation(defaultVarLocation), m_WUPreVarLocation(defaultVarLocation), m_WUPostVarLocation(defaultVarLocation), m_WUExtraGlobalParamLocation(defaultExtraGlobalParamLocation), 
        m_PSVarLocation(defaultVarLocation),  m_PSExtraGlobalParamLocation(defaultExtraGlobalParamLocation), m_SparseConnectivityLocation(defaultSparseConnectivityLocation),
        m_FusedPSTarget(nullptr), m_FusedWUPreTarget(nullptr), m_FusedWUPostTarget(nullptr), m_FusedPreOutputTarget(nullptr), m_PostTargetVar("Isyn"), m_PreTargetVar("Isyn")
{
    // Validate names
    Utils::validatePopName(name, "Synapse group");
  
    
    // Check additional local variable reference constraints
    Models::checkLocalVarReferences(getPSInitialiser().getNeuronVarReferences(), getPSInitialiser().getSnippet()->getNeuronVarRefs(),
                                    getTrgNeuronGroup(), "Postsynaptic model variable references can only point to postsynaptic neuron group.");
    Models::checkLocalVarReferences(getWUInitialiser().getPreNeuronVarReferences(), getWUInitialiser().getSnippet()->getPreNeuronVarRefs(),
                                    getSrcNeuronGroup(), "Weight update model presynaptic variable references can only point to presynaptic neuron group.");
    Models::checkLocalVarReferences(getWUInitialiser().getPostNeuronVarReferences(), getWUInitialiser().getSnippet()->getPostNeuronVarRefs(),
                                    getTrgNeuronGroup(), "Weight update model postsynaptic variable references can only point to postsynaptic neuron group.");
    
    // If connectivity is procedural
    if(m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) {
        // If there's a toeplitz initialiser, give an error
        if(!Utils::areTokensEmpty(m_ToeplitzConnectivityInitialiser.getDiagonalBuildCodeTokens())) {
            throw std::runtime_error("Cannot use procedural connectivity with toeplitz initialisation snippet");
        }

        // If there's no row build code, give an error
        if(Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getRowBuildCodeTokens())) {
            throw std::runtime_error("Cannot use procedural connectivity without specifying a connectivity initialisation snippet with row building code");
        }

        // If there's column build code, give an error
        if(!Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getColBuildCodeTokens())) {
            throw std::runtime_error("Cannot use procedural connectivity with connectivity initialisation snippets with column building code");
        }

        // If the weight update model has code for postsynaptic-spike triggered updating, give an error
        if(!Utils::areTokensEmpty(getWUInitialiser().getPostLearnCodeTokens())) {
            throw std::runtime_error("Procedural connectivity cannot be used for synapse groups with postsynaptic spike-triggered learning");
        }

        // If weight update model has code for continuous synapse dynamics, give error
        // **THINK** this would actually be pretty trivial to implement
        if (!Utils::areTokensEmpty(getWUInitialiser().getSynapseDynamicsCodeTokens())) {
            throw std::runtime_error("Procedural connectivity cannot be used for synapse groups with continuous synapse dynamics");
        }
    }
    // Otherwise, if WEIGHTS are procedural e.g. in the case of DENSE_PROCEDURALG, give error if RNG is required for weights
    else if(m_MatrixType & SynapseMatrixWeight::PROCEDURAL) {
        if(Utils::isRNGRequired(getWUInitialiser().getVarInitialisers())) {
            throw std::runtime_error("Procedural weights used without procedural connectivity cannot currently access RNG.");
        }
    }
    
    // If synapse group has Toeplitz connectivity
    if(m_MatrixType & SynapseMatrixConnectivity::TOEPLITZ) {
        // Give an error if there is sparse connectivity initialiser code
        if(!Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getRowBuildCodeTokens()) 
           || !Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getColBuildCodeTokens())) 
        {
            throw std::runtime_error("Cannot use TOEPLITZ connectivity with sparse connectivity initialisation snippet.");
        }

        // Give an error if there isn't toeplitz connectivity initialiser code
        if(Utils::areTokensEmpty(m_ToeplitzConnectivityInitialiser.getDiagonalBuildCodeTokens())) {
            throw std::runtime_error("TOEPLITZ connectivity requires toeplitz connectivity initialisation snippet.");
        }

        // Give an error if connectivity initialisation snippet uses RNG
        if(Utils::isRNGRequired(m_ToeplitzConnectivityInitialiser.getDiagonalBuildCodeTokens())) {
            throw std::runtime_error("TOEPLITZ connectivity cannot currently access RNG.");
        }

        // If the weight update model has code for postsynaptic-spike triggered updating, give an error
        if(!Utils::areTokensEmpty(getWUInitialiser().getPostLearnCodeTokens())) {
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
    if(!m_KernelSize.empty() && (m_MatrixType != SynapseMatrixType::PROCEDURAL) && (m_MatrixType != SynapseMatrixType::TOEPLITZ)
       && (m_MatrixType != SynapseMatrixType::SPARSE) && (m_MatrixType != SynapseMatrixType::PROCEDURAL_KERNELG)) 
    {
        throw std::runtime_error("BITMASK connectivity can only be used with weight update models without variables like StaticPulseConstantWeight.");
    }

    // If connectivity is dense and there is connectivity initialiser code, give error
    if((m_MatrixType & SynapseMatrixConnectivity::DENSE) 
       && (!Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getRowBuildCodeTokens()) 
           || !Utils::areTokensEmpty(m_SparseConnectivityInitialiser.getColBuildCodeTokens()))) 
    {
        throw std::runtime_error("Cannot use DENSE connectivity with connectivity initialisation snippet.");
    }

    // If synapse group uses sparse or procedural connectivity but no kernel size is provided, 
    // check that no variable's initialisation snippets require a kernel
    if(((m_MatrixType == SynapseMatrixType::SPARSE) || (m_MatrixType == SynapseMatrixType::PROCEDURAL)) &&
       m_KernelSize.empty() && std::any_of(getWUInitialiser().getVarInitialisers().cbegin(), getWUInitialiser().getVarInitialisers().cend(), 
                                           [](const auto &v) { return v.second.isKernelRequired(); }))
    {
        throw std::runtime_error("Variable initialisation snippets which use $(id_kernel) must be used with a connectivity initialisation snippet which specifies how kernel size is calculated.");
    }

    // Check that the source neuron group supports the desired number of delay steps
    srcNeuronGroup->checkNumDelaySlots(delaySteps);
}
//----------------------------------------------------------------------------
void SynapseGroup::finalise(double dt)
{
    // Finalise derived parameters in Init objects
    m_PSInitialiser.finalise(dt);
    m_WUInitialiser.finalise(dt);
    m_SparseConnectivityInitialiser.finalise(dt);
    m_ToeplitzConnectivityInitialiser.finalise(dt);

    // Loop through presynaptic variable references
    for(const auto &v : getWUInitialiser().getPreNeuronVarReferences()) {
        // If variable reference is referenced in synapse code, mark variable 
        // reference target as requiring queuing on source neuron group
        if(Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSimCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getEventCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostLearnCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSynapseDynamicsCodeTokens()))
        {
            getSrcNeuronGroup()->setVarQueueRequired(v.second.getVarName());
        }
    }
    
    // Loop through postsynaptic variable references
    for(const auto &v : getWUInitialiser().getPostNeuronVarReferences()) {
        // If variable reference is referenced in synapse code, mark variable 
        // reference target as requiring queuing on target neuron group
        if(Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSimCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getEventCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostLearnCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSynapseDynamicsCodeTokens()))
        {
            getTrgNeuronGroup()->setVarQueueRequired(v.second.getVarName());
        }
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSBeFused() const
{
    // If any postsynaptic model variables aren't initialised to constant values, this synapse group's postsynaptic model can't be merged
    // **NOTE** hash check will compare these constant values
    if(std::any_of(getPSInitialiser().getVarInitialisers().cbegin(), getPSInitialiser().getVarInitialisers().cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    // **NOTE** this is kind of silly as, if it's not referenced in either of 
    // these code strings, there wouldn't be a lot of point in a PSM EGP existing!
    for(const auto &egp : getPSInitialiser().getSnippet()->getExtraGlobalParams()) {
        // If this EGP is referenced in decay code, return false
        if(Utils::isIdentifierReferenced(egp.name, getPSInitialiser().getDecayCodeTokens())) {
            return false;
        }
        
        // If this EGP is referenced in apply input code, return false
        if(Utils::isIdentifierReferenced(egp.name, getPSInitialiser().getApplyInputCodeTokens())) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getPSInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic
        if(isPSParameterDynamic(p.name)) {
            // If this parameter is referenced in decay code, return false
            if(Utils::isIdentifierReferenced(p.name, getPSInitialiser().getDecayCodeTokens())) {
                return false;
            }
        
            // If this parameter is referenced in apply input code, return false
            if(Utils::isIdentifierReferenced(p.name, getPSInitialiser().getApplyInputCodeTokens())) {
                return false;
            }
        }
    }
    
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUMPreUpdateBeFused() const
{
    // If any presynaptic variables aren't initialised to constant values, this synapse group's presynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    if(std::any_of(getWUInitialiser().getPreVarInitialisers().cbegin(), getWUInitialiser().getPreVarInitialisers().cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    for(const auto &egp : getWUInitialiser().getSnippet()->getExtraGlobalParams()) {
        // If this EGP is referenced in presynaptic spike code, return false
        if(Utils::isIdentifierReferenced(egp.name, getWUInitialiser().getPreSpikeCodeTokens())) {
            return false;
        }
        
        // If this EGP is referenced in presynaptic dynamics code, return false
        if(Utils::isIdentifierReferenced(egp.name, getWUInitialiser().getPreDynamicsCodeTokens())) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic
        if(isWUParameterDynamic(p.name)) {
            // If this parameter is referenced in presynaptic spike code, return false
            if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPreSpikeCodeTokens())) {
                return false;
            }
        
            // If this parameter is referenced in presynaptic dynamics code, return false
            if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPreDynamicsCodeTokens())) {
                return false;
            }
        }
    }

    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUMPostUpdateBeFused() const
{
    // If any postsynaptic variables aren't initialised to constant values, this synapse group's postsynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    if(std::any_of(getWUInitialiser().getPostVarInitialisers().cbegin(), getWUInitialiser().getPostVarInitialisers().cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    for(const auto &egp : getWUInitialiser().getSnippet()->getExtraGlobalParams()) {
        // If this EGP is referenced in postsynaptic spike code, return false
        if(Utils::isIdentifierReferenced(egp.name, getWUInitialiser().getPostSpikeCodeTokens())) {
            return false;
        }
        
        // If this EGP is referenced in postsynaptic dynamics code, return false
        if(Utils::isIdentifierReferenced(egp.name, getWUInitialiser().getPostDynamicsCodeTokens())) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic
        if(isWUParameterDynamic(p.name)) {
            // If this parameter is referenced in postsynaptic spike code, return false
            if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPostSpikeCodeTokens())) {
                return false;
            }
        
            // If this parameter is referenced in postsynaptic dynamics code, return false
            if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPostDynamicsCodeTokens())) {
                return false;
            }
        }
    }
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::isDendriticDelayRequired() const
{
    // If addToPostDelay function is used in sim code, return true
    if(Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getSimCodeTokens())) {
        return true;
    }

    // If addToPostDelay function is used in event code, return true
    if(Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getEventCodeTokens())) {
        return true;
    }

    // If addToPostDelay function is used in synapse dynamics, return tru
    if(Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getSynapseDynamicsCodeTokens())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPresynapticOutputRequired() const
{
    // If addToPre function is used in sim code, return true
    if(Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getSimCodeTokens())) {
        return true;
    }

    // If addToPre function is used in event code, return true
    if(Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getEventCodeTokens())) {
        return true;
    }

    // If addToPre function is used in learn post code, return true
    if(Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getPostLearnCodeTokens())) {
        return true;
    }

    // If addToPre function is used in synapse dynamics, return tru
    if(Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getSynapseDynamicsCodeTokens())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostsynapticOutputRequired() const
{
    if(isDendriticDelayRequired()) {
        return true;
    }
    else {
        // If addToPost function is used in sim code, return true
        if(Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getSimCodeTokens())) {
            return true;
        }

        // If addToPost function is used in event code, return true
        if(Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getEventCodeTokens())) {
            return true;
        }

        // If addToPost function is used in synapse dynamics, return tru
        if(Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getSynapseDynamicsCodeTokens())) {
            return true;
        }

        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::isProceduralConnectivityRNGRequired() const
{
    if(m_MatrixType & SynapseMatrixConnectivity::PROCEDURAL) {
        return m_SparseConnectivityInitialiser.isRNGRequired();
    }
    else if(m_MatrixType & SynapseMatrixConnectivity::TOEPLITZ) {
        return m_ToeplitzConnectivityInitialiser.isRNGRequired();
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUInitRNGRequired() const
{
    // If initialising the weight update variables require an RNG, return true
    if(Utils::isRNGRequired(getWUInitialiser().getVarInitialisers())) {
        return true;
    }

    // Return true if matrix has sparse or bitmask connectivity and an RNG is required to initialise connectivity
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && m_SparseConnectivityInitialiser.isRNGRequired());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPSVarInitRequired() const
{
    return std::any_of(getPSInitialiser().getVarInitialisers().cbegin(), getPSInitialiser().getVarInitialisers().cend(),
                       [](const auto &init)
                       { 
                           return !Utils::areTokensEmpty(init.second.getCodeTokens());
                       });
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse or kernel state variables, 
    // return true if any of them have initialisation code which doesn't require a kernel
    if ((getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) || (getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        return std::any_of(getWUInitialiser().getVarInitialisers().cbegin(), getWUInitialiser().getVarInitialisers().cend(),
                           [](const auto &init)
                           { 
                               return !Utils::areTokensEmpty(init.second.getCodeTokens()) && !init.second.isKernelRequired();
                           });
    }
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUPreVarInitRequired() const
{
    return std::any_of(getWUInitialiser().getPreVarInitialisers().cbegin(), getWUInitialiser().getPreVarInitialisers().cend(),
                       [](const auto &init)
                       { 
                           return !Utils::areTokensEmpty(init.second.getCodeTokens());
                       });
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUPostVarInitRequired() const
{
    return std::any_of(getWUInitialiser().getPostVarInitialisers().cbegin(), getWUInitialiser().getPostVarInitialisers().cend(),
                       [](const auto &init)
                       { 
                           return !Utils::areTokensEmpty(init.second.getCodeTokens());
                       });
}
//----------------------------------------------------------------------------
bool SynapseGroup::isSparseConnectivityInitRequired() const
{
    // Return true if the matrix type is sparse or bitmask 
    // and there is code to initialise sparse connectivity 
    return (((m_MatrixType & SynapseMatrixConnectivity::SPARSE) || (m_MatrixType & SynapseMatrixConnectivity::BITMASK))
            && (!Utils::areTokensEmpty(getConnectivityInitialiser().getRowBuildCodeTokens()) 
                || !Utils::areTokensEmpty(getConnectivityInitialiser().getColBuildCodeTokens())));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPreTimeReferenced(const std::string &identifier) const
{
    return (Utils::isIdentifierReferenced(identifier, getWUInitialiser().getEventCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostLearnCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreDynamicsCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreSpikeCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSimCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostTimeReferenced(const std::string &identifier) const
{
    return (Utils::isIdentifierReferenced(identifier, getWUInitialiser().getEventCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostLearnCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostDynamicsCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostSpikeCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSimCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPreOutputBeFused() const
{
    // There are no variables or other non-constant objects, so these can presumably always be fused
    return true;
}
//----------------------------------------------------------------------------
const Type::ResolvedType &SynapseGroup::getSparseIndType() const
{
    // If narrow sparse inds are enabled
    if(m_NarrowSparseIndEnabled) {
        // If number of target neurons can be represented using a uint8, use this type
        const unsigned int numTrgNeurons = getTrgNeuronGroup()->getNumNeurons();
        if(numTrgNeurons <= std::numeric_limits<uint8_t>::max()) {
            return Type::Uint8;
        }
        // Otherwise, if they can be represented as a uint16, use this type
        else if(numTrgNeurons <= std::numeric_limits<uint16_t>::max()) {
            return Type::Uint16;
        }
    }

    // Otherwise, use 32-bit int
    return Type::Uint32;
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getHashDigest(), hash);
    Utils::updateHash(getDelaySteps(), hash);
    Utils::updateHash(getBackPropDelaySteps(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Type::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getNumThreadsPerSpike(), hash);
    Utils::updateHash(isEventThresholdReTestRequired(), hash);
    Utils::updateHash(getSpanType(), hash);
    Utils::updateHash(isPSModelFused(), hash);
    Utils::updateHash(getSrcNeuronGroup()->getNumDelaySlots(), hash);
    Utils::updateHash(getTrgNeuronGroup()->getNumDelaySlots(), hash);
    Utils::updateHash(getMatrixType(), hash);
    m_WUDynamicParams.updateHash(hash);

    // If weights are procedural, include variable initialiser hashes
    if(getMatrixType() & SynapseMatrixWeight::PROCEDURAL) {
        for(const auto &w : getWUInitialiser().getVarInitialisers()) {
            Utils::updateHash(w.first, hash);
            Utils::updateHash(w.second.getHashDigest(), hash);
        }
    }

    // If connectivity is procedural, include connectivitiy initialiser hash
    if(getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) {
        Utils::updateHash(getConnectivityInitialiser().getHashDigest(), hash);
    }

    // If connectivity is Toepltiz, include Toeplitz connectivitiy initialiser hash
    if(getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) {
        Utils::updateHash(getToeplitzConnectivityInitialiser().getHashDigest(), hash);
    }

    // Loop through presynaptic neuron variable references
    for(const auto &v : getWUInitialiser().getPreNeuronVarReferences()) {
        // Update hash with whether variable references require delay
        Utils::updateHash((v.second.getDelayNeuronGroup() == nullptr), hash);

        // Update hash with target variable dimensions as this effects indexing code
        Utils::updateHash(v.second.getVarDims(), hash);
    }

    // Loop through postsynapatic neuron variable references
    for(const auto &v : getWUInitialiser().getPostNeuronVarReferences()) {
        // Update hash with whether variable references require delay
        Utils::updateHash((v.second.getDelayNeuronGroup() == nullptr), hash);

        // Update hash with target variable dimensions as this effects indexing code
        Utils::updateHash(v.second.getVarDims(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPreHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getHashDigest(), hash);
    Utils::updateHash((getDelaySteps() != 0), hash);
    m_WUDynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getWUInitialiser().getPreNeuronVarReferences()) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getHashDigest(), hash);
    Utils::updateHash((getBackPropDelaySteps() != 0), hash);
    m_WUDynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getWUInitialiser().getPostNeuronVarReferences()) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPSInitialiser().getSnippet()->getHashDigest(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getPostTargetVar(), hash);
    m_PSDynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getPSInitialiser().getNeuronVarReferences()) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSFuseHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPSInitialiser().getSnippet()->getHashDigest(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Utils::updateHash(getPostTargetVar(), hash);
    Utils::updateHash(getPSInitialiser().getParams(), hash);
    Utils::updateHash(getPSInitialiser().getDerivedParams(), hash);
    
    // Loop through PSM variable initialisers and hash first parameter.
    // Due to SynapseGroup::canPSBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    for(const auto &w : getPSInitialiser().getVarInitialisers()) {
        assert(w.second.getParams().size() == 1);
        Type::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getPSInitialiser().getNeuronVarReferences()) {
        Utils::updateHash(v.second.getVarName(), hash);
    };
    
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
    Utils::updateHash(getWUInitialiser().getSnippet()->getPreHashDigest(), hash);
    Utils::updateHash(getDelaySteps(), hash);

    // Loop through presynaptic variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPreUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    for(const auto &w : getWUInitialiser().getPreVarInitialisers()) {
        assert(w.second.getParams().size() == 1);
        Type::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPreSpikeCodeTokens())
           || Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPreDynamicsCodeTokens())) 
        {
            Type::updateHash(getWUInitialiser().getParams().at(p.name), hash);
        }
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &d : getWUInitialiser().getSnippet()->getDerivedParams()) {
        if(Utils::isIdentifierReferenced(d.name, getWUInitialiser().getPreSpikeCodeTokens())
           || Utils::isIdentifierReferenced(d.name, getWUInitialiser().getPreDynamicsCodeTokens()))
        {
            Type::updateHash(getWUInitialiser().getDerivedParams().at(d.name), hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostFuseHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getPostHashDigest(), hash);
    Utils::updateHash(getBackPropDelaySteps(), hash);

    // Loop through postsynaptic variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPostUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    for(const auto &w : getWUInitialiser().getPostVarInitialisers()) {
        assert(w.second.getParams().size() == 1);
        Type::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
       if(Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPostSpikeCodeTokens())
           || Utils::isIdentifierReferenced(p.name, getWUInitialiser().getPostDynamicsCodeTokens())) 
        {
            Type::updateHash(getWUInitialiser().getParams().at(p.name), hash);
        }
    }

    // Loop through weight update model parameters and, if they are referenced
    // in presynaptic spike or dynamics code, include their value in hash
    for(const auto &d : getWUInitialiser().getSnippet()->getDerivedParams()) {
        if(Utils::isIdentifierReferenced(d.name, getWUInitialiser().getPostSpikeCodeTokens())
           || Utils::isIdentifierReferenced(d.name, getWUInitialiser().getPostDynamicsCodeTokens())) 
        {
            Type::updateHash(getWUInitialiser().getDerivedParams().at(d.name), hash);
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
    Type::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getWUInitialiser().getSnippet()->getVars(), hash);

    Utils::updateHash(Utils::areTokensEmpty(getWUInitialiser().getSynapseDynamicsCodeTokens()), hash);
    Utils::updateHash(Utils::areTokensEmpty(getWUInitialiser().getPostLearnCodeTokens()), hash);

    // Include variable initialiser hashes
    for(const auto &w : getWUInitialiser().getVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPreInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getPreVars(), hash);

    // Include presynaptic variable initialiser hashes
    for(const auto &w : getWUInitialiser().getPreVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPostInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getWUInitialiser().getSnippet()->getPostVars(), hash);

    // Include postsynaptic variable initialiser hashes
    for(const auto &w : getWUInitialiser().getPostVarInitialisers()) {
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
    Utils::updateHash(getPSInitialiser().getSnippet()->getVars(), hash);

    // Include postsynaptic model variable initialiser hashes
    for(const auto &p : getPSInitialiser().getVarInitialisers()) {
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
    Utils::updateHash(getConnectivityInitialiser().getHashDigest(), hash);
    Utils::updateHash(getMatrixType(), hash);
    Type::updateHash(getSparseIndType(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getConnectivityHostInitHashDigest() const
{
    return getConnectivityInitialiser().getHashDigest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getInSynLocation(), hash);
    Utils::updateHash(getDendriticDelayLocation(), hash);
    Utils::updateHash(getSparseConnectivityLocation(), hash);
    m_WUVarLocation.updateHash(hash);
    m_WUPreVarLocation.updateHash(hash);
    m_WUPostVarLocation.updateHash(hash);
    m_PSVarLocation.updateHash(hash);
    m_WUExtraGlobalParamLocation.updateHash(hash);
    m_PSExtraGlobalParamLocation.updateHash(hash);
    return hash.get_digest();
}
}   // namespace GeNN
