#include "synapseGroup.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
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
    const auto additionalInputVars = getTrgNeuronGroup()->getModel()->getAdditionalInputVars();
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
    const auto additionalInputVars = getSrcNeuronGroup()->getModel()->getAdditionalInputVars();
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
    if(maxDendriticDelayTimesteps < 1) {
        throw std::runtime_error("setMaxDendriticDelayTimesteps: A minimum of one dendritic delay timestep is required.");
    }
    m_MaxDendriticDelayTimesteps = maxDendriticDelayTimesteps;
}
//----------------------------------------------------------------------------
void SynapseGroup::setAxonalDelaySteps(unsigned int timesteps)
{
    m_AxonalDelaySteps = timesteps;

    m_SrcNeuronGroup->checkNumDelaySlots(m_AxonalDelaySteps);
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
bool SynapseGroup::isPreSpikeRequired() const
{
    return !Utils::areTokensEmpty(getWUInitialiser().getPreSpikeSynCodeTokens());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPreSpikeEventRequired() const
{
     return !Utils::areTokensEmpty(getWUInitialiser().getPreEventSynCodeTokens());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostSpikeRequired() const
{
     return !Utils::areTokensEmpty(getWUInitialiser().getPostSpikeSynCodeTokens());
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostSpikeEventRequired() const
{
     return !Utils::areTokensEmpty(getWUInitialiser().getPostEventSynCodeTokens());
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
bool SynapseGroup::isPostSpikeEventTimeRequired() const
{
    return isPostTimeReferenced("set_post");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPrevPostSpikeTimeRequired() const
{
    return isPostTimeReferenced("prev_st_post");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPrevPostSpikeEventTimeRequired() const
{
    return isPostTimeReferenced("prev_set_post");
}
//----------------------------------------------------------------------------
bool SynapseGroup::isZeroCopyEnabled() const
{
    if(m_OutputLocation & VarLocationAttribute::ZERO_COPY) {
        return true;
    }

    if(m_DendriticDelayLocation & VarLocationAttribute::ZERO_COPY) {
        return true;
    }
    
    if(m_SparseConnectivityLocation & VarLocationAttribute::ZERO_COPY) {
        return true;
    }

    // If there are any variables or EGPs implemented in zero-copy mode return true
    return (m_PSVarLocation.anyZeroCopy() || m_PSExtraGlobalParamLocation.anyZeroCopy()
            || m_WUVarLocation.anyZeroCopy() || m_WUPreVarLocation.anyZeroCopy() 
            || m_WUPostVarLocation.anyZeroCopy() || m_WUExtraGlobalParamLocation.anyZeroCopy());
}
//----------------------------------------------------------------------------
SynapseGroup::SynapseGroup(const std::string &name, SynapseMatrixType matrixType,
                           const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser,
                           const InitToeplitzConnectivitySnippet::Init &toeplitzInitialiser,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation,
                           VarLocation defaultSparseConnectivityLocation, bool defaultNarrowSparseIndEnabled)
    :   m_Name(name), m_ParallelismHint(ParallelismHint::POSTSYNAPTIC), m_NumThreadsPerSpike(1), m_AxonalDelaySteps(0), m_BackPropDelaySteps(0),
        m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup), 
        m_NarrowSparseIndEnabled(defaultNarrowSparseIndEnabled),
        m_OutputLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUInitialiser(wumInitialiser), m_PSInitialiser(psmInitialiser), m_SparseConnectivityInitialiser(connectivityInitialiser),  m_ToeplitzConnectivityInitialiser(toeplitzInitialiser), 
        m_WUVarLocation(defaultVarLocation), m_WUPreVarLocation(defaultVarLocation), m_WUPostVarLocation(defaultVarLocation), m_WUExtraGlobalParamLocation(defaultExtraGlobalParamLocation), 
        m_PSVarLocation(defaultVarLocation),  m_PSExtraGlobalParamLocation(defaultExtraGlobalParamLocation), m_SparseConnectivityLocation(defaultSparseConnectivityLocation),
        m_FusedPSTarget(nullptr), m_FusedPreSpikeTarget(nullptr), m_FusedPostSpikeTarget(nullptr), m_FusedPreSpikeEventTarget(nullptr), m_FusedPostSpikeEventTarget(nullptr),
        m_FusedWUPreTarget(nullptr), m_FusedWUPostTarget(nullptr), m_FusedPreOutputTarget(nullptr), m_PostTargetVar("Isyn"), m_PreTargetVar("Isyn")
{
    // 'Resolve' local variable references
    Models::resolveVarReferences(getPSInitialiser().getNeuronVarReferences(),
                                 m_PSNeuronVarReferences, getTrgNeuronGroup(),
                                 Models::VarReference::createVarRef)
    // Validate names
    Utils::validatePopName(name, "Synapse group");
    

    // Check variable reference types
    Models::checkVarReferenceTypes(getPreNeuronVarReferences(), getSnippet()->getPreNeuronVarRefs());
    Models::checkVarReferenceTypes(getPostNeuronVarReferences(), getSnippet()->getPostNeuronVarRefs());
    Models::checkVarReferenceTypes(getNeuronVarReferences(), getSnippet()->getNeuronVarRefs());

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
        if(isPostSpikeRequired()) {
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
        if(!Utils::areTokensEmpty(getWUInitialiser().getPostSpikeSynCodeTokens())) {
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
 }
//----------------------------------------------------------------------------
void SynapseGroup::setFusedPSTarget(const NeuronGroup *ng, const SynapseGroup &target)
{
    assert(ng == getTrgNeuronGroup());
    m_FusedPSTarget = &target; 
}
//----------------------------------------------------------------------------
void SynapseGroup::setFusedSpikeTarget(const NeuronGroup *ng, const SynapseGroup &target)
{ 
    if(ng == getSrcNeuronGroup()) {
        m_FusedPreSpikeTarget = &target; 
    }
    else {
        assert(ng == getTrgNeuronGroup());
        m_FusedPostSpikeTarget = &target; 
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setFusedSpikeEventTarget(const NeuronGroup *ng, const SynapseGroup &target)
{
    if(ng == getSrcNeuronGroup()) {
        m_FusedPreSpikeEventTarget = &target; 
    }
    else {
        assert(ng == getTrgNeuronGroup());
        m_FusedPostSpikeEventTarget = &target; 
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setFusedWUPrePostTarget(const NeuronGroup *ng, const SynapseGroup &target)
{ 
    if(ng == getSrcNeuronGroup()) {
        m_FusedWUPreTarget = &target;
    }
    else {
        assert(ng == getTrgNeuronGroup());    
        m_FusedWUPostTarget = &target; 
    }
}
//----------------------------------------------------------------------------
void SynapseGroup::setFusedPreOutputTarget(const NeuronGroup *ng, const SynapseGroup &target)
{ 
    assert(ng == getSrcNeuronGroup());
    m_FusedPreOutputTarget = &target; 
}
//----------------------------------------------------------------------------
void SynapseGroup::finalise(double dt)
{
    // Finalise derived parameters in Init objects
    m_PSInitialiser.finalise(dt);
    m_WUInitialiser.finalise(dt);
    m_SparseConnectivityInitialiser.finalise(dt);
    m_ToeplitzConnectivityInitialiser.finalise(dt);

    // Determine whether any postsynaptic neuron variable references 
    // are accessed with heterogeneous delays in synapse code
    bool heterogeneousVarDelay = std::any_of(getWUInitialiser().getPostNeuronVarReferences().cbegin(), 
                                             getWUInitialiser().getPostNeuronVarReferences().cend(),
                                             [this](const auto &v)
                                             { 
                                                return getWUInitialiser().isVarHeterogeneouslyDelayedInSynCode(v.first);
                                             });
    for(const auto &v : getWUInitialiser().getSnippet()->getPostVars()) {
        if(getWUInitialiser().isVarHeterogeneouslyDelayedInSynCode(v.name)) {
            m_HeterogeneouslyDelayedWUPostVars.insert(v.name);
            heterogeneousVarDelay = true;
        }
    }
    // If there are any dendritically delayed variables, ensure postsynaptic 
    // neuron group has enough delay slots to encompass maximum dendritic delay timesteps
    if(heterogeneousVarDelay) {
        m_TrgNeuronGroup->checkNumDelaySlots(getMaxDendriticDelayTimesteps());
    }

     // If weight update uses dendritic delay but maximum number of delay timesteps hasn't been specified
    if((heterogeneousVarDelay || isDendriticOutputDelayRequired()) && !m_MaxDendriticDelayTimesteps.has_value()) {
        throw std::runtime_error("Synapse group '" + getName() + "' uses a weight update model with heterogeneous dendritic delays but maximum dendritic delay timesteps has not been set");
    }

    // Loop through presynaptic variable references
    for(const auto &v : getWUInitialiser().getPreNeuronVarReferences()) {
        // If variable reference is referenced in synapse code, mark variable 
        // reference target as requiring queuing on source neuron group
        if(Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPreSpikeSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPreEventSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostEventSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostSpikeSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSynapseDynamicsCodeTokens()))
        {
            getSrcNeuronGroup()->setVarQueueRequired(v.second.getVarName());
        }
    }
    
    // Loop through postsynaptic variable references
    for(const auto &v : getWUInitialiser().getPostNeuronVarReferences()) {
        // If variable reference is referenced in synapse code, mark variable 
        // reference target as requiring queuing on target neuron group
        // **NOTE** this will also detect delayed references
        if(Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPreSpikeSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPreEventSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostEventSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getPostSpikeSynCodeTokens())
           || Utils::isIdentifierReferenced(v.first, getWUInitialiser().getSynapseDynamicsCodeTokens()))
        {
            getTrgNeuronGroup()->setVarQueueRequired(v.second.getVarName());
        }
    }

    // If synapse group has axonal delaysa
    if(getAxonalDelaySteps() > 0) {
        // If it has presynaptic spike triggered code, mark source neuron group as requiring a spike queue
        if(isPreSpikeRequired()) {
            getSrcNeuronGroup()->setSpikeQueueRequired();
        }
        // If it has presynaptic spike-like-event triggered code, mark source neuron group as requiring a spike event queue
        if(isPreSpikeEventRequired()) {
            getSrcNeuronGroup()->setSpikeEventQueueRequired();
        }
    }

    // If synapse group has backpropagation delays
    if(getBackPropDelaySteps() > 0) {
        // If it has postsynaptic spike triggered code, mark target neuron group as requiring a spike queue
        if(isPostSpikeRequired()) {
            getTrgNeuronGroup()->setSpikeQueueRequired();
        }
        // If it has postsynaptic spike-like-event triggered code, mark target neuron group as requiring a spike queue
        if(isPostSpikeEventRequired()) {
            getTrgNeuronGroup()->setSpikeEventQueueRequired();
        }
    }
}
//----------------------------------------------------------------------------
const SynapseGroup &SynapseGroup::getFusedSpikeTarget(const NeuronGroup *ng) const
{ 
    if(ng == getSrcNeuronGroup()) {
        return m_FusedPreSpikeTarget ? *m_FusedPreSpikeTarget : *this; 
    }
    else {
        assert(ng == getTrgNeuronGroup());
        return m_FusedPostSpikeTarget ? *m_FusedPostSpikeTarget : *this;
    }
}
//----------------------------------------------------------------------------
const SynapseGroup &SynapseGroup::getFusedSpikeEventTarget(const NeuronGroup *ng) const
{ 
    if(ng == getSrcNeuronGroup()) {
        return m_FusedPreSpikeEventTarget ? *m_FusedPreSpikeEventTarget : *this; 
    }
    else {
        assert(ng == getTrgNeuronGroup());
        return m_FusedPostSpikeEventTarget ? *m_FusedPostSpikeEventTarget : *this;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPSBeFused(const NeuronGroup *ng) const
{
    assert(ng == getTrgNeuronGroup());

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
        // If this EGP is referenced in sim code, return false
        if(Utils::isIdentifierReferenced(egp.name, getPSInitialiser().getSimCodeTokens())) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getPSInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic
        if(isPSParamDynamic(p.name)) {
            // If this parameter is referenced in sim code, return false
            if(Utils::isIdentifierReferenced(p.name, getPSInitialiser().getSimCodeTokens())) {
                return false;
            }
        }
    }
    
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUSpikeEventBeFused(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    // If any postsynaptic variables aren't initialised to constant values, this synapse group's postsynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    const auto &varInitialisers = presynaptic ? getWUInitialiser().getPreVarInitialisers() : getWUInitialiser().getPostVarInitialisers();
    if(std::any_of(varInitialisers.cbegin(), varInitialisers.cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    //const auto &eventThresholdCodeTokens = presynaptic ? getWUInitialiser().getPreEventThresholdCodeTokens() : getWUInitialiser().getPostEventThresholdCodeTokens();
    const auto &eventThresholdCodeTokens = getWUInitialiser().getPreEventThresholdCodeTokens();
    for(const auto &egp : getWUInitialiser().getSnippet()->getExtraGlobalParams()) {
        // If this EGP is referenced in event threshold code, return false
        if(Utils::isIdentifierReferenced(egp.name, eventThresholdCodeTokens)) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic and is referenced in event threshold code, return false
        if(isWUParamDynamic(p.name) && Utils::isIdentifierReferenced(p.name, eventThresholdCodeTokens)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::canWUMPrePostUpdateBeFused(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    // If any postsynaptic variables aren't initialised to constant values, this synapse group's postsynaptic update can't be merged
    // **NOTE** hash check will compare these constant values
    const auto &varInitialisers = presynaptic ? getWUInitialiser().getPreVarInitialisers() : getWUInitialiser().getPostVarInitialisers();
    if(std::any_of(varInitialisers.cbegin(), varInitialisers.cend(), 
                   [](const auto &v){ return (dynamic_cast<const InitVarSnippet::Constant*>(v.second.getSnippet()) == nullptr); }))
    {
        return false;
    }
    
    // Loop through EGPs
    const auto &spikeCodeTokens = presynaptic ? getWUInitialiser().getPreSpikeCodeTokens() : getWUInitialiser().getPostSpikeCodeTokens();
    const auto &dynamicsCodeTokens = presynaptic ? getWUInitialiser().getPreDynamicsCodeTokens() : getWUInitialiser().getPostDynamicsCodeTokens();
    for(const auto &egp : getWUInitialiser().getSnippet()->getExtraGlobalParams()) {
        // If this EGP is referenced in postsynaptic spike code, return false
        if(Utils::isIdentifierReferenced(egp.name, spikeCodeTokens)) {
            return false;
        }
        
        // If this EGP is referenced in postsynaptic dynamics code, return false
        if(Utils::isIdentifierReferenced(egp.name, dynamicsCodeTokens)) {
            return false;
        }
    }

    // Loop through parameters
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        // If parameter is dynamic
        if(isWUParamDynamic(p.name)) {
            // If this parameter is referenced in postsynaptic spike code, return false
            if(Utils::isIdentifierReferenced(p.name, spikeCodeTokens)) {
                return false;
            }
        
            // If this parameter is referenced in postsynaptic dynamics code, return false
            if(Utils::isIdentifierReferenced(p.name, dynamicsCodeTokens)) {
                return false;
            }
        }
    }
    return true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::isDendriticOutputDelayRequired() const
{
    return (Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getPreSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getPreEventSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getPostEventSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPostDelay", getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isWUPostVarHeterogeneouslyDelayed(const std::string &var) const
{
    return (m_HeterogeneouslyDelayedWUPostVars.count(var) == 0) ? false : true;
}
//----------------------------------------------------------------------------
bool SynapseGroup::areAnyWUPostVarHeterogeneouslyDelayed() const
{
    return !m_HeterogeneouslyDelayedWUPostVars.empty();
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPresynapticOutputRequired() const
{
    return (Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getPreSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getPreEventSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getPostEventSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getPostSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced("addToPre", getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostsynapticOutputRequired() const
{
    if(isDendriticOutputDelayRequired()) {
        return true;
    }
    else {
        return (Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getPreSpikeSynCodeTokens())
                || Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getPreEventSynCodeTokens())
                || Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getPostEventSynCodeTokens())
                || Utils::isIdentifierReferenced("addToPost", getWUInitialiser().getSynapseDynamicsCodeTokens()));
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
            && (!Utils::areTokensEmpty(getSparseConnectivityInitialiser().getRowBuildCodeTokens()) 
                || !Utils::areTokensEmpty(getSparseConnectivityInitialiser().getColBuildCodeTokens())));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPreTimeReferenced(const std::string &identifier) const
{
    return (Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreEventSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostEventSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreDynamicsCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreSpikeCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::isPostTimeReferenced(const std::string &identifier) const
{
    return (Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreEventSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostEventSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostEventThresholdCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostDynamicsCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPostSpikeCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getPreSpikeSynCodeTokens())
            || Utils::isIdentifierReferenced(identifier, getWUInitialiser().getSynapseDynamicsCodeTokens()));
}
//----------------------------------------------------------------------------
bool SynapseGroup::canPreOutputBeFused(const NeuronGroup *ng) const
{
    // There are no variables or other non-constant objects, so these can presumably always be fused
    assert(ng == getSrcNeuronGroup());
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
    Utils::updateHash(getAxonalDelaySteps(), hash);
    Utils::updateHash(getBackPropDelaySteps(), hash);
    Utils::updateHash(getMaxDendriticDelayTimesteps(), hash);
    Type::updateHash(getSparseIndType(), hash);
    Utils::updateHash(getNumThreadsPerSpike(), hash);
    Utils::updateHash(getParallelismHint(), hash);
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
        Utils::updateHash(getSparseConnectivityInitialiser().getHashDigest(), hash);
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
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPrePostHashDigest(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    boost::uuids::detail::sha1 hash;
    if(presynaptic) {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPreHashDigest(), hash);
        Utils::updateHash((getAxonalDelaySteps() != 0), hash);
    }
    else {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPostHashDigest(), hash);
        Utils::updateHash((getBackPropDelaySteps() != 0), hash);
        Utils::updateHash(m_HeterogeneouslyDelayedWUPostVars, hash);
        
    }
    m_WUDynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    const auto &neuronVarReferences = presynaptic ? getWUInitialiser().getPreNeuronVarReferences() : getWUInitialiser().getPostNeuronVarReferences();
    for(const auto &v : neuronVarReferences) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSHashDigest(const NeuronGroup *ng) const
{
    assert(ng == getTrgNeuronGroup());

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
boost::uuids::detail::sha1::digest_type SynapseGroup::getSpikeHashDigest(const NeuronGroup *ng) const
{
    assert((ng == getSrcNeuronGroup()) 
           || (ng == getTrgNeuronGroup()));
    boost::uuids::detail::sha1 hash;
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUSpikeEventHashDigest(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    boost::uuids::detail::sha1 hash;
    if(presynaptic) {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPreEventHashDigest(), hash);
        Utils::updateHash((getAxonalDelaySteps() != 0), hash);
    }
    else {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPostEventHashDigest(), hash);
        Utils::updateHash((getBackPropDelaySteps() != 0), hash);
    }
    m_WUDynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    const auto &neuronVarReferences = presynaptic ? getWUInitialiser().getPreNeuronVarReferences() : getWUInitialiser().getPostNeuronVarReferences();
    for(const auto &v : neuronVarReferences) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSFuseHashDigest(const NeuronGroup *ng) const
{
    assert(ng == getTrgNeuronGroup());

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
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUSpikeEventFuseHashDigest(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    boost::uuids::detail::sha1 hash;
    if(presynaptic) {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPreEventHashDigest(), hash);
        Utils::updateHash(getAxonalDelaySteps(), hash);
    }
    else {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPostEventHashDigest(), hash);
        Utils::updateHash(getBackPropDelaySteps(), hash);
    }

    // Loop through variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPreUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    const auto &varInitialisers = presynaptic ? getWUInitialiser().getPreVarInitialisers() : getWUInitialiser().getPostVarInitialisers();
    for(const auto &w : varInitialisers) {
        assert(w.second.getParams().size() == 1);
        Type::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    const auto &neuronVarReferences = presynaptic ? getWUInitialiser().getPreNeuronVarReferences() : getWUInitialiser().getPostNeuronVarReferences();
    for(const auto &v : neuronVarReferences) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    // Loop through weight update model parameters and, if they are referenced
    // in appropriate event threshold code, include their value in hash
    const auto &eventThresholdCodeTokens = presynaptic ? getWUInitialiser().getPreEventThresholdCodeTokens() : getWUInitialiser().getPostEventThresholdCodeTokens();
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        if(Utils::isIdentifierReferenced(p.name, eventThresholdCodeTokens)){
            Type::updateHash(getWUInitialiser().getParams().at(p.name), hash);
        }
    }

    // Loop through weight update model parameters and, if they are referenced
    // in appropriate event threshold code, include their value in hash
    for(const auto &d : getWUInitialiser().getSnippet()->getDerivedParams()) {
        if(Utils::isIdentifierReferenced(d.name, eventThresholdCodeTokens)){
            Type::updateHash(getWUInitialiser().getDerivedParams().at(d.name), hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPrePostFuseHashDigest(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    boost::uuids::detail::sha1 hash;
    if(presynaptic) {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPreHashDigest(), hash);
        Utils::updateHash(getAxonalDelaySteps(), hash);
    }
    else {
        Utils::updateHash(getWUInitialiser().getSnippet()->getPostHashDigest(), hash);
        Utils::updateHash(getBackPropDelaySteps(), hash);
        Utils::updateHash(m_HeterogeneouslyDelayedWUPostVars, hash);
    }

    // Loop through variable initialisers and hash first parameter.
    // Due to SynapseGroup::canWUMPreUpdateBeFused, all initialiser snippets
    // will be constant and have a single parameter containing the value
    const auto &varInitialisers = presynaptic ? getWUInitialiser().getPreVarInitialisers() : getWUInitialiser().getPostVarInitialisers();
    for(const auto &w : varInitialisers) {
        assert(w.second.getParams().size() == 1);
        Type::updateHash(w.second.getParams().at("constant"), hash);
    }

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    const auto &neuronVarReferences = presynaptic ? getWUInitialiser().getPreNeuronVarReferences() : getWUInitialiser().getPostNeuronVarReferences();
    for(const auto &v : neuronVarReferences) {
        Utils::updateHash(v.second.getVarName(), hash);
    };

    // Loop through weight update model parameters and, if they are referenced
    // in appropriate spike or dynamics code, include their value in hash
    const auto &spikeCodeTokens = presynaptic ? getWUInitialiser().getPreSpikeCodeTokens() : getWUInitialiser().getPostSpikeCodeTokens();
    const auto &dynamicsCodeTokens = presynaptic ? getWUInitialiser().getPreDynamicsCodeTokens() : getWUInitialiser().getPostDynamicsCodeTokens();
    for(const auto &p : getWUInitialiser().getSnippet()->getParams()) {
        if(Utils::isIdentifierReferenced(p.name, spikeCodeTokens)
           || Utils::isIdentifierReferenced(p.name, dynamicsCodeTokens)) 
        {
            Type::updateHash(getWUInitialiser().getParams().at(p.name), hash);
        }
    }

    // Loop through weight update model derived parameters and, if they are referenced
    // in appropriate spike or dynamics code, include their value in hash
    for(const auto &d : getWUInitialiser().getSnippet()->getDerivedParams()) {
        if(Utils::isIdentifierReferenced(d.name, spikeCodeTokens)
           || Utils::isIdentifierReferenced(d.name, dynamicsCodeTokens))
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
    Utils::updateHash(isPostSpikeRequired() || isPostSpikeEventRequired(), hash);

    // Include variable initialiser hashes
    for(const auto &w : getWUInitialiser().getVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getWUPrePostInitHashDigest(const NeuronGroup *ng) const
{
    const bool presynaptic = (ng == getSrcNeuronGroup());
    assert(presynaptic || (ng == getTrgNeuronGroup()));

    boost::uuids::detail::sha1 hash;
    const auto vars = presynaptic ? getWUInitialiser().getSnippet()->getPreVars() : getWUInitialiser().getSnippet()->getPostVars();
    Utils::updateHash(vars, hash);

    // Include presynaptic variable initialiser hashes
    const auto &varInitialisers = presynaptic ? getWUInitialiser().getPreVarInitialisers() : getWUInitialiser().getPostVarInitialisers();
    for(const auto &w : varInitialisers) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }

    if(!presynaptic) {
        Utils::updateHash(m_HeterogeneouslyDelayedWUPostVars, hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPSInitHashDigest(const NeuronGroup *ng) const
{
    assert(ng == getTrgNeuronGroup());

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
boost::uuids::detail::sha1::digest_type SynapseGroup::getPreOutputInitHashDigest(const NeuronGroup *ng) const
{
    assert(ng == getSrcNeuronGroup());

    boost::uuids::detail::sha1 hash;
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getPreOutputHashDigest(const NeuronGroup *ng) const
{
    assert(ng == getSrcNeuronGroup());

    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getPreTargetVar(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getConnectivityInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getSparseConnectivityInitialiser().getHashDigest(), hash);
    Utils::updateHash(getMatrixType(), hash);
    Type::updateHash(getSparseIndType(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getConnectivityHostInitHashDigest() const
{
    return getSparseConnectivityInitialiser().getHashDigest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroup::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getOutputLocation(), hash);
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
