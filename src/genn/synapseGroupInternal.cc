#include "synapseGroupInternal.h"

// Standard includes
#include <algorithm>
#include <cmath>
#include <iostream>

// GeNN includes
#include "neuronGroupInternal.h"
#include "gennUtils.h"

//------------------------------------------------------------------------
// SynapseGroupInternal
//------------------------------------------------------------------------
SynapseGroupInternal::SynapseGroupInternal(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser, 
                                           VarLocation defaultVarLocation, VarLocation defaultSparseConnectivityLocation)
:   SynapseGroup(name, matrixType, delaySteps, wu, wuParams, wuVarInitialisers, wuPreVarInitialisers, wuPostVarInitialisers, 
                 ps, psParams, psVarInitialisers, srcNeuronGroup, trgNeuronGroup, 
                 connectivityInitialiser, defaultVarLocation, defaultSparseConnectivityLocation),
    m_EventThresholdReTestRequired(false), m_PSModelTargetName(name)
{
    
    // Check that the source neuron group supports the desired number of delay steps
    srcNeuronGroup->checkNumDelaySlots(delaySteps);

    // Add references to target and source neuron groups
    trgNeuronGroup->addInSyn(this);
    srcNeuronGroup->addOutSyn(this);
}

void SynapseGroupInternal::initDerivedParams(double dt)
{
    auto wuDerivedParams = getWUModel()->getDerivedParams();
    auto psDerivedParams = getPSModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_WUDerivedParams.reserve(wuDerivedParams.size());
    m_PSDerivedParams.reserve(psDerivedParams.size());

    // Loop through WU derived parameters
    for(const auto &d : wuDerivedParams) {
        m_WUDerivedParams.push_back(d.second(getWUParams(), dt));
    }

    // Loop through PSM derived parameters
    for(const auto &d : psDerivedParams) {
        m_PSDerivedParams.push_back(d.second(getPSParams(), dt));
    }

    initInitialiserDerivedParams(dt);
}

bool SynapseGroupInternal::isTrueSpikeRequired() const
{
    return !getWUModel()->getSimCode().empty();
}

bool SynapseGroupInternal::isSpikeEventRequired() const
{
     return !getWUModel()->getEventCode().empty();
}

bool SynapseGroupInternal::isDendriticDelayRequired() const
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

bool SynapseGroupInternal::isPSInitRNGRequired() const
{
    // If initialising the postsynaptic variables require an RNG, return true
    return Utils::isInitRNGRequired(getPSVarInitialisers());
}

bool SynapseGroupInternal::isWUInitRNGRequired() const
{
    // If initialising the weight update variables require an RNG, return true
    if(Utils::isInitRNGRequired(getWUVarInitialisers())) {
        return true;
    }

    // Return true if the var init mode we're querying is the one used for sparse connectivity and the connectivity initialiser requires an RNG
    return Utils::isRNGRequired(getConnectivityInitialiser().getSnippet()->getRowBuildCode());
}

bool SynapseGroupInternal::isPSVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of them have initialisation code
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
        return std::any_of(getPSVarInitialisers().cbegin(), getPSVarInitialisers().cend(),
                           [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
    }
    else {
        return false;
    }
}

bool SynapseGroupInternal::isWUVarInitRequired() const
{
    // If this synapse group has per-synapse state variables,
    // return true if any of them have initialisation code
    if (getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
        return std::any_of(getWUVarInitialisers().cbegin(), getWUVarInitialisers().cend(),
                           [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
    }
    else {
        return false;
    }
}

bool SynapseGroupInternal::isWUPreVarInitRequired() const
{
    return std::any_of(getWUPreVarInitialisers().cbegin(), getWUPreVarInitialisers().cend(),
                       [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}

bool SynapseGroupInternal::isWUPostVarInitRequired() const
{
    return std::any_of(getWUPostVarInitialisers().cbegin(), getWUPostVarInitialisers().cend(),
                       [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}

bool SynapseGroupInternal::isSparseConnectivityInitRequired() const
{
    // Return true if there is code to initialise sparse connectivity on device
    return !getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty();
}

bool SynapseGroupInternal::isInitRequired() const
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

bool SynapseGroupInternal::isSparseInitRequired() const
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

std::string SynapseGroupInternal::getPresynapticAxonalDelaySlot(const std::string &devPrefix) const
{
    assert(getSrcNeuronGroup()->isDelayRequired());

    if(getDelaySteps() == 0) {
        return devPrefix + "spkQuePtr" + getSrcNeuronGroup()->getName();
    }
    else {
        return "((" + devPrefix + "spkQuePtr" + getSrcNeuronGroup()->getName() + " + " + std::to_string(getSrcNeuronGroup()->getNumDelaySlots() - getDelaySteps()) + ") % " + std::to_string(getSrcNeuronGroup()->getNumDelaySlots()) + ")";
    }
}

std::string SynapseGroupInternal::getPostsynapticBackPropDelaySlot(const std::string &devPrefix) const
{
    assert(getTrgNeuronGroup()->isDelayRequired());

    if(getBackPropDelaySteps() == 0) {
        return devPrefix + "spkQuePtr" + getTrgNeuronGroup()->getName();
    }
    else {
        return "((" + devPrefix + "spkQuePtr" + getTrgNeuronGroup()->getName() + " + " + std::to_string(getTrgNeuronGroup()->getNumDelaySlots() - getBackPropDelaySteps()) + ") % " + std::to_string(getTrgNeuronGroup()->getNumDelaySlots()) + ")";
    }
}

std::string SynapseGroupInternal::getDendriticDelayOffset(const std::string &devPrefix, const std::string &offset) const
{
    assert(isDendriticDelayRequired());

    if(offset.empty()) {
        return "(" + devPrefix + "denDelayPtr" + getPSModelTargetName() + " * " + std::to_string(getTrgNeuronGroup()->getNumNeurons()) + ") + ";
    }
    else {
        return "(((" + devPrefix + "denDelayPtr" + getPSModelTargetName() + " + " + offset + ") % " + std::to_string(getMaxDendriticDelayTimesteps()) + ") * " + std::to_string(getTrgNeuronGroup()->getNumNeurons()) + ") + ";
    }
}
