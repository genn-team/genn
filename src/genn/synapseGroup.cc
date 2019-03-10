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
SynapseGroup::SynapseGroup(const std::string name, SynapseMatrixType matrixType, unsigned int delaySteps,
                           const WeightUpdateModels::Base *wu, const std::vector<double> &wuParams, const std::vector<Models::VarInit> &wuVarInitialisers, const std::vector<Models::VarInit> &wuPreVarInitialisers, const std::vector<Models::VarInit> &wuPostVarInitialisers,
                           const PostsynapticModels::Base *ps, const std::vector<double> &psParams, const std::vector<Models::VarInit> &psVarInitialisers,
                           NeuronGroupInternal *srcNeuronGroup, NeuronGroupInternal *trgNeuronGroup,
                           const InitSparseConnectivitySnippet::Init &connectivityInitialiser, 
                           VarLocation defaultVarLocation, VarLocation defaultSparseConnectivityLocation)
    :   m_Name(name), m_SpanType(SpanType::POSTSYNAPTIC), m_DelaySteps(delaySteps), m_BackPropDelaySteps(0),
        m_MaxDendriticDelayTimesteps(1), m_MatrixType(matrixType),  m_SrcNeuronGroup(srcNeuronGroup), m_TrgNeuronGroup(trgNeuronGroup),
        m_InSynLocation(defaultVarLocation),  m_DendriticDelayLocation(defaultVarLocation),
        m_WUModel(wu), m_WUParams(wuParams), m_WUVarInitialisers(wuVarInitialisers), m_WUPreVarInitialisers(wuPreVarInitialisers), m_WUPostVarInitialisers(wuPostVarInitialisers),
        m_PSModel(ps), m_PSParams(psParams), m_PSVarInitialisers(psVarInitialisers),
        m_WUVarLocation(wuVarInitialisers.size(), defaultVarLocation), m_WUPreVarLocation(wuPreVarInitialisers.size(), defaultVarLocation),
        m_WUPostVarLocation(wuPostVarInitialisers.size(), defaultVarLocation), m_PSVarLocation(psVarInitialisers.size(), defaultVarLocation),
        m_ConnectivityInitialiser(connectivityInitialiser), m_SparseConnectivityLocation(defaultSparseConnectivityLocation)
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

int SynapseGroup::getClusterHostID() const
{
    return m_TrgNeuronGroup->getClusterHostID();
}

int SynapseGroup::getClusterDeviceID() const
{
    return m_TrgNeuronGroup->getClusterDeviceID();
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

void SynapseGroup::initInitialiserDerivedParams(double dt)
{
    // Initialise derived parameters for WU variable initialisers
    for(auto &v : m_WUVarInitialisers) {
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

    // Initialise derived parameters for PSM variable initialisers
    for(auto &v : m_PSVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise any derived connectivity initialiser parameters
    m_ConnectivityInitialiser.initDerivedParams(dt);
}