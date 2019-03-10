#include "neuronGroup.h"

// ------------------------------------------------------------------------
// NeuronGroup
// ------------------------------------------------------------------------
void NeuronGroup::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getNeuronModel()->getVarIndex(varName)] = loc;
}

VarLocation NeuronGroup::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getNeuronModel()->getVarIndex(varName)];
}


bool NeuronGroup::isZeroCopyEnabled() const
{
    // If any bits of spikes require zero-copy return true
    if((m_SpikeLocation & VarLocation::ZERO_COPY) || (m_SpikeEventLocation & VarLocation::ZERO_COPY) || (m_SpikeTimeLocation & VarLocation::ZERO_COPY)) {
        return true;
    }

    // If there are any variables implemented in zero-copy mode return true
    if(std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
        [](VarLocation loc){ return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}

void NeuronGroup::initInitialiserDerivedParams(double dt)
{
    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}