#include "customUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomUpdateBase
//------------------------------------------------------------------------
void CustomUpdateBase::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateBase::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                        [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}
//----------------------------------------------------------------------------
void CustomUpdateBase::initDerivedParams(double dt)
{
    auto derivedParams = getCustomUpdateModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(getParams(), dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isInitRNGRequired() const
{
    // If initialising the neuron variables require an RNG, return true
    if(Utils::isRNGRequired(getVarInitialisers())) {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canBeMerged(const CustomUpdateBase &other) const
{
    return (getCustomUpdateModel()->canBeMerged(other.getCustomUpdateModel())
            && (getUpdateGroupName() == other.getUpdateGroupName())
            && (isBatched() == other.isBatched()));
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::canInitBeMerged(const CustomUpdateBase &other) const
{
     // If both groups have the same variables
    if((getCustomUpdateModel()->getVars() == other.getCustomUpdateModel()->getVars())
       && (isBatched() == other.isBatched())) {
        // if any of the variable's initialisers can't be merged, return false
        for(size_t i = 0; i < getVarInitialisers().size(); i++) {
            if(!getVarInitialisers()[i].canBeMerged(other.getVarInitialisers()[i])) {
                return false;
            }
        }
        
        return true;
    }
    else {
        return false;
    }
}

//----------------------------------------------------------------------------
// CustomUpdate
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName,
                           const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                           const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::VarReference> &varReferences,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    : CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_Size(varReferences.empty() ? 0 : varReferences.front().getSize()), m_DelayNeuronGroup(nullptr)
{
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    checkVarReferenceTypes(m_VarReferences);

    // Give error if any sizes differ
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                   [this](const Models::VarReference &v) { return v.getSize() != m_Size; }))
    {
        throw std::runtime_error("All referenced variables must have the same size.");
    }
}
//----------------------------------------------------------------------------
void CustomUpdate::finalize(unsigned int batchSize)
{
    // Because batch size might be set at any point and which neuron 
    // variables are queued is only calculated during Modelspec::finalize, 
    // these checks cannot be performed in the constructor

    // If any variable references have delays
    auto delayRef = std::find_if(m_VarReferences.cbegin(), m_VarReferences.cend(),
                                 [](const Models::VarReference &v) { return v.getDelayNeuronGroup() != nullptr; });
    if(delayRef != m_VarReferences.cend()) {
        // Set the delay neuron group 
        m_DelayNeuronGroup = delayRef->getDelayNeuronGroup();

        // If any of the variable references are delayed with a different group, give an error
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [this](const Models::VarReference &v) { return (v.getDelayNeuronGroup() != nullptr) && (v.getDelayNeuronGroup() != m_DelayNeuronGroup); }))
        {
            throw std::runtime_error("Referenced variables with delays in custom update '" + getName() + "' must all refer to same neuron group.");
        }
    }

    // Determine whether custom update is batched
    finalizeBatched(batchSize, m_VarReferences);
}
//----------------------------------------------------------------------------
bool CustomUpdate::canBeMerged(const CustomUpdate &other) const
{
    // If the two groups' models can be merged and they either both have no delay neuron group
    // or both do and delay neuron groups have the same number of delay slots
    const bool delayed = (getDelayNeuronGroup() != nullptr);
    const bool otherDelayed = (other.getDelayNeuronGroup() != nullptr);
    if(CustomUpdateBase::canBeMerged(other)
       && ((!delayed && !otherDelayed) 
           || (delayed && otherDelayed && getDelayNeuronGroup()->getNumDelaySlots() == other.getDelayNeuronGroup()->getNumDelaySlots())))
    {
        // Return whether the variables with delays match
        return std::equal(getVarReferences().cbegin(), getVarReferences().cend(), other.getVarReferences().cbegin(),
                          [](const Models::VarReference &a, const Models::VarReference &b)
                          {
                              return ((a.getDelayNeuronGroup() == nullptr) == (b.getDelayNeuronGroup() == nullptr));
                          });
    }

    return false;
}

//----------------------------------------------------------------------------
// CustomUpdateWU
//----------------------------------------------------------------------------
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                               const CustomUpdateModels::Base *customUpdateModel, const std::vector<double> &params,
                               const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::WUVarReference> &varReferences,
                               VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<const SynapseGroupInternal*>(m_VarReferences.front().getSynapseGroup()))
{
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    checkVarReferenceTypes(m_VarReferences);

    // Give error if references point to different synapse groups
    // **NOTE** this could be relaxed for dense
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const Models::WUVarReference &v) 
                    { 
                        return (v.getSynapseGroup() != m_SynapseGroup); 
                    }))
    {
        throw std::runtime_error("All referenced variables must belong to the same synapse group.");
    }

    // If this is a transpose operation
    if(isTransposeOperation()) {
        // Give error if any of the variable references aren't dense
        // **NOTE** there's no reason NOT to implement sparse transpose
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const Models::WUVarReference &v) 
                       {
                           return !(v.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE); 
                       }))
        {
            throw std::runtime_error("Custom updates that perform a transpose operation can currently only be used on DENSE synaptic matrices.");
        }

        // If there's more than one variable with a transpose give error
        // **NOTE** there's no reason NOT to allow multiple transposes, it just gets a little tricky with shared memory allocations
        if(std::count_if(m_VarReferences.cbegin(), m_VarReferences.cend(),
                        [](const Models::WUVarReference &v) { return v.getTransposeSynapseGroup() != nullptr; }) > 1)
        {
            throw std::runtime_error("Each custom update can only calculate the tranpose of a single variable,");
        }
    }
}
//----------------------------------------------------------------------------
void CustomUpdateWU::finalize(unsigned int batchSize)
{
    finalizeBatched(batchSize, m_VarReferences);
}
//----------------------------------------------------------------------------
bool CustomUpdateWU::isTransposeOperation() const
{
    // Transpose opetation is required if any variable references have a transpose
    return std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const Models::WUVarReference &v) { return (v.getTransposeSynapseGroup() != nullptr); });
}
//----------------------------------------------------------------------------
bool CustomUpdateWU::canBeMerged(const CustomUpdateWU &other) const
{
    // If the two groups' models can be merged and their connectivity is the same
    if(CustomUpdateBase::canBeMerged(other)
       && (getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()) == getSynapseMatrixConnectivity(other.getSynapseGroup()->getMatrixType()))
       && (getSynapseGroup()->getSparseIndType() == other.getSynapseGroup()->getSparseIndType()))
    {
        // Return whether the variables with transposes match
        return std::equal(getVarReferences().cbegin(), getVarReferences().cend(), other.getVarReferences().cbegin(),
                          [](const Models::WUVarReference &a, const Models::WUVarReference &b)
                          {
                              return ((a.getTransposeSynapseGroup() == nullptr) == (b.getTransposeSynapseGroup() == nullptr));
                          });
    }

    return false;
}
//----------------------------------------------------------------------------
bool CustomUpdateWU::canInitBeMerged(const CustomUpdateWU &other) const
{
    return (CustomUpdateBase::canInitBeMerged(other)
            && (getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()) == getSynapseMatrixConnectivity(other.getSynapseGroup()->getMatrixType()))
            && (getSynapseGroup()->getSparseIndType() == other.getSynapseGroup()->getSparseIndType()));
}