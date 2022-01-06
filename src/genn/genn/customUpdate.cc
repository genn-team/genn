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

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.emplace(d.name, d.func(getParams(), dt));
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
void CustomUpdateBase::updateHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getCustomUpdateModel()->getHashDigest(), hash);
    Utils::updateHash(getUpdateGroupName(), hash);
    Utils::updateHash(isBatched(), hash);
}
//----------------------------------------------------------------------------
void CustomUpdateBase::updateInitHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getCustomUpdateModel()->getVars(), hash);
    Utils::updateHash(isBatched(), hash);

    // Include variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.getHashDigest(), hash);
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateBase::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
}

//----------------------------------------------------------------------------
// CustomUpdate
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName,
                           const CustomUpdateModels::Base *customUpdateModel, const Snippet::ParamValues &params,
                           const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::VarReference> &varReferences,
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    : CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_Size(varReferences.empty() ? 0 : varReferences.front().getSize()), m_DelayNeuronGroup(nullptr)
{
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    checkVarReferences(m_VarReferences);

    // Give error if any sizes differ
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                   [this](const Models::VarReference &v) { return v.getSize() != m_Size; }))
    {
        throw std::runtime_error("All referenced variables must have the same size.");
    }
}
//----------------------------------------------------------------------------
void CustomUpdate::finalize()
{
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
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdate::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    CustomUpdateBase::updateHash(hash);

    // Update hash with whether delay is required
    const bool delayed = (getDelayNeuronGroup() != nullptr);
    Utils::updateHash(delayed, hash);

    // If it is, also update hash with number of delay slots
    if(delayed) {
        Utils::updateHash(getDelayNeuronGroup()->getNumDelaySlots(), hash);
    }

    // Update hash with whether variable references require delay
    for(const auto &v : getVarReferences()) {
        Utils::updateHash((v.getDelayNeuronGroup() == nullptr), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdate::getInitHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    CustomUpdateBase::updateInitHash(hash);
    return hash.get_digest();
}

//----------------------------------------------------------------------------
// CustomUpdateWU
//----------------------------------------------------------------------------
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                               const CustomUpdateModels::Base *customUpdateModel, const Snippet::ParamValues &params,
                               const std::vector<Models::VarInit> &varInitialisers, const std::vector<Models::WUVarReference> &varReferences,
                               VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<const SynapseGroupInternal*>(m_VarReferences.front().getSynapseGroup()))
{
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    checkVarReferences(m_VarReferences);

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
        // Check that it isn't also a reduction
        if(getCustomUpdateModel()->isReduction()) {
            throw std::runtime_error("Custom updates cannot perform both transpose and reduction operations.");
        }

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
bool CustomUpdateWU::isTransposeOperation() const
{
    // Transpose opetation is required if any variable references have a transpose
    return std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const Models::WUVarReference &v) { return (v.getTransposeSynapseGroup() != nullptr); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWU::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    CustomUpdateBase::updateHash(hash);

    Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
    Utils::updateHash(getSynapseGroup()->getSparseIndType(), hash);

    // Update hash with whether variable references require transpose
    for(const auto &v : getVarReferences()) {
        Utils::updateHash((v.getTransposeSynapseGroup() == nullptr), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWU::getInitHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    CustomUpdateBase::updateInitHash(hash);

    Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
    Utils::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    return hash.get_digest();
}
