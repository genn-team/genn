#include "customConnectivityUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//------------------------------------------------------------------------
// CustomConnectivityUpdate
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPreVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getPreVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPostVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomConnectivityUpdateModel()->getPostVarIndex(varName)] = loc;
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getVarIndex(varName)];
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getPreVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getPreVarIndex(varName)];
}
//------------------------------------------------------------------------
VarLocation CustomConnectivityUpdate::getPostVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomConnectivityUpdateModel()->getPostVarIndex(varName)];
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPreVarInitRequired() const
{
    return std::any_of(m_PreVarInitialisers.cbegin(), m_PreVarInitialisers.cend(),
                       [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPostVarInitRequired() const
{
    return std::any_of(m_PostVarInitialisers.cbegin(), m_PostVarInitialisers.cend(),
                       [](const Models::VarInit &init){ return !init.getSnippet()->getCode().empty(); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isRowSimRNGRequired() const
{
    return Utils::isRNGRequired(getCustomConnectivityUpdateModel()->getRowUpdateCode());
}
//------------------------------------------------------------------------
CustomConnectivityUpdate::CustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, SynapseGroupInternal *synapseGroup,
                                                   const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel,
                                                   const std::vector<double> &params, const std::vector<Models::VarInit> &varInitialisers,
                                                   const std::vector<Models::VarInit> &preVarInitialisers, const std::vector<Models::VarInit> &postVarInitialisers,
                                                   const std::vector<Models::WUVarReference> &varReferences, const std::vector<Models::VarReference> &preVarReferences,
                                                   const std::vector<Models::VarReference> &postVarReferences, VarLocation defaultVarLocation,
                                                   VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_SynapseGroup(synapseGroup), m_CustomConnectivityUpdateModel(customConnectivityUpdateModel),
    m_Params(params), m_VarInitialisers(varInitialisers), m_PreVarInitialisers(preVarInitialisers), m_PostVarInitialisers(postVarInitialisers),
    m_VarLocation(varInitialisers.size(), defaultVarLocation), m_PreVarLocation(preVarInitialisers.size()), m_PostVarLocation(postVarInitialisers.size()),
    m_ExtraGlobalParamLocation(customConnectivityUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_PreVarReferences(preVarReferences), m_PostVarReferences(postVarReferences)
{
    // If connectivity is bitmask, give error if any variables are referenced
    if (getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        if (!getVarInitialisers().empty() || !getVarReferences().empty()) {
            throw std::runtime_error("Custom connectivity updates attached to synapse groups with BITMASK connectivity cannot have any per-synapse variables or variable references");
        }
    }
    // Otherwise, give error if synapse group has unsupported connectivity type
    else if (!(getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE)) {
        throw std::runtime_error("Custom connectivity updates can only be attached to synapse groups with BITMASK or SPARSE connectivity.");
    }

    // If synapse group doesn't have individual weights, give error if there are any per-synapse variable references
    if (!(getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && !getVarReferences().empty()) {
        throw std::runtime_error("Custom connectivity updates attached to synapse groups without INDIVIDUAL connectivity cannot have any per-synapse variable references");
    }

    // Check variable reference types
    Models::checkVarReferences(m_VarReferences, getCustomConnectivityUpdateModel()->getVarRefs());
    Models::checkVarReferences(m_PreVarReferences, getCustomConnectivityUpdateModel()->getPreVarRefs());
    Models::checkVarReferences(m_PostVarReferences, getCustomConnectivityUpdateModel()->getPostVarRefs());

    // Give error if any WU var references aren't pointing to synapse group
    if (std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const Models::WUVarReference &v) { return v.getSynapseGroup() != m_SynapseGroup; }))
    {
        throw std::runtime_error("All referenced synaptic variables must refer to same synapse group.");
    }

    // Give error if any presynaptic variable reference sizes differ from source neuron group
    if(std::any_of(m_PreVarReferences.cbegin(), m_PreVarReferences.cend(),
                   [this](const Models::VarReference &v) { return v.getSize() != m_SynapseGroup->getSrcNeuronGroup()->getNumNeurons(); }))
    {
        throw std::runtime_error("All referenced presynaptic variables must have the same size as presynaptic population.");
    }

    // Give error if any postsynaptic variable reference sizes differ from target neuron group
    if(std::any_of(m_PostVarReferences.cbegin(), m_PostVarReferences.cend(),
                   [this](const Models::VarReference &v) { return v.getSize() != m_SynapseGroup->getTrgNeuronGroup()->getNumNeurons(); }))
    {
        throw std::runtime_error("All referenced postsynaptic variables must have the same size as postsynaptic population.");
    }

    // Validate names
    Utils::validatePopName(name, "Custom connectivity update");
    Utils::validatePopName(updateGroupName, "Custom connectivity update group name");
    getCustomConnectivityUpdateModel()->validate();
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::initDerivedParams(double dt)
{
    auto derivedParams = getCustomConnectivityUpdateModel()->getDerivedParams();

    // Reserve vector to hold derived parameters
    m_DerivedParams.reserve(derivedParams.size());

    // Loop through derived parameters
    for (const auto &d : derivedParams) {
        m_DerivedParams.push_back(d.func(getParams(), dt));
    }

    // Initialise derived parameters for synaptic variable initialisers
    for (auto &v : m_VarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for presynaptic variable initialisers
    for (auto &v : m_PreVarInitialisers) {
        v.initDerivedParams(dt);
    }

    // Initialise derived parameters for postsynaptic variable initialisers
    for (auto &v : m_PostVarInitialisers) {
        v.initDerivedParams(dt);
    }
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::finalize(unsigned int batchSize)
{
    // If model is batched we need to check all variable references 
    // are SHARED as, connectivity itself is always SHARED
    if (batchSize > 1) {
        // If any referenced synaptic variables aren't shared, give error
        if (std::any_of(getVarReferences().cbegin(), getVarReferences().cend(),
                        [](const Models::WUVarReference &v) { return (getVarAccessDuplication(v.getVar().access) != VarAccessDuplication::SHARED); }))
        {
            throw std::runtime_error("Synaptic variables referenced by CustomConnectivityUpdate must be SHARED across batches");
        }

        // If any referenced presynaptic variables aren't shared, give error
        if (std::any_of(getPreVarReferences().cbegin(), getPreVarReferences().cend(),
                        [](const Models::VarReference &v) { return (getVarAccessDuplication(v.getVar().access) != VarAccessDuplication::SHARED); }))
        {
            throw std::runtime_error("Presynaptic variables referenced by CustomConnectivityUpdate must be SHARED across batches");
        }

        // If any referenced presynaptic variables aren't shared, give error
        if (std::any_of(getPostVarReferences().cbegin(), getPostVarReferences().cend(),
                        [](const Models::VarReference &v) { return (getVarAccessDuplication(v.getVar().access) != VarAccessDuplication::SHARED); }))
        {
            throw std::runtime_error("Postsynaptic variables referenced by CustomConnectivityUpdate must be SHARED across batches");
        }
    }
    // If any variable references have delays
    /*auto delayRef = std::find_if(m_VarReferences.cbegin(), m_VarReferences.cend(),
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
    }*/
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPreVarInitRNGRequired() const
{
    return Utils::isRNGRequired(getPreVarInitialisers());
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPostVarInitRNGRequired() const
{
    return Utils::isRNGRequired(getPostVarInitialisers());
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isVarInitRNGRequired() const
{
    return Utils::isRNGRequired(getVarInitialisers());
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isZeroCopyEnabled() const
{
    // If there are any synaptic variables implemented in zero-copy mode return true
    if (std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                    [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any presynaptic variables implemented in zero-copy mode return true
    if (std::any_of(m_PreVarLocation.begin(), m_PreVarLocation.end(),
                    [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    // If there are any presynaptic variables implemented in zero-copy mode return true
    if (std::any_of(m_PostVarLocation.begin(), m_PostVarLocation.end(),
                    [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); }))
    {
        return true;
    }

    return false;
}
//------------------------------------------------------------------------
std::vector<Models::WUVarReference> CustomConnectivityUpdate::getDependentVariables() const
{
    // 1) Get synapse group var references
    // Add hashes of weight update model variable types and duplication modes to vector
    std::vector<Models::WUVarReference> dependentVars;
    const auto &vars = getSynapseGroup()->getWUModel()->getVars();
    std::transform(vars.cbegin(), vars.cend(), std::back_inserter(dependentVars),
                   [this](const Models::Base::Var &v)
                   { 
                       return Models::WUVarReference(getSynapseGroup(), v.name);
                   });
    
    // **TODO** skip variables already referenced by variable references
    // > Could point to any of these
    // Loop through custom updates which reference this synapse group
    for(auto *c : getSynapseGroup()->getCustomUpdateReferences()) {
        // Add hashes of custom update var types and duplication modes to vector
        const auto &vars = c->getCustomUpdateModel()->getVars();
        std::transform(vars.cbegin(), vars.cend(), std::back_inserter(dependentVars),
                       [c](const Models::Base::Var &v)
                       { 
                           return Models::WUVarReference(c, v.name);
                       });
    }
    
    // Loop through custom connectivity updates which reference this synapse group
    for(auto *c : getSynapseGroup()->getCustomConnectivityUpdateReferences()) {
        // Skip this custom connectivity update group
        if(c == this) {
            continue;
        }
        
        // Add hashes of custom connectivity update var types and duplication modes to vector
        const auto &vars = c->getCustomConnectivityUpdateModel()->getVars();
        std::transform(vars.cbegin(), vars.cend(), std::back_inserter(dependentVars),
                       [c](const Models::Base::Var &v)
                       { 
                           return Models::WUVarReference(c, v.name);
                       });
    }

    return dependentVars;
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCustomConnectivityUpdateModel()->getHashDigest(), hash);
    Utils::updateHash(getUpdateGroupName(), hash);

    // Because it adds and removes synapses, connectivity update has to update 
    // ALL variables associated with synapse group being modified as well as 
    // with custom WU updates and this and other custom connectivity updates.
    // Therefore, for custom connectivity updates to be merged, 
    // the unordered types and number of these variables should match
    const auto dependentVars = getDependentVariables();

    // Build vector of hashes of variable types and duplication modes
    std::vector<boost::uuids::detail::sha1::digest_type> varTypeDigests;
    std::transform(dependentVars.cbegin(), dependentVars.cend(), std::back_inserter(varTypeDigests),
                   [](const Models::WUVarReference &v)
                   {
                       boost::uuids::detail::sha1 hash;  
                       Utils::updateHash(v.getVar().type, hash);
                       Utils::updateHash(getVarAccessDuplication(v.getVar().access), hash);
                       return hash.get_digest();
                   });
    
    // Sort digests
    std::sort(varTypeDigests.begin(), varTypeDigests.end());

    // Concatenate the digests to the hash
    Utils::updateHash(varTypeDigests, hash);
    
    
    /*getCustomConnectivityUpdateReferences;
    using SynapseGroup::getCustomUpdateReferences*/
    /*// Update hash with whether delay is required
    const bool delayed = (getDelayNeuronGroup() != nullptr);
    Utils::updateHash(delayed, hash);

    // If it is, also update hash with number of delay slots
    if(delayed) {
        Utils::updateHash(getDelayNeuronGroup()->getNumDelaySlots(), hash);
    }

    // Update hash with whether variable references require delay
    for(const auto &v : getVarReferences()) {
        Utils::updateHash((v.getDelayNeuronGroup() == nullptr), hash);
    }*/

    return hash.get_digest();
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCustomConnectivityUpdateModel()->getVars(), hash);
    Utils::updateHash(getCustomConnectivityUpdateModel()->getPreVars(), hash);
    Utils::updateHash(getCustomConnectivityUpdateModel()->getPostVars(), hash);

    // Include synaptic variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.getHashDigest(), hash);
    }

    // Include presynaptic variable initialiser hashes
    for(const auto &w : getPreVarInitialisers()) {
        Utils::updateHash(w.getHashDigest(), hash);
    }

    // Include postsynaptic variable initialiser hashes
    for(const auto &w : getPostVarInitialisers()) {
        Utils::updateHash(w.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_PreVarLocation, hash);
    Utils::updateHash(m_PostVarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
}
