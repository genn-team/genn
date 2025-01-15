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
#include "type.h"

using namespace GeNN;

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
void updateVarRefDelayHash(const NeuronGroup *delayGroup, const std::map<std::string, Models::VarReference> &varRefs, 
                           boost::uuids::detail::sha1 &hash)
{
    // Update hash with whether delay is required
    const bool delayed = (delayGroup != nullptr);
    Utils::updateHash(delayed, hash);

    // If it is, also update hash with number of delay slots
    if(delayed) {
        Utils::updateHash(delayGroup->getNumDelaySlots(), hash);
    }

    // Update hash with whether presynaptic variable references require delay
    for(const auto &v : varRefs) {
        Utils::updateHash((v.second.getDelayNeuronGroup() == nullptr), hash);
    }
}
}   // Anonymous namespace

//------------------------------------------------------------------------
// CustomConnectivityUpdate
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getModel()->getVar(varName)) {
        throw std::runtime_error("Unknown custom connectivity update model variable '" + varName + "'");
    }
    m_VarLocation.set(varName, loc); 
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPreVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getModel()->getPreVar(varName)) {
        throw std::runtime_error("Unknown custom connectivity update model presynaptic variable '" + varName + "'");
    }
    m_PreVarLocation.set(varName, loc); 
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::setPostVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getModel()->getPostVar(varName)) {
        throw std::runtime_error("Unknown custom connectivity update model postsynaptic variable '" + varName + "'");
    }
    m_PostVarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdate::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getModel()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown custom connectivity update model extra global parameter '" + paramName + "'");
    }
    m_ExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdate::setParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getModel()->getParam(paramName)) {
        throw std::runtime_error("Unknown custom connectivity update model parameter '" + paramName + "'");
    }
    m_DynamicParams.set(paramName, dynamic); 
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const auto &v){ return !Utils::areTokensEmpty(v.second.getCodeTokens()); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPreVarInitRequired() const
{
    return std::any_of(m_PreVarInitialisers.cbegin(), m_PreVarInitialisers.cend(),
                       [](const auto &v){ return !Utils::areTokensEmpty(v.second.getCodeTokens()); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isPostVarInitRequired() const
{
    return std::any_of(m_PostVarInitialisers.cbegin(), m_PostVarInitialisers.cend(),
                       [](const auto &v){ return !Utils::areTokensEmpty(v.second.getCodeTokens()); });
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::canModifyConnectivity() const
{
    return (Utils::isIdentifierReferenced("remove_synapse", getRowUpdateCodeTokens())
            || Utils::isIdentifierReferenced("add_synapse", getRowUpdateCodeTokens()));
}
//------------------------------------------------------------------------
CustomConnectivityUpdate::CustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, SynapseGroupInternal *synapseGroup,
                                                   const CustomConnectivityUpdateModels::Base *customConnectivityUpdateModel,
                                                   const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                                                   const std::map<std::string, InitVarSnippet::Init> &preVarInitialisers, const std::map<std::string, InitVarSnippet::Init> &postVarInitialisers,
                                                   const std::map<std::string, Models::WUVarReference> &varReferences, const std::map<std::string, Models::VarReference> &preVarReferences,
                                                   const std::map<std::string, Models::VarReference> &postVarReferences, const std::map<std::string, Models::EGPReference> &egpReferences,
                                                   VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_SynapseGroup(synapseGroup), m_Model(customConnectivityUpdateModel),
    m_Params(params), m_VarInitialisers(varInitialisers), m_PreVarInitialisers(preVarInitialisers), m_PostVarInitialisers(postVarInitialisers),
    m_VarLocation(defaultVarLocation), m_PreVarLocation(defaultVarLocation), m_PostVarLocation(defaultVarLocation), 
    m_ExtraGlobalParamLocation(defaultExtraGlobalParamLocation), m_VarReferences(varReferences), m_PreVarReferences(preVarReferences), 
    m_PostVarReferences(postVarReferences), m_EGPReferences(egpReferences), m_PreDelayNeuronGroup(nullptr), m_PostDelayNeuronGroup(nullptr)
{
    
    // Validate names
    Utils::validatePopName(name, "Custom connectivity update");
    Utils::validatePopName(updateGroupName, "Custom connectivity update group name");
    getModel()->validate(getParams(), getVarInitialisers(), getPreVarInitialisers(),
                         getPostVarInitialisers(), getVarReferences(), getPreVarReferences(),
                         getPostVarReferences(), getEGPReferences(), "Custom connectivity update " + getName());

    // Scan custom connectivity update model code strings
    m_RowUpdateCodeTokens = Utils::scanCode(getModel()->getRowUpdateCode(), 
                                            "Custom connectivity update '" + getName() + "' row update code");
    m_HostUpdateCodeTokens = Utils::scanCode(getModel()->getHostUpdateCode(), 
                                             "Custom connectivity update '" + getName() + "' host update code");

    // Give error if synapse group has unsupported connectivity type
    if (!(getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE)) {
        throw std::runtime_error("Custom connectivity updates can only be attached to synapse groups with SPARSE connectivity.");
    }

    // Check variable reference types
    Models::checkVarReferenceTypes(m_VarReferences, getModel()->getVarRefs());
    Models::checkVarReferenceTypes(m_PreVarReferences, getModel()->getPreVarRefs());
    Models::checkVarReferenceTypes(m_PostVarReferences, getModel()->getPostVarRefs());

    // Check EGP reference types
    Models::checkEGPReferenceTypes(m_EGPReferences, getModel()->getExtraGlobalParamRefs());

    // Give error if any WU var references aren't pointing to synapse group
    if (std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const auto &v) { return v.second.getSynapseGroup() != m_SynapseGroup; }))
    {
        throw std::runtime_error("All referenced synaptic variables must refer to same synapse group.");
    }

    // Give error if any presynaptic variable reference sizes differ from source neuron group
    if(std::any_of(m_PreVarReferences.cbegin(), m_PreVarReferences.cend(),
                   [this](const auto&v) { return v.second.getNumNeurons() != m_SynapseGroup->getSrcNeuronGroup()->getNumNeurons(); }))
    {
        throw std::runtime_error("All referenced presynaptic variables must have the same size as presynaptic population.");
    }

    // Give error if any postsynaptic variable reference sizes differ from target neuron group
    if(std::any_of(m_PostVarReferences.cbegin(), m_PostVarReferences.cend(),
                   [this](const auto &v) { return v.second.getNumNeurons() != m_SynapseGroup->getTrgNeuronGroup()->getNumNeurons(); }))
    {
        throw std::runtime_error("All referenced postsynaptic variables must have the same size as postsynaptic population.");
    }
}
//------------------------------------------------------------------------
void CustomConnectivityUpdate::finalise(double dt, unsigned int batchSize)
{
    // Loop through derived parameters
    auto derivedParams = getModel()->getDerivedParams();
    for(const auto &d : derivedParams) {
        m_DerivedParams.emplace(d.name, d.func(getParams(), dt));
    }

    // Finalise derived parameters for synaptic variable initialisers
    for (auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }

    // Finalise derived parameters for presynaptic variable initialisers
    for (auto &v : m_PreVarInitialisers) {
        v.second.finalise(dt);
    }

    // Finalise derived parameters for postsynaptic variable initialisers
    for (auto &v : m_PostVarInitialisers) {
        v.second.finalise(dt);
    }

    // If model is batched we need to check all variable references 
    // are SHARED as, connectivity itself is always SHARED
    if (batchSize > 1) {
        // If any referenced presynaptic variables are batched, give error
        if (std::any_of(getPreVarReferences().cbegin(), getPreVarReferences().cend(),
                        [](const auto &v) 
                        { 
                            return (v.second.getVarDims() & VarAccessDim::BATCH); 
                        }))
        {
            throw std::runtime_error("Presynaptic variables referenced by CustomConnectivityUpdate must be SHARED across batches");
        }

        // If any referenced presynaptic variables aren't shared, give error
        if (std::any_of(getPostVarReferences().cbegin(), getPostVarReferences().cend(),
                        [](const auto &v) 
                        { 
                            return (v.second.getVarDims() & VarAccessDim::BATCH); 
                        }))
        {
            throw std::runtime_error("Postsynaptic variables referenced by CustomConnectivityUpdate must be SHARED across batches");
        }
    }
    
    // Get neuron groups to use for pre and postsynaptic variable reference delays
    m_PreDelayNeuronGroup = getVarRefDelayGroup(getPreVarReferences(), "presynaptic");
    m_PostDelayNeuronGroup = getVarRefDelayGroup(getPostVarReferences(), "postsynaptic");
}
//------------------------------------------------------------------------
bool CustomConnectivityUpdate::isZeroCopyEnabled() const
{
    // If there are any variables or EGPs implemented in zero-copy mode return true
    return (m_VarLocation.anyZeroCopy() || m_PreVarLocation.anyZeroCopy() 
            || m_PostVarLocation.anyZeroCopy() || m_ExtraGlobalParamLocation.anyZeroCopy());
}
//------------------------------------------------------------------------
std::vector<Models::WUVarReference> CustomConnectivityUpdate::getDependentVariables() const
{
    // Build set of 'manual' variable references
    // If variables are already referenced by this mechanism they shouldn't be included in dependent variables
    std::set<Models::WUVarReference> manualReferences;
    std::transform(getVarReferences().cbegin(), getVarReferences().cend(), std::inserter(manualReferences, manualReferences.end()),
                   [](const auto &r){ return r.second; });

    // If our synapse group has individual or kernel weights
    std::vector<Models::WUVarReference> dependentVars;
    if ((getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)
        || (getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL))
    {
        // Loop through synapse group variables
        for (const auto &v : getSynapseGroup()->getWUInitialiser().getSnippet()->getVars()) {
            // Create reference to variable
            auto ref = Models::WUVarReference::createWUVarReference(getSynapseGroup(), v.name);

            // Add to dependent variables if it isn't already 'manually' referenced
            if (manualReferences.find(ref) == manualReferences.cend()) {
                dependentVars.emplace_back(ref);
            }
        }
    }
    
    // > Could point to any of these
    // Loop through custom updates which reference this synapse group
    for(auto *c : getSynapseGroup()->getCustomUpdateReferences()) {
        // Loop through custom update variables
        for (const auto &v : c->getModel()->getVars()) {
            // Create reference to variable
            auto ref = Models::WUVarReference::createWUVarReference(c, v.name);

            // Add to dependent variables if it isn't already 'manually' referenced
            if (manualReferences.find(ref) == manualReferences.cend()) {
                dependentVars.emplace_back(ref);
            }
        }
    }
    
    // Loop through custom connectivity updates which reference this synapse group
    for(auto *c : getSynapseGroup()->getCustomConnectivityUpdateReferences()) {
        // Skip this custom connectivity update group
        if(c == this) {
            continue;
        }
        
        // Loop through custom connectivity update variables
        for (const auto &v : c->getModel()->getVars()) {
            // Create reference to variable
            auto ref = Models::WUVarReference::createWUVarReference(c, v.name);

            // Add to dependent variables if it isn't already 'manually' referenced
            if (manualReferences.find(ref) == manualReferences.cend()) {
                dependentVars.emplace_back(ref);
            }
        }
    }

    return dependentVars;
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getModel()->getHashDigest(), hash);
    Utils::updateHash(getUpdateGroupName(), hash);

    Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
    Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    m_DynamicParams.updateHash(hash);

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
                       Type::updateHash(v.getVarStorageType(), hash);
                       Utils::updateHash(v.getVarDims(), hash);
                       return hash.get_digest();
                   });
    
    // Sort digests
    std::sort(varTypeDigests.begin(), varTypeDigests.end());

    // Concatenate the digests to the hash
    Utils::updateHash(varTypeDigests, hash);
    
    // Update hash with delay information for pre and postsynaptic variable references
    updateVarRefDelayHash(getPreDelayNeuronGroup(), getPreVarReferences(), hash);
    updateVarRefDelayHash(getPostDelayNeuronGroup(), getPostVarReferences(), hash);

    // Update hash with duplication mode of synaptic variable references
    for(const auto &v : getVarReferences()) {
        Utils::updateHash(v.second.getVarDims(), hash);
    }

    return hash.get_digest();
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getRemapHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    Utils::updateHash(getUpdateGroupName(), hash);

    Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
    Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    

    return hash.get_digest();
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getModel()->getVars(), hash);
    Utils::updateHash(getModel()->getPreVars(), hash);
    Utils::updateHash(getModel()->getPostVars(), hash);

    // Include synaptic variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.second.getHashDigest(), hash);
    }

    // Include presynaptic variable initialiser hashes
    for(const auto &w : getPreVarInitialisers()) {
        Utils::updateHash(w.second.getHashDigest(), hash);
    }

    // Include postsynaptic variable initialiser hashes
    for(const auto &w : getPostVarInitialisers()) {
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdate::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    m_VarLocation.updateHash(hash);
    m_PreVarLocation.updateHash(hash);
    m_PostVarLocation.updateHash(hash);
    m_ExtraGlobalParamLocation.updateHash(hash);
    return hash.get_digest();
}
//------------------------------------------------------------------------
NeuronGroup *CustomConnectivityUpdate::getVarRefDelayGroup(const std::map<std::string, Models::VarReference> &varRefs, 
                                                           const std::string &errorContext) const
{
    // If any variable references have delays
    // **YUCK** copy and paste from CustomUpdate::finalise
    auto delayRef = std::find_if(varRefs.cbegin(), varRefs.cend(),
                                 [](const auto &v) { return v.second.getDelayNeuronGroup() != nullptr; });
    if(delayRef != varRefs.cend()) {
        // If any of the variable references are delayed with a different group, give an error
        if(std::any_of(varRefs.cbegin(), varRefs.cend(),
                       [delayRef](const auto &v) { return (v.second.getDelayNeuronGroup() != nullptr) && (v.second.getDelayNeuronGroup() != delayRef->second.getDelayNeuronGroup()); }))
        {
            throw std::runtime_error("Referenced " + errorContext + " variables with delays in custom connectivity update '" + getName() + "' must all refer to same neuron group.");
        }
        
        // Return the delay neuron group 
        return delayRef->second.getDelayNeuronGroup();
    }
    else {
        return nullptr;
    }
}
