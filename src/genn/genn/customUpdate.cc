#include "customUpdate.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "currentSource.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "type.h"

//------------------------------------------------------------------------
// GeNN::CustomUpdateBase
//------------------------------------------------------------------------
namespace GeNN
{
//------------------------------------------------------------------------
void CustomUpdateBase::setVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getCustomUpdateModel()->getVar(varName)) {
        throw std::runtime_error("Unknown custom update model variable '" + varName + "'");
    }
    m_VarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void CustomUpdateBase::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getCustomUpdateModel()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown custom update model extra global parameter '" + paramName + "'");
    }
    m_ExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void CustomUpdateBase::setParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getCustomUpdateModel()->getParam(paramName)) {
        throw std::runtime_error("Unknown custom update model parameter '" + paramName + "'");
    }
    m_DynamicParams.set(paramName, dynamic); 
}
//------------------------------------------------------------------------
bool CustomUpdateBase::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const auto &init){ return !Utils::areTokensEmpty(init.second.getCodeTokens()); });
}
//----------------------------------------------------------------------------
CustomUpdateBase::CustomUpdateBase(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *customUpdateModel, 
                                   const std::unordered_map<std::string, Type::NumericValue> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                                   const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomUpdateModel(customUpdateModel), m_Params(params), 
    m_VarInitialisers(varInitialisers), m_EGPReferences(egpReferences), m_VarLocation(defaultVarLocation),
    m_ExtraGlobalParamLocation(defaultExtraGlobalParamLocation), m_Dims{0}
{
    // Validate names
    Utils::validatePopName(name, "Custom update");
    Utils::validatePopName(updateGroupName, "Custom update group name");

    // Loop through all extra global parameter references
    for (const auto &modelEGPRef : getCustomUpdateModel()->getExtraGlobalParamRefs()) {
        const auto egpRef = egpReferences.at(modelEGPRef.name);

        // Check types of extra global parameter references against those specified in model
        // **THINK** this is rather conservative but I think not allowing "scalar" and whatever happens to be scalar type is ok
        if (egpRef.getEGP().type != modelEGPRef.type) {
            throw std::runtime_error("Incompatible type for extra global parameter reference '" + modelEGPRef.name + "'");
        }
    }
    // Scan custom update model code string
    m_UpdateCodeTokens = Utils::scanCode(getCustomUpdateModel()->getUpdateCode(), 
                                         "Custom update '" + getName() + "' update code");
}
//----------------------------------------------------------------------------
void CustomUpdateBase::finalise(double dt)
{
    auto derivedParams = getCustomUpdateModel()->getDerivedParams();

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.emplace(d.name, d.func(getParams(), dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isInitRNGRequired() const
{
    // Return whether initialising variables require an RNG
    return Utils::isRNGRequired(getVarInitialisers());
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isZeroCopyEnabled() const
{
    return (m_VarLocation.anyZeroCopy() || m_ExtraGlobalParamLocation.anyZeroCopy());
}
//----------------------------------------------------------------------------
bool CustomUpdateBase::isModelReduction() const
{
    // Return true if any variables have REDUCE flag in their access mode
    const auto vars = getCustomUpdateModel()->getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                    [](const auto &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        return true;
    }

    // Return true if any variable references have REDUCE flag in their access mode
    const auto varRefs = getCustomUpdateModel()->getVarRefs();
    if(std::any_of(varRefs.cbegin(), varRefs.cend(),
                    [](const auto &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        return true;
    }

    return false;
}
//----------------------------------------------------------------------------
void CustomUpdateBase::updateHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getCustomUpdateModel()->getHashDigest(), hash);
    Utils::updateHash(getUpdateGroupName(), hash);
    Utils::updateHash(getDims(), hash);
    m_DynamicParams.updateHash(hash);
}
//----------------------------------------------------------------------------
void CustomUpdateBase::updateInitHash(boost::uuids::detail::sha1 &hash) const
{
    Utils::updateHash(getCustomUpdateModel()->getVars(), hash);
    Utils::updateHash(getDims(), hash);

    // Include variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateBase::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    m_VarLocation.updateHash(hash);
    m_ExtraGlobalParamLocation.updateHash(hash);
    return hash.get_digest();
}

//----------------------------------------------------------------------------
// CustomUpdate
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName,
                           const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, Type::NumericValue> &params,
                           const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::VarReference> &varReferences,
                           const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, egpReferences, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_NumNeurons(varReferences.empty() ? 0 : varReferences.begin()->second.getNumNeurons()), m_DelayNeuronGroup(nullptr)
{
    // Validate parameters, variables and variable references
    getCustomUpdateModel()->validate(getParams(), getVarInitialisers(), getVarReferences(), "Custom update " + getName());

    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    Models::checkVarReferenceTypes(m_VarReferences, getCustomUpdateModel()->getVarRefs());

    // Check only one type of reduction is specified
    const bool batchReduction = isBatchReduction();
    const bool neuronReduction = isNeuronReduction();
    if (batchReduction && neuronReduction) {
        throw std::runtime_error("Custom updates cannot perform batch and neuron reductions simultaneously.");
    }
    // Otherwise, if model specifies reduction operations but none are correctly configured
    else if(isModelReduction() && !batchReduction && !neuronReduction) {
        throw std::runtime_error("Custom updates uses reduction model but shape is incorrect.");
    }

    // Give error if any sizes differ
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                   [this](const auto &v) { return v.second.getNumNeurons() != m_NumNeurons; }))
    {
        throw std::runtime_error("All referenced variables must have the same size.");
    }
}
//----------------------------------------------------------------------------
void CustomUpdate::finalise(double dt, unsigned int batchSize)
{
    // Superclass
    CustomUpdateBase::finalise(dt);

    // Check variable reference batching
    checkVarReferenceDims(m_VarReferences, batchSize);

    // If any variable references have delays
    auto delayRef = std::find_if(m_VarReferences.cbegin(), m_VarReferences.cend(),
                                 [](const auto &v) { return v.second.getDelayNeuronGroup() != nullptr; });
    if(delayRef != m_VarReferences.cend()) {
        // Set the delay neuron group 
        m_DelayNeuronGroup = delayRef->second.getDelayNeuronGroup();

        // If any of the variable references are delayed with a different group, give an error
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [this](const auto &v) { return (v.second.getDelayNeuronGroup() != nullptr) && (v.second.getDelayNeuronGroup() != m_DelayNeuronGroup); }))
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

    // Loop through variable references
    for(const auto &v : getVarReferences()) {
        // Update hash with whether variable references require delay
        Utils::updateHash((v.second.getDelayNeuronGroup() == nullptr), hash);

        // Update hash with target variable dimensions as this effects indexing code
        Utils::updateHash(v.second.getVarDims(), hash);
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
// GeNN::CustomUpdateWU
//----------------------------------------------------------------------------
CustomUpdateWU::CustomUpdateWU(const std::string &name, const std::string &updateGroupName,
                               const CustomUpdateModels::Base *customUpdateModel, const std::unordered_map<std::string, Type::NumericValue> &params,
                               const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, const std::unordered_map<std::string, Models::WUVarReference> &varReferences,
                               const std::unordered_map<std::string, Models::EGPReference> &egpReferences, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   CustomUpdateBase(name, updateGroupName, customUpdateModel, params, varInitialisers, egpReferences, defaultVarLocation, defaultExtraGlobalParamLocation),
    m_VarReferences(varReferences), m_SynapseGroup(m_VarReferences.empty() ? nullptr : static_cast<SynapseGroupInternal*>(m_VarReferences.begin()->second.getSynapseGroup()))
{
    // Validate parameters, variables and variable references
    getCustomUpdateModel()->validate(getParams(), getVarInitialisers(), getVarReferences(), "Custom update " + getName());

    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

    // Check variable reference types
    Models::checkVarReferenceTypes(m_VarReferences, getCustomUpdateModel()->getVarRefs());

    // If model specifies reduction operations but none are correctly configured
    if(isModelReduction() && !isBatchReduction()) {
        throw std::runtime_error("Custom updates uses reduction model but shape is incorrect.");
    }

    // Give error if custom update model includes any shared neuron variables
    // **NOTE** because there's no way to reference neuron variables with WUVarReferences, 
    // this safely checks for attempts to do neuron reductions
    const auto vars = getCustomUpdateModel()->getVars();
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const auto &v)
                    {
                        return (v.access == CustomUpdateVarAccess::READ_ONLY_SHARED_NEURON);
                    }))
    {
        throw std::runtime_error("Custom weight updates cannot use models with SHARED_NEURON variables.");
    }

    // Give error if references point to different synapse groups
    // **NOTE** this could be relaxed for dense
    if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                    [this](const auto &v) 
                    { 
                        return (v.second.getSynapseGroup() != m_SynapseGroup); 
                    }))
    {
        throw std::runtime_error("All referenced variables must belong to the same synapse group.");
    }

    // If this is a transpose operation
    if(isTransposeOperation()) {
        // Check that it isn't also a reduction
        if(isBatchReduction()) {
            throw std::runtime_error("Custom weight updates cannot perform both transpose and batch reduction operations.");
        }

        // Give error if any of the variable references aren't dense
        // **NOTE** there's no reason NOT to implement sparse transpose
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const auto &v) 
                       {
                           return !(v.second.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE); 
                       }))
        {
            throw std::runtime_error("Custom weight updates that perform a transpose operation can currently only be used on DENSE synaptic matrices.");
        }

        // If there's more than one variable with a transpose give error
        // **NOTE** there's no reason NOT to allow multiple transposes, it just gets a little tricky with shared memory allocations
        if(std::count_if(m_VarReferences.cbegin(), m_VarReferences.cend(),
                        [](const auto &v) { return v.second.getTransposeSynapseGroup() != nullptr; }) > 1)
        {
            throw std::runtime_error("Each custom weight update can only calculate the tranpose of a single variable,");
        }
    }
}
//----------------------------------------------------------------------------
void CustomUpdateWU::finalise(double dt, unsigned int batchSize)
{
    // Superclass
    CustomUpdateBase::finalise(dt);

    // Check variable reference types
    checkVarReferenceDims(m_VarReferences, batchSize);
}
//----------------------------------------------------------------------------
bool CustomUpdateWU::isTransposeOperation() const
{
    // Transpose opetation is required if any variable references have a transpose
    return std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const auto &v) { return (v.second.getTransposeSynapseGroup() != nullptr); });
}
//----------------------------------------------------------------------------
const std::vector<unsigned int> &CustomUpdateWU::getKernelSize() const 
{ 
    return getSynapseGroup()->getKernelSize(); 
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWU::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    CustomUpdateBase::updateHash(hash);

    Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
    Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);

    // Loop through variable references
    for(const auto &v : getVarReferences()) {
        // Update hash with whether variable references require transpose
        Utils::updateHash((v.second.getTransposeSynapseGroup() == nullptr), hash);

        // Update hash with dimensionality of target variable dimensions as this effects indexing code
        Utils::updateHash(v.second.getVarDims(), hash);
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
    Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    return hash.get_digest();
}
}   // namespace GeNN
