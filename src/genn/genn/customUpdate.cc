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
// GeNN::CustomUpdate
//------------------------------------------------------------------------
namespace GeNN
{
void CustomUpdate::setVarLocation(const std::string &varName, VarLocation loc)
{
    m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdate::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCustomUpdateModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
bool CustomUpdate::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const auto &init){ return !Utils::areTokensEmpty(init.second.getCodeTokens()); });
}
//----------------------------------------------------------------------------
CustomUpdate::CustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *customUpdateModel, 
                           const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                           const std::unordered_map<std::string, Models::VarReference> &varReferences, const std::unordered_map<std::string, Models::EGPReference> &egpReferences, 
                           VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_UpdateGroupName(updateGroupName), m_CustomUpdateModel(customUpdateModel), m_Params(params), 
    m_VarInitialisers(varInitialisers), m_VarReferences(varReferences), m_EGPReferences(egpReferences), 
    m_VarLocation(varInitialisers.size(), defaultVarLocation), 
    m_ExtraGlobalParamLocation(customUpdateModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
    m_Dims{0}, m_DelayNeuronGroup(nullptr), m_SynapseGroup(nullptr)
{
     // Validate names
    Utils::validatePopName(name, "Custom update");
    Utils::validatePopName(updateGroupName, "Custom update group name");

    // Validate parameters, variables and variable references
    getCustomUpdateModel()->validate(getParams(), getVarInitialisers(), getVarReferences(), "Custom update " + getName());

    // Check variable reference types
    Models::checkVarReferenceTypes(m_VarReferences, getCustomUpdateModel()->getVarRefs());
   
    if(varReferences.empty()) {
        throw std::runtime_error("Custom update models must reference variables.");
    }

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
void CustomUpdate::finalise(double dt, unsigned int batchSize)
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

    // Loop through variable references and determine dimensionality of update
    // **NOTE** this must be done at finalise-time as custom updates need to be sorted 
    // in dependency order so references between them are correctly resolves
    m_Dims = VarAccessDim{0};
    for(const auto &v : m_VarReferences) {
        const auto varDims = v.second.getVarDims();

        // If variable is batched, set BATCH dimension in custom update dimensions
        if(varDims & VarAccessDim::BATCH) {
            m_Dims = m_Dims | VarAccessDim::BATCH;
        }

        // If variable is per-neuron, set NEURON dimension in custom update dimensions
        if(varDims & VarAccessDim::NEURON) {
            m_Dims = m_Dims | VarAccessDim::NEURON;
        }

        // If variable is per-synapse, set pre and postsynaptic flags in custom update dimensions
        const bool pre = (varDims & VarAccessDim::PRE_NEURON);
        const bool post = (varDims & VarAccessDim::POST_NEURON);
        if(pre && post) {
            m_Dims = m_Dims | VarAccessDim::PRE_NEURON | VarAccessDim::POST_NEURON;
        }
        // Otherwise, if variable is per-neuron, set neuron flag in custom update dimensions
        // **TODO** when we support row and column reductions, this needs to change a little
        else if(pre || post) {
            m_Dims = m_Dims | VarAccessDim::NEURON;
        }
    }

    // If there is some mixture of neuron and synapse flags
    if((m_Dims & VarAccessDim::NEURON) && ((m_Dims & VarAccessDim::PRE_NEURON) || (m_Dims & VarAccessDim::POST_NEURON))) {
        throw std::runtime_error("Custom update references ambiguous mix of neuron and synapse variables.");
    }

    // Loop through all variable references
    for(const auto &modelVarRef : getCustomUpdateModel()->getVarRefs()) {
        const auto varRef = getVarReferences().at(modelVarRef.name);

        // If the batchedness of the references variable doesn't match that
        // of the custom update, check its access mode isn't read-write
        // **TODO** 
        if(((m_Dims & VarAccessDim::BATCH) != (varRef.getVarDims() & VarAccessDim::BATCH))
            && (modelVarRef.access & VarAccessModeAttribute::READ_WRITE))
        {
            throw std::runtime_error("Variable references to lower-dimensional variables cannot be read-write.");
        }
    }

    // If custom update is synaptic i.e. across both pre and postsynaptic axes
    if(isSynaptic()) {
        // **TODO** check any non-synaptic variables/var references are read-only
        
        // If model specifies reduction operations but none are correctly configured
        if(isModelReduction() && !isReduction(VarAccessDim::BATCH)) {
            throw std::runtime_error("Custom updates uses reduction model but shape is incorrect.");
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
            if(isReduction(VarAccessDim::BATCH)) {
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
    // Otherwise, it's across a single axis
    else {
        // Get size of first variable reference
        m_Size = m_VarReferences.begin()->second.getSize();
        assert(m_Size.has_value());

        // Give error if any sizes differ
        if(std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [this](const auto &v) { return v.second.getSize() != m_Size; }))
        {
            throw std::runtime_error("All referenced variables must have the same size.");
        }

        // Check only one type of reduction is specified
        const bool batchReduction = isReduction(VarAccessDim::BATCH);
        const bool neuronReduction = isReduction(VarAccessDim::NEURON);
        if (batchReduction && neuronReduction) {
            throw std::runtime_error("Custom updates cannot perform batch and neuron reductions simultaneously.");
        }
        // Otherwise, if model specifies reduction operations but none are correctly configured
        else if(isModelReduction() && !batchReduction && !neuronReduction) {
            throw std::runtime_error("Custom updates uses reduction model but shape is incorrect.");
        }

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
}
//----------------------------------------------------------------------------
bool CustomUpdate::isInitRNGRequired() const
{
    // Return whether initialising variables require an RNG
    return Utils::isRNGRequired(getVarInitialisers());
}
//----------------------------------------------------------------------------
bool CustomUpdate::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
}
//----------------------------------------------------------------------------
bool CustomUpdate::isModelReduction() const
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
bool CustomUpdate::isTransposeOperation() const
{
    // Transpose opetation is required if any variable references have a transpose
    return std::any_of(m_VarReferences.cbegin(), m_VarReferences.cend(),
                       [](const auto &v) { return (v.second.getTransposeSynapseGroup() != nullptr); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdate::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCustomUpdateModel()->getHashDigest(), hash);
    Utils::updateHash(getUpdateGroupName(), hash);
    Utils::updateHash(getDims(), hash);

    // If custom update is synaptic
    if(isSynaptic()) {
        Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
        Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    }
    else {
        // Update hash with whether delay is required
        const bool delayed = (getDelayNeuronGroup() != nullptr);
        Utils::updateHash(delayed, hash);

        // If it is, also update hash with number of delay slots
        if(delayed) {
            Utils::updateHash(getDelayNeuronGroup()->getNumDelaySlots(), hash);
        }
    }

    // Loop through variable references
    for(const auto &v : getVarReferences()) {
        // Update hash with whether variable references require transpose
        Utils::updateHash((v.second.getTransposeSynapseGroup() == nullptr), hash);

        // Update hash with whether variable references require delay
        Utils::updateHash((v.second.getDelayNeuronGroup() == nullptr), hash);

        // Update hash with dimensionality of target variable dimensions as this effects indexing code
        Utils::updateHash(v.second.getVarDims(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdate::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCustomUpdateModel()->getVars(), hash);
    Utils::updateHash(getDims(), hash);

    if(isSynaptic()) {
        Utils::updateHash(getSynapseMatrixConnectivity(getSynapseGroup()->getMatrixType()), hash);
        Type::updateHash(getSynapseGroup()->getSparseIndType(), hash);
    }

    // Include variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdate::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
bool CustomUpdate::isReduction(VarAccessDim reduceDim) const
{
    // Return true if any variables have REDUCE flag in their access mode and have reduction dimension 
    // **NOTE** this is correct because custom update variable access types are defined subtractively
    const auto vars = getCustomUpdateModel()->getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                    [reduceDim](const Models::Base::CustomUpdateVar &v)
                    { 
                        return ((v.access & VarAccessModeAttribute::REDUCE) 
                                && (static_cast<unsigned int>(v.access) & static_cast<unsigned int>(reduceDim)));
                    }))
    {
        return true;
    }

    // Loop through all variable references
    for(const auto &modelVarRef : getCustomUpdateModel()->getVarRefs()) {
        // If custom update model reduces into this variable reference 
        // and the variable it targets doesn't have reduction dimension
        const auto &varRef = getVarReferences().at(modelVarRef.name);
        if ((modelVarRef.access & VarAccessModeAttribute::REDUCE) 
            && !(varRef.getVarDims() & reduceDim)) 
        {
            return true;
        }
    }

    return false;
}
//----------------------------------------------------------------------------
std::vector<CustomUpdate*> CustomUpdate::getReferencedCustomUpdates() const
{
    // Loop through variable references
    std::vector<CustomUpdate*> references;
    for(const auto &v : getVarReferences()) {
        // If a custom update is referenced, add to set
        auto *refCU = v.second.getReferencedCustomUpdate();
        if(refCU != nullptr) {
            references.push_back(refCU);
        }
    }

    // Return set
    return references;
}
}   // namespace GeNN
