/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/
// Standard C++ includes
#include <algorithm>
#include <numeric>
#include <typeinfo>

// Standard C includes
#include <cstdio>
#include <cmath>
#include <cassert>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "modelSpec.h"

// GeNN transpiler includes
#include "transpiler/parser.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//! Use Kahn's algorithm to sort custom updates topologically based on inter-custom update references
template<typename G>
std::vector<G*> getSortedCustomUpdates(std::map<std::string, G> &customUpdates)
{
    // Loop through custom updates
    std::set<G*> startCustomUpdate;
    std::multimap<G*, G*> referencingCustomUpdates;
    std::map<G*, int> inDegree;
    for(auto &c : customUpdates) {
        // Get vector of other custom updates referenced by this one and use to build reverse lookup structure
        const auto referenced = c.second.getReferencedCustomUpdates();
        for(auto *r : referenced) {
            referencingCustomUpdates.emplace(static_cast<G*>(r), &c.second);
        }
        
        // Store this as initial in-degree
        inDegree.emplace(&c.second, referenced.size());

        // If it doesn't references any other custom updates, it's a possible starting node
        if(referenced.empty()) {
            startCustomUpdate.insert(&c.second);
        }
    }

    // Loop through start nodes
    std::vector<G*> sortedCustomUpdates;
    while(!startCustomUpdate.empty()) {
        // Move custom update from star set to sorted 
        auto *cu = *startCustomUpdate.begin();
        sortedCustomUpdates.push_back(cu);
        startCustomUpdate.erase(startCustomUpdate.begin());

        // Loop through all custom updates whch reference this one
        auto const [refBegin, refEnd] = referencingCustomUpdates.equal_range(cu);
        for(auto r = refBegin; r != refEnd; r++) {
            // Decrease in-degree
            int &rInDegree = inDegree.at(r->second);
            rInDegree--;

            // If in-degree has reached zero, add to starting set
            if(rInDegree == 0) {
                startCustomUpdate.insert(r->second);
            }
        }
    }

    // If any custom updates end up with an in-degree greater than 0
    if(std::any_of(inDegree.cbegin(), inDegree.cend(),
                   [](const auto &cu){ return cu.second > 0; }))
    {
        throw std::runtime_error("Custom update variable references cannot form a cycle");
    }

    // Check that all custom updates have made it into the sorted list
    assert(sortedCustomUpdates.size() == customUpdates.size());

    // Return sorted custom updates
    return sortedCustomUpdates;
}
}   // Anonymous namespace

// ---------------------------------------------------------------------------
// GeNN::ModelSpec
// ---------------------------------------------------------------------------
namespace GeNN
{
ModelSpec::ModelSpec()
:   m_Precision(Type::Float), m_TimePrecision(std::nullopt), m_DT(0.5), m_TimingEnabled(false), m_Seed(0),
    m_DefaultVarLocation(VarLocation::HOST_DEVICE), m_DefaultExtraGlobalParamLocation(VarLocation::HOST_DEVICE),
    m_DefaultSparseConnectivityLocation(VarLocation::HOST_DEVICE), m_DefaultNarrowSparseIndEnabled(false),
    m_FusePostsynapticModels(false), m_FusePrePostWeightUpdateModels(false), m_BatchSize(1)
{
}
// ---------------------------------------------------------------------------
ModelSpec::~ModelSpec() 
{
}
// ---------------------------------------------------------------------------
void ModelSpec::setPrecision(const Type::UnresolvedType &precision)
{
    // Resolve type
    // **NOTE** no type context as that would be circular!
    const auto resolved = precision.resolve({});
    if (!resolved.isNumeric()) {
        throw std::runtime_error("Only numeric types can be used for precision");
    }
    else {
        if (resolved.getNumeric().isIntegral) {
            throw std::runtime_error("Only floating point types can be used for precision");
        }
        m_Precision = resolved;
    }
}
// ---------------------------------------------------------------------------
void ModelSpec::setTimePrecision(const Type::UnresolvedType &timePrecision)
{ 
    // Resolve type
    // **NOTE** no type context as that would be circular!
    const auto resolved = timePrecision.resolve({});
    if (!resolved.isNumeric()) {
        throw std::runtime_error("Only numeric types can be used for timeprecision");
    }
    else {
        if (resolved.getNumeric().isIntegral) {
            throw std::runtime_error("Only floating point types can be used for time precision");
        }
        m_TimePrecision = resolved; 
    }
}
// ---------------------------------------------------------------------------
unsigned int ModelSpec::getNumNeurons() const
{
    // Return sum of local neuron group sizes
    return std::accumulate(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(), 0,
                           [](unsigned int total, const NeuronGroupValueType &n)
                           {
                               return total + n.second.getNumNeurons();
                           });
}
// ---------------------------------------------------------------------------
NeuronGroup *ModelSpec::findNeuronGroup(const std::string &name)
{
    // If a matching local neuron group is found, return it
    auto localNeuronGroup = m_LocalNeuronGroups.find(name);
    if(localNeuronGroup != m_LocalNeuronGroups.cend()) {
        return &localNeuronGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("Neuron group '" + name + "' not found");
    }
}
// ---------------------------------------------------------------------------
const NeuronGroup *ModelSpec::findNeuronGroup(const std::string &name) const
{
    // If a matching local neuron group is found, return it
    auto localNeuronGroup = m_LocalNeuronGroups.find(name);
    if(localNeuronGroup != m_LocalNeuronGroups.cend()) {
        return &localNeuronGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("Neuron group '" + name + "' not found");
    }
}
// ---------------------------------------------------------------------------
NeuronGroup *ModelSpec::addNeuronPopulation(const std::string &name, unsigned int size, const NeuronModels::Base *model,
                                            const ParamValues &paramValues, const VarValues &varInitialisers)
{
    // Add neuron group to map
    auto result = m_LocalNeuronGroups.try_emplace(
        name,
        name, size, model,
        paramValues, varInitialisers, 
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation);

    if(!result.second) {
        throw std::runtime_error("Cannot add a neuron population with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
SynapseGroup *ModelSpec::findSynapseGroup(const std::string &name)
{
    // If a matching local synapse group is found, return it
    auto synapseGroup = m_LocalSynapseGroups.find(name);
    if(synapseGroup != m_LocalSynapseGroups.cend()) {
        return &synapseGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("Synapse group '" + name + "' not found");
    }
}
// ---------------------------------------------------------------------------
const SynapseGroup *ModelSpec::findSynapseGroup(const std::string &name) const
{
    // If a matching local synapse group is found, return it
    auto synapseGroup = m_LocalSynapseGroups.find(name);
    if(synapseGroup != m_LocalSynapseGroups.cend()) {
        return &synapseGroup->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("Synapse group '" + name + "' not found");
    }
}
// ---------------------------------------------------------------------------
CurrentSource *ModelSpec::findCurrentSource(const std::string &name)
{
    // If a matching local current source is found, return it
    auto localCurrentSource = m_LocalCurrentSources.find(name);
    if(localCurrentSource != m_LocalCurrentSources.cend()) {
        return &localCurrentSource->second;
    }
    // Otherwise, error
    else {
        throw std::runtime_error("Current source " + name + " not found, aborting ...");
    }
}
// ---------------------------------------------------------------------------
CurrentSource *ModelSpec::addCurrentSource(const std::string &currentSourceName, const CurrentSourceModels::Base *model, NeuronGroup *neuronGroup, 
                                           const ParamValues &paramValues, const VarValues &varInitialisers, const LocalVarReferences &neuronVarReferences)
{
    // Add current source to map
    auto *neuronGroupInternal = static_cast<NeuronGroupInternal*>(neuronGroup);
    auto result = m_LocalCurrentSources.try_emplace(
        currentSourceName,
        currentSourceName, model, paramValues,
        varInitialisers, neuronVarReferences, neuronGroupInternal, 
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation);

    if(!result.second) {
        throw std::runtime_error("Cannot add a current source with duplicate name:" + currentSourceName);
    }
    else {
        neuronGroupInternal->injectCurrent(&result.first->second);
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomUpdate *ModelSpec::addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model,
                                         const ParamValues &paramValues, const VarValues &varInitialisers,
                                         const VarReferences &varReferences, const EGPReferences &egpReferences)
{
    // Add neuron group to map
    auto result = m_CustomUpdates.try_emplace(
        name,
        name, updateGroupName, model,
        paramValues, varInitialisers, varReferences, egpReferences,
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation);

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomConnectivityUpdate *ModelSpec::addCustomConnectivityUpdate(const std::string &name, const std::string &updateGroupName, 
                                                                 SynapseGroup *synapseGroup, const CustomConnectivityUpdateModels::Base *model, 
                                                                 const ParamValues &paramValues, const VarValues &varInitialisers,
                                                                 const VarValues &preVarInitialisers, const VarValues &postVarInitialisers,
                                                                 const WUVarReferences &varReferences, const VarReferences &preVarReferences,
                                                                 const VarReferences &postVarReferences, const EGPReferences &egpReferences)
{
    // Add custom connectivity update to 
    auto result = m_CustomConnectivityUpdates.try_emplace(
        name,
        name, updateGroupName, static_cast<SynapseGroupInternal*>(synapseGroup), model,
        paramValues, varInitialisers, preVarInitialisers, postVarInitialisers, 
        varReferences, preVarReferences, postVarReferences, egpReferences,
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation);

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom connectivity update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
CustomUpdateWU *ModelSpec::addCustomUpdate(const std::string &name, const std::string &updateGroupName, const CustomUpdateModels::Base *model, 
                                           const ParamValues &paramValues, const VarValues &varInitialisers,
                                           const WUVarReferences &varReferences, const EGPReferences &egpReferences)
{
    // Add neuron group to map
    auto result = m_CustomWUUpdates.try_emplace(
        name,
        name, updateGroupName, model,
        paramValues, varInitialisers, varReferences, egpReferences,
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation);

    if(!result.second) {
        throw std::runtime_error("Cannot add a custom update with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
// ---------------------------------------------------------------------------
void ModelSpec::finalise()
{
    // Build type context
    m_TypeContext = {{"scalar", getPrecision()}, {"timepoint", getTimePrecision()}};

    // Sort custom updates and custom WU updates so whether each one 
    // should be batched or not gets correctly resolved
    const auto sortedCustomUpdates = getSortedCustomUpdates(m_CustomUpdates);
    const auto sortedCustomWUUpdates = getSortedCustomUpdates(m_CustomWUUpdates);
    
    LOGD_GENN << "Custom update finalise order";
    for(const auto *c : sortedCustomUpdates) {
        LOGD_GENN << "\t" << c->getName();
    }

    LOGD_GENN << "Custom WU update finalise order";
    for(const auto *c : sortedCustomWUUpdates) {
        LOGD_GENN << "\t" << c->getName();
    }

    // Finalise neuron groups
    for(auto &n : m_LocalNeuronGroups) {
        n.second.finalise(m_DT);
    }

    // Finalise synapse groups
    for(auto &s : m_LocalSynapseGroups) {
        s.second.finalise(m_DT);
    }

    // Finalise current sources
    for(auto &cs : m_LocalCurrentSources) {
        cs.second.finalise(m_DT);
    }

    // Finalise custom update groups
    // **NOTE** needs to be after synapse groups are finalised 
    // so which vars are delayed has been established
    for(auto *c : sortedCustomUpdates) {
        c->finalise(m_DT, m_BatchSize);
    }

    // Finalise custom WUM update groups
    for(auto *c : sortedCustomWUUpdates) {
        c->finalise(m_DT, m_BatchSize);
    }

    // Finalize custom connectivity update groups
    // **NOTE** needs to be after synapse groups are finalised 
    // so which vars are delayed has been established and after custom
    // updates are finalised so which variables are batched has been established
    for (auto &c : m_CustomConnectivityUpdates) {
        c.second.finalise(m_DT, m_BatchSize);
    }

    // Merge incoming postsynaptic models
    for(auto &n : m_LocalNeuronGroups) {
        n.second.fusePrePostSynapses(m_FusePostsynapticModels, m_FusePrePostWeightUpdateModels);
    }
}
// ---------------------------------------------------------------------------
bool ModelSpec::zeroCopyInUse() const
{
    // If any neuron groups use zero copy return true
    if(std::any_of(std::begin(m_LocalNeuronGroups), std::end(m_LocalNeuronGroups),
                   [](const NeuronGroupValueType &n){ return n.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any current sources use zero copy return true
    if(std::any_of(std::begin(m_LocalCurrentSources), std::end(m_LocalCurrentSources),
                   [](const CurrentSourceValueType &c){ return c.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

    // If any synapse groups use zero copy return true
    if(std::any_of(std::begin(m_LocalSynapseGroups), std::end(m_LocalSynapseGroups),
                   [](const SynapseGroupValueType &s){ return s.second.isZeroCopyEnabled(); }))
    {
        return true;
    }

     // If any custom updates use zero copy return true
     if(std::any_of(std::begin(m_CustomUpdates), std::end(m_CustomUpdates),
                   [](const CustomUpdateValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

     // If any custom WU updates use zero copy return true
     if(std::any_of(std::begin(m_CustomWUUpdates), std::end(m_CustomWUUpdates),
                   [](const CustomUpdateWUValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

     // If any custom connectivity updates use zero copy return true
     if(std::any_of(std::begin(m_CustomConnectivityUpdates), std::end(m_CustomConnectivityUpdates),
                   [](const CustomConnectivityUpdateValueType &c){ return c.second.isZeroCopyEnabled(); }))
     {
        return true;
     }

    return false;
}
// ---------------------------------------------------------------------------
bool ModelSpec::isRecordingInUse() const
{
    return std::any_of(m_LocalNeuronGroups.cbegin(), m_LocalNeuronGroups.cend(),
                       [](const NeuronGroupValueType &n) { return n.second.isRecordingEnabled(); });
}
// ---------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpec::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    Utils::updateHash(getName(), hash);
    Type::updateHash(getPrecision(), hash);
    Type::updateHash(getTimePrecision(), hash);
    Utils::updateHash(getDT(), hash);
    Utils::updateHash(isTimingEnabled(), hash);
    Utils::updateHash(getBatchSize(), hash);
    Utils::updateHash(getSeed(), hash);

    return hash.get_digest();
}
// ---------------------------------------------------------------------------
std::set<std::string> ModelSpec::getCustomUpdateGroupNames(bool includeTranspose, bool includeNonTranspose) const
{
    std::set<std::string> customUpdateGroups;
    if(includeNonTranspose) {
        std::transform(getCustomUpdates().cbegin(), getCustomUpdates().cend(),
                       std::inserter(customUpdateGroups, customUpdateGroups.begin()),
                       [](const ModelSpec::CustomUpdateValueType &v) { return v.second.getUpdateGroupName(); });
        std::transform(getCustomConnectivityUpdates().cbegin(), getCustomConnectivityUpdates().cend(),
                       std::inserter(customUpdateGroups, customUpdateGroups.begin()),
                       [](const ModelSpec::CustomConnectivityUpdateValueType &v) { return v.second.getUpdateGroupName(); });
    }

    // Loop through custom updates
    for(const auto &c : getCustomWUUpdates()) {
        if(c.second.isTransposeOperation()) {
            if(includeTranspose) {
                customUpdateGroups.insert(c.second.getUpdateGroupName());
            }
        }
        else if(includeNonTranspose) {
            customUpdateGroups.insert(c.second.getUpdateGroupName());
        }
    }

    return customUpdateGroups;
}
// ---------------------------------------------------------------------------
SynapseGroup *ModelSpec::addSynapsePopulation(const std::string &name, SynapseMatrixType mtype, NeuronGroup *src, NeuronGroup *trg,
                                              const WeightUpdateModels::Init &wumInitialiser, const PostsynapticModels::Init &psmInitialiser, 
                                              const InitSparseConnectivitySnippet::Init &connectivityInitialiser, 
                                              const InitToeplitzConnectivitySnippet::Init &toeplitzConnectivityInitialiser)
{
    // Add synapse group to map
    auto result = m_LocalSynapseGroups.try_emplace(
        name,
        name, mtype,
        wumInitialiser, psmInitialiser, 
        static_cast<NeuronGroupInternal*>(src), static_cast<NeuronGroupInternal*>(trg),
        connectivityInitialiser, toeplitzConnectivityInitialiser, 
        m_DefaultVarLocation, m_DefaultExtraGlobalParamLocation,
        m_DefaultSparseConnectivityLocation, m_DefaultNarrowSparseIndEnabled);

    if(!result.second) {
        throw std::runtime_error("Cannot add a synapse population with duplicate name:" + name);
    }
    else {
        return &result.first->second;
    }
}
}   // namespace GeNN