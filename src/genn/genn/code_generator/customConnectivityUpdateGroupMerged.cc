#include "code_generator/customConnectivityUpdateGroupMerged.h"

// Standard C++ includes
#include <list>

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMergedBase
//----------------------------------------------------------------------------
CustomConnectivityUpdateGroupMergedBase::CustomConnectivityUpdateGroupMergedBase(size_t index, const std::string &precision,
                                                                                 const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   GroupMerged<CustomConnectivityUpdateInternal>(index, precision, groups)
{
    addField("unsigned int", "numSrcNeurons",
            [](const CustomConnectivityUpdateInternal &cg, size_t) 
            {
                const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons()); 
            });

    addField("unsigned int", "numTrgNeurons",
            [](const CustomConnectivityUpdateInternal &cg, size_t)
            { 
                const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                return std::to_string(sgInternal->getTrgNeuronGroup()->getNumNeurons()); 
            });

    // Add heterogeneous custom update model parameters
    addHeterogeneousParams<CustomConnectivityUpdateGroupMergedBase>(
        getArchetype().getCustomConnectivityUpdateModel()->getParamNames(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); },
        &CustomConnectivityUpdateGroupMergedBase::isParamHeterogeneous);

    // Add heterogeneous weight update model CustomConnectivityUpdateGroupMerged parameters
    addHeterogeneousDerivedParams<CustomConnectivityUpdateGroupMergedBase>(
        getArchetype().getCustomConnectivityUpdateModel()->getDerivedParams(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); },
        &CustomConnectivityUpdateGroupMergedBase::isDerivedParamHeterogeneous);
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMergedBase::isParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMergedBase::isDerivedParamHeterogeneous(const std::string &name) const
{
    return isParamValueHeterogeneous(name, [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); });
}

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateGroupMerged::name = "CustomConnectivityUpdate";
//----------------------------------------------------------------------------
CustomConnectivityUpdateGroupMerged::CustomConnectivityUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomConnectivityUpdateGroupMergedBase(index, precision, groups)
{
    // Reserve vector of vectors to hold variables to update for all custom connectivity update groups, in archetype order
    m_SortedDependentVars.reserve(getGroups().size());

    // Loop through groups
    for(const auto &g : getGroups()) {
        // Get group's dependent variables
        const auto dependentVars = g.get().getDependentVariables();
        
        // Convert to list and sort
        // **NOTE** WUVarReferences are non-assignable so can't be sorted in a vector
        std::list<Models::WUVarReference> dependentVarsList(dependentVars.cbegin(), dependentVars.cend());
        dependentVarsList.sort([](const Models::WUVarReference &a, const Models::WUVarReference &b)
                               {  
                                   boost::uuids::detail::sha1 hashA;  
                                   Utils::updateHash(a.getVar().type, hashA);
                                   Utils::updateHash(getVarAccessDuplication(a.getVar().access), hashA);

                                   boost::uuids::detail::sha1 hashB;
                                   Utils::updateHash(b.getVar().type, hashB);
                                   Utils::updateHash(getVarAccessDuplication(b.getVar().access), hashB);

                                   return (hashA.get_digest() < hashB.get_digest());
                                });

        // Add vector for this groups update vars
        m_SortedDependentVars.emplace_back(dependentVarsList.cbegin(), dependentVarsList.cend());
    }
    
    // Check all vectors are the same size
    assert(std::all_of(m_SortedDependentVars.cbegin(), m_SortedDependentVars.cend(),
                       [this](const std::vector<Models::WUVarReference> &vars)
                       {
                           return (vars.size() == m_SortedDependentVars.front().size());
                       }));
    
    
    addField("unsigned int", "rowStride",
            [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
            { 
                const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
            });
    
    
    assert(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE);
    addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind", 
                [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
                { 
                    return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
                });

    addField("unsigned int*", "rowLength",
                [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
                { 
                    return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
                });
    
    // If some presynaptic variables are delayed, add delay pointer
    if (getArchetype().getPreDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "preSpkQuePtr", 
                 [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getPreDelayNeuronGroup()->getName(); 
                 });
    }

    // If some postsynaptic variables are delayed, add delay pointer
    if (getArchetype().getPostDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "postSpkQuePtr", 
                 [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getPostDelayNeuronGroup()->getName(); 
                 });
    }
    
    // If this backend requires per-population RNGs and this group requires one
    if(backend.isPopulationRNGRequired() && getArchetype().isRowSimRNGRequired()){
        addPointerField(backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rowRNG");
    }

    // Add variables to struct
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    addVars(cm->getVars(), backend.getDeviceVarPrefix());
    addVars(cm->getPreVars(), backend.getDeviceVarPrefix());
    addVars(cm->getPostVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    addVarReferences(cm->getVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getVarReferences(); });
    addVarReferences(cm->getPreVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getPreVarReferences(); });
    addVarReferences(cm->getPostVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getPostVarReferences(); });

    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    
    // Loop through sorted dependent variables
    for(size_t i = 0; i < getSortedArchetypeDependentVars().size(); i++) {
        addField(getSortedArchetypeDependentVars().at(i).getVar().type + "*", "_dependentVar" + std::to_string(i), 
                 [i, &backend, this](const CustomConnectivityUpdateInternal&, size_t g) 
                 { 
                     const auto &varRef = m_SortedDependentVars[g][i];
                     return backend.getDeviceVarPrefix() + varRef.getVar().name + varRef.getTargetName(); 
                 });
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const CustomConnectivityUpdateInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomConnectivityUpdateInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    // Update hash with each group's parameters, derived parameters and variable references
    updateHash([](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); }, hash);
    updateHash([](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const CustomConnectivityUpdateInternal &cg) { return cg.getVarReferences(); }, hash);
    updateHash([](const CustomConnectivityUpdateInternal &cg) { return cg.getPreVarReferences(); }, hash);
    updateHash([](const CustomConnectivityUpdateInternal &cg) { return cg.getPostVarReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdateGroupMerged::generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    Substitutions updateSubs(&popSubs);

    // Add substitutions for number of pre and postsynaptic neurons
    updateSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
    updateSubs.addVarSubstitution("num_post", "group->numTrgNeurons");

    // Define synapse loop function
    // **NOTE** index is signed integer so generated code can safely use j-- to process same synapse again
    // **YUCK** ideally id_post, id_syn, remove_synapse and all synaptic and postsynaptic variable substitutions would only be allowable within this scope but that's not currently possible
    updateSubs.addFuncSubstitution("for_each_synapse", 1, "for(int j = 0; j < group->rowLength[" + updateSubs["id_pre"] + "]; j++){ const unsigned int idx = rowStartIdx + j; $(0) }");

    updateSubs.addVarSubstitution("id_post", "group->ind[rowStartIdx + j]");
    updateSubs.addVarSubstitution("id_syn", "idx");

    // Get variables which will need to be manipulated when adding and removing synapses
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    const auto &ccuVars = cm->getVars();
    const auto &ccuVarRefs = cm->getVarRefs();
    const auto &dependentVars = getSortedArchetypeDependentVars();

    // Determine if any
    const bool modelBatched = (modelMerged.getModel().getBatchSize() > 1);
    const bool anyBatched = (modelBatched && (std::any_of(getArchetype().getVarReferences().cbegin(), getArchetype().getVarReferences().cend(), 
                                                          [](const auto &v){ return v.second.isDuplicated(); })
                                              || std::any_of(dependentVars.cbegin(), dependentVars.cend(),
                                                             [](const auto &v){ return v.isDuplicated(); })));

    // Calculate index of start of row
    os << "const unsigned int rowStartIdx = " << updateSubs["id_pre"] << " * group->rowStride;" << std::endl;

    // If any variables are batched
    if (anyBatched) {
        os << "const unsigned int synStride = group->numSrcNeurons * group->rowStride;" << std::endl;
    }

    // Generate code to add a synapse to this row
    std::stringstream addSynapseStream;
    CodeStream addSynapse(addSynapseStream);
    {
        CodeStream::Scope b(addSynapse);

        // Assert that there is space to add synapse
        backend.genAssert(addSynapse, "group->rowLength[" + updateSubs["id_pre"] + "] < group->rowStride");

        // Calculate index to insert synapse
        addSynapse << "const unsigned newIdx = rowStartIdx + group->rowLength[" << updateSubs["id_pre"] << "];" << std::endl;

        // Set postsynaptic target to parameter 0
        addSynapse << "group->ind[newIdx] = $(0);" << std::endl;
 
        // Use subsequent parameters to initialise new synapse's custom connectivity update model variables
        for (size_t i = 0; i < ccuVars.size(); i++) {
            addSynapse << "group->" << ccuVars[i].name << "[newIdx] = $(" << (1 + i) << ");" << std::endl;
        }

        // Use subsequent parameters to initialise new synapse's variables referenced via the custom connectivity update
        for (size_t i = 0; i < ccuVarRefs.size(); i++) {
            // If model is batched and this variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && getArchetype().getVarReferences().at(ccuVarRefs[i].name).isDuplicated()) 
            {
                // Copy parameter into a register (just incase it's e.g. a RNG call) and copy into all batches
                addSynapse << "const " << ccuVarRefs[i].type << " _" << ccuVarRefs[i].name << "Val = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
                addSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    addSynapse << "group->" << ccuVarRefs[i].name << "[(b * synStride) + newIdx] = _" << ccuVarRefs[i].name << "Val;" << std::endl;
                }
            }
            // Otherwise, write parameter straight into var reference
            else {
                addSynapse << "group->" << ccuVarRefs[i].name << "[newIdx] = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
            }
        }
        
        // Loop through any other dependent variables
        for (size_t i = 0; i < dependentVars.size(); i++) {
            // If model is batched and this dependent variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && dependentVars.at(i).isDuplicated())
            {
                // Loop through all batches and zero
                addSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    addSynapse << "group->_dependentVar" << i << "[(b * synStride) + newIdx] = 0;" << std::endl;
                }
            }
            // Otherwise, zero var reference
            else {
                addSynapse << "group->_dependentVar" << i << "[newIdx] = 0;" << std::endl;
            }
        }

        // Increment row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        addSynapse << "group->rowLength[" << updateSubs["id_pre"] << "]++;" << std::endl;
    }

    // Add function substitution with parameters to initialise custom connectivity update variables and variable references
    updateSubs.addFuncSubstitution("add_synapse", 1 + ccuVars.size() + ccuVarRefs.size(), addSynapseStream.str());

    // Generate code to remove a synapse from this row
    std::stringstream removeSynapseStream;
    CodeStream removeSynapse(removeSynapseStream);
    {
        CodeStream::Scope b(removeSynapse);

        // Calculate index we want to copy synapse from
        removeSynapse << "const unsigned lastIdx = rowStartIdx + group->rowLength[" << updateSubs["id_pre"] << "] - 1;" << std::endl;

        // Copy postsynaptic target from end of row over synapse to be deleted
        removeSynapse << "group->ind[idx] = group->ind[lastIdx];" << std::endl;

        // Copy custom connectivity update variables from end of row over synapse to be deleted
        for (size_t i = 0; i < ccuVars.size(); i++) {
            removeSynapse << "group->" << ccuVars[i].name << "[idx] = group->" << ccuVars[i].name << "[lastIdx];" << std::endl;
        }
        
        // Loop through variable references
        for (size_t i = 0; i < ccuVarRefs.size(); i++) {
            // If model is batched and this variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) 
                && getArchetype().getVarReferences().at(ccuVarRefs[i].name).isDuplicated())
            {
                // Loop through all batches and copy custom connectivity update variable references from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(addSynapse);
                    removeSynapse << "group->" << ccuVarRefs[i].name << "[(b * synStride) + idx] = group->" << ccuVarRefs[i].name << "[(b * synStride) + lastIdx];" << std::endl;
                }
            }
            // Otherwise, copy custom connectivity update variable references from end of row over synapse to be deleted
            else {
                removeSynapse << "group->" << ccuVarRefs[i].name << "[idx] = group->" << ccuVarRefs[i].name << "[lastIdx];" << std::endl;
            }
        }
        
        // Loop through any other dependent variables
        for (size_t i = 0; i < dependentVars.size(); i++) {
            // If model is batched and this dependent variable is duplicated
            if ((modelMerged.getModel().getBatchSize() > 1) && dependentVars.at(i).isDuplicated()) {
                // Loop through all batches and copy dependent variable from end of row over synapse to be deleted
                removeSynapse << "for(int b = 0; b < " << modelMerged.getModel().getBatchSize() << "; b++)";
                {
                    CodeStream::Scope b(removeSynapse);
                    removeSynapse << "group->_dependentVar" << i << "[(b * synStride) + idx] = group->_dependentVar" << i << "[(b * synStride) + lastIdx];" << std::endl;
                }
            }
            // Otherwise, copy dependent variable from end of row over synapse to be deleted
            else {
                removeSynapse << "group->_dependentVar" << i << "[idx] = group->_dependentVar" << i << "[lastIdx];" << std::endl;
            }
        }

        // Decrement row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        removeSynapse << "group->rowLength[" << updateSubs["id_pre"] << "]--;" << std::endl;

        // Decrement loop counter so synapse j will get processed
        removeSynapse << "j--;" << std::endl;
    }
    updateSubs.addFuncSubstitution("remove_synapse", 0, removeSynapseStream.str());

    // **TODO** presynaptic variables and variable references could be read into registers at start of row
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "group->", "[" + updateSubs["id_syn"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPreVars(), "", "group->", "[" + updateSubs["id_pre"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPostVars(), "", "group->", "[" + updateSubs["id_post"] + "]");

    // Substitute in variable references, filtering out those which are duplicated
    const auto &variableRefs = getArchetype().getVarReferences();
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "group->", 
                                      [&updateSubs](VarAccessMode, const std::string&) { return "[" + updateSubs["id_syn"] + "]"; },
                                      [modelBatched, &variableRefs](VarAccessMode, const std::string &name) 
                                      {
                                          return !modelBatched || !variableRefs.at(name).isDuplicated(); 
                                      });

    // Substitute in (potentially delayed) presynaptic variable references
    const auto &preVariableRefs = getArchetype().getPreVarReferences();
    updateSubs.addVarNameSubstitution(cm->getPreVarRefs(), "", "group->", 
                                      [&preVariableRefs, &updateSubs](VarAccessMode, const std::string &name)
                                      { 
                                          if(preVariableRefs.at(name).getDelayNeuronGroup() != nullptr) {
                                              return "[preDelayOffset + " + updateSubs["id_pre"] + "]"; 
                                          }
                                          else {
                                              return "[" + updateSubs["id_pre"] + "]"; 
                                          }
                                      });
    
    // Substitute in (potentially delayed) postsynaptic variable references
    const auto &postVariableRefs = getArchetype().getPostVarReferences();
    updateSubs.addVarNameSubstitution(cm->getPostVarRefs(), "", "group->",
                                      [&postVariableRefs, &updateSubs](VarAccessMode, const std::string &name)
                                      { 
                                          if(postVariableRefs.at(name).getDelayNeuronGroup() != nullptr) {
                                              return "[postDelayOffset + " + updateSubs["id_post"] + "]"; 
                                          }
                                          else {
                                              return "[" + updateSubs["id_post"] + "]"; 
                                          }
                                      });

    updateSubs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                         [this](const std::string &name) { return isParamHeterogeneous(name);  },
                                         "", "group->");
    updateSubs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                       [this](const std::string &name) { return isDerivedParamHeterogeneous(name);  },
                                       "", "group->");
    updateSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");

    // Apply substitutons to row update code and write out
    std::string code = cm->getRowUpdateCode();
    updateSubs.applyCheckUnreplaced(code, "custom connectivity update : merged" + std::to_string(getIndex()));
    code = ensureFtype(code, modelMerged.getModel().getPrecision());
    os << code;
}

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityHostUpdateGroupMerged::name = "CustomConnectivityHostUpdate";
//----------------------------------------------------------------------------
CustomConnectivityHostUpdateGroupMerged::CustomConnectivityHostUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                                 const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomConnectivityUpdateGroupMergedBase(index, precision, groups)
{
    // Add pre and postsynaptic variables
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    addVars(backend, cm->getPreVars(), &CustomConnectivityUpdateInternal::getPreVarLocation);
    addVars(backend, cm->getPostVars(), &CustomConnectivityUpdateInternal::getPostVarLocation);

    // Add host extra global parameters
    for(const auto &e : cm->getExtraGlobalParams()) {
        addField(e.type, e.name,
                 [e](const CustomConnectivityUpdateInternal &g, size_t) { return e.name + g.getName(); },
                 GroupMergedFieldType::HOST_DYNAMIC);

        if(Utils::isTypePointer(e.type) && !backend.getDeviceVarPrefix().empty()) {
            addField(e.type, backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const CustomConnectivityUpdateInternal &g, size_t)
                     {
                         return backend.getDeviceVarPrefix() + e.name + g.getName();
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
             
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    CodeStream::Scope b(os);
    os << "// merged custom connectivity host update group " << getIndex() << std::endl;
    os << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(os);

        // Get reference to group
        os << "const auto *group = &mergedCustomConnectivityHostUpdateGroup" << getIndex() << "[g]; " << std::endl;

        // Create substitutions
        const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
        Substitutions subs;
        subs.addVarSubstitution("rng", "hostRNG");
        subs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        subs.addVarSubstitution("num_post", "group->numTrgNeurons");
        subs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");
        subs.addVarNameSubstitution(cm->getPreVars(), "", "group->");
        subs.addVarNameSubstitution(cm->getPostVars(), "", "group->");
        subs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                       [this](const std::string &p) { return isParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                     [this](const std::string & p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        for(const auto &egp : cm->getExtraGlobalParams()) {
            // If EGP is a pointer
            if(Utils::isTypePointer(egp.type)) {
                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genExtraGlobalParamPush(push, egp.type, egp.name,
                                                VarLocation::HOST_DEVICE, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("push" + egp.name + "ToDevice", 1, pushStream.str());

                // Generate code to pull this EGP with count specified by $(0)
                std::stringstream pullStream;
                CodeStream pull(pullStream);
                backend.genExtraGlobalParamPull(pull, egp.type, egp.name,
                                                VarLocation::HOST_DEVICE, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("pull" + egp.name + "FromDevice", 1, pullStream.str());
            }
        }

        addVarPushPullFuncSubs(backend, subs, cm->getPreVars(), "group->numSrcNeurons",
                               &CustomConnectivityUpdateInternal::getPreVarLocation);
        addVarPushPullFuncSubs(backend, subs, cm->getPostVars(), "group->numTrgNeurons",
                               &CustomConnectivityUpdateInternal::getPostVarLocation);

        // Apply substitutons to row update code and write out
        std::string code = cm->getHostUpdateCode();
        subs.applyCheckUnreplaced(code, "custom connectivity host update : merged" + std::to_string(getIndex()));
        code = ensureFtype(code, modelMerged.getModel().getPrecision());
        os << code;
    }
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                                                     const Models::Base::VarVec &vars, const std::string &count,
                                                                     VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const) const
{
    // Loop through variables
    for(const auto &v : vars) {
        // If var is located on the host
        const auto loc = (getArchetype().*getVarLocationFn)(v.name);
        if (loc & VarLocation::HOST) {
            // Generate code to push this variable
            // **YUCK** these EGP functions should probably just be called dynamic or something
            std::stringstream pushStream;
            CodeStream push(pushStream);
            backend.genExtraGlobalParamPush(push, v.type + "*", v.name,
                                            loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("push" + v.name + "ToDevice", 0, pushStream.str());

            // Generate code to pull this variable
            // **YUCK** these EGP functions should probably just be called dynamic or something
            std::stringstream pullStream;
            CodeStream pull(pullStream);
            backend.genExtraGlobalParamPull(pull, v.type + "*", v.name,
                                            loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("pull" + v.name + "FromDevice", 0, pullStream.str());
        }
    }
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVars(const BackendBase &backend, const Models::Base::VarVec &vars,
                                                      VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const)
{
    // Loop through variables
    for(const auto &v : vars) {
        // If var is located on the host
        if ((getArchetype().*getVarLocationFn)(v.name) & VarLocation::HOST) {
            addField(v.type + "*", v.name,
                    [v](const CustomConnectivityUpdateInternal &g, size_t) { return v.name + g.getName(); },
                    GroupMergedFieldType::HOST);

            if(!backend.getDeviceVarPrefix().empty()) {
                addField(v.type + "*", backend.getDeviceVarPrefix() + v.name,
                         [v, &backend](const CustomConnectivityUpdateInternal &g, size_t)
                         {
                             return backend.getDeviceVarPrefix() + v.name + g.getName();
                         });
            }
        }
    }
}