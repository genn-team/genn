#include "code_generator/customConnectivityUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateGroupMerged::name = "CustomConnectivityUpdate";
//----------------------------------------------------------------------------
CustomConnectivityUpdateGroupMerged::CustomConnectivityUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   GroupMerged<CustomConnectivityUpdateInternal>(index, precision, groups)
{
    // Reserve vector of vectors to hold variables to update for all custom connectivity update groups, in archetype order
    m_SortedUpdateVars.reserve(getGroups().size());

    // Loop through groups
    for(const auto &g : getGroups()) {
        // Add vector for this groups update vars
        m_SortedUpdateVars.emplace_back();
        
        // Add tuple for each synapse variable with (full) name, type and access
        const auto &vars = g.get().getSynapseGroup()->getWUModel()->getVars();
        std::transform(vars.cbegin(), vars.cend(), std::back_inserter(m_SortedUpdateVars.back()),
                       [g](const Models::Base::Var &v)
                       { 
                           return std::make_tuple(v.name + g.get().getSynapseGroup()->getName(), v.type, getVarAccessDuplication(v.access));
                       });
        
        // Add tuple for each custom update variable with (full) name, type and access
        for(const auto *c : g.get().getSynapseGroup()->getCustomUpdateReferences()) {
            const auto &vars = c->getCustomUpdateModel()->getVars();
            std::transform(vars.cbegin(), vars.cend(), std::back_inserter(m_SortedUpdateVars.back()),
                           [c](const Models::Base::Var &v)
                           { 
                               return std::make_tuple(v.name + c->getName(), v.type, getVarAccessDuplication(v.access));
                           });
        }
        
        // Add tuple for each custom connectivity update variable with (full) name, type and access
        for(const auto *c : g.get().getSynapseGroup()->getCustomConnectivityUpdateReferences()) {
            // Skip references to underlying synapse group from group
            if (c == &g.get()) {
                continue;
            }
            const auto &vars = c->getCustomConnectivityUpdateModel()->getVars();
            std::transform(vars.cbegin(), vars.cend(), std::back_inserter(m_SortedUpdateVars.back()),
                           [c](const Models::Base::Var &v)
                           { 
                               return std::make_tuple(v.name + c->getName(), v.type, getVarAccessDuplication(v.access));
                           });
        }
        
        // Sort update variables
        std::sort(m_SortedUpdateVars.back().begin(), m_SortedUpdateVars.back().end(),
                  [](const UpdateVar &a, const UpdateVar &b)
                  {
                      // Get hash of a's type ane duplication
                      // **NOTE** name is irrelevant
                      boost::uuids::detail::sha1 hashA;  
                      Utils::updateHash(std::get<1>(a), hashA);
                      Utils::updateHash(std::get<2>(a), hashA);
                        
                      // Get hash of b's type ane duplication
                      // **NOTE** name is irrelevant
                      boost::uuids::detail::sha1 hashB;
                      Utils::updateHash(std::get<1>(b), hashB);
                      Utils::updateHash(std::get<2>(b), hashB);
                        
                      // Compare digest
                      return (hashA.get_digest() < hashB.get_digest());
                  });

    }
    
    // Check all vectors are the same size
    assert(std::all_of(m_SortedUpdateVars.cbegin(), m_SortedUpdateVars.cend(),
                       [this](const std::vector<UpdateVar> &vars)
                       {
                           return (vars.size() == m_SortedUpdateVars.front().size());
                       }));
    
    
    addField("unsigned int", "rowStride",
            [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
            { 
                const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
            });
    
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

    // If synapse group has sparse connectivity
    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
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
    }
    else if (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
         addField("uint32_t*", "gp",
                  [&backend](const CustomConnectivityUpdateInternal &cg, size_t) 
                  { 
                      return backend.getDeviceVarPrefix() + "gp" + cg.getSynapseGroup()->getName(); 
                  });
    }
    
    // If some variables are delayed, add delay pointer
    /*if (getArchetype().getDelayNeuronGroup() != nullptr) {
        addField("unsigned int*", "spkQuePtr", 
                 [&backend](const CustomUpdateInternal &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                 });
    }*/

    // Add heterogeneous custom update model parameters
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    addHeterogeneousParams<CustomConnectivityUpdateGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); },
        &CustomConnectivityUpdateGroupMerged::isParamHeterogeneous);

    // Add heterogeneous weight update model CustomConnectivityUpdateGroupMerged parameters
    addHeterogeneousDerivedParams<CustomConnectivityUpdateGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); },
        &CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous);

    // Add variables to struct
    addVars(cm->getVars(), backend.getDeviceVarPrefix());
    addVars(cm->getPreVars(), backend.getDeviceVarPrefix());
    addVars(cm->getPostVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    addVarReferences(cm->getVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getVarReferences(); });
    addVarReferences(cm->getPreVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getPostVarReferences(); });
    addVarReferences(cm->getPostVarRefs(), backend.getDeviceVarPrefix(),
                     [](const CustomConnectivityUpdateInternal &cg) { return cg.getPostVarReferences(); });

    // Add EGPs to struct
    this->addEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    
    // Loop through sorted update variables
    for(size_t i = 0; i < getSortedArchetypeUpdateVars().size(); i++) {
        addField(std::get<1>(getSortedArchetypeUpdateVars().at(i)) + "*", "_updateVar" + std::to_string(i), 
                 [i, &backend, this](const CustomConnectivityUpdateInternal&, size_t g) 
                 { 
                     return backend.getDeviceVarPrefix() + std::get<0>(m_SortedUpdateVars[g][i]);
                 });
    }
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMerged::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); });
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
void CustomConnectivityUpdateGroupMerged::generateUpdate(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    Substitutions updateSubs(&popSubs);

    // Calculate index of start of row
    os << "const unsigned int rowStartIdx = " << updateSubs["id_pre"] << " * group->rowStride;" << std::endl;

    // Add substitutions for number of pre and postsynaptic neurons
    updateSubs.addVarSubstitution("num_pre", "group->numSrcNeurons");
    updateSubs.addVarSubstitution("num_post", "group->numTrgNeurons");

    // Define synapse loop function
    // **NOTE** index is signed integer so generated code can safely use i-- to process same synapse again
    // **YUCK** ideally id_post, id_syn, remove_synapse and all synaptic and postsynaptic variable substitutions would only be allowable within this scope but that's not currently possible
    updateSubs.addFuncSubstitution("for_each_synapse", 1, "for(int i = 0; i < group->rowLength[" + updateSubs["id_pre"] + "]; i++){ const unsigned int idx = rowStartIdx + i; $(0) }");

    updateSubs.addVarSubstitution("id_post", "group->ind[rowStartIdx + i]");
    updateSubs.addVarSubstitution("id_syn", "idccuVarsx");

    // Get variables which will need to be manipulated when adding and removing synapses
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    const auto &updateVars = getSortedArchetypeUpdateVars();

    // Generate code to add a synapse to this row
    std::stringstream addSynapseStream;
    CodeStream addSynapse(addSynapseStream);
    {
        CodeStream::Scope b(addSynapse);

        // Calculate index to insert synapse
        addSynapse << "const unsigned newIdx = rowStartIdx + group->rowLength[" << updateSubs["id_pre"] << "];" << std::endl;

        // Set postsynaptic target to parameter 0
        addSynapse << "group->ind[newIdx] = $(0);" << std::endl;
 
        // **THINK** update vars shouldn't include our custom connectivity update variables OR  any referenced already by custom connectivity update
        // THESE variables should be initialised via parameters, remainder should be zerod
        // This is also better as it means $(add_synapse) calls won't be broken by model changes
        
        // Use subsequent parameters to initialise new synapse's custom connectivity update model variables
        /*for (size_t i = 0; i < ccuVars.size(); i++) {
            addSynapse << "group->" << ccuVars[i].name << "[newIdx] = $(" << (1 + i) << ");" << std::endl;
        }

        // Use subsequent parameters to initialise new synapse's weight update model variables
        for (size_t i = 0; i < wumVars.size(); i++) {
            addSynapse << "group->_" << wumVars[i].name << "[newIdx] = $(" << (1 + ccuVars.size() + i) << ");" << std::endl;
        }*/

        // Increment row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        addSynapse << "group->rowLength[" << updateSubs["id_pre"] << "]++;" << std::endl;
    }

    // Add function substitution
    updateSubs.addFuncSubstitution("add_synapse", 1 /*+ ccuVars.size() + wumVars.size()*/, addSynapseStream.str());

    // Generate code to remove a synapse from this row
    std::stringstream removeSynapseStream;
    CodeStream removeSynapse(removeSynapseStream);
    {
        CodeStream::Scope b(removeSynapse);

        // Calculate index we want to copy synapse from
        removeSynapse << "const unsigned lastIdx = rowStartIdx + group->rowLength[" << updateSubs["id_pre"] << "] - 1;" << std::endl;

        // Copy postsynaptic target from end of row over synapse to be deleted
        removeSynapse << "group->ind[idx] = group->ind[lastIdx];" << std::endl;

        // Copy update variables from end of row over synapse to be deleted
        for (size_t i = 0; i < updateVars.size(); i++) {
            removeSynapse << "group->_updateVar" << i << "[idx] = group->_updateVar" << i << "[lastIdx];" << std::endl;
        }

        // Decrement row length
        // **NOTE** this will also effect any forEachSynapse loop currently in operation
        removeSynapse << "group->rowLength[" << updateSubs["id_pre"] << "]--;" << std::endl;

        // Decrement loop counter so synapse i will get processed
        addSynapse << "i--;" << std::endl;
    }
    updateSubs.addFuncSubstitution("remove_synapse", 0, removeSynapseStream.str());
    
    // **TODO** presynaptic variables and variable references could be read into registers at start of row
    // **TODO** delays
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "group->", "[" + updateSubs["id_syn"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPreVars(), "", "group->", "[" + updateSubs["id_pre"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPostVars(), "", "group->", "[" + updateSubs["id_post"] + "]");
    
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "group->", "[" + updateSubs["id_syn"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPreVarRefs(), "", "group->", "[" + updateSubs["id_pre"] + "]");
    updateSubs.addVarNameSubstitution(cm->getPostVarRefs(), "", "group->", "[" + updateSubs["id_post"] + "]");
    updateSubs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                         [this](size_t i) { return isParamHeterogeneous(i);  },
                                         "", "group->");
    updateSubs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                       [this](size_t i) { return isDerivedParamHeterogeneous(i);  },
                                       "", "group->");
    updateSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");

    // Apply substitutons to row update code and write out
    std::string code = cm->getRowUpdateCode();
    updateSubs.applyCheckUnreplaced(code, "custom connectivity update : merged" + std::to_string(getIndex()));
    code = ensureFtype(code, modelMerged.getModel().getPrecision());
    os << code;
}
//----------------------------------------------------------------------------
std::string CustomConnectivityUpdateGroupMerged::getPrePostVarRefIndex(bool delay, const std::string &index) const
{
    return delay ? ("delayOffset + " + index) : index;
}

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityHostUpdateGroupMerged::name = "CustomConnectivityHostUpdate";
//----------------------------------------------------------------------------
CustomConnectivityHostUpdateGroupMerged::CustomConnectivityHostUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
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
    const auto *cm = getArchetype().getCustomConnectivityUpdateModel();
    addHeterogeneousParams<CustomConnectivityUpdateGroupMerged>(
        cm->getParamNames(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); },
        &CustomConnectivityUpdateGroupMerged::isParamHeterogeneous);

    // Add heterogeneous weight update model CustomConnectivityUpdateGroupMerged parameters
    addHeterogeneousDerivedParams<CustomConnectivityUpdateGroupMerged>(
        cm->getDerivedParams(), "",
        [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); },
        &CustomConnectivityUpdateGroupMerged::isDerivedParamHeterogeneous);

    // Add pre and postsynaptic variables
    addVars(backend, cm->getPreVars(), &CustomConnectivityUpdateInternal::getPreVarLocation);
    addVars(backend, cm->getPostVars(), &CustomConnectivityUpdateInternal::getPostVarLocation);

    // **TODO** add device and host pre and post vars; var refs and EGPsaddVars

    // Add host extra global parameters
    for(const auto &e : cm->getExtraGlobalParams()) {
        addField(e.type, e.name,
                 [e](const CustomConnectivityUpdateInternal &g, size_t) { return e.name + g.getName(); },
                 FieldType::Host);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(e.type, backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const CustomConnectivityUpdateInternal &g, size_t)
                     {
                         return backend.getDeviceVarPrefix() + e.name + g.getName();
                     });
        }
    }
             
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged&) const
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
                                       [this](size_t p) { return isParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                     [this](size_t p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        //**TODO** template helpers and use for EGPs too
        const auto egps = cm->getExtraGlobalParams();
        for(size_t i = 0; i < egps.size(); i++) {
            const auto loc = VarLocation::HOST_DEVICE;// **HACK** getArchetype().getExtraGlobalParamLocation(i);
            // If EGP is a pointer and located on the host
            if(Utils::isTypePointer(egps[i].type) && (loc & VarLocation::HOST)) {
                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genExtraGlobalParamPush(push, egps[i].type, egps[i].name,
                                                loc, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("push" + egps[i].name, 1, pushStream.str());

                // Generate code to pull this EGP with count specified by $(0)
                std::stringstream pullStream;
                CodeStream pull(pullStream);
                backend.genExtraGlobalParamPull(pull, egps[i].type, egps[i].name,
                                                loc, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("pull" + egps[i].name, 1, pullStream.str());
            }
        }

        addVarPushPullFuncSubs(backend, subs, cm->getPreVars(), "group->numSrcNeurons",
                               &CustomConnectivityUpdateInternal::getPreVarLocation);
        addVarPushPullFuncSubs(backend, subs, cm->getPostVars(), "group->numTrgNeurons",
                               &CustomConnectivityUpdateInternal::getPostVarLocation);
    }
}
//----------------------------------------------------------------------------
bool CustomConnectivityHostUpdateGroupMerged::isParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomConnectivityUpdateInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomConnectivityHostUpdateGroupMerged::isDerivedParamHeterogeneous(size_t index) const
{
    return isParamValueHeterogeneous(index, [](const CustomConnectivityUpdateInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                                                     const Models::Base::VarVec &vars, const std::string &count,
                                                                     VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(size_t) const) const
{
    // Loop through variables
    for (size_t i = 0; i < vars.size(); i++) {
        // If var is located on the host
        const auto loc = (getArchetype().*getVarLocationFn)(i);
        if (loc & VarLocation::HOST) {
            // Generate code to push this variable
            // **YUCK** these EGP functions should probably just be called dynamic or something
            std::stringstream pushStream;
            CodeStream push(pushStream);
            backend.genExtraGlobalParamPush(push, vars[i].type, vars[i].name,
                                            loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("push" + vars[i].name, 0, pushStream.str());

            // Generate code to pull this variable
            // **YUCK** these EGP functions should probably just be called dynamic or something
            std::stringstream pullStream;
            CodeStream pull(pullStream);
            backend.genExtraGlobalParamPull(pull, vars[i].type, vars[i].name,
                                            loc, count, "group->");

            // Add substitution
            subs.addFuncSubstitution("pull" + vars[i].name, 0, pullStream.str());
        }
    }
}
//----------------------------------------------------------------------------
void CustomConnectivityHostUpdateGroupMerged::addVars(const BackendBase &backend, const Models::Base::VarVec &vars,
                                                      VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(size_t) const)
{
    // Loop through variables
    for (size_t i = 0; i < vars.size(); i++) {
        // If var is located on the host
        const auto var = vars[i];
        if ((getArchetype().*getVarLocationFn)(i) & VarLocation::HOST) {
            addField(var.type + "*", var.name,
                    [var](const CustomConnectivityUpdateInternal &g, size_t) { return var.name + g.getName(); },
                    FieldType::Host);

            if(!backend.getDeviceVarPrefix().empty()) {
                addField(var.type + "*", backend.getDeviceVarPrefix() + var.name,
                         [var, &backend](const CustomConnectivityUpdateInternal &g, size_t)
                         {
                             return backend.getDeviceVarPrefix() + var.name + g.getName();
                         });
            }
        }
    }
}
