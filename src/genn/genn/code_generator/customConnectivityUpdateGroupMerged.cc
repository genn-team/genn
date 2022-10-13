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
void CustomConnectivityUpdateGroupMerged::generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{

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

    // **TODO** add device and host pre and post vars; var refs and EGPs

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
        subs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                       [this](size_t p) { return isParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                     [this](size_t p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
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

        // **TODO** move into helper addPushPullFuncSubs
        const auto preVars = cm->getPreVars();
        for (size_t i = 0; i < preVars.size(); i++) {
            // If var is located on the host
            const auto loc = getArchetype().getPreVarLocation(i);
            if (loc & VarLocation::HOST) {
                // Generate code to push this variable
                // **YUCK** these EGP functions should probably just be called dynamic or something
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genExtraGlobalParamPush(push, preVars[i].type, preVars[i].name,
                                                loc, "group->numSrcNeurons", "group->");

                // Add substitution
                subs.addFuncSubstitution("push" + preVars[i].name, 0, pushStream.str());

                // Generate code to pull this variable
                // **YUCK** these EGP functions should probably just be called dynamic or something
                std::stringstream pullStream;
                CodeStream pull(pullStream);
                backend.genExtraGlobalParamPull(pull, preVars[i].type, preVars[i].name,
                                                loc, "group->numSrcNeurons", "group->");

                // Add substitution
                subs.addFuncSubstitution("pull" + preVars[i].name, 0, pullStream.str());
            }
        }
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