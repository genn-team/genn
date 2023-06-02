#include "code_generator/neuronUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::CurrentSource
//----------------------------------------------------------------------------
// **TODO**
// * field suffix (string) and value suffix (function to get suffix from group) common to everything in group - GroupMerged fields?
// * without nasty combined groups, getParams and getDerivedParams functions can use pointers to members
// * pre and post neuron stuff in synapse update group merged can also be child classes
NeuronUpdateGroupMerged::CurrentSource::CurrentSource(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                      const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups)
:   GroupMerged<CurrentSourceInternal>(index, typeContext, groups)
{
    const std::string suffix =  "CS" + std::to_string(getIndex());

    // Add variables
    for(const auto &var : getArchetype().getCurrentSourceModel()->getVars()) {
        addPointerField(var.type, var.name + suffix, 
                        backend.getDeviceVarPrefix() + var.name);
    }
    
    // Add parameters and derived parameters
    addHeterogeneousParams<CurrentSource>(
        getArchetype().getCurrentSourceModel()->getParamNames(), suffix,
        [](const auto &cs) { return cs.getParams(); },
        &CurrentSource::isParamHeterogeneous);
    addHeterogeneousDerivedParams<CurrentSource>(
        getArchetype().getCurrentSourceModel()->getDerivedParams(), suffix,
        [](const auto &cs) { return cs.getDerivedParams(); },
        &CurrentSource::isDerivedParamHeterogeneous);

    // Add EGPs
    for(const auto &egp : getArchetype().getCurrentSourceModel()->getExtraGlobalParams()) {
        addPointerField(egp.type, egp.name + suffix, 
                        backend.getDeviceVarPrefix() + egp.name,
                        GroupMergedFieldType::DYNAMIC);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::CurrentSource::generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    os << "// current source " << getIndex() << std::endl;

    // Read current source variables into registers
    const std::string suffix =  "CS" + std::to_string(getIndex());
    for(const auto &v : getArchetype().getCurrentSourceModel()->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type.resolve(getTypeContext()).getName() << " lcs" << v.name << " = " << "group->" << v.name << suffix << "[";
        os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
    }

    Substitutions currSourceSubs(&popSubs);
    currSourceSubs.addFuncSubstitution("injectCurrent", 1, "Isyn += $(0)");
    currSourceSubs.addVarNameSubstitution(getArchetype().getCurrentSourceModel()->getVars(), "", "lcs");
    currSourceSubs.addParamValueSubstitution(getArchetype().getCurrentSourceModel()->getParamNames(), getArchetype().getParams(),
                                             [this](const std::string &p) { return isParamHeterogeneous(p);  },
                                             "", "group->", suffix);
    currSourceSubs.addVarValueSubstitution(getArchetype().getCurrentSourceModel()->getDerivedParams(), getArchetype().getDerivedParams(),
                                           [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  },
                                           "", "group->", suffix);
    currSourceSubs.addVarNameSubstitution(getArchetype().getCurrentSourceModel()->getExtraGlobalParams(), "", "group->", suffix);

    std::string iCode = getArchetype().getCurrentSourceModel()->getInjectionCode();
    currSourceSubs.applyCheckUnreplaced(iCode, "injectionCode : merged" + getIndex());
    //iCode = ensureFtype(iCode, model.getPrecision());
    os << iCode << std::endl;

    // Write read/write variables back to global memory
    for(const auto &v : getArchetype().getCurrentSourceModel()->getVars()) {
        if(v.access & VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << suffix << "[";
            os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), currSourceSubs["id"]);
            os << "] = lcs" << v.name << ";" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::CurrentSource::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash<CurrentSource>(&CurrentSource::isParamReferenced, 
                                   [](const CurrentSourceInternal &g) { return g.getParams(); }, hash);
    updateParamHash<CurrentSource>(&CurrentSource::isParamReferenced, 
                                   [](const CurrentSourceInternal &g) { return g.getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::CurrentSource::isParamHeterogeneous(const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const CurrentSourceInternal &cs) { return cs.getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::CurrentSource::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const CurrentSourceInternal &cs) { return cs.getDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::CurrentSource::isParamReferenced(const std::string &paramName) const
{
    return GroupMerged<CurrentSourceInternal>::isParamReferenced({getArchetype().getCurrentSourceModel()->getInjectionCode()},
                                                                 paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynPSM
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::InSynPSM::InSynPSM(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                            const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "InSyn" + std::to_string(getIndex());

    // Add pointer to insyn
    addField(getScalarType().createPointer(), "inSyn" + suffix,
             [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "inSyn" + g.getFusedPSVarSuffix(); });
    
    // Add pointer to dendritic delay buffer if required
    if(getArchetype().isDendriticDelayRequired()) {
        addField(getScalarType().createPointer(), "denDelay" + suffix,
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelay" + g.getFusedPSVarSuffix(); });

        addField(Type::Uint32.createPointer(), "denDelayPtr" + suffix,
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelayPtr" + g.getFusedPSVarSuffix(); });
    }

    // Add pointers to state variable
    // **FUSE**
    for(const auto &var : getArchetype().getPSModel()->getVars()) {
        addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedPSVarSuffix(); });
    }

    // Add any heterogeneous postsynaptic model parameters
    addHeterogeneousParams<InSynPSM>(
        getArchetype().getPSModel()->getParamNames(), suffix,
        [](const auto &sg) { return sg.getPSParams(); },
        &InSynPSM::isParamHeterogeneous);

    // Add any heterogeneous postsynaptic mode derived parameters
    addHeterogeneousDerivedParams<InSynPSM>(
        getArchetype().getPSModel()->getDerivedParams(), suffix,
        [](const auto &sg) { return sg.getPSDerivedParams(); },
        &InSynPSM::isDerivedParamHeterogeneous);

    // Add EGPs
    for(const auto &egp : getArchetype().getPSModel()->getExtraGlobalParams()) {
        addField(egp.type.resolve(getTypeContext()).createPointer(), egp.name + suffix,
                [&backend, egp](const auto &g, size_t) { return backend.getDeviceVarPrefix() + egp.name + g.getFusedPSVarSuffix(); },
                GroupMergedFieldType::DYNAMIC);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                 const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "InSyn" + std::to_string(getIndex());
    const auto *psm = getArchetype().getPSModel();

    os << "// pull inSyn values in a coalesced access" << std::endl;
    os << "scalar linSyn = group->inSynInSyn" << getIndex() << "[";
    os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, popSubs["id"]);
    os << "];" << std::endl;

    // If dendritic delay is required
    if (getArchetype().isDendriticDelayRequired()) {
        // Get reference to dendritic delay buffer input for this timestep
        os << backend.getPointerPrefix() << "scalar *denDelayFront = ";
        os << "&group->denDelay" << suffix << "[(*group->denDelayPtr" << suffix << " * group->numNeurons) + ";
        os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, popSubs["id"]);
        os << "];" << std::endl;

        // Add delayed input from buffer into inSyn
        os << "linSyn += *denDelayFront;" << std::endl;

        // Zero delay buffer slot
        os << "*denDelayFront = " << modelMerged.scalarExpr(0.0) << ";" << std::endl;
    }

    // Pull postsynaptic model variables in a coalesced access
    for (const auto &v : psm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type.resolve(getTypeContext()).getName() << " lps" << v.name << " = group->" << v.name << suffix << "[";
        os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]);
        os << "];" << std::endl;
    }

    Substitutions inSynSubs(&popSubs);
    inSynSubs.addVarSubstitution("inSyn", "linSyn");
        
    // Allow synapse group's PS output var to override what Isyn points to
    inSynSubs.addVarSubstitution("Isyn", getArchetype().getPSTargetVar(), true);
    inSynSubs.addVarNameSubstitution(psm->getVars(), "", "lps");

    inSynSubs.addParamValueSubstitution(psm->getParamNames(), getArchetype().getPSParams(),
                                        [this](const std::string &p) { return isParamHeterogeneous(p);  },
                                        "", "group->", suffix);
    inSynSubs.addVarValueSubstitution(psm->getDerivedParams(), getArchetype().getPSDerivedParams(),
                                      [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  },
                                      "", "group->", suffix);
    inSynSubs.addVarNameSubstitution(psm->getExtraGlobalParams(), "", "group->", suffix);

    // Apply substitutions to current converter code
    std::string psCode = psm->getApplyInputCode();
    inSynSubs.applyCheckUnreplaced(psCode, "postSyntoCurrent : merged " + getIndex());
    //psCode = ensureFtype(psCode, model.getPrecision());

    // Apply substitutions to decay code
    std::string pdCode = psm->getDecayCode();
    inSynSubs.applyCheckUnreplaced(pdCode, "decayCode : merged " + getIndex());
    //pdCode = ensureFtype(pdCode, model.getPrecision());

    if (!psm->getSupportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()) <<  ";" << std::endl;
    }

    if (!psm->getSupportCode().empty() && !backend.supportsNamespace()) {
        psCode = disambiguateNamespaceFunction(psm->getSupportCode(), psCode, modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()));
        pdCode = disambiguateNamespaceFunction(psm->getSupportCode(), pdCode, modelMerged.getPostsynapticDynamicsSupportCodeNamespace(psm->getSupportCode()));
    }

    os << psCode << std::endl;
    os << pdCode << std::endl;

    if (!psm->getSupportCode().empty()) {
        os << CodeStream::CB(29) << " // namespace bracket closed" << std::endl;
    }

    // Write back linSyn
    os << "group->inSyn" << suffix << "[";
    os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, inSynSubs["id"]);
    os << "] = linSyn;" << std::endl;

    // Copy any non-readonly postsynaptic model variables back to global state variables dd_V etc
    for (const auto &v : psm->getVars()) {
        if(v.access & VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << suffix << "[";
            os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), inSynSubs["id"]);
            os << "]" << " = lps" << v.name << ";" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash<InSynPSM>(&InSynPSM::isParamReferenced, 
                               [](const SynapseGroupInternal &g) { return g.getPSParams(); }, hash);
    updateParamHash<InSynPSM>(&InSynPSM::isParamReferenced, 
                              [](const SynapseGroupInternal &g) { return g.getPSDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isParamHeterogeneous(const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isParamReferenced(const std::string &paramName) const
{
    return GroupMerged<SynapseGroupInternal>::isParamReferenced(
        {getArchetype().getPSModel()->getApplyInputCode(), getArchetype().getPSModel()->getDecayCode()},
        paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::OutSynPreOutput::OutSynPreOutput(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                          const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());

    addField(getScalarType().createPointer(), "revInSyn" + suffix,
             [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "revInSyn" + g.getFusedPreOutputSuffix(); });
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynPreOutput::generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                        const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());
     
    os << getArchetype().getPreTargetVar() << "+= ";
    os << "group->revInSyn" << suffix << "[";
    os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, popSubs["id"]);
    os << "];" << std::endl;
    os << "group->revInSyn" << suffix << "[";
    os << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, popSubs["id"]);
    os << "]= " << modelMerged.scalarExpr(0.0) << ";" << std::endl;
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::InSynWUMPostCode::InSynWUMPostCode(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                            const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());

    // Add postsynaptic variables
    for(const auto &var : getArchetype().getWUModel()->getPostVars()) {
        addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                 [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedWUPostVarSuffix(); });
    }
    
    // Add parameters and derived parameters
    addHeterogeneousParams<InSynWUMPostCode>(
        getArchetype().getWUModel()->getParamNames(), suffix,
        [](const auto &sg) { return sg.getWUParams(); },
        &InSynWUMPostCode::isParamHeterogeneous);
    addHeterogeneousDerivedParams<InSynWUMPostCode>(
        getArchetype().getWUModel()->getDerivedParams(), suffix,
        [](const auto &sg) { return sg.getWUDerivedParams(); },
        &InSynWUMPostCode::isDerivedParamHeterogeneous);

    // Add EGPs
    for(const auto &egp : getArchetype().getWUModel()->getExtraGlobalParams()) {
        addField(egp.type.resolve(getTypeContext()).createPointer(), egp.name + suffix,
                 [&backend, egp](const auto &g, size_t) { return backend.getDeviceVarPrefix() + egp.name + g.getFusedWUPostVarSuffix(); },
                 GroupMergedFieldType::DYNAMIC);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, Substitutions &popSubs, bool dynamicsNotSpike) const
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // If this code string isn't empty
    std::string code = dynamicsNotSpike ? getArchetype().getWUModel()->getPostDynamicsCode() : getArchetype().getWUModel()->getPostSpikeCode();
    if(!code.empty()) {
        Substitutions subs(&popSubs);

        // Fetch postsynaptic variables from global memory
        os << "// perform WUM update required for merged" << getIndex() << std::endl;
        const auto vars = getArchetype().getWUModel()->getPostVars();
        const bool delayed = (getArchetype().getBackPropDelaySteps() != NO_DELAY);
        for(const auto &v : vars) {
            if(v.access & VarAccessMode::READ_ONLY) {
                os << "const ";
            }
            os << v.type.resolve(getTypeContext()).getName() << " l" << v.name << " = group->" << v.name << suffix << "[";
            os << ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "];" << std::endl;
        }

        subs.addParamValueSubstitution(getArchetype().getWUModel()->getParamNames(), getArchetype().getWUParams(),
                                       [this](const std::string &p) { return isParamHeterogeneous(p); },
                                       "", "group->", suffix);
        subs.addVarValueSubstitution(getArchetype().getWUModel()->getDerivedParams(), getArchetype().getWUDerivedParams(),
                                     [this](const std::string &p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->", suffix);
        subs.addVarNameSubstitution(getArchetype().getWUModel()->getExtraGlobalParams(), "", "group->", suffix);
        subs.addVarNameSubstitution(vars, "", "l");

        neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), "", "_post", "", "", "", dynamicsNotSpike,
                                          [&ng](const std::string &p) { return ng.isParamHeterogeneous(p); },
                                          [&ng](const std::string &p) { return ng.isDerivedParamHeterogeneous(p); },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          {
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          { 
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          });

        // Perform standard substitutions
        subs.applyCheckUnreplaced(code, "spikeCode : merged" + getIndex());
        //code = ensureFtype(code, precision);
        os << code;

        // Write back postsynaptic variables into global memory
        for(const auto &v : vars) {
            // If state variables is read/write - meaning that it may have been updated - or it is delayed -
            // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
            // back to global state variables dd_V etc
            if((v.access & VarAccessMode::READ_WRITE) || delayed) {
                os << "group->" << v.name << suffix << "[";
                os << ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "] = l" << v.name << ";" << std::endl;
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::genCopyDelayedVars(CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    if(getArchetype().getBackPropDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPostDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPostVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                os << "group->" << v.name << suffix << "[";
                os << ng.getWriteVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]);
                os << "] = ";

                os << "group->" << v.name << suffix << "[";
                os << ng.getReadVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]);
                os << "];" << std::endl;
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash<InSynWUMPostCode>(&InSynWUMPostCode::isParamReferenced, 
                                      [](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
    updateParamHash<InSynWUMPostCode>(&InSynWUMPostCode::isParamReferenced, 
                                      [](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isParamHeterogeneous(const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isParamReferenced(const std::string &paramName) const
{
    return GroupMerged<SynapseGroupInternal>::isParamReferenced(
        {getArchetype().getWUModel()->getPostDynamicsCode(), getArchetype().getWUModel()->getPostSpikeCode()},
        paramName);
}

 //----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::OutSynWUMPreCode::OutSynWUMPreCode(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                            const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());

    // Add presynaptic variables
    for(const auto &var : getArchetype().getWUModel()->getPreVars()) {
        addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                 [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedWUPreVarSuffix(); });
    }
    
    // Add parameters and derived parameters
    addHeterogeneousParams<OutSynWUMPreCode>(
        getArchetype().getWUModel()->getParamNames(), suffix,
        [](const auto &sg) { return sg.getWUParams(); },
        &OutSynWUMPreCode::isParamHeterogeneous);
    addHeterogeneousDerivedParams<OutSynWUMPreCode>(
        getArchetype().getWUModel()->getDerivedParams(), suffix,
        [](const auto &sg) { return sg.getWUDerivedParams(); },
        &OutSynWUMPreCode::isDerivedParamHeterogeneous);

    // Add EGPs
    for(const auto &egp : getArchetype().getWUModel()->getExtraGlobalParams()) {
        addField(egp.type.resolve(getTypeContext()).createPointer(), egp.name + suffix,
                 [&backend, egp](const auto &g, size_t) { return backend.getDeviceVarPrefix() + egp.name + g.getFusedWUPreVarSuffix(); },
                 GroupMergedFieldType::DYNAMIC);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, Substitutions &popSubs, bool dynamicsNotSpike) const
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // If this code string isn't empty
    std::string code = dynamicsNotSpike ? getArchetype().getWUModel()->getPreDynamicsCode() : getArchetype().getWUModel()->getPreSpikeCode();
    if(!code.empty()) {
        Substitutions subs(&popSubs);

        // Fetch presynaptic variables from global memory
        os << "// perform WUM update required for merged" << getIndex() << std::endl;
        const auto vars = getArchetype().getWUModel()->getPreVars();
        const bool delayed = (getArchetype().getDelaySteps() != NO_DELAY);
        for(const auto &v : vars) {
            if(v.access & VarAccessMode::READ_ONLY) {
                os << "const ";
            }
            os << v.type.resolve(getTypeContext()).getName() << " l" << v.name << " = group->" << v.name << suffix << "[";
            os << ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "];" << std::endl;
        }

        subs.addParamValueSubstitution(getArchetype().getWUModel()->getParamNames(), getArchetype().getWUParams(),
                                       [this](const std::string &p) { return isParamHeterogeneous(p); },
                                       "", "group->", suffix);
        subs.addVarValueSubstitution(getArchetype().getWUModel()->getDerivedParams(), getArchetype().getWUDerivedParams(),
                                     [this](const std::string &p) { return isDerivedParamHeterogeneous(p); },
                                     "", "group->", suffix);
        subs.addVarNameSubstitution(getArchetype().getWUModel()->getExtraGlobalParams(), "", "group->", suffix);
        subs.addVarNameSubstitution(vars, "", "l");

        neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), "", "_pre", "", "", "", dynamicsNotSpike,
                                          [&ng](const std::string &p) { return ng.isParamHeterogeneous(p); },
                                          [&ng](const std::string &p) { return ng.isDerivedParamHeterogeneous(p); },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          {
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          { 
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          });

        // Perform standard substitutions
        subs.applyCheckUnreplaced(code, "spikeCode : merged" + getIndex());
        //code = ensureFtype(code, precision);
        os << code;

        // Write back presynaptic variables into global memory
        for(const auto &v : vars) {
            // If state variables is read/write - meaning that it may have been updated - or it is delayed -
            // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
            // back to global state variables dd_V etc
            if((v.access & VarAccessMode::READ_WRITE) || delayed) {
                os << "group->" << v.name << suffix << "[";
                os << ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), subs["id"]) << "] = l" << v.name << ";" << std::endl;
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::genCopyDelayedVars(CodeStream &os, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    if(getArchetype().getDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPreDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPreVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                os << "group->" << v.name << suffix << "[";
                os << ng.getWriteVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]);
                os << "] = ";

                os << "group->" << v.name << suffix << "[";
                os << ng.getReadVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), popSubs["id"]);
                os << "];" << std::endl;
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash<OutSynWUMPreCode>(&OutSynWUMPreCode::isParamReferenced, 
                                      [](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
    updateParamHash<OutSynWUMPreCode>(&OutSynWUMPreCode::isParamReferenced, 
                                      [](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isParamHeterogeneous(const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return (isParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); }));
 
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isParamReferenced(const std::string &paramName) const
{
    return GroupMerged<SynapseGroupInternal>::isParamReferenced(
        {getArchetype().getWUModel()->getPreDynamicsCode(), getArchetype().getWUModel()->getPreSpikeCode()},
        paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronUpdateGroupMerged::name = "NeuronUpdate";
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                                 const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, backend, groups)
{
    using namespace Type;

    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, typeContext, backend,
                             &NeuronGroupInternal::getFusedPSMInSyn,
                             &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, typeContext, backend, 
                             &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, typeContext, backend,
                             &NeuronGroupInternal::getCurrentSources,
                             &CurrentSourceInternal::getHashDigest);


    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostCodeGroups, typeContext, backend,
                             &NeuronGroupInternal::getFusedInSynWithPostCode,
                             &SynapseGroupInternal::getWUPostHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreCodeGroups, typeContext, backend, 
                             &NeuronGroupInternal::getFusedOutSynWithPreCode,
                             &SynapseGroupInternal::getWUPreHashDigest);

    if(backend.isPopulationRNGRequired() && getArchetype().isSimRNGRequired()) {
        addPointerField(*backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rng");
    }

    // Add variables and extra global parameters
    addVars(getArchetype().getNeuronModel()->getVars(), backend.getDeviceVarPrefix());
    addEGPs(getArchetype().getNeuronModel()->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    // Add heterogeneous neuron model parameters
    addHeterogeneousParams<NeuronGroupMergedBase>(
        getArchetype().getNeuronModel()->getParamNames(), "",
        [](const NeuronGroupInternal &ng) { return ng.getParams(); },
        &NeuronGroupMergedBase::isParamHeterogeneous);

    // Add heterogeneous neuron model derived parameters
    addHeterogeneousDerivedParams<NeuronGroupMergedBase>(
        getArchetype().getNeuronModel()->getDerivedParams(), "",
        [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); },
        &NeuronGroupMergedBase::isDerivedParamHeterogeneous);

    // Loop through neuron groups
    std::vector<std::vector<SynapseGroupInternal *>> eventThresholdSGs;
    for(const auto &g : getGroups()) {
        // Reserve vector for this group's children
        eventThresholdSGs.emplace_back();

        // Add synapse groups 
        for(const auto &s : g.get().getSpikeEventCondition()) {
            if(s.synapseStateInThresholdCode) {
                eventThresholdSGs.back().push_back(s.synapseGroup);
            }
        }
    }

    // Loop through all spike event conditions
    size_t i = 0;
    for(const auto &s : getArchetype().getSpikeEventCondition()) {
        // If threshold condition references any synapse state
        if(s.synapseStateInThresholdCode) {
            const auto wum = s.synapseGroup->getWUModel();

            // Loop through all EGPs in synapse group 
            const auto sgEGPs = wum->getExtraGlobalParams();
            for(const auto &egp : sgEGPs) {
                // If EGP is referenced in event threshold code
                if(s.eventThresholdCode.find("$(" + egp.name + ")") != std::string::npos) {
                    const std::string prefix = backend.getDeviceVarPrefix();
                    addField(egp.type.resolve(getTypeContext()).createPointer(), egp.name + "EventThresh" + std::to_string(i),
                             [eventThresholdSGs, prefix, egp, i](const auto &, size_t groupIndex)
                             {
                                 return prefix + egp.name + eventThresholdSGs.at(groupIndex).at(i)->getName();
                             },
                             GroupMergedFieldType::DYNAMIC);
                }
            }

            // Loop through all presynaptic variables in synapse group 
            const auto sgPreVars = wum->getPreVars();
            for(const auto &var : sgPreVars) {
                // If variable is referenced in event threshold code
                if(s.eventThresholdCode.find("$(" + var.name + ")") != std::string::npos) {
                    addField(var.type.resolve(getTypeContext()).createPointer(), var.name + "EventThresh" + std::to_string(i),
                             [&backend, eventThresholdSGs, var, i](const auto&, size_t groupIndex)
                             {
                                 return backend.getDeviceVarPrefix() + var.name + eventThresholdSGs.at(groupIndex).at(i)->getName();
                             });
                }
            }
            i++;
        }
    }

    if(getArchetype().isSpikeRecordingEnabled()) {
        // Add field for spike recording
        addField(Uint32.createPointer(), "recordSpk",
                 [&backend](const auto &ng, size_t) 
                 { 
                     return backend.getDeviceVarPrefix() + "recordSpk" + ng.getName(); 
                 },
                 GroupMergedFieldType::DYNAMIC);
    }

    if(getArchetype().isSpikeEventRecordingEnabled()) {
        // Add field for spike event recording
        addField(Uint32.createPointer(), "recordSpkEvent",
                 [&backend](const auto &ng, size_t)
                 {
                     return backend.getDeviceVarPrefix() + "recordSpkEvent" + ng.getName(); 
                 },
                 GroupMergedFieldType::DYNAMIC);
    }

}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with each group's neuron count
    updateHash([](const NeuronGroupInternal &g) { return g.getNumNeurons(); }, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with each group's parameters and derived parameters
    updateHash([](const NeuronGroupInternal &g) { return g.getParams(); }, hash);
    updateHash([](const NeuronGroupInternal &g) { return g.getDerivedParams(); }, hash);
    
    // Update hash with child groups
    for (const auto &cs : getMergedCurrentSourceGroups()) {
        cs.updateHash(hash);
    }
    for(const auto &sg : getMergedInSynPSMGroups()) {
        sg.updateHash(hash);
    }
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        sg.updateHash(hash);
    }
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        sg.updateHash(hash);
    }

    return hash.get_digest();
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateNeuronUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs,
                                                   BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitTrueSpike,
                                                   BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();

    // Generate code to copy neuron state into local variable
    for(const auto &v : nm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type.resolve(getTypeContext()).getName() << " l" << v.name << " = group->" << v.name << "[";
        const bool delayed = (getArchetype().isVarQueueRequired(v.name) && getArchetype().isDelayRequired());
        os << getReadVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "];" << std::endl;
    }

    // Also read spike and spike-like-event times into local variables if required
    if(getArchetype().isSpikeTimeRequired()) {
        os << "const timepoint lsT = group->sT[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isPrevSpikeTimeRequired()) {
        os << "const timepoint lprevST = group->prevST[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        os << "const timepoint lseT = group->seT[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        os <<  "const timepoint lprevSET = group->prevSET[";
        os << getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "];" << std::endl;
    }
    os << std::endl;

    // If neuron model sim code references ISyn (could still be the case if there are no incoming synapses)
    // OR any incoming synapse groups have post synaptic models which reference $(Isyn), declare it
    if (nm->getSimCode().find("$(Isyn)") != std::string::npos ||
        std::any_of(getArchetype().getFusedPSMInSyn().cbegin(), getArchetype().getFusedPSMInSyn().cend(),
                    [](const SynapseGroupInternal *sg)
                    {
                        return (sg->getPSModel()->getApplyInputCode().find("$(Isyn)") != std::string::npos
                                || sg->getPSModel()->getDecayCode().find("$(Isyn)") != std::string::npos);
                    }))
    {
        os << "scalar Isyn = 0;" << std::endl;
    }

    Substitutions neuronSubs(&popSubs);
    neuronSubs.addVarSubstitution("Isyn", "Isyn");

    if(getArchetype().isSpikeTimeRequired()) {
        neuronSubs.addVarSubstitution("sT", "lsT");
    }
    if(getArchetype().isPrevSpikeTimeRequired()) {
        neuronSubs.addVarSubstitution("prev_sT", "lprevST");
    }
    if(getArchetype().isSpikeEventTimeRequired()) {
        neuronSubs.addVarSubstitution("seT", "lseT");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        neuronSubs.addVarSubstitution("prev_seT", "lprevSET");
    }
    neuronSubs.addVarNameSubstitution(nm->getAdditionalInputVars());
    addNeuronModelSubstitutions(neuronSubs);

    // Initialise any additional input variables supported by neuron model
    for (const auto &a : nm->getAdditionalInputVars()) {
        // Apply substitutions to value
        std::string value = a.value;
        neuronSubs.applyCheckUnreplaced(value, "neuron additional input var : merged" + std::to_string(getIndex()));
        //value = ensureFtype(value, modelMerged.getModel().getPrecision());

        os << a.type.resolve(getTypeContext()).getName() << " " << a.name << " = " << value << ";" << std::endl;
    }

    // Loop through incoming synapse groups
    for(const auto &sg : getMergedInSynPSMGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }

    // Loop through outgoing synapse groups with presynaptic output
    for (const auto &sg : getMergedOutSynPreOutputGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }
 
    // Loop through all of neuron group's current sources
    for (const auto &cs : getMergedCurrentSourceGroups()) {
        CodeStream::Scope b(os);
        cs.generate(backend, os, *this, modelMerged, popSubs);
    }

    if (!nm->getSupportCode().empty() && backend.supportsNamespace()) {
        os << "using namespace " << modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()) <<  ";" << std::endl;
    }

    // If a threshold condition is provided
    std::string thCode = nm->getThresholdConditionCode();
    if (!thCode.empty()) {
        os << "// test whether spike condition was fulfilled previously" << std::endl;

        neuronSubs.applyCheckUnreplaced(thCode, "thresholdConditionCode : merged" + std::to_string(getIndex()));
        //thCode= ensureFtype(thCode, model.getPrecision());

        if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
            thCode = disambiguateNamespaceFunction(nm->getSupportCode(), thCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
        }

        if (nm->isAutoRefractoryRequired()) {
            os << "const bool oldSpike = (" << thCode << ");" << std::endl;
        }
    }
    // Otherwise, if any outgoing synapse groups have spike-processing code
    /*else if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                        [](const SynapseGroupInternal *sg){ return !sg->getWUModel()->getSimCode().empty(); }))
    {
        LOGW_CODE_GEN << "No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << getName() << "\" was provided. There will be no spikes detected in this population!";
    }*/

    os << "// calculate membrane potential" << std::endl;
    std::string sCode = nm->getSimCode();
    neuronSubs.applyCheckUnreplaced(sCode, "simCode : merged" + std::to_string(getIndex()));
    //sCode = ensureFtype(sCode, model.getPrecision());

    if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
        sCode = disambiguateNamespaceFunction(nm->getSupportCode(), sCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
    }

    os << sCode << std::endl;

    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs, true);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs, true);
    }

    // look for spike type events first.
    if (getArchetype().isSpikeEventRequired()) {
        // Create local variable
        os << "bool spikeLikeEvent = false;" << std::endl;

        // Loop through outgoing synapse populations that will contribute to event condition code
        size_t i = 0;
        for(const auto &spkEventCond : getArchetype().getSpikeEventCondition()) {
            // Replace of parameters, derived parameters and extraglobalsynapse parameters
            Substitutions spkEventCondSubs(&popSubs);

            // If this spike event condition requires synapse state
            if(spkEventCond.synapseStateInThresholdCode) {
                // Substitute EGPs
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getExtraGlobalParams(), "", "group->", "EventThresh" + std::to_string(i));

                // Substitute presynaptic variables
                const bool delayed = (spkEventCond.synapseGroup->getDelaySteps() != NO_DELAY);
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUModel()->getPreVars(), "", "group->",
                                                        [&popSubs, batchSize, delayed, i, this](VarAccess a, const std::string&)
                                                        { 
                                                            return "EventThresh" + std::to_string(i) + "[" + getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), popSubs["id"]) + "]";
                                                        });
                i++;
            }
            addNeuronModelSubstitutions(spkEventCondSubs, "_pre");

            std::string eCode = spkEventCond.eventThresholdCode;
            spkEventCondSubs.applyCheckUnreplaced(eCode, "neuronSpkEvntCondition : merged" + std::to_string(getIndex()));
            //eCode = ensureFtype(eCode, model.getPrecision());

            // Open scope for spike-like event test
            os << CodeStream::OB(31);

            // Use presynaptic update namespace if required
            if (!spkEventCond.supportCode.empty() && backend.supportsNamespace()) {
                os << "using namespace " << modelMerged.getPresynapticUpdateSupportCodeNamespace(spkEventCond.supportCode) << ";" << std::endl;
            }

            // Substitute with namespace functions
            if (!spkEventCond.supportCode.empty() && !backend.supportsNamespace()) {
                eCode = disambiguateNamespaceFunction(spkEventCond.supportCode, eCode, modelMerged.getPresynapticUpdateSupportCodeNamespace(spkEventCond.supportCode));
            }

            // Combine this event threshold test with
            os << "spikeLikeEvent |= (" << eCode << ");" << std::endl;

            // Close scope for spike-like event test
            os << CodeStream::CB(31);
        }

        os << "// register a spike-like event" << std::endl;
        os << "if (spikeLikeEvent)";
        {
            CodeStream::Scope b(os);
            genEmitSpikeLikeEvent(os, *this, popSubs);
        }

        // If spike-like-event timing is required and they aren't updated after update, copy spike-like-event time from register
        if(getArchetype().isDelayRequired() && (getArchetype().isSpikeEventTimeRequired() || getArchetype().isPrevSpikeEventTimeRequired())) {
            os << "else";
            CodeStream::Scope b(os);

            if(getArchetype().isSpikeEventTimeRequired()) {
                os << "group->seT[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lseT;" << std::endl;
            }
            if(getArchetype().isPrevSpikeEventTimeRequired()) {
                os << "group->prevSET[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevSET;" << std::endl;
            }
        }
    }

    // test for true spikes if condition is provided
    if (!thCode.empty()) {
        os << "// test for and register a true spike" << std::endl;
        if (nm->isAutoRefractoryRequired()) {
            os << "if ((" << thCode << ") && !(oldSpike))";
        }
        else {
            os << "if (" << thCode << ")";
        }
        {
            CodeStream::Scope b(os);
            genEmitTrueSpike(os, *this, popSubs);

            // add after-spike reset if provided
            if (!nm->getResetCode().empty()) {
                std::string rCode = nm->getResetCode();
                neuronSubs.applyCheckUnreplaced(rCode, "resetCode : merged" + std::to_string(getIndex()));
                //rCode = ensureFtype(rCode, model.getPrecision());

                os << "// spike reset code" << std::endl;
                os << rCode << std::endl;
            }
        }

        // Spike triggered variables don't need to be copied
        // if delay isn't required as there's only one copy of them
        if(getArchetype().isDelayRequired()) {
            // **FIXME** there is a corner case here where, if pre or postsynaptic variables have no update code
            // but there are delays they won't get copied. It might make more sense (and tidy up several things
            // to instead build merged neuron update groups based on inSynWithPostVars/outSynWithPreVars instead.
            
            // Are there any outgoing synapse groups with presynaptic code
            // which have axonal delay and no presynaptic dynamics
            const bool preVars = std::any_of(getMergedOutSynWUMPreCodeGroups().cbegin(), getMergedOutSynWUMPreCodeGroups().cend(),
                                             [](const OutSynWUMPreCode &sg)
                                             {
                                                 return ((sg.getArchetype().getDelaySteps() != NO_DELAY)
                                                         && sg.getArchetype().getWUModel()->getPreDynamicsCode().empty());
                                             });

            // Are there any incoming synapse groups with postsynaptic code
            // which have back-propagation delay and no postsynaptic dynamics
            const bool postVars = std::any_of(getMergedInSynWUMPostCodeGroups().cbegin(), getMergedInSynWUMPostCodeGroups().cend(),
                                              [](const auto &sg)
                                              {
                                                  return ((sg.getArchetype().getBackPropDelaySteps() != NO_DELAY)
                                                           && sg.getArchetype().getWUModel()->getPostDynamicsCode().empty());
                                              });

            // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
            if(getArchetype().isSpikeTimeRequired() || getArchetype().isPrevSpikeTimeRequired() || preVars || postVars) {
                os << "else";
                CodeStream::Scope b(os);

                // If spike times are required, copy times from register
                if(getArchetype().isSpikeTimeRequired()) {
                    os << "group->sT[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lsT;" << std::endl;
                }

                // If previous spike times are required, copy times from register
                if(getArchetype().isPrevSpikeTimeRequired()) {
                    os << "group->prevST[" << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, popSubs["id"]) << "] = lprevST;" << std::endl;
                }

                // Loop through outgoing synapse groups with some sort of presynaptic code
                for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
                    sg.genCopyDelayedVars(os, *this, modelMerged, popSubs);
                }

                // Loop through outgoing synapse groups with some sort of postsynaptic code
                for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
                    sg.genCopyDelayedVars(os, *this, modelMerged, popSubs);
                }
            }
        }
    }

    // Loop through neuron state variables
    for(const auto &v : nm->getVars()) {
        // If state variables is read/writes - meaning that it may have been updated - or it is delayed -
        // meaning that it needs to be copied into next delay slot whatever - copy neuron state variables
        // back to global state variables dd_V etc  
        const bool delayed = (getArchetype().isVarQueueRequired(v.name) && getArchetype().isDelayRequired());
        if((v.access & VarAccessMode::READ_WRITE) || delayed) {
            os << "group->" << v.name << "[";
            os << getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(v.access), popSubs["id"]) << "] = l" << v.name << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs, false);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(os);
        sg.generate(backend, os, *this, modelMerged, popSubs, false);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
        return (batchSize == 1) ? "0" : "batch";
    }
    else if(varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
        return index;
    }
    else {
        return "batchOffset + " + index;
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    if(delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "readDelaySlot" : "readBatchDelaySlot";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return "readDelayOffset + " + index;
        }
        else {
            return "readBatchDelayOffset + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    if(delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "writeDelaySlot" : "writeBatchDelaySlot";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return "writeDelayOffset + " + index;
        }
        else {
            return "writeBatchDelayOffset + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::addNeuronModelSubstitutions(Substitutions &substitution, const std::string &sourceSuffix, const std::string &destSuffix) const
{
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
    substitution.addVarNameSubstitution(nm->getVars(), sourceSuffix, "l", destSuffix);
    substitution.addParamValueSubstitution(nm->getParamNames(), getArchetype().getParams(), 
                                           [this](const std::string &p) { return isParamHeterogeneous(p);  },
                                           sourceSuffix, "group->");
    substitution.addVarValueSubstitution(nm->getDerivedParams(), getArchetype().getDerivedParams(), 
                                         [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  },
                                         sourceSuffix, "group->");
    substitution.addVarNameSubstitution(nm->getExtraGlobalParams(), sourceSuffix, "group->");
}
