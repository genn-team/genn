#include "code_generator/neuronUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/prettyPrinter.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::CurrentSource
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::CurrentSource::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                      unsigned int batchSize)
{
    const std::string fieldSuffix =  "CS" + std::to_string(getIndex());
    const auto *cm = getArchetype().getCurrentSourceModel();

    // Create new environment to add current source fields to neuron update group 
    EnvironmentGroupMergedField<CurrentSource, NeuronUpdateGroupMerged> csEnv(env, *this, ng);
    
    csEnv.getStream() << "// current source " << getIndex() << std::endl;

    // Substitute parameter and derived parameter names
    csEnv.addParams(cm->getParams(), fieldSuffix, &CurrentSourceInternal::getParams, &CurrentSource::isParamHeterogeneous);
    csEnv.addDerivedParams(cm->getDerivedParams(), fieldSuffix, &CurrentSourceInternal::getDerivedParams, &CurrentSource::isDerivedParamHeterogeneous);
    csEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), "", fieldSuffix);

    // Add neuron variable references
    csEnv.addLocalVarRefs<CurrentSourceNeuronVarRefAdapter>(true);

    // Define inject current function
    csEnv.add(Type::ResolvedType::createFunction(Type::Void, {getScalarType()}), 
              "injectCurrent", "$(_" + getArchetype().getTargetVar() + ") += $(0)");

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CurrentSourceVarAdapter, CurrentSource, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), csEnv, fieldSuffix, "l", false,
        [batchSize, &ng](const std::string&, VarAccess d)
        {
            return ng.getVarIndex(batchSize, getVarAccessDim(d), "$(id)");
        });

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Current source '" + getArchetype().getName() + "' injection code");
    prettyPrintStatements(getArchetype().getInjectionCodeTokens(), getTypeContext(), varEnv, errorHandler);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::CurrentSource::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const CurrentSourceInternal &g) { return g.getParams(); }, hash);
    updateParamHash([](const CurrentSourceInternal &g) { return g.getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::CurrentSource::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const CurrentSourceInternal &cs) { return cs.getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::CurrentSource::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const CurrentSourceInternal &cs) { return cs.getDerivedParams(); });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynPSM
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::generate(const BackendBase &backend, EnvironmentExternalBase &env,
                                                 NeuronUpdateGroupMerged &ng, unsigned int batchSize)
{
    const std::string fieldSuffix =  "InSyn" + std::to_string(getIndex());
    const auto *psm = getArchetype().getPSInitialiser().getSnippet();

    // Create new environment to add PSM fields to neuron update group 
    EnvironmentGroupMergedField<InSynPSM, NeuronUpdateGroupMerged> psmEnv(env, *this, ng);

    // Add inSyn
    psmEnv.addField(getScalarType().createPointer(), "_out_post", "outPost" + fieldSuffix,
                    [](const auto &runtime, const auto &g, size_t){ return runtime.getArray(g.getFusedPSTarget(), "outPost"); });

    // Read into local variable
    const std::string idx = ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    psmEnv.getStream() << "// postsynaptic model " << getIndex() << std::endl;
    psmEnv.printLine(getScalarType().getName() + " linSyn = $(_out_post)[" + idx + "];");

    // If dendritic delay is required
    if (getArchetype().isDendriticDelayRequired()) {
        // Add dendritic delay buffer and pointer into it
        psmEnv.addField(getScalarType().createPointer(), "_den_delay", "denDelay" + fieldSuffix,
                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPSTarget(), "denDelay");});
        psmEnv.addField(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr" + fieldSuffix,
                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPSTarget(), "denDelayPtr");});

        // Get reference to dendritic delay buffer input for this timestep
        psmEnv.printLine(backend.getPointerPrefix() + getScalarType().getName() + " *denDelayFront = &$(_den_delay)[(*$(_den_delay_ptr) * $(num_neurons)) + " + idx + "];");

        // Add delayed input from buffer into inSyn
        psmEnv.getStream() << "linSyn += *denDelayFront;" << std::endl;

        // Zero delay buffer slot
        psmEnv.getStream() << "*denDelayFront = " << Type::writeNumeric(0.0, getScalarType()) << ";" << std::endl;
    }

    // Add parameters, derived parameters and extra global parameters to environment
    psmEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getPSInitialiser, &InSynPSM::isParamHeterogeneous);
    psmEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getPSInitialiser, &InSynPSM::isDerivedParamHeterogeneous);
    psmEnv.addExtraGlobalParams(psm->getExtraGlobalParams(), "", fieldSuffix);
    
    // Add neuron variable references
    psmEnv.addLocalVarRefs<SynapsePSMNeuronVarRefAdapter>(true);

    // **TODO** naming convention
    psmEnv.add(getScalarType(), "inSyn", "linSyn");
        
    // Allow synapse group's PS output var to override what Isyn points to
    psmEnv.add(getScalarType(), "Isyn", "$(_" + getArchetype().getPostTargetVar() + ")");

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<SynapsePSMVarAdapter, InSynPSM, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), psmEnv, fieldSuffix, "l", false,
        [batchSize, &ng](const std::string&, VarAccess d)
        {
            return ng.getVarIndex(batchSize, getVarAccessDim(d), "$(id)");
        });

    // Pretty print code back to environment
    Transpiler::ErrorHandler applyInputErrorHandler("Synapse group '" + getArchetype().getName() + "' postsynaptic model apply input code");
    prettyPrintStatements(getArchetype().getPSInitialiser().getApplyInputCodeTokens(), getTypeContext(), varEnv, applyInputErrorHandler);

    Transpiler::ErrorHandler decayErrorHandler("Synapse group '" + getArchetype().getName() + "' postsynaptic model decay code");
    prettyPrintStatements(getArchetype().getPSInitialiser().getDecayCodeTokens(), getTypeContext(), varEnv, decayErrorHandler);

    // Write back linSyn
    varEnv.printLine("$(_out_post)[" + ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)") + "] = linSyn;");
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getPSInitialiser().getParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getPSInitialiser().getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSInitialiser().getDerivedParams(); });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynPreOutput::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                        unsigned int batchSize)
{
    const std::string fieldSuffix =  "OutSyn" + std::to_string(getIndex());
    
    // Create new environment to add out syn fields to neuron update group 
    EnvironmentGroupMergedField<OutSynPreOutput, NeuronUpdateGroupMerged> outSynEnv(env, *this, ng);
    
    outSynEnv.addField(getScalarType().createPointer(), "_out_pre", "outPre" + fieldSuffix,
                       [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPreOutputTarget(), "outPre"); });

    // Add reverse insyn variable to 
    const std::string idx = ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    outSynEnv.printLine(getArchetype().getPreTargetVar() + " += $(_out_pre)[" + idx + "];");

    // Zero it again
    outSynEnv.printLine("$(_out_pre)[" + idx + "] = " + Type::writeNumeric(0.0, getScalarType()) + ";");
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                         unsigned int batchSize, bool dynamicsNotSpike)
{
    const std::string fieldSuffix =  "InSynWUMPost" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUInitialiser().getSnippet();

    // If there are any statements to execute here
    const auto &tokens = dynamicsNotSpike ? getArchetype().getWUInitialiser().getPostDynamicsCodeTokens() : getArchetype().getWUInitialiser().getPostSpikeCodeTokens();
    if(!Utils::areTokensEmpty(tokens)) {
        // Create new environment to add out syn fields to neuron update group 
        EnvironmentGroupMergedField<InSynWUMPostCode, NeuronUpdateGroupMerged> synEnv(env, *this, ng);

        synEnv.getStream() << "// postsynaptic weight update " << getIndex() << std::endl;
        
        // Add parameters, derived parameters and extra global parameters to environment
        synEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, &InSynWUMPostCode::isParamHeterogeneous);
        synEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, &InSynWUMPostCode::isDerivedParamHeterogeneous);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), "", fieldSuffix);

        // If we're generating dynamics code, add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPostNeuronVarRefAdapter>(true);
  
        // Create an environment which caches variables in local variables if they are accessed
        // **NOTE** always copy variables if synapse group is delayed
        const bool delayed = (getArchetype().getBackPropDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, InSynWUMPostCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, delayed, &synEnv, &ng](const std::string&, VarAccess d)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, delayed, &synEnv, &ng](const std::string&, VarAccess d)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [delayed](const std::string&, VarAccess)
            {
                return delayed;
            });

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model postsynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                                                   unsigned int batchSize)
{
    // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    if(getArchetype().getBackPropDelaySteps() != NO_DELAY && Utils::areTokensEmpty(getArchetype().getWUInitialiser().getPostDynamicsCodeTokens())) {
        // Loop through variables and copy between read and write delay slots
        // **YUCK** this a bit sketchy as fields may not have been added - could add fields here but need to guarantee uniqueness
        for(const auto &v : getArchetype().getWUInitialiser().getSnippet()->getPostVars()) {
            if(getVarAccessMode(v.access) == VarAccessMode::READ_WRITE) {
                const VarAccessDim varDims = getVarAccessDim(v.access);
                env.print("group->" + v.name + suffix + "[" + ng.getWriteVarIndex(true, batchSize, varDims, "$(id)") + "] = ");
                env.printLine("group->" + v.name + suffix + "[" + ng.getReadVarIndex(true, batchSize, varDims, "$(id)") + "];");
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getDerivedParams(); });
}

 //----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                         unsigned int batchSize, bool dynamicsNotSpike)
{
    const std::string fieldSuffix =  "OutSynWUMPre" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUInitialiser().getSnippet();
    
    // If there are any statements to execute here
    const auto &tokens = dynamicsNotSpike ? getArchetype().getWUInitialiser().getPreDynamicsCodeTokens() : getArchetype().getWUInitialiser().getPreSpikeCodeTokens();
    if(!Utils::areTokensEmpty(tokens)) {
        // Create new environment to add out syn fields to neuron update group 
        EnvironmentGroupMergedField<OutSynWUMPreCode, NeuronUpdateGroupMerged> synEnv(env, *this, ng);

        synEnv.getStream() << "// presynaptic weight update " << getIndex() << std::endl;
        
        // Add parameters, derived parameters and extra global parameters to environment
        synEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, &OutSynWUMPreCode::isParamHeterogeneous);
        synEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, &OutSynWUMPreCode::isDerivedParamHeterogeneous);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), "", fieldSuffix);

        // If we're generating dynamics code, add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPreNeuronVarRefAdapter>(true);

        // Create an environment which caches variables in local variables if they are accessed
        // **NOTE** always copy variables if synapse group is delayed
        const bool delayed = (getArchetype().getDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPreVarAdapter, OutSynWUMPreCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, delayed, &ng](const std::string&, VarAccess d)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, delayed, &ng](const std::string&, VarAccess d)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [delayed](const std::string&, VarAccess)
            {
                return delayed;
            });     

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model presynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                                                   unsigned int batchSize)
{
    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    if(getArchetype().getDelaySteps() != NO_DELAY && Utils::areTokensEmpty(getArchetype().getWUInitialiser().getPreDynamicsCodeTokens())) {
        // Loop through variables and copy between read and write delay slots
        // **YUCK** this a bit sketchy as fields may not have been added - could add fields here but need to guarantee uniqueness
        for(const auto &v : getArchetype().getWUInitialiser().getSnippet()->getPreVars()) {
            if(getVarAccessMode(v.access) == VarAccessMode::READ_WRITE) {
                const VarAccessDim varDims = getVarAccessDim(v.access);
                env.print("group->" + v.name + suffix + "[" + ng.getWriteVarIndex(true, batchSize, varDims, "$(id)") + "] = ");
                env.printLine("group->" + v.name + suffix + "[" + ng.getReadVarIndex(true, batchSize, varDims, "$(id)") + "];");
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUInitialiser().getDerivedParams(); });
 
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronUpdateGroupMerged::name = "NeuronUpdate";
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, 
                                                 const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, groups)
{
    // Loop through neuron groups
    /*std::vector<std::vector<SynapseGroupInternal *>> eventThresholdSGs;
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
            const auto wum = s.synapseGroup->getWUInitialiser().getSnippet();

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
    }*/

    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, getTypeContext(), &NeuronGroupInternal::getFusedPSMInSyn, &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, getTypeContext(), &NeuronGroupInternal::getFusedPreOutputOutSyn, &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, getTypeContext(), &NeuronGroupInternal::getCurrentSources, &CurrentSourceInternal::getHashDigest);

    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostCodeGroups, getTypeContext(), &NeuronGroupInternal::getFusedInSynWithPostCode, &SynapseGroupInternal::getWUPostHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreCodeGroups, getTypeContext(), &NeuronGroupInternal::getFusedOutSynWithPreCode, &SynapseGroupInternal::getWUPreHashDigest);
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
    for (const auto &sg : getMergedInSynWUMPostCodeGroups()) {
        sg.updateHash(hash);
    }
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        sg.updateHash(hash);
    }

    return hash.get_digest();
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateNeuronUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize,
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitTrueSpike,
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent)
{
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
 
    // Add default input variable
    // **NOTE** this is hidden as only their chosen target gets exposed to PSM and current source
    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> neuronChildEnv(env, *this);
    neuronChildEnv.add(getScalarType(), "_Isyn", "Isyn",
                       {neuronChildEnv.addInitialiser(getScalarType().getName() + " Isyn = 0;")});

    // Add additional input variables
    // **NOTE** these are hidden as only their chosen target gets exposed to PSM and currnet source
    for (const auto &v : nm->getAdditionalInputVars()) {
        const auto resolvedType = v.type.resolve(getTypeContext());
        neuronChildEnv.add(resolvedType, "_" + v.name, "_" + v.name,
                           {neuronChildEnv.addInitialiser(resolvedType.getName() + " _" + v.name + " = " + Type::writeNumeric(v.value, resolvedType) + ";")});
    }

    // Create an environment which caches neuron variable fields in local variables if they are accessed
    // **NOTE** we do this right at the top so that local copies can be used by child groups
    // **NOTE** always copy variables if variable is delayed
    EnvironmentLocalFieldVarCache<NeuronVarAdapter, NeuronUpdateGroupMerged> neuronChildVarEnv(
        *this, getTypeContext(), neuronChildEnv, "l", true,
        [batchSize, this](const std::string &varName, VarAccess d)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
        },
        [batchSize, this](const std::string &varName, VarAccess d)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
        },
        [this](const std::string &varName, VarAccess)
        {
            return (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
        });


    // Loop through incoming synapse groups
    for(auto &sg : m_MergedInSynPSMGroups) {
        CodeStream::Scope b(neuronChildVarEnv.getStream());
        sg.generate(backend, neuronChildVarEnv, *this, batchSize);
    }

    // Loop through outgoing synapse groups with presynaptic output
    for (auto &sg : m_MergedOutSynPreOutputGroups) {
        CodeStream::Scope b(neuronChildVarEnv.getStream());
        sg.generate(neuronChildVarEnv, *this, batchSize);
    }
 
    // Loop through all of neuron group's current sources
    for (auto &cs : m_MergedCurrentSourceGroups) {
        CodeStream::Scope b(neuronChildVarEnv.getStream());
        cs.generate(neuronChildVarEnv, *this, batchSize);
    }

    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> neuronEnv(neuronChildVarEnv, *this); 

    // Expose read-only Isyn
    neuronEnv.add(getScalarType().addConst(), "Isyn", "$(_Isyn)");

    // Expose read-only additional input variables
    for (const auto &v : nm->getAdditionalInputVars()) {
        const auto resolvedType = v.type.resolve(getTypeContext()).addConst();
        neuronEnv.add(resolvedType, v.name, "$(_" + v.name + ")");
    }

    // Expose neuron variables
    neuronEnv.addVarExposeAliases<NeuronVarAdapter>();

    // Substitute parameter and derived parameter names
    neuronEnv.addParams(nm->getParams(), "", &NeuronGroupInternal::getParams, &NeuronUpdateGroupMerged::isParamHeterogeneous);
    neuronEnv.addDerivedParams(nm->getDerivedParams(), "", &NeuronGroupInternal::getDerivedParams, &NeuronUpdateGroupMerged::isDerivedParamHeterogeneous);
    neuronEnv.addExtraGlobalParams(nm->getExtraGlobalParams());
    
    // Substitute spike times
    const std::string timePrecision = getTimeType().getName();
    const std::string spikeTimeReadIndex = getReadVarIndex(getArchetype().isDelayRequired(), batchSize, 
                                                           VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    neuronEnv.add(getTimeType().addConst(), "st", "lsT", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lsT = $(_st)[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(getTimeType().addConst(), "prev_st", "lprevST", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lprevST = $(_prev_st)[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(getTimeType().addConst(), "set", "lseT", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lseT = $(_set)[" + spikeTimeReadIndex+ "];")});
    neuronEnv.add(getTimeType().addConst(), "prev_set", "lprevSET", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lprevSET = $(_prev_set)[" + spikeTimeReadIndex + "];")});

    // If a threshold condition is provided
    if (!Utils::areTokensEmpty(getArchetype().getThresholdConditionCodeTokens())) {
        neuronEnv.getStream() << "// test whether spike condition was fulfilled previously" << std::endl;

        if (nm->isAutoRefractoryRequired()) {
            neuronEnv.getStream() << "const bool oldSpike = (";

            Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' threshold condition code");
            prettyPrintExpression(getArchetype().getThresholdConditionCodeTokens(), getTypeContext(), neuronEnv, errorHandler);
            
            neuronEnv.getStream() << ");" << std::endl;
        }
    }
    // Otherwise, if any outgoing synapse groups have spike-processing code
    /*else if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                        [](const SynapseGroupInternal *sg){ return !sg->getWUInitialiser().getSnippet()->getSimCode().empty(); }))
    {
        LOGW_CODE_GEN << "No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << getName() << "\" was provided. There will be no spikes detected in this population!";
    }*/

    neuronEnv.getStream() << "// calculate membrane potential" << std::endl;

    Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' sim code");
    prettyPrintStatements(getArchetype().getSimCodeTokens(), getTypeContext(), neuronEnv, errorHandler);

    {
        // Generate var update for outgoing synaptic populations with presynaptic update code
        // **NOTE** we want to use the child environment where variables etc are hidden but 
        // actually print into the neuron environment so update happens at the right place
        EnvironmentGroupMergedField<NeuronUpdateGroupMerged> neuronWUChildEnv(
            neuronChildVarEnv, neuronEnv.getStream(), *this);
        for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
            CodeStream::Scope b(neuronWUChildEnv.getStream());
            sg.generate(neuronWUChildEnv, *this, batchSize, true);
        }

        // Generate var update for incoming synaptic populations with postsynaptic code
        for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
            CodeStream::Scope b(neuronWUChildEnv.getStream());
            sg.generate(neuronWUChildEnv, *this, batchSize, true);
        }
    }

    // look for spike type events first.
    /*if (getArchetype().isSpikeEventRequired()) {
        // Create local variable
        neuronVarEnv.getStream() << "bool spikeLikeEvent = false;" << std::endl;

        // Loop through outgoing synapse populations that will contribute to event condition code
        size_t i = 0;
        for(const auto &spkEventCond : getArchetype().getSpikeEventCondition()) {
            // Replace of parameters, derived parameters and extraglobalsynapse parameters
            Substitutions spkEventCondSubs(&popSubs);

            // If this spike event condition requires synapse state
            if(spkEventCond.synapseStateInThresholdCode) {
                // Substitute EGPs
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUInitialiser().getSnippet()->getExtraGlobalParams(), "", "group->", "EventThresh" + std::to_string(i));

                // Substitute presynaptic variables
                const bool delayed = (spkEventCond.synapseGroup->getDelaySteps() != NO_DELAY);
                spkEventCondSubs.addVarNameSubstitution(spkEventCond.synapseGroup->getWUInitialiser().getSnippet()->getPreVars(), "", "group->",
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
    }*/

    // test for true spikes if condition is provided
    if (!Utils::areTokensEmpty(getArchetype().getThresholdConditionCodeTokens())) {
        neuronEnv.getStream() << "// test for and register a true spike" << std::endl;
        neuronEnv.getStream() << "if ((";
        
        Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' threshold condition code");
        prettyPrintExpression(getArchetype().getThresholdConditionCodeTokens(), getTypeContext(), neuronEnv, errorHandler);
            
        neuronEnv.getStream() << ")";
        if (nm->isAutoRefractoryRequired()) {
            neuronEnv.getStream() << " && !oldSpike";
        }
        neuronEnv.getStream() << ")";
        {
            CodeStream::Scope b(neuronEnv.getStream());
            genEmitTrueSpike(neuronEnv, *this);

            // add after-spike reset if provided
            if (!Utils::areTokensEmpty(getArchetype().getResetCodeTokens())) {
                neuronEnv.getStream() << "// spike reset code" << std::endl;
                
                Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' reset code");
                prettyPrintStatements(getArchetype().getResetCodeTokens(), getTypeContext(), neuronEnv, errorHandler);
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
                                                         && Utils::areTokensEmpty(sg.getArchetype().getWUInitialiser().getPreDynamicsCodeTokens()));
                                             });

            // Are there any incoming synapse groups with postsynaptic code
            // which have back-propagation delay and no postsynaptic dynamics
            const bool postVars = std::any_of(getMergedInSynWUMPostCodeGroups().cbegin(), getMergedInSynWUMPostCodeGroups().cend(),
                                              [](const auto &sg)
                                              {
                                                  return ((sg.getArchetype().getBackPropDelaySteps() != NO_DELAY)
                                                           && Utils::areTokensEmpty(sg.getArchetype().getWUInitialiser().getPostDynamicsCodeTokens()));
                                              });

            // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
            if(getArchetype().isSpikeTimeRequired() || getArchetype().isPrevSpikeTimeRequired() || preVars || postVars) {
                neuronEnv.getStream() << "else";
                CodeStream::Scope b(neuronEnv.getStream());

                // If spike times are required, copy times from register
                if(getArchetype().isSpikeTimeRequired()) {
                    neuronEnv.printLine("$(_st)[" + getWriteVarIndex(true, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)") + "] = $(st);");
                }

                // If previous spike times are required, copy times from register
                if(getArchetype().isPrevSpikeTimeRequired()) {
                    neuronEnv.printLine("$(_prev_st)[" + getWriteVarIndex(true, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)") + "] = $(prev_st);");
                }

                // Loop through outgoing synapse groups with some sort of presynaptic code
                for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
                    sg.genCopyDelayedVars(neuronEnv, *this, batchSize);
                }

                // Loop through incoming synapse groups with some sort of presynaptic code
                for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
                    sg.genCopyDelayedVars(neuronEnv, *this, batchSize);
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(env, *this, batchSize, false);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(env, *this, batchSize, false);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDim varDims, 
                                                 const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    if (!(varDims & VarAccessDim::ELEMENT)) {
        return batched ? "$(batch)" : "0";
    }
    else if(batched) {
        return "$(_batch_offset) + " + index;
    }
    else {
        return index;
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getReadVarIndex(bool delay, unsigned int batchSize, 
                                                     VarAccessDim varDims, const std::string &index) const
{
    if(delay) {
        const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
        if (!(varDims & VarAccessDim::ELEMENT)) {
            return batched ? "$(_read_batch_delay_slot)" : "$(_read_delay_slot)";
        }
        else if(batched) {
            return "$(_read_batch_delay_offset) + " + index;
        }
        else {
            return "$(_read_delay_offset) + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDims, index);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getWriteVarIndex(bool delay, unsigned int batchSize, 
                                                      VarAccessDim varDims, const std::string &index) const
{
    if(delay) {
        const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
        if (!(varDims & VarAccessDim::ELEMENT)) {
            return batched ? "$(_write_batch_delay_slot)" : "$(_write_delay_slot)";
        }
        else if (batched) {
            return "$(_write_batch_delay_offset) + " + index;
        }
        else {
            return "$(_write_delay_offset) + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDims, index);
    }
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const NeuronGroupInternal &ng) { return ng.getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::isDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const NeuronGroupInternal &ng) { return ng.getDerivedParams(); });
}


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::genSpikeQueueUpdate(EnvironmentExternalBase &env, unsigned int batchSize) const
{
    // Update spike queue
    if(getArchetype().isDelayRequired()) {
        env.printLine("*$(_spk_que_ptr) = (*$(_spk_que_ptr) + 1) % " + std::to_string(getArchetype().getNumDelaySlots()) + ";");
    }

    if(batchSize > 1) {
        env.getStream() << "for(unsigned int batch = 0; batch < " << batchSize << "; batch++)" << CodeStream::OB(1);
    }
    if(getArchetype().isSpikeEventRequired()) {
        if(getArchetype().isDelayRequired()) {
            env.getStream() << env["_spk_cnt_evnt"] << "[*" << env["_spk_que_ptr"];
            if(batchSize > 1) {
                env.getStream() << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
            }
            env.getStream() << "] = 0; " << std::endl;
        }
        else {
            env.getStream() << env["_spk_cnt_evnt"] << "[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
        }
    }

    if(getArchetype().isTrueSpikeRequired()) {
        if(getArchetype().isDelayRequired()) {
            env.getStream() << env["_spk_cnt"] << "[*" << env["_spk_que_ptr"];
            if(batchSize > 1) {
                env.getStream() << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
            }
            env.getStream() << "] = 0; " << std::endl;
        }
        else {
            env.getStream() << env["_spk_cnt"] << "[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
        }
    }

    if(batchSize > 1) {
        env.getStream() << CodeStream::CB(1);
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";

