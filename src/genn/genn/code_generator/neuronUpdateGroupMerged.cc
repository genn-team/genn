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
void NeuronUpdateGroupMerged::CurrentSource::generate(const BackendBase &backend, EnvironmentExternalBase &env,
                                                      NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    const std::string fieldSuffix =  "CS" + std::to_string(getIndex());
    const auto *cm = getArchetype().getCurrentSourceModel();

    // Create new environment to add current source fields to neuron update group 
    EnvironmentGroupMergedField<CurrentSource, NeuronUpdateGroupMerged> csEnv(env, *this, ng);
    
    csEnv.getStream() << "// current source " << getIndex() << std::endl;

    // Substitute parameter and derived parameter names
    csEnv.addParams(cm->getParamNames(), fieldSuffix, &CurrentSourceInternal::getParams, &CurrentSource::isParamHeterogeneous);
    csEnv.addDerivedParams(cm->getDerivedParams(), fieldSuffix, &CurrentSourceInternal::getDerivedParams, &CurrentSource::isDerivedParamHeterogeneous);
    csEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", fieldSuffix);

    // Define inject current function
    csEnv.add(Type::ResolvedType::createFunction(Type::Void, {modelMerged.getModel().getPrecision()}), 
              "injectCurrent", "$(Isyn) += $(0)");

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CurrentSourceVarAdapter, CurrentSource, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), csEnv, backend.getDeviceVarPrefix(), fieldSuffix, "l",
        [&modelMerged, &ng](const std::string&, VarAccessDuplication d)
        {
            return ng.getVarIndex(modelMerged.getModel().getBatchSize(), d, "$(id)");
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
                                                 NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    const std::string fieldSuffix =  "InSyn" + std::to_string(getIndex());
    const auto *psm = getArchetype().getPSModel();

    // Create new environment to add PSM fields to neuron update group 
    EnvironmentGroupMergedField<InSynPSM, NeuronUpdateGroupMerged> psmEnv(env, *this, ng);

    // Add inSyn
    psmEnv.addField(getScalarType().createPointer(), "_out_post", "outPost" + fieldSuffix,
                    [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPost" + g.getFusedPSVarSuffix(); });

    // Read into local variable
    const std::string idx = ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, "$(id)");
    psmEnv.getStream() << "// postsynaptic model " << getIndex() << std::endl;
    psmEnv.printLine(getScalarType().getName() + " linSyn = $(_out_post)[" + idx + "];");

    // If dendritic delay is required
    if (getArchetype().isDendriticDelayRequired()) {
        // Add dendritic delay buffer and pointer into it
        psmEnv.addField(getScalarType().createPointer(), "_den_delay", "denDelay" + fieldSuffix,
                        [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelay" + g.getFusedPSVarSuffix();});
        psmEnv.addField(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr" + fieldSuffix,
                        [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "denDelayPtr" + g.getFusedPSVarSuffix();});

        // Get reference to dendritic delay buffer input for this timestep
        psmEnv.printLine(backend.getPointerPrefix() + getScalarType().getName() + " *denDelayFront = &$(_den_delay)[(*$(_den_delay_ptr) * $(num_neurons)) + " + idx + "];");

        // Add delayed input from buffer into inSyn
        psmEnv.getStream() << "linSyn += *denDelayFront;" << std::endl;

        // Zero delay buffer slot
        psmEnv.getStream() << "*denDelayFront = " << writePreciseLiteral(0.0, getScalarType()) << ";" << std::endl;
    }

    // Add parameters, derived parameters and extra global parameters to environment
    psmEnv.addParams(psm->getParamNames(), fieldSuffix, &SynapseGroupInternal::getPSParams, &InSynPSM::isParamHeterogeneous);
    psmEnv.addDerivedParams(psm->getDerivedParams(), fieldSuffix, &SynapseGroupInternal::getPSDerivedParams, &InSynPSM::isDerivedParamHeterogeneous);
    psmEnv.addExtraGlobalParams(psm->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", fieldSuffix);
    
    // **TODO** naming convention
    psmEnv.add(modelMerged.getModel().getPrecision(), "inSyn", "linSyn");
        
    // Allow synapse group's PS output var to override what Isyn points to
    psmEnv.add(modelMerged.getModel().getPrecision(), "Isyn", getArchetype().getPSTargetVar());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<SynapsePSMVarAdapter, InSynPSM, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), psmEnv, backend.getDeviceVarPrefix(), fieldSuffix, "l",
        [&modelMerged, &ng](const std::string&, VarAccessDuplication d)
        {
            return ng.getVarIndex(modelMerged.getModel().getBatchSize(), d, "$(id)");
        });

    // Pretty print code back to environment
    Transpiler::ErrorHandler applyInputErrorHandler("Synapse group '" + getArchetype().getName() + "' postsynaptic model apply input code");
    prettyPrintStatements(getArchetype().getPSApplyInputCodeTokens(), getTypeContext(), varEnv, applyInputErrorHandler);

    Transpiler::ErrorHandler decayErrorHandler("Synapse group '" + getArchetype().getName() + "' postsynaptic model decay code");
    prettyPrintStatements(getArchetype().getPSDecayCodeTokens(), getTypeContext(), varEnv, decayErrorHandler);

    // Write back linSyn
    varEnv.printLine("$(_out_post)[" + ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, "$(id)") + "] = linSyn;");
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getPSParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getPSDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynPSM::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getPSDerivedParams(); });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynPreOutput::generate(const BackendBase &backend, EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, 
                                                        const ModelSpecMerged &modelMerged)
{
    const std::string fieldSuffix =  "OutSyn" + std::to_string(getIndex());
    
    // Create new environment to add out syn fields to neuron update group 
    EnvironmentGroupMergedField<OutSynPreOutput, NeuronUpdateGroupMerged> outSynEnv(env, *this, ng);
    
    outSynEnv.addField(modelMerged.getModel().getPrecision().createPointer(), "_out_pre", "outPre" + fieldSuffix,
                       [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPre" + g.getFusedPreOutputSuffix(); });

    // Add reverse insyn variable to 
    const std::string idx = ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, "$(id)");
    outSynEnv.printLine(getArchetype().getPreTargetVar() + " += $(_out_pre)[" + idx + "];");

    // Zero it again
    outSynEnv.printLine("$(_out_pre)[" + idx + "] = " + writePreciseLiteral(0.0, getScalarType()) + ";");
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::generate(const BackendBase &backend, EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, bool dynamicsNotSpike)
{
    const std::string fieldSuffix =  "InSynWUMPost" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUModel();

    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // If there are any statements to execute here
    const auto &tokens = dynamicsNotSpike ? getArchetype().getWUPostDynamicsCodeTokens() : getArchetype().getWUPostSpikeCodeTokens();
    if(!Utils::areTokensEmpty(tokens)) {
        // Create new environment to add out syn fields to neuron update group 
        EnvironmentGroupMergedField<InSynWUMPostCode, NeuronUpdateGroupMerged> synEnv(env, *this, ng);

        synEnv.getStream() << "// postsynaptic weight update " << getIndex() << std::endl;
        
        // Add parameters, derived parameters and extra global parameters to environment
        synEnv.addParams(wum->getParamNames(), fieldSuffix, &SynapseGroupInternal::getWUParams, &InSynWUMPostCode::isParamHeterogeneous);
        synEnv.addDerivedParams(wum->getDerivedParams(), fieldSuffix, &SynapseGroupInternal::getWUDerivedParams, &InSynWUMPostCode::isDerivedParamHeterogeneous);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", fieldSuffix);

        // Create an environment which caches variables in local variables if they are accessed
        const bool delayed = (getArchetype().getBackPropDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, InSynWUMPostCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, backend.getDeviceVarPrefix(), fieldSuffix, "l",
            [batchSize, delayed, &synEnv, &ng](const std::string&, VarAccessDuplication d)
            {
                return ng.getReadVarIndex(delayed, batchSize, d, "$(id)");
            },
            [batchSize, delayed, &synEnv, &ng](const std::string&, VarAccessDuplication d)
            {
                return ng.getWriteVarIndex(delayed, batchSize, d, "$(id)");
            });

        /*neuronSubstitutionsInSynapticCode(varEnv, &ng.getArchetype(), "", "_post", "", "", "", dynamicsNotSpike,
                                          [&ng](const std::string &p) { return ng.isParamHeterogeneous(p); },
                                          [&ng](const std::string &p) { return ng.isDerivedParamHeterogeneous(p); },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                            s              {
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          { 
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          });*/

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model postsynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged)
{
    // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    if(getArchetype().getBackPropDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPostDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPostVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                const unsigned int batchSize = modelMerged.getModel().getBatchSize();
                env.print("$(" + v.name + ")[" + ng.getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), "$(id)") + "] = ");
                env.printLine("$(" + v.name + ")[" + ng.getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), "$(id)") + "];");
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::InSynWUMPostCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); });
}

 //----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::generate(const BackendBase &backend, EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, bool dynamicsNotSpike)
{
    const std::string fieldSuffix =  "OutSynWUMPre" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUModel();
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
    
    // If there are any statements to execute here
    const auto &tokens = dynamicsNotSpike ? getArchetype().getWUPreDynamicsCodeTokens() : getArchetype().getWUPreSpikeCodeTokens();
    if(!Utils::areTokensEmpty(tokens)) {
        // Create new environment to add out syn fields to neuron update group 
        EnvironmentGroupMergedField<OutSynWUMPreCode, NeuronUpdateGroupMerged> synEnv(env, *this, ng);

        synEnv.getStream() << "// postsynaptic weight update " << getIndex() << std::endl;
        
        // Add parameters, derived parameters and extra global parameters to environment
        synEnv.addParams(wum->getParamNames(), fieldSuffix, &SynapseGroupInternal::getWUParams, &OutSynWUMPreCode::isParamHeterogeneous);
        synEnv.addDerivedParams(wum->getDerivedParams(), fieldSuffix, &SynapseGroupInternal::getWUDerivedParams, &OutSynWUMPreCode::isDerivedParamHeterogeneous);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", fieldSuffix);

        // Create an environment which caches variables in local variables if they are accessed
        const bool delayed = (getArchetype().getDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPreVarAdapter, OutSynWUMPreCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, backend.getDeviceVarPrefix(), fieldSuffix, "l",
            [batchSize, delayed, &ng](const std::string&, VarAccessDuplication d)
            {
                return ng.getReadVarIndex(delayed, batchSize, d, "$(id)");
            },
            [batchSize, delayed, &ng](const std::string&, VarAccessDuplication d)
            {
                return ng.getWriteVarIndex(delayed, batchSize, d, "$(id)");
            });     
        
        /*neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), "", "_pre", "", "", "", dynamicsNotSpike,
                                          [&ng](const std::string &p) { return ng.isParamHeterogeneous(p); },
                                          [&ng](const std::string &p) { return ng.isDerivedParamHeterogeneous(p); },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          {
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          },
                                          [&subs, &ng, batchSize](bool delay, VarAccessDuplication varDuplication) 
                                          { 
                                              return ng.getReadVarIndex(delay, batchSize, varDuplication, subs["id"]); 
                                          });*/

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model presynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged)
{
    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    if(getArchetype().getDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPreDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPreVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                const unsigned int batchSize = modelMerged.getModel().getBatchSize();
                env.print("$(" + v.name + ")[" + ng.getWriteVarIndex(true, batchSize, getVarAccessDuplication(v.access), "$(id)") + "] = ");
                env.printLine("$(" + v.name + ")[" + ng.getReadVarIndex(true, batchSize, getVarAccessDuplication(v.access), "$(id)") + "];");
            }
        }
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); });
}
//----------------------------------------------------------------------------
bool NeuronUpdateGroupMerged::OutSynWUMPreCode::isDerivedParamHeterogeneous( const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); });
 
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
void NeuronUpdateGroupMerged::generateNeuronUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged,
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitTrueSpike,
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent)
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
 
    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> neuronEnv(env, *this);

    // Add default input variable
    neuronEnv.add(modelMerged.getModel().getPrecision(), "Isyn", "Isyn",
                  {neuronEnv.addInitialiser(getScalarType().getName() + " Isyn = 0;")});

    // **NOTE** arbitrary code in param value to be deprecated
    for (const auto &v : nm->getAdditionalInputVars()) {
        const auto resolvedType = v.type.resolve(getTypeContext());
        neuronEnv.add(resolvedType, v.name, "_" + v.name,
                      {neuronEnv.addInitialiser(resolvedType.getName() + " _" + v.name + " = " + v.value + ";")});
    }

    // Substitute parameter and derived parameter names
    neuronEnv.addParams(nm->getParamNames(), "", &NeuronGroupInternal::getParams, &NeuronUpdateGroupMerged::isParamHeterogeneous);
    neuronEnv.addDerivedParams(nm->getDerivedParams(), "", &NeuronGroupInternal::getDerivedParams, &NeuronUpdateGroupMerged::isDerivedParamHeterogeneous);
    neuronEnv.addExtraGlobalParams(nm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    
    // Substitute spike times
    const std::string timePrecision = modelMerged.getModel().getTimePrecision().getName();
    const std::string spikeTimeReadIndex = getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, "$(id)");
    neuronEnv.add(getTimeType().addConst(), "sT", "lsT", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lsT = $(_spk_time)[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(getTimeType().addConst(), "prev_sT", "lprevST", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lprevST = $(_prev_spk_time)[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(getTimeType().addConst(), "seT", "lseT", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lseT = $(_spk_evnt_time)[" + spikeTimeReadIndex+ "];")});
    neuronEnv.add(getTimeType().addConst(), "prev_seT", "lprevSET", 
                  {neuronEnv.addInitialiser("const " + timePrecision + " lprevSET = $(_prev_spk_evnt_time)[" + spikeTimeReadIndex + "];")});

    // Create an environment which caches variables in local variables if they are accessed
    // **NOTE** we do this right at the top so that local copies can be used by child groups
    EnvironmentLocalVarCache<NeuronVarAdapter, NeuronUpdateGroupMerged> neuronVarEnv(
        *this, *this, getTypeContext(), neuronEnv, backend.getDeviceVarPrefix(), "", "l",
        [batchSize, &neuronEnv, this](const std::string &varName, VarAccessDuplication d)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getReadVarIndex(delayed, batchSize, d, "$(id)") ;
        },
        [batchSize, &neuronEnv, this](const std::string &varName, VarAccessDuplication d)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getWriteVarIndex(delayed, batchSize, d, "$(id)") ;
        });


    // Loop through incoming synapse groups
    for(auto &sg : m_MergedInSynPSMGroups) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged);
    }

    // Loop through outgoing synapse groups with presynaptic output
    for (auto &sg : m_MergedOutSynPreOutputGroups) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged);
    }
 
    // Loop through all of neuron group's current sources
    for (auto &cs : m_MergedCurrentSourceGroups) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        cs.generate(backend, neuronVarEnv, *this, modelMerged);
    }


    // If a threshold condition is provided
    if (!nm->getThresholdConditionCode().empty()) {
        neuronVarEnv.getStream() << "// test whether spike condition was fulfilled previously" << std::endl;
        
        //if (!nm->getSupportCode().empty() && !backend.supportsNamespace()) {
        //    thCode = disambiguateNamespaceFunction(nm->getSupportCode(), thCode, modelMerged.getNeuronUpdateSupportCodeNamespace(nm->getSupportCode()));
        //}

        if (nm->isAutoRefractoryRequired()) {
            neuronVarEnv.getStream() << "const bool oldSpike = (";

            Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' threshold condition code");
            prettyPrintExpression(getArchetype().getThresholdConditionCodeTokens(), getTypeContext(), neuronVarEnv, errorHandler);
            
            neuronVarEnv.getStream() << ");" << std::endl;
        }
    }
    // Otherwise, if any outgoing synapse groups have spike-processing code
    /*else if(std::any_of(getOutSyn().cbegin(), getOutSyn().cend(),
                        [](const SynapseGroupInternal *sg){ return !sg->getWUModel()->getSimCode().empty(); }))
    {
        LOGW_CODE_GEN << "No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << getName() << "\" was provided. There will be no spikes detected in this population!";
    }*/

    neuronVarEnv.getStream() << "// calculate membrane potential" << std::endl;

    Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' sim code");
    prettyPrintStatements(getArchetype().getSimCodeTokens(), getTypeContext(), neuronVarEnv, errorHandler);

    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged, true);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged, true);
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
    }*/

    // test for true spikes if condition is provided
    if (!nm->getThresholdConditionCode().empty()) {
        neuronVarEnv.getStream() << "// test for and register a true spike" << std::endl;
        neuronVarEnv.getStream() << "if ((";
        
        Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' threshold condition code");
        prettyPrintExpression(getArchetype().getThresholdConditionCodeTokens(), getTypeContext(), neuronVarEnv, errorHandler);
            
        neuronVarEnv.getStream() << ")";
        if (nm->isAutoRefractoryRequired()) {
            neuronVarEnv.getStream() << " && !oldSpike";
        }
        neuronVarEnv.getStream() << ")";
        {
            CodeStream::Scope b(neuronVarEnv.getStream());
            genEmitTrueSpike(neuronVarEnv, *this);

            // add after-spike reset if provided
            if (!nm->getResetCode().empty()) {
                neuronVarEnv.getStream() << "// spike reset code" << std::endl;
                
                Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' reset code");
                prettyPrintStatements(getArchetype().getResetCodeTokens(), getTypeContext(), neuronVarEnv, errorHandler);
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
                neuronVarEnv.getStream() << "else";
                CodeStream::Scope b(neuronVarEnv.getStream());

                // If spike times are required, copy times from register
                if(getArchetype().isSpikeTimeRequired()) {
                    neuronVarEnv.printLine("$(_spk_time)[" + getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, "$(id)") + "] = $(sT);");
                }

                // If previous spike times are required, copy times from register
                if(getArchetype().isPrevSpikeTimeRequired()) {
                    neuronVarEnv.printLine("$(_prev_spk_time)[" + getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, "$(id)") + "] = $(prev_sT);");
                }

                // Loop through outgoing synapse groups with some sort of presynaptic code
                for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
                    sg.genCopyDelayedVars(neuronVarEnv, *this, modelMerged);
                }

                // Loop through incoming synapse groups with some sort of presynaptic code
                for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
                    sg.genCopyDelayedVars(neuronVarEnv, *this, modelMerged);
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (auto &sg : m_MergedOutSynWUMPreCodeGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(backend, env, *this, modelMerged, false);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (auto &sg : m_MergedInSynWUMPostCodeGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(backend, env, *this, modelMerged, false);
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
        return (batchSize == 1) ? "0" : "$(batch)";
    }
    else if(varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
        return index;
    }
    else {
        return "$(_batch_offset) " + index;
    }
}
//--------------------------------------------------------------------------
std::string NeuronUpdateGroupMerged::getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    if(delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "$(_read_delay_slot)" : "$(_read_batch_delay_slot)";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return "$(_read_delay_offset) + " + index;
        }
        else {
            return "$(_read_batch_delay_offset) + " + index;
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
            return (batchSize == 1) ? "$(_write_delay_slot)" : "$(_write_batch_delay_slot)";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return "$(_write_delay_offset) + " + index;
        }
        else {
            return "$(_write_batch_delay_offset) + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDuplication, index);
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
void NeuronSpikeQueueUpdateGroupMerged::genMergedGroupSpikeCountReset(EnvironmentExternalBase &env, unsigned int batchSize) const
{
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

    if(getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired()) {
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

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";

