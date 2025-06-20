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
    const auto *cm = getArchetype().getModel();

    // Create new environment to add current source fields to neuron update group
    EnvironmentGroupMergedField<CurrentSource, NeuronUpdateGroupMerged> csEnv(env, *this, ng);

    csEnv.getStream() << "// current source " << getIndex() << std::endl;

    // Substitute parameter and derived parameter names
    csEnv.addParams(cm->getParams(), fieldSuffix, &CurrentSourceInternal::getParams,
                    &CurrentSourceInternal::isParamDynamic);
    csEnv.addDerivedParams(cm->getDerivedParams(), fieldSuffix, &CurrentSourceInternal::getDerivedParams);
    csEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), "", fieldSuffix);

    // Add neuron variable references
    csEnv.addLocalVarRefs<CurrentSourceNeuronVarRefAdapter>(true);

    // Define inject current function
    csEnv.add(Type::ResolvedType::createFunction(Type::Void, {getScalarType()}),
              "injectCurrent", "$(_" + getArchetype().getTargetVar() + ") += $(0)");

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CurrentSourceVarAdapter, CurrentSource, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), csEnv, fieldSuffix, "l", false,
        [batchSize, &ng](const std::string&, VarAccess d, bool)
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
                    [](const auto &runtime, const auto &g, size_t){ return runtime.getArray(g, "outPost"); });

    // Read into local variable
    const std::string idx = ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    psmEnv.getStream() << "// postsynaptic model " << getIndex() << std::endl;
    psmEnv.printLine(getScalarType().getName() + " linSyn = $(_out_post)[" + idx + "];");

    // If dendritic delay is required
    if (getArchetype().isDendriticOutputDelayRequired()) {
        // Add dendritic delay buffer and pointer into it
        psmEnv.addField(getScalarType().createPointer(), "_den_delay", "denDelay" + fieldSuffix,
                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "denDelay");});
        psmEnv.addField(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr" + fieldSuffix,
                        [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "denDelayPtr");});

        // Get reference to dendritic delay buffer input for this timestep
        const std::string denOffset = (batchSize > 1) ? "($(_batch_offset) * " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") + $(id)" : "$(id)";
        psmEnv.printLine(getScalarType().getName() + " *denDelayFront = &$(_den_delay)[(*$(_den_delay_ptr) * $(num_neurons)) + " + denOffset + "];");

        // Add delayed input from buffer into inSyn
        psmEnv.getStream() << "linSyn += *denDelayFront;" << std::endl;

        // Zero delay buffer slot
        psmEnv.getStream() << "*denDelayFront = " << Type::writeNumeric(0.0, getScalarType()) << ";" << std::endl;
    }

    // Add parameters, derived parameters and extra global parameters to environment
    psmEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getPSInitialiser, 
                                &SynapseGroupInternal::isPSParamDynamic);
    psmEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getPSInitialiser);
    psmEnv.addExtraGlobalParams(psm->getExtraGlobalParams(), "", fieldSuffix);
    
    // Add neuron variable references
    psmEnv.addLocalVarRefs<SynapsePSMNeuronVarRefAdapter>(true);

    // **TODO** naming convention
    psmEnv.add(getScalarType(), "inSyn", "linSyn");
    
    // Define inject current function
    psmEnv.add(Type::ResolvedType::createFunction(Type::Void, {getScalarType()}),
              "injectCurrent", "$(_" + getArchetype().getPostTargetVar() + ") += $(0)");

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<SynapsePSMVarAdapter, InSynPSM, NeuronUpdateGroupMerged> varEnv(
        *this, ng, getTypeContext(), psmEnv, fieldSuffix, "l", false,
        [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
        {
            return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
        },
        [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
        {
            return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
        });

    // Pretty print code back to environment
    Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' postsynaptic model apply sim code");
    prettyPrintStatements(getArchetype().getPSInitialiser().getSimCodeTokens(), getTypeContext(), varEnv, errorHandler);

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
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynPreOutput::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                        unsigned int batchSize)
{
    const std::string fieldSuffix =  "OutSyn" + std::to_string(getIndex());
    
    // Create new environment to add out syn fields to neuron update group 
    EnvironmentGroupMergedField<OutSynPreOutput, NeuronUpdateGroupMerged> outSynEnv(env, *this, ng);
    
    outSynEnv.addField(getScalarType().createPointer(), "_out_pre", "outPre" + fieldSuffix,
                       [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "outPre"); });

    // Add reverse insyn variable to 
    const std::string idx = ng.getVarIndex(batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    outSynEnv.printLine("$(_" + getArchetype().getPreTargetVar() + ") += $(_out_pre)[" + idx + "];");

    // Zero it again
    outSynEnv.printLine("$(_out_pre)[" + idx + "] = " + Type::writeNumeric(0.0, getScalarType()) + ";");
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::SynSpike
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::SynSpike::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                 BackendBase::HandlerEnv genUpdate)
{
    CodeStream::Scope b(env.getStream());
    const std::string fieldSuffix =  "SynSpike" + std::to_string(getIndex());

    // Add fields to environment
    EnvironmentGroupMergedField<NeuronUpdateGroupMerged::SynSpike, NeuronUpdateGroupMerged> groupEnv(env, *this, ng);

    groupEnv.addField(Type::Uint32.createPointer(), "_spk_cnt", "spkCnt" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCnt"); });
    groupEnv.addField(Type::Uint32.createPointer(), "_spk", "spk" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "Spk"); });

    // Call callback to generate update
    genUpdate(groupEnv);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::SynSpikeEvent
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::SynSpikeEvent::generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                      BackendBase::GroupHandlerEnv<SynSpikeEvent> genUpdate)
{
    CodeStream::Scope b(env.getStream());
    const std::string fieldSuffix =  "SynSpikeEvent" + std::to_string(getIndex());

    // Add fields to environment
    EnvironmentGroupMergedField<SynSpikeEvent, NeuronUpdateGroupMerged> groupEnv(env, *this, ng);

    groupEnv.addField(getTimeType().createPointer(), "_set", "seT" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i){ return runtime.getFusedEventArray(ng, i, g, "SET"); });

    groupEnv.addField(Type::Uint32.createPointer(), "_record_spk_event", "recordSpkEvent" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i){ return runtime.getFusedEventArray(ng, i, g, "RecordSpkEvent"); });

    groupEnv.addField(Type::Uint32.createPointer(), "_spk_cnt_event", "spkCntEvent" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCntEvent"); });
    groupEnv.addField(Type::Uint32.createPointer(), "_spk_event", "spkEvent" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkEvent"); });

    // Call callback to generate update
    genUpdate(groupEnv, *this);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::SynSpikeEvent::generateEventCondition(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                                    unsigned int batchSize, BackendBase::GroupHandlerEnv<SynSpikeEvent> genEmitSpikeLikeEvent)
{
    const std::string fieldSuffix =  "SynSpikeEvent" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUInitialiser().getSnippet();

    // Create new environment to add out syn fields to neuron update group 
    EnvironmentGroupMergedField<SynSpikeEvent, NeuronUpdateGroupMerged> synEnv(env, *this, ng);

    synEnv.getStream() << "// spike event condition " << getIndex() << std::endl;
    
    synEnv.addField(getTimeType().createPointer(), "_set", "seT" + fieldSuffix,
                    [&ng](const auto &runtime, const auto &g, size_t i){ return runtime.getFusedEventArray(ng, i, g, "SET"); });
    synEnv.addField(getTimeType().createPointer(), "_prev_set", "prevSET" + fieldSuffix,
                    [&ng](const auto &runtime, const auto &g, size_t i){ return runtime.getFusedEventArray(ng, i, g, "PrevSET"); });
     
    // Expose spike event times to neuron code
    const std::string timePrecision = getTimeType().getName();
    const std::string spikeTimeReadIndex = ng.getReadVarIndex(ng.getArchetype().isSpikeEventDelayRequired(), batchSize,
                                                              VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    synEnv.add(getTimeType().addConst(), "set", "lseT", 
               {synEnv.addInitialiser("const " + timePrecision + " lseT = $(_set)[" + spikeTimeReadIndex + "];")});
    synEnv.add(getTimeType().addConst(), "prev_set", "lprevSET", 
               {synEnv.addInitialiser("const " + timePrecision + " lprevSET = $(_prev_set)[" + spikeTimeReadIndex + "];")});

    // Add parameters, derived parameters and extra global parameters to environment
    synEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, 
                                &SynapseGroupInternal::isWUParamDynamic);
    synEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser);
    synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), "", fieldSuffix);

    // **NOTE** for an incoming and outgoing synapse group to be merged together, the getPreEventHashDigest and getPostEventHashDigest 
    // of their weight update model must match which means the pre and post event threshold condition must be the 
    // same and they must refernece the same variables so we just need to pick the right one on the archetype and it'll be fine
    if(getArchetype().getSrcNeuronGroup() == &ng.getArchetype()) {
        // Add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPreNeuronVarRefAdapter>(true);

        // Create an environment which caches variables in local variables if they are accessed
        EnvironmentLocalVarCache<SynapseWUPreVarAdapter, SynSpikeEvent, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            });

        generateEventConditionInternal(varEnv, ng, batchSize, genEmitSpikeLikeEvent,
                                       getArchetype().getWUInitialiser().getPreEventThresholdCodeTokens(),
                                       "Synapse group '" + getArchetype().getName() + "' presynaptic event threshold condition");

    }
    else {
        // Add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPostNeuronVarRefAdapter>(true);

        // Create an environment which caches variables in local variables if they are accessed
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, SynSpikeEvent, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            }); 

        generateEventConditionInternal(varEnv, ng, batchSize, genEmitSpikeLikeEvent,
                                       getArchetype().getWUInitialiser().getPostEventThresholdCodeTokens(),
                                       "Synapse group '" + getArchetype().getName() + "' postsynaptic event threshold condition");
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::SynSpikeEvent::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &g) { return g.getWUInitialiser().getDerivedParams(); }, hash);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::SynSpikeEvent::generateEventConditionInternal(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                                            unsigned int batchSize, BackendBase::GroupHandlerEnv<SynSpikeEvent> genEmitSpikeLikeEvent,
                                                                            const std::vector<Transpiler::Token> &conditionTokens, const std::string &errorContext)
{
    Transpiler::ErrorHandler errorHandler(errorContext);
    env.print("if((");    
    prettyPrintExpression(conditionTokens, getTypeContext(), env, errorHandler);
    env.print("))");
    {
        CodeStream::Scope b(env.getStream());
        genEmitSpikeLikeEvent(env, *this);
    }
    // If delays are required and event times are required
    if(ng.getArchetype().isSpikeEventDelayRequired()
       && (ng.getArchetype().isSpikeEventTimeRequired() || ng.getArchetype().isPrevSpikeEventTimeRequired())) 
    {
        env.print("else");
        {
            CodeStream::Scope b(env.getStream());

            // If spike times are required, copy times from register
            if(ng.getArchetype().isSpikeEventTimeRequired()) {
                env.printLine("$(_set)[" + ng.getWriteVarIndex(true, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)") + "] = $(set);");
            }

            // If previous spike times are required, copy times from register
            if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                env.printLine("$(_prev_set)[" + ng.getWriteVarIndex(true, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)") + "] = $(prev_set);");
            }
        }
    }
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
        synEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, 
                                    &SynapseGroupInternal::isWUParamDynamic);
        synEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), "", fieldSuffix);

        // If we're generating dynamics code, add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPostNeuronVarRefAdapter>(true);

        // Substitute spike time
        // **NOTE** previous spike time is meaningless in neuron kernel
        const std::string spikeTimeReadIndex = ng.getReadVarIndex(ng.getArchetype().isSpikeDelayRequired(), batchSize,
                                                                  VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
        synEnv.add(getTimeType().addConst(), "st_post", "lsTPost", 
                   {synEnv.addInitialiser("const " + getTimeType().getName() + " lsTPost = $(_st)[" + spikeTimeReadIndex + "];")});

        // Create an environment which caches variables in local variables if they are accessed
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, InSynWUMPostCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            });

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model postsynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::genCopyDelayedVars(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                                   unsigned int batchSize)
{
    // If this group has no postsynaptic dynamics (which will already perform this copying)
    if(Utils::areTokensEmpty(getArchetype().getWUInitialiser().getPostDynamicsCodeTokens())) {
         // Create environment and add fields for variable 
        EnvironmentGroupMergedField<InSynWUMPostCode, NeuronUpdateGroupMerged> varEnv(env, *this, ng);
        varEnv.addVarPointers<SynapseWUPostVarAdapter>("InSynWUMPost" + std::to_string(getIndex()), false);
        
        // Loop through variables
        for(const auto &v : getArchetype().getWUInitialiser().getSnippet()->getPostVars()) {
            // If variable is delayed, copy between read and write delay slots
            if(getArchetype().getBackPropDelaySteps() != 0 
               || getArchetype().isWUPostVarHeterogeneouslyDelayed(v.name)) 
            {
                const VarAccessDim varDims = getVarAccessDim(v.access);
                varEnv.print("$(" + v.name + ")[" + ng.getWriteVarIndex(true, batchSize, varDims, "$(id)") + "] = ");
                varEnv.printLine("$(" + v.name + ")[" + ng.getReadVarIndex(true, batchSize, varDims, "$(id)") + "];");
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
        synEnv.addInitialiserParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser, 
                                    &SynapseGroupInternal::isWUParamDynamic);
        synEnv.addInitialiserDerivedParams(fieldSuffix, &SynapseGroupInternal::getWUInitialiser);
        synEnv.addExtraGlobalParams(wum->getExtraGlobalParams(), "", fieldSuffix);

        // If we're generating dynamics code, add local neuron variable references
        synEnv.addLocalVarRefs<SynapseWUPreNeuronVarRefAdapter>(true);

        // Substitute spike time
        // **NOTE** previous spike time is meaningless in neuron kernel
        const std::string spikeTimeReadIndex = ng.getReadVarIndex(ng.getArchetype().isSpikeDelayRequired(), batchSize,
                                                                  VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
        synEnv.add(getTimeType().addConst(), "st_pre", "lsTPre", 
                   {synEnv.addInitialiser("const " + getTimeType().getName() + " lsTPre = $(_st)[" + spikeTimeReadIndex + "];")});

        // Create an environment which caches variables in local variables if they are accessed
        EnvironmentLocalVarCache<SynapseWUPreVarAdapter, OutSynWUMPreCode, NeuronUpdateGroupMerged> varEnv(
            *this, ng, getTypeContext(), synEnv, fieldSuffix, "l", false,
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            },
            [batchSize, &ng](const std::string&, VarAccess d, bool delayed)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)");
            });     

        const std::string context = dynamicsNotSpike ? "dynamics" : "spike";
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' weight update model presynaptic " + context + " code");
        prettyPrintStatements(tokens, getTypeContext(), varEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::genCopyDelayedVars(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                                                   unsigned int batchSize)
{
    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
    if(getArchetype().getAxonalDelaySteps() != 0 && Utils::areTokensEmpty(getArchetype().getWUInitialiser().getPreDynamicsCodeTokens())) {
        // Create environment and add fields for variable 
        EnvironmentGroupMergedField<OutSynWUMPreCode, NeuronUpdateGroupMerged> varEnv(env, *this, ng);
        varEnv.addVarPointers<SynapseWUPreVarAdapter>("OutSynWUMPre" + std::to_string(getIndex()), false);

        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUInitialiser().getSnippet()->getPreVars()) {
            if(getVarAccessMode(v.access) == VarAccessMode::READ_WRITE) {
                const VarAccessDim varDims = getVarAccessDim(v.access);
                varEnv.print("$(" + v.name + ")[" + ng.getWriteVarIndex(true, batchSize, varDims, "$(id)") + "] = ");
                varEnv.printLine("$(" + v.name + ")[" + ng.getReadVarIndex(true, batchSize, varDims, "$(id)") + "];");
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
// GeNN::CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronUpdateGroupMerged::name = "NeuronUpdate";
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, 
                                                 const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, groups)
{
    // Build vector of child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, getTypeContext(), &NeuronGroupInternal::getFusedPSMInSyn, &SynapseGroupInternal::getPSHashDigest);

    // Build vector of child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, getTypeContext(), &NeuronGroupInternal::getFusedPreOutputOutSyn, &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, getTypeContext(), &NeuronGroupInternal::getCurrentSources, &CurrentSourceInternal::getHashDigest);

    // Build vector of child group's spikes
    orderNeuronGroupChildren(m_MergedSpikeGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpike, &SynapseGroupInternal::getSpikeHashDigest);

    // Build vector of child group's spike evnets
    orderNeuronGroupChildren(m_MergedSpikeEventGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpikeEvent, &SynapseGroupInternal::getWUSpikeEventHashDigest);

    // Build vector of hild group's incoming synapse groups with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostCodeGroups, getTypeContext(), &NeuronGroupInternal::getFusedInSynWithPostCode, &SynapseGroupInternal::getWUPrePostHashDigest);

    // Build vector of child group's outgoing synapse groups with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreCodeGroups, getTypeContext(), &NeuronGroupInternal::getFusedOutSynWithPreCode, &SynapseGroupInternal::getWUPrePostHashDigest);
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
    for(const auto &sg : getMergedSpikeEventGroups()) {
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
                                                   BackendBase::HandlerEnv genEmitTrueSpike,
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged::SynSpikeEvent> genEmitSpikeLikeEvent)
{
    const NeuronModels::Base *nm = getArchetype().getModel();
 
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
    EnvironmentLocalVarCache<NeuronVarAdapter, NeuronUpdateGroupMerged> neuronChildVarEnv(
        *this, *this, getTypeContext(), neuronChildEnv, "", "l", true,
        [batchSize, this](const std::string&, VarAccess d, bool delayed)
        {
            return getReadVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
        },
        [batchSize, this](const std::string&, VarAccess d, bool delayed)
        {
            return getWriteVarIndex(delayed, batchSize, getVarAccessDim(d), "$(id)") ;
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
    neuronEnv.addParams(nm->getParams(), "", &NeuronGroupInternal::getParams,
                        &NeuronGroupInternal::isParamDynamic);
    neuronEnv.addDerivedParams(nm->getDerivedParams(), "", &NeuronGroupInternal::getDerivedParams);
    neuronEnv.addExtraGlobalParams(nm->getExtraGlobalParams());
    
    // Substitute spike time
    // **NOTE** previous spike time is meaningless in neuron kernel
    const std::string spikeTimeReadIndex = getReadVarIndex(getArchetype().isSpikeDelayRequired(), batchSize,
                                                           VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
    neuronEnv.add(getTimeType().addConst(), "st", "lsT", 
                  {neuronEnv.addInitialiser("const " + getTimeType().getName() + " lsT = $(_st)[" + spikeTimeReadIndex + "];")});

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

    // **YUCK** because we want local variables declared in sim code to be accessible in 
    // threshold condition and reset, manually create internal environments here
    Transpiler::TypeChecker::EnvironmentInternal neuronTypeCheckEnv(neuronEnv);
    Transpiler::PrettyPrinter::EnvironmentInternal neuronPrettyPrintEnv(neuronEnv);
    
    Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' sim code");
    prettyPrintStatements(getArchetype().getSimCodeTokens(), getTypeContext(), 
                          neuronTypeCheckEnv, neuronPrettyPrintEnv, errorHandler);

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

         // Generate spike event conditions and generation
        for(auto &sg : m_MergedSpikeEventGroups) {
            CodeStream::Scope b(neuronWUChildEnv.getStream());
            sg.generateEventCondition(neuronWUChildEnv, *this, 
                                      batchSize, genEmitSpikeLikeEvent);
        }
    }

    // test for true spikes if condition is provided
    if (!Utils::areTokensEmpty(getArchetype().getThresholdConditionCodeTokens())) {
        neuronEnv.getStream() << "// test for and register a true spike" << std::endl;
        neuronEnv.getStream() << "if ((";
        
        Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' threshold condition code");
        prettyPrintExpression(getArchetype().getThresholdConditionCodeTokens(), getTypeContext(), 
                              neuronTypeCheckEnv, neuronPrettyPrintEnv, errorHandler);
            
        neuronEnv.getStream() << ")";
        if (nm->isAutoRefractoryRequired()) {
            neuronEnv.getStream() << " && !oldSpike";
        }
        neuronEnv.getStream() << ")";
        {
            CodeStream::Scope b(neuronEnv.getStream());
            genEmitTrueSpike(neuronEnv);

            // add after-spike reset if provided
            if (!Utils::areTokensEmpty(getArchetype().getResetCodeTokens())) {
                neuronEnv.getStream() << "// spike reset code" << std::endl;
                
                Transpiler::ErrorHandler errorHandler("Neuron group '" + getArchetype().getName() + "' reset code");
                prettyPrintStatements(getArchetype().getResetCodeTokens(), getTypeContext(), 
                                      neuronTypeCheckEnv, neuronPrettyPrintEnv, errorHandler);
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
                                                 return ((sg.getArchetype().getAxonalDelaySteps() != 0)
                                                         && Utils::areTokensEmpty(sg.getArchetype().getWUInitialiser().getPreDynamicsCodeTokens()));
                                             });

            // Are there any incoming synapse groups with postsynaptic code
            // which have back-propagation delay or heterogeneous delays and no postsynaptic dynamics
            const bool postVars = std::any_of(getMergedInSynWUMPostCodeGroups().cbegin(), getMergedInSynWUMPostCodeGroups().cend(),
                                              [](const auto &sg)
                                              {
                                                  return (((sg.getArchetype().getBackPropDelaySteps() != 0)
                                                            || sg.getArchetype().areAnyWUPostVarHeterogeneouslyDelayed())
                                                           && Utils::areTokensEmpty(sg.getArchetype().getWUInitialiser().getPostDynamicsCodeTokens()));
                                              });

            // If spike times, presynaptic variables or postsynaptic variables are required, add if clause
            if((getArchetype().isSpikeTimeRequired() && getArchetype().isSpikeQueueRequired()) 
               || (getArchetype().isPrevSpikeTimeRequired() && getArchetype().isSpikeEventQueueRequired()) 
               || preVars || postVars) 
            {
                neuronEnv.getStream() << "else";
                CodeStream::Scope b(neuronEnv.getStream());
                
                // If spike times are required, copy times between delay slots
                if(getArchetype().isSpikeQueueRequired()) {
                    const std::string spikeTimeWriteIndex = getWriteVarIndex(true, batchSize, VarAccessDim::BATCH | VarAccessDim::ELEMENT, "$(id)");
                    if(getArchetype().isSpikeTimeRequired()) {
                        neuronEnv.printLine("$(_st)[" + spikeTimeWriteIndex + "] = $(st);");
                    }

                    // If previous spike times are required, copy times between delay slots
                    if(getArchetype().isPrevSpikeTimeRequired()) {
                        neuronEnv.printLine("$(_prev_st)[" + spikeTimeWriteIndex + "] = $(_prev_st)[" + spikeTimeReadIndex + "];");
                    }
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
void NeuronUpdateGroupMerged::generateSpikes(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate)
{
    // Loop through merged spike groups
    for(auto &s : m_MergedSpikeGroups) {
        s.generate(env, *this, genUpdate);
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateSpikeEvents(EnvironmentExternalBase &env, BackendBase::GroupHandlerEnv<SynSpikeEvent> genUpdate)
{
    // Loop through merged spike event groups
    for(auto &s : m_MergedSpikeEventGroups) {
        s.generate(env, *this, genUpdate);
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
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::SynSpike
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::SynSpike::generate(EnvironmentExternalBase &env, NeuronSpikeQueueUpdateGroupMerged &ng,
                                                           unsigned int batchSize)
{
    env.getStream() << "// spike queue update " << getIndex() << std::endl;
    const std::string fieldSuffix =  "SynSpike" + std::to_string(getIndex());

    // Add spike count and spikes to environment
    EnvironmentGroupMergedField<SynSpike, NeuronSpikeQueueUpdateGroupMerged> synSpkEnv(env, *this, ng);
    synSpkEnv.addField(Type::Uint32.createPointer(), "_spk_cnt", "spkCnt" + fieldSuffix,
                        [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCnt"); });
    synSpkEnv.addField(Type::Uint32.createPointer(), "_spk", "spk" + fieldSuffix,
                        [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "Spk"); });

    // Update spike count
    if(ng.getArchetype().isSpikeDelayRequired()) {
        synSpkEnv.print("$(_spk_cnt)[*$(_spk_que_ptr)");
        if(batchSize > 1) {
            synSpkEnv.print(" + (batch * " + std::to_string(ng.getArchetype().getNumDelaySlots()) + ")");
        }
        synSpkEnv.printLine("] = 0; ");
    }
    else {
        if(batchSize > 1) {
            synSpkEnv.printLine("$(_spk_cnt)[batch] = 0;");
        }
        else {
            synSpkEnv.printLine("$(_spk_cnt)[0] = 0;");
        }
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::SynSpike
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::SynSpikeEvent::generate(EnvironmentExternalBase &env, NeuronSpikeQueueUpdateGroupMerged &ng,
                                                                unsigned int batchSize)
{
    env.getStream() << "// spike event queue update " << getIndex() << std::endl;
    const std::string fieldSuffix =  "SynSpikeEvent" + std::to_string(getIndex());

    // Add spike count and spikes to environment
    EnvironmentGroupMergedField<SynSpikeEvent, NeuronSpikeQueueUpdateGroupMerged> synSpkEnv(env, *this, ng);
    synSpkEnv.addField(Type::Uint32.createPointer(), "_spk_cnt_event", "spkCntEvent" + fieldSuffix,
                        [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCntEvent"); });
    synSpkEnv.addField(Type::Uint32.createPointer(), "_spk_event", "spk_event" + fieldSuffix,
                        [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkEvent"); });

    // Update spike count
    if(ng.getArchetype().isSpikeEventDelayRequired()) {
        synSpkEnv.print("$(_spk_cnt_event)[*$(_spk_que_ptr)");
        if(batchSize > 1) {
            synSpkEnv.print(" + (batch * " + std::to_string(ng.getArchetype().getNumDelaySlots()) + ")");
        }
        synSpkEnv.printLine("] = 0; ");
    }
    else {
        if(batchSize > 1) {
            synSpkEnv.printLine("$(_spk_cnt_event)[batch] = 0;");
        }
        else {
            synSpkEnv.printLine("$(_spk_cnt_event)[0] = 0;");
        }
    }
}


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
NeuronSpikeQueueUpdateGroupMerged::NeuronSpikeQueueUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                                                     const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, groups)
{
    // Build vector of child group's spikes
    orderNeuronGroupChildren(m_MergedSpikeGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpike, &SynapseGroupInternal::getSpikeHashDigest);

    // Build vector of child group's spike events
    orderNeuronGroupChildren(m_MergedSpikeEventGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpikeEvent, &SynapseGroupInternal::getWUSpikeEventHashDigest);
}
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::genSpikeQueueUpdate(EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Update spike queue
    if(getArchetype().isDelayRequired()) {
        env.printLine("*$(_spk_que_ptr) = (*$(_spk_que_ptr) + 1) % " + std::to_string(getArchetype().getNumDelaySlots()) + ";");
    }

    // Start loop around batches if required
    if(batchSize > 1) {
        env.getStream() << "for(unsigned int batch = 0; batch < " << batchSize << "; batch++)" << CodeStream::OB(1);
    }

    // Loop through groups with spikes and generate update code
    for (auto &sg : m_MergedSpikeGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(env, *this, batchSize);
    }

    // Loop through groups with spike events and generate update code
    for (auto &sg : m_MergedSpikeEventGroups) {
        CodeStream::Scope b(env.getStream());
        sg.generate(env, *this, batchSize);
    }

    // End loop around batches if required
    if(batchSize > 1) {
        env.getStream() << CodeStream::CB(1);
    }
}


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged::SynSpike
//----------------------------------------------------------------------------
void NeuronPrevSpikeTimeUpdateGroupMerged::SynSpike::generate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                                                              BackendBase::HandlerEnv genUpdate)
{
    CodeStream::Scope b(env.getStream());
    const std::string fieldSuffix =  "PrevSpikeTime" + std::to_string(getIndex());

    // Add fields to environment
    EnvironmentGroupMergedField<SynSpike, NeuronPrevSpikeTimeUpdateGroupMerged> groupEnv(env, *this, ng);
    groupEnv.addField(Type::Uint32.createPointer(), "_spk_cnt", "spkCnt" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCnt"); });
    groupEnv.addField(Type::Uint32.createPointer(), "_spk", "spk" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "Spk"); });

    // Call callback to generate update
    genUpdate(groupEnv);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged::SynSpikeEvent
//----------------------------------------------------------------------------
void NeuronPrevSpikeTimeUpdateGroupMerged::SynSpikeEvent::generate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                                                                   BackendBase::HandlerEnv genUpdate)
{
    CodeStream::Scope b(env.getStream());
    const std::string fieldSuffix =  "PrevSpikeEventTime" + std::to_string(getIndex());

    // Add fields to environment
    EnvironmentGroupMergedField<SynSpikeEvent, NeuronPrevSpikeTimeUpdateGroupMerged> groupEnv(env, *this, ng);
    groupEnv.addField(getTimeType().createPointer(), "_prev_set", "prevSET" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "PrevSET"); });
    groupEnv.addField(Type::Uint32.createPointer(), "_spk_cnt_event", "spkCntEvent" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCntEvent"); });
    groupEnv.addField(Type::Uint32.createPointer(), "_spk_event", "spkEvent" + fieldSuffix,
                      [&ng](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkEvent"); });

    // Call callback to generate update
    genUpdate(groupEnv);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";
NeuronPrevSpikeTimeUpdateGroupMerged::NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                                                           const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, groups)
{
    assert(m_MergedSpikeGroups.size() <= 1);

    // Build vector of child group's spikes
    // **TODO** correct hash
    orderNeuronGroupChildren(m_MergedSpikeGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpike, &SynapseGroupInternal::getSpikeHashDigest);
    orderNeuronGroupChildren(m_MergedSpikeEventGroups, getTypeContext(), &NeuronGroupInternal::getFusedSpikeEvent, &SynapseGroupInternal::getSpikeHashDigest);
}
//----------------------------------------------------------------------------
void NeuronPrevSpikeTimeUpdateGroupMerged::generateSpikes(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate)
{
    // Loop through merged groups
    for(auto &s : m_MergedSpikeGroups) {
        s.generate(env, *this, genUpdate);
    }
}
//----------------------------------------------------------------------------
void NeuronPrevSpikeTimeUpdateGroupMerged::generateSpikeEvents(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate)
{
    // Loop through merged groups
    for(auto &s : m_MergedSpikeEventGroups) {
        s.generate(env, *this, genUpdate);
    }
}
