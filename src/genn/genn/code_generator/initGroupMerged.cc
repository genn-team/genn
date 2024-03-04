#include "code_generator/initGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"
#include "code_generator/standardLibrary.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/prettyPrinter.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void genVariableFill(EnvironmentExternalBase &env, const std::string &target, const std::string &value, const std::string &idx, const std::string &stride,
                     VarAccessDim varDims, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDims & VarAccessDim::BATCH) ? batchSize : 1) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        env.printLine("$(" + target + ")[$(" + idx + ")] = " + value + ";");
    }
    // Otherwise
    else {
        env.getStream() << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(" + target + ")[(d * " + stride + ") + $(" + idx + ")] = " + value + ";");
        }
    }
}
//--------------------------------------------------------------------------
void genScalarFill(EnvironmentExternalBase &env, const std::string &target, const std::string &value,
                   VarAccessDim varDims, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDims & VarAccessDim::BATCH) ? batchSize : 1) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        env.printLine("$(" + target + ")[0] = " + value + ";");
    }
    // Otherwise
    else {
        env.getStream() << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(env.getStream());
            env.printLine("$(" + target + ")[d] = " + value + ";");
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void genInitEvents(const BackendBase &backend, EnvironmentGroupMergedField<G, NeuronInitGroupMerged> &env, 
                   NeuronInitGroupMerged &ng, const std::string &fieldSuffix, bool trueSpike, unsigned int batchSize)
{
    // Add spike count
    const std::string suffix = trueSpike ? "" : "Event";
    env.addField(Type::Uint32.createPointer(), "_event_cnt", "spkCnt" + suffix + fieldSuffix,
                 [&ng, suffix](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "SpkCnt" + suffix); });
    
    env.addField(Type::Uint32.createPointer(), "_event", "spk" + suffix + fieldSuffix,
                 [&ng, suffix](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(ng, i, g, "Spk" + suffix); });

    // Generate code to zero spikes across all delay slots and batches
    backend.genVariableInit(env, "num_neurons", "id",
        [batchSize, &ng] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, "_event", "0", "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT,
                            batchSize, ng.getArchetype().isDelayRequired(), ng.getArchetype().getNumDelaySlots());
        });
    
    // Generate code to zero spike count across all delay slots and batches
    backend.genPopVariableInit(env,
        [batchSize, &ng] (EnvironmentExternalBase &spikeCountEnv)
        {
            genScalarFill(spikeCountEnv, "_event_cnt", "0", VarAccessDim::BATCH | VarAccessDim::ELEMENT, 
                          batchSize, ng.getArchetype().isDelayRequired(), ng.getArchetype().getNumDelaySlots());
        });
}
//------------------------------------------------------------------------
template<typename G>
void genInitEventTime(const BackendBase &backend, EnvironmentExternalBase &env, G &group, NeuronInitGroupMerged &fieldGroup, 
                      const std::string &fieldSuffix, const std::string &varName, unsigned int batchSize)
{
    EnvironmentGroupMergedField<G, NeuronInitGroupMerged> timeEnv(env, group, fieldGroup);
    timeEnv.addField(group.getTimeType().createPointer(), "_time", varName + fieldSuffix,
                     [&fieldGroup, varName](const auto &runtime, const auto &g, size_t i) { return runtime.getFusedEventArray(fieldGroup, i, g, varName); });


    // Generate variable initialisation code
    backend.genVariableInit(timeEnv, "num_neurons", "id",
        [batchSize, varName, &fieldGroup] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, "_time", "-TIME_MAX", "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT, 
                            batchSize, fieldGroup.getArchetype().isDelayRequired(), fieldGroup.getArchetype().getNumDelaySlots());
        });
}
//------------------------------------------------------------------------
void genInitEventTime(const BackendBase &backend, EnvironmentExternalBase &env, NeuronInitGroupMerged &group, 
                      const std::string &varName, unsigned int batchSize)
{
    // Generate variable initialisation code
    backend.genVariableInit(env, "num_neurons", "id",
        [batchSize, varName, &group] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, varName, "-TIME_MAX", "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT, 
                            batchSize, group.getArchetype().isDelayRequired(), group.getArchetype().getNumDelaySlots());
        });
}
//------------------------------------------------------------------------
template<typename A, typename F, typename G>
void genInitNeuronVarCode(const BackendBase &backend, EnvironmentExternalBase &env,
                          G &group, F &fieldGroup, const std::string &fieldSuffix, 
                          const std::string &count, unsigned int numDelaySlots, unsigned int batchSize)
{
    A adaptor(group.getArchetype());
    for (const auto &var : adaptor.getDefs()) {
        // If there is any initialisation code
        const auto resolvedType = var.type.resolve(group.getTypeContext());
        const auto &varInit = adaptor.getInitialisers().at(var.name);
        if (!Utils::areTokensEmpty(varInit.getCodeTokens())) {
            CodeStream::Scope b(env.getStream());

            // Substitute in parameters and derived parameters for initialising variables
            EnvironmentGroupMergedField<G, F> varEnv(env, group, fieldGroup);
            varEnv.template addVarInitParams<A>(&G::isVarInitParamHeterogeneous, var.name, fieldSuffix);
            varEnv.template addVarInitDerivedParams<A>(&G::isVarInitDerivedParamHeterogeneous, var.name, fieldSuffix);
            varEnv.addExtraGlobalParams(varInit.getSnippet()->getExtraGlobalParams(), var.name, fieldSuffix);

            // Add field for variable itself
            varEnv.addField(resolvedType.createPointer(), "_value", var.name + fieldSuffix,
                            [var](const auto &runtime, const auto &g, size_t) 
                            { 
                                return runtime.getArray(g, var.name); 
                            });

            // If variable has NEURON axis
            const VarAccessDim varDims = adaptor.getVarDims(var);
            if (varDims & VarAccessDim::ELEMENT) {
                backend.genVariableInit(
                    varEnv, count, "id",
                    [&adaptor, &fieldGroup, &fieldSuffix, &group, &var, &resolvedType, &varInit, batchSize, count, numDelaySlots, varDims]
                    (EnvironmentExternalBase &env)
                    {
                        // Generate initial value into temporary variable
                        EnvironmentGroupMergedField<G, F> varInitEnv(env, group, fieldGroup);
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Group '" + group.getArchetype().getName() + "' variable '" + var.name + "' init code");
                        prettyPrintStatements(varInit.getCodeTokens(), group.getTypeContext(), varInitEnv, errorHandler);

                        // Fill value across all delay slots and batches
                        genVariableFill(varInitEnv, "_value", "$(value)", "id", "$(" + count + ")",
                                        varDims, batchSize, adaptor.isVarDelayed(var.name), numDelaySlots);
                    });
            }
            // Otherwise
            else {
                backend.genPopVariableInit(
                    varEnv,
                    [&adaptor, &fieldGroup, &fieldSuffix, &group, &resolvedType, &var, &varInit, batchSize, numDelaySlots, varDims]
                    (EnvironmentExternalBase &env)
                    {
                        // Generate initial value into temporary variable
                        EnvironmentGroupMergedField<G, F> varInitEnv(env, group, fieldGroup);
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Group '" + group.getArchetype().getName() + "' variable '" + var.name + "' init code");
                        prettyPrintStatements(varInit.getCodeTokens(), group.getTypeContext(), varInitEnv, errorHandler);
                        
                        // Fill value across all delay slots and batches
                        genScalarFill(varInitEnv, "_value", "$(value)", varDims,
                                      batchSize, adaptor.isVarDelayed(var.name), numDelaySlots);
                    });
            }
        }
            
    }
}
//------------------------------------------------------------------------
template<typename A, typename G>
void genInitNeuronVarCode(const BackendBase &backend, EnvironmentExternalBase &env,
                          G &group, const std::string &fieldSuffix, 
                          const std::string &count, unsigned int numDelaySlots, unsigned int batchSize)
{
    genInitNeuronVarCode<A, G, G>(backend, env, group, group, fieldSuffix, count, numDelaySlots, batchSize);
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
template<typename A, typename G, typename V>
void genInitWUVarCode(EnvironmentExternalBase &env, G &group,
                      const std::string &stride, unsigned int batchSize, bool kernel,
                      V genSynapseVariableRowInitFn)
{
    A adaptor(group.getArchetype());
    for (const auto &var : adaptor.getDefs()) {
        // If this variable has any initialisation code and doesn't require a kernel (in this case it will be initialised elsewhere)
        const auto resolvedType = var.type.resolve(group.getTypeContext());
        const auto &varInit = adaptor.getInitialisers().at(var.name);
        if(!Utils::areTokensEmpty(varInit.getCodeTokens()) && (varInit.isKernelRequired() == kernel)) {
            CodeStream::Scope b(env.getStream());

            // Substitute in parameters and derived parameters for initialising variables
            EnvironmentGroupMergedField<G> varEnv(env, group);
            varEnv.template addVarInitParams<A>(&G::isVarInitParamHeterogeneous, var.name);
            varEnv.template addVarInitDerivedParams<A>(&G::isVarInitDerivedParamHeterogeneous, var.name);
            varEnv.addExtraGlobalParams(varInit.getSnippet()->getExtraGlobalParams(), var.name);

            // Add field for variable itself
            varEnv.addField(resolvedType.createPointer(), "_value", var.name,
                            [var](const auto &runtime, const auto &g, size_t) 
                            { 
                                return runtime.getArray(g, var.name); 
                            });

            // Generate target-specific code to initialise variable
            genSynapseVariableRowInitFn(varEnv,
                [&adaptor, &group, &resolvedType, &stride, &var, &varInit, batchSize]
                (EnvironmentExternalBase &env)
                {
                    // Generate initial value into temporary variable
                    EnvironmentGroupMergedField<G> varInitEnv(env, group);
                    varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                    varInitEnv.add(resolvedType, "value", "initVal");

                    // Pretty print variable initialisation code
                    Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                    prettyPrintStatements(varInit.getCodeTokens(), group.getTypeContext(), varInitEnv, errorHandler);

                    // Fill value across all batches
                    genVariableFill(varInitEnv, "_value", "$(value)", "id_syn", stride,
                                    adaptor.getVarDims(var), batchSize);
                });
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::CurrentSource
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::CurrentSource::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                    NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    genInitNeuronVarCode<CurrentSourceVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "CS" + std::to_string(getIndex()), 
        "num_neurons", 0, batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::SynSpike
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::SynSpike::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                               NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    const std::string fieldSuffix =  "SynSpike" + std::to_string(getIndex());

    // Create environment for group
    EnvironmentGroupMergedField<SynSpike, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Initialise spikes
    genInitEvents(backend, groupEnv, ng, fieldSuffix, true, batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::SynSpikeEvent
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::SynSpikeEvent::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                    NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    const std::string fieldSuffix =  "SynSpikeEvent" + std::to_string(getIndex());

    // Create environment for group
    EnvironmentGroupMergedField<SynSpikeEvent, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Initialise spike-like events
    genInitEvents(backend, groupEnv, ng, fieldSuffix, false, batchSize);

    // Initialize spike-like-event times
    if(ng.getArchetype().isSpikeEventTimeRequired()) {
        genInitEventTime(backend, groupEnv, *this, ng, fieldSuffix, "SET", batchSize);
    }

    // Initialize previous spike-like-event times
    if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
        genInitEventTime(backend, groupEnv, *this, ng, fieldSuffix, "PrevSET", batchSize);
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::InSynPSM
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynPSM::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                               NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    const std::string fieldSuffix =  "InSyn" + std::to_string(getIndex());

    // Create environment for group
    EnvironmentGroupMergedField<InSynPSM, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Add field for InSyn and zero
    groupEnv.addField(getScalarType().createPointer(), "_out_post", "outPost" + fieldSuffix,
                      [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "outPost"); });
    backend.genVariableInit(groupEnv, "num_neurons", "id",
        [batchSize, this] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, "_out_post", Type::writeNumeric(0.0, getScalarType()), 
                            "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT, batchSize);

        });

    // If dendritic delays are required
    if(getArchetype().isDendriticDelayRequired()) {
        // Add field for dendritic delay buffer and zero
        groupEnv.addField(getScalarType().createPointer(), "_den_delay", "denDelay" + fieldSuffix,
                          [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "denDelay"); });
        backend.genVariableInit(groupEnv, "num_neurons", "id",
            [batchSize, this](EnvironmentExternalBase &varEnv)
            {
                genVariableFill(varEnv, "_den_delay", Type::writeNumeric(0.0, getScalarType()),
                                "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT, 
                                batchSize, true, getArchetype().getMaxDendriticDelayTimesteps());
            });

        // Add field for dendritic delay pointer and zero
        groupEnv.addField(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr" + fieldSuffix,
                          [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "denDelayPtr"); });
        backend.genPopVariableInit(groupEnv,
            [](EnvironmentExternalBase &varEnv)
            {
                varEnv.getStream() << "*" << varEnv["_den_delay_ptr"] << " = 0;" << std::endl;
            });
    }

    genInitNeuronVarCode<SynapsePSMVarAdapter, NeuronInitGroupMerged>(
        backend, groupEnv, *this, ng, fieldSuffix, "num_neurons", 0, batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynPreOutput::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                      NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    // Create environment for group
    EnvironmentGroupMergedField<OutSynPreOutput, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Add 
    groupEnv.addField(getScalarType().createPointer(), "_out_pre", "outPreOutSyn" + std::to_string(getIndex()),
                      [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "outPre"); });
    backend.genVariableInit(groupEnv, "num_neurons", "id",
                            [batchSize, this] (EnvironmentExternalBase &varEnv)
                            {
                                genVariableFill(varEnv, "_out_pre", Type::writeNumeric(0.0, getScalarType()),
                                                "id", "$(num_neurons)", VarAccessDim::BATCH | VarAccessDim::ELEMENT, batchSize);
                            });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::InSynWUMPostVars
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynWUMPostVars::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    genInitNeuronVarCode<SynapseWUPostVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "InSynWUMPost" + std::to_string(getIndex()), "num_neurons", 
        getArchetype().getTrgNeuronGroup()->getNumDelaySlots(), batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynWUMPreVars
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynWUMPreVars::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       NeuronInitGroupMerged &ng, unsigned int batchSize)
{
    genInitNeuronVarCode<SynapseWUPreVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "OutSynWUMPre" + std::to_string(getIndex()), "num_neurons", 
        getArchetype().getSrcNeuronGroup()->getNumDelaySlots(), batchSize);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronInitGroupMerged::name = "NeuronInit";
//----------------------------------------------------------------------------
NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                             const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   InitGroupMergedBase<NeuronGroupMergedBase, NeuronVarAdapter>(index, typeContext, groups)
{
    // Build vector of child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, typeContext,
                             &NeuronGroupInternal::getFusedPSMInSyn,
                             &SynapseGroupInternal::getPSInitHashDigest );

    // Build vector of child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, typeContext,
                             &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             &SynapseGroupInternal::getPreOutputInitHashDigest );

    // Build vector of child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, typeContext,
                             &NeuronGroupInternal::getCurrentSources,
                             &CurrentSourceInternal::getInitHashDigest );

    // Build vector of child group's spikes
    orderNeuronGroupChildren(m_MergedSpikeGroups, getTypeContext(), 
                             &NeuronGroupInternal::getFusedSpike, 
                             &SynapseGroupInternal::getSpikeHashDigest);

    // Build vector of child group's spike events
    // **TODO** correct hash
    orderNeuronGroupChildren(m_MergedSpikeEventGroups, getTypeContext(), 
                             &NeuronGroupInternal::getFusedSpikeEvent,
                             &SynapseGroupInternal::getWUSpikeEventHashDigest);

    // Build vector of child group's incoming synapse groups
    // with postsynaptic weight update model variable, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostVarGroups, typeContext,
                             &NeuronGroupInternal::getFusedInSynWithPostVars,
                             &SynapseGroupInternal::getWUPrePostInitHashDigest);

    // Build vector of child group's outgoing synapse groups
    // with presynaptic weight update model variables, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreVarGroups, typeContext,
                             &NeuronGroupInternal::getFusedOutSynWithPreVars,
                             &SynapseGroupInternal::getWUPrePostInitHashDigest); 
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with standard archetype hash and var init parameters and derived parameters
    updateBaseHash(hash);

    // Update hash with each group's neuron count
    updateHash([](const NeuronGroupInternal &g) { return g.getNumNeurons(); }, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with child groups
    for (const auto &cs : getMergedCurrentSourceGroups()) {
        cs.updateHash(hash);
    }
    for(const auto &sg : getMergedInSynPSMGroups()) {
        sg.updateHash(hash);
    }
    for (const auto &sg : getMergedInSynWUMPostVarGroups()) {
        sg.updateHash(hash);
    }
    for (const auto &sg : getMergedOutSynWUMPreVarGroups()) {
        sg.updateHash(hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create environment for group
    EnvironmentGroupMergedField<NeuronInitGroupMerged> groupEnv(env, *this);
  
    // If neuron group requires delays
    if(getArchetype().isDelayRequired()) {
        // Add spike queue pointer field and zero
        groupEnv.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                          [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "spkQuePtr"); });
        backend.genPopVariableInit(groupEnv,
            [](EnvironmentExternalBase &varEnv)
            {
                varEnv.printLine("*$(_spk_que_ptr) = 0;");
            });
    }

    // Initialize spike times
    if(getArchetype().isSpikeTimeRequired()) {
        genInitEventTime(backend, groupEnv, *this, "_st", batchSize);
    }

    // Initialize previous spike times
    if(getArchetype().isPrevSpikeTimeRequired()) {
        genInitEventTime(backend, groupEnv, *this, "_prev_st", batchSize);
    }

    // Initialise neuron variables
    genInitNeuronVarCode<NeuronVarAdapter>(backend, groupEnv, *this, "", "num_neurons", 
                                           getArchetype().getNumDelaySlots(), batchSize);

    // Generate initialisation code for child groups
    for (auto &cs : m_MergedCurrentSourceGroups) {
        cs.generate(backend, groupEnv, *this, batchSize);
    }
    for (auto &sg : m_MergedSpikeGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }
    for (auto &sg : m_MergedSpikeEventGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }
    for(auto &sg : m_MergedInSynPSMGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }
    for (auto &sg : m_MergedOutSynPreOutputGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }  
    for (auto &sg : m_MergedOutSynWUMPreVarGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }
    for (auto &sg : m_MergedInSynWUMPostVarGroups) {
        sg.generate(backend, groupEnv, *this, batchSize);
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseInitGroupMerged::name = "SynapseInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with standard archetype hash and var init parameters and derived parameters
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getWUInitHashDigest(), hash);

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create environment for group
    EnvironmentGroupMergedField<SynapseInitGroupMerged> groupEnv(env, *this);

    // If we're using non-kernel weights, generate loop over source neurons
    const bool kernel = (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL);
    if (!kernel) {
        groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
        groupEnv.getStream() << CodeStream::OB(1);    
        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
    }

    // Generate initialisation code
    const std::string stride = kernel ? "$(_kernel_size)" : "$(num_pre) * $(_row_stride)";
    genInitWUVarCode<SynapseWUVarAdapter>(groupEnv, *this, stride, batchSize, false,
                                          [&backend, kernel, this](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
                                          {
                                              if (kernel) {
                                                  backend.genKernelSynapseVariableInit(varInitEnv, *this, handler);
                                              }
                                              else {
                                                  backend.genDenseSynapseVariableRowInit(varInitEnv, handler);
                                              }
                                          });

    // If we're using non-kernel weights, close loop
    if (!kernel) {
        groupEnv.getStream() << CodeStream::CB(1);
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseSparseInitGroupMerged::name = "SynapseSparseInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseSparseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with standard archetype hash and var init parameters and derived parameters
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getWUInitHashDigest(), hash);

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create environment for group
    genInitWUVarCode<SynapseWUVarAdapter>(
        env, *this, "$(num_pre) * $(_row_stride)", batchSize, false,
        [&backend](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            backend.genSparseSynapseVariableRowInit(varInitEnv, handler); 
        });
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityInitGroupMerged::name = "SynapseConnectivityInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseConnectivityInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype connectivity initialisation hash
    Utils::updateHash(getArchetype().getConnectivityInitHashDigest(), hash);

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);

    // Update hash with connectivity parameters and derived parameters
    updateParamHash([](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);
    updateParamHash([](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);

    if(!getArchetype().getKernelSize().empty()) {
        updateHash([](const SynapseGroupInternal &g) { return g.getKernelSize(); }, hash);

        // Update hash with each group's variable initialisation parameters and derived parameters
        updateVarInitParamHash<SynapseWUVarAdapter>(hash);
        updateVarInitDerivedParamHash<SynapseWUVarAdapter>(hash);
    }   
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseRowInit(EnvironmentExternalBase &env)
{
    genInitConnectivity(env, true);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseColumnInit(EnvironmentExternalBase &env)
{
    genInitConnectivity(env, false);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateKernelInit(EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create environment for group
    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(env, *this);

    // Add substitution
    // **TODO** dependencies on kernel fields
    groupEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                 {groupEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(*this) + ";")});

    // Initialise single (hence empty lambda function) synapse variable
    genInitWUVarCode<SynapseWUVarAdapter>(
        groupEnv, *this, "$(num_pre) * $(_row_stride)", batchSize, true,
        [](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            handler(varInitEnv);
        });
}
//----------------------------------------------------------------------------
bool SynapseConnectivityInitGroupMerged::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, 
                                     [&varName](const auto &g)
                                     { 
                                         return SynapseWUVarAdapter(g).getInitialisers().at(varName).getParams(); 
                                     });
}
//----------------------------------------------------------------------------
bool SynapseConnectivityInitGroupMerged::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, 
                                    [&varName](const auto &g) 
                                    { 
                                        return SynapseWUVarAdapter(g).getInitialisers().at(varName).getDerivedParams();
                                    });
}
//----------------------------------------------------------------------------
bool SynapseConnectivityInitGroupMerged::isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseConnectivityInitGroupMerged::isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); });
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::genInitConnectivity(EnvironmentExternalBase &env, bool rowNotColumns)
{
    // Create environment for group
    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(env, *this);

    // Substitute in parameters and derived parameters for initialising connectivity
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    groupEnv.addInitialiserParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                  &SynapseConnectivityInitGroupMerged::isSparseConnectivityInitParamHeterogeneous);
    groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                         &SynapseConnectivityInitGroupMerged::isSparseConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(connectInit.getSnippet()->getExtraGlobalParams(), "SparseConnect", "");

    const std::string context = rowNotColumns ? "row" : "column";
    Transpiler::ErrorHandler errorHandler("Synapse group sparse connectivity '" + getArchetype().getName() + "' " + context + " build code");
    prettyPrintStatements(rowNotColumns ? connectInit.getRowBuildCodeTokens() : connectInit.getColBuildCodeTokens(), 
                          getTypeContext(), groupEnv, errorHandler);
}


// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//-------------------------------------------------------------------------
void SynapseConnectivityHostInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // Add standard library to environment
    EnvironmentLibrary envStdLib(env, StandardLibrary::getMathsFunctions());

    // Add host RNG functions to environment
    EnvironmentLibrary envRandom(envStdLib, StandardLibrary::getHostRNGFunctions(getScalarType()));

    // Add standard host assert function to environment
    EnvironmentExternal envAssert(envRandom);
    envAssert.add(Type::Assert, "assert", "assert($(0))");

    CodeStream::Scope b(envAssert.getStream());
    envAssert.getStream() << "// merged synapse connectivity host init group " << getIndex() << std::endl;
    envAssert.getStream() << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(envAssert.getStream());

        // Get reference to group
        envAssert.getStream() << "const auto *group = &mergedSynapseConnectivityHostInitGroup" << getIndex() << "[g]; " << std::endl;
        
        // Create environment for group
        EnvironmentGroupMergedField<SynapseConnectivityHostInitGroupMerged> groupEnv(envAssert, *this);
        const auto &connectInit = getArchetype().getConnectivityInitialiser();

        // If matrix type is procedural then initialized connectivity init snippet will potentially be used with multiple threads per spike. 
        // Otherwise it will only ever be used for initialization which uses one thread per row
        const size_t numThreads = (getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) ? getArchetype().getNumThreadsPerSpike() : 1;

        // Create substitutions
        groupEnv.addField(Type::Uint32.addConst(), "num_pre",
                          Type::Uint32, "numSrcNeurons", 
                          [](const auto &, const SynapseGroupInternal &sg, size_t) { return sg.getSrcNeuronGroup()->getNumNeurons(); });
        groupEnv.addField(Type::Uint32.addConst(), "num_post",
                          Type::Uint32, "numTrgNeurons", 
                          [](const auto &, const SynapseGroupInternal &sg, size_t) { return sg.getTrgNeuronGroup()->getNumNeurons(); });
        groupEnv.add(Type::Uint32.addConst(), "num_threads", std::to_string(numThreads));

        groupEnv.addInitialiserParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                      &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);
        groupEnv.addInitialiserDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                             &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

        // Loop through EGPs
        for(const auto &egp : connectInit.getSnippet()->getExtraGlobalParams()) {
            // If EGP is located on the host
            const auto loc = VarLocation::HOST_DEVICE;//getArchetype().getSparseConnectivityExtraGlobalParamLocation(egp.name);
            if(loc & VarLocationAttribute::HOST) {
                const auto resolvedType = egp.type.resolve(getTypeContext());
                assert(!resolvedType.isPointer());
                const auto pointerType = resolvedType.createPointer();
                const auto pointerToPointerType = pointerType.createPointer();

                // Add field for host pointer
                // **NOTE** none of these need to be dynamic as they are allocated once and pushed with all other merged groups
                groupEnv.addField(pointerToPointerType, "_" + egp.name, egp.name,
                                  [egp](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, egp.name + "SparseConnect"); },
                                  "", GroupMergedFieldType::HOST);

                // Add substitution for direct access to field
                groupEnv.add(pointerType, egp.name, "*$(_" + egp.name + ")");

                // If backend requires seperate device objects, add additional (private) field)
                if(backend.isArrayDeviceObjectRequired()) {
                    groupEnv.addField(pointerToPointerType, "_d_" + egp.name,
                                      "d_" + egp.name,
                                      [egp](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, egp.name + "SparseConnect"); });
                }

                // If backend requires seperate host objects, add additional (private) field)
                if(backend.isArrayHostObjectRequired()) {
                    groupEnv.addField(pointerToPointerType, "_h_" + egp.name,
                                      "h_" + egp.name,
                                      [egp](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, egp.name  + "SparseConnect"); },
                                      "", GroupMergedFieldType::HOST_OBJECT);
                }

                // Generate code to allocate this EGP with count specified by $(0)
                // **NOTE** we generate these with a pointer type as the fields are pointer to pointer
                std::stringstream allocStream;
                CodeGenerator::CodeStream alloc(allocStream);
                backend.genLazyVariableDynamicAllocation(alloc, 
                                                         pointerType, egp.name,
                                                         loc, "$(0)");

                // Add substitution
                groupEnv.add(Type::AllocatePushPullEGP, "allocate" + egp.name, allocStream.str());

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genLazyVariableDynamicPush(push, 
                                                   pointerType, egp.name,
                                                   loc, "$(0)");


                // Add substitution
                groupEnv.add(Type::AllocatePushPullEGP, "push" + egp.name, pushStream.str());
            }
        }
        Transpiler::ErrorHandler errorHandler("Synapse group '" + getArchetype().getName() + "' sparse connectivity host init code");
        prettyPrintStatements(connectInit.getHostInitCodeTokens(), getTypeContext(), groupEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg){ return sg.getConnectivityInitialiser().getParams(); });
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); });
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateInitGroupMerged::name = "CustomUpdateInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with size of custom update
    updateHash([](const CustomUpdateInternal &cg) { return cg.getNumNeurons(); }, hash);

    return hash.get_digest();
}
// ----------------------------------------------------------------------------
void CustomUpdateInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    // Create environment for group
    EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> groupEnv(env, *this);

    // Expose batch size
    const unsigned int updateBatchSize = (getArchetype().getDims() & VarAccessDim::BATCH) ? batchSize : 1;
    groupEnv.add(Type::Uint32.addConst(), "num_batch", std::to_string(updateBatchSize));

    // Initialise custom update variables
    genInitNeuronVarCode<CustomUpdateVarAdapter>(backend, groupEnv, *this, "", "num_neurons", 1, 
                                                 updateBatchSize);
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateInitGroupMerged::name = "CustomWUUpdateInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomWUUpdateInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // If underlying synapse group has kernel weights, update hash with kernel size
    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        updateHash([](const auto &g) { return g.getSynapseGroup()->getKernelSize(); }, hash);
    }
    // Otherwise, update hash with sizes of pre and postsynaptic neuron groups
    else {
        updateHash([](const auto &cg) 
                   {
                       return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
                   }, hash);

        updateHash([](const auto &cg) 
                   {
                       return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
                   }, hash);


        updateHash([](const auto &cg)
                   {
                       return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getMaxConnections(); 
                   }, hash);
    }

    return hash.get_digest();
}
// ----------------------------------------------------------------------------
void CustomWUUpdateInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> groupEnv(env, *this);

    const bool kernel = (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL);
    if(!kernel) {
        groupEnv.print("for(unsigned int i = 0; i < $(num_pre); i++)");
        groupEnv.getStream() << CodeStream::OB(3);
        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
    }
    
    // Expose batch size
    const unsigned int updateBatchSize = (getArchetype().getDims() & VarAccessDim::BATCH) ? batchSize : 1;
    groupEnv.add(Type::Uint32.addConst(), "num_batch", std::to_string(updateBatchSize));

    // Loop through rows
    const std::string stride = kernel ? "$(_kernel_size)" : "$(num_pre) * $(_row_stride)";
    genInitWUVarCode<CustomUpdateVarAdapter>(
        groupEnv, *this, stride, updateBatchSize, false,
        [&backend, kernel, this](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            if (kernel) {
                backend.genKernelCustomUpdateVariableInit(varInitEnv, *this, handler);
            }
            else {
                backend.genDenseSynapseVariableRowInit(varInitEnv, handler);
            }
    
        });

    if(!kernel) {
        groupEnv.getStream() << CodeStream::CB(3);
    }
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateSparseInitGroupMerged::name = "CustomWUUpdateSparseInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomWUUpdateSparseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    
    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups; and max row length
    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const auto& cg)
               {
                   return cg.getSynapseGroup()->getMaxConnections();
               }, hash);

    return hash.get_digest();
}
// ----------------------------------------------------------------------------
void CustomWUUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize)
{
    EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> groupEnv(env, *this);
 
    // Expose batch size
    const unsigned int updateBatchSize = (getArchetype().getDims() & VarAccessDim::BATCH) ? batchSize : 1;
    groupEnv.add(Type::Uint32.addConst(), "num_batch", std::to_string(updateBatchSize));

    genInitWUVarCode<CustomUpdateVarAdapter>(
        groupEnv, *this, "$(num_pre) * $(_row_stride)", updateBatchSize, false,
        [&backend](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            return backend.genSparseSynapseVariableRowInit(varInitEnv, handler); 
        });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdatePreInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdatePreInitGroupMerged::name = "CustomConnectivityUpdatePreInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdatePreInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with size of custom update
    updateHash([](const auto &cg) 
               { 
                   return cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); 
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePreInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int)
{
    genInitNeuronVarCode<CustomConnectivityUpdatePreVarAdapter>(backend, env, *this, "", "num_neurons", 0, 1);
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdatePostInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdatePostInitGroupMerged::name = "CustomConnectivityUpdatePostInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdatePostInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with size of custom update
    updateHash([](const auto &cg)
               {
                   return cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePostInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int)
{
    // Initialise presynaptic custom connectivity update variables
    genInitNeuronVarCode<CustomConnectivityUpdatePostVarAdapter>(backend, env, *this, "", "num_neurons", 0, 1);
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateSparseInitGroupMerged::name = "CustomConnectivityUpdateSparseInit";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdateSparseInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups; and max row length
    updateHash([](const CustomConnectivityUpdateInternal &cg)
               {
                   return static_cast<const SynapseGroupInternal *>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomConnectivityUpdateInternal &cg)
               {
                   return static_cast<const SynapseGroupInternal *>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomConnectivityUpdateInternal &cg)
               {
                   return cg.getSynapseGroup()->getMaxConnections();
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int)
{
    // Initialise custom connectivity update variables
    genInitWUVarCode<CustomConnectivityUpdateVarAdapter>(
        env, *this, "$(num_pre) * $(_row_stride)", 1, false,
        [&backend](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            return backend.genSparseSynapseVariableRowInit(varInitEnv, handler);
        });
}
