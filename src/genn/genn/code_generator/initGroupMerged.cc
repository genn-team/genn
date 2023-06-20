#include "code_generator/initGroupMerged.h"

// GeNN code generator includes
#include "code_generator/groupMergedTypeEnvironment.h"
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

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void genVariableFill(EnvironmentExternalBase &env, const std::string &target, const std::string &value, const std::string &idx, const std::string &stride,
                     VarAccessDuplication varDuplication, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDuplication == VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        env.getStream() <<  env[target] << "[" << env[idx] << "] = " << value << ";" << std::endl;
    }
    // Otherwise
    else {
        env.getStream() << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(env.getStream());
            env.getStream() << env[target] << "[(d * " << stride << ") + " << env[idx] << "] = " << value << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void genScalarFill(EnvironmentExternalBase &env, const std::string &target, const std::string &value,
                   VarAccessDuplication varDuplication, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDuplication == VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        env.getStream() << env[target] << "[0] = " << value << ";" << std::endl;
    }
    // Otherwise
    else {
        env.getStream() << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(env.getStream());
            env.getStream() << env[target] << "[d] = " << value << ";" << std::endl;
        }
    }
}
//------------------------------------------------------------------------
template<typename A, typename F, typename G>
void genInitNeuronVarCode(const BackendBase &backend, EnvironmentExternalBase &env,
                          G &group, F &fieldGroup, const std::string &fieldSuffix, 
                          const std::string &count, size_t numDelaySlots, unsigned int batchSize)
{
    A adaptor(group.getArchetype());
    for (const auto &var : adaptor.getDefs()) {
        // If there is any initialisation code
        const auto resolvedType = var.type.resolve(group.getTypeContext());
        const auto &varInit = adaptor.getInitialisers().at(var.name);
        const auto *snippet = varInit.getSnippet();
        if (!snippet->getCode().empty()) {
            CodeStream::Scope b(env.getStream());

            // Substitute in parameters and derived parameters for initialising variables
            EnvironmentGroupMergedField<G, F> varEnv(env, group, fieldGroup);
            varEnv.addVarInitParams<A>(&G::isVarInitParamHeterogeneous, fieldSuffix);
            varEnv.addVarInitDerivedParams<A>(&G::isVarInitDerivedParamHeterogeneous, fieldSuffix);
            varEnv.addExtraGlobalParams(snippet->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name, fieldSuffix);

            // Add field for variable itself
            varEnv.addField(resolvedType.createPointer(), "_value", var.name + fieldSuffix,
                            [&backend, var](const auto &g, size_t) 
                            { 
                                return backend.getDeviceVarPrefix() + var.name + A(g).getNameSuffix(); 
                            });

            // If variable is shared between neurons
            if (getVarAccessDuplication(var.access) == VarAccessDuplication::SHARED_NEURON) {
                backend.genPopVariableInit(
                    varEnv,
                    [&adaptor, &fieldGroup, &fieldSuffix, &group, &resolvedType, &var, batchSize, numDelaySlots, snippet]
                    (EnvironmentExternalBase &env)
                    {
                        // Generate initial value into temporary variable
                        EnvironmentGroupMergedField<G, F> varInitEnv(env, group, fieldGroup);
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                        prettyPrintStatements(snippet->getCode(), group.getTypeContext(), varInitEnv, errorHandler);
                        
                        // Fill value across all delay slots and batches
                        genScalarFill(varInitEnv, "value", "initVal", getVarAccessDuplication(var.access),
                                      batchSize, adaptor.isVarDelayed(var.name), numDelaySlots);
                    });
            }
            // Otherwise
            else {
                backend.genVariableInit(
                    varEnv, count, "id",
                    [&adaptor, &fieldGroup, &fieldSuffix, &group, &var, &resolvedType, batchSize, count, numDelaySlots, snippet]
                    (EnvironmentExternalBase &env)
                    {
                        // Generate initial value into temporary variable
                        EnvironmentGroupMergedField<G, F> varInitEnv(env, group, fieldGroup);
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                        prettyPrintStatements(snippet->getCode(), group.getTypeContext(), varInitEnv, errorHandler);

                        // Fill value across all delay slots and batches
                        genVariableFill(varInitEnv, "value", "initVal", "id", count,
                                        getVarAccessDuplication(var.access), batchSize, adaptor.isVarDelayed(var.name), numDelaySlots);
                    });
            }
        }
            
    }
}
//------------------------------------------------------------------------
template<typename A, typename G>
void genInitNeuronVarCode(const BackendBase &backend, EnvironmentExternalBase &env,
                          G &group, const std::string &fieldSuffix, 
                          const std::string &count, size_t numDelaySlots, unsigned int batchSize)
{
    genInitNeuronVarCode<A, G, G>(backend, env, group, group, fieldSuffix, count, numDelaySlots, batchSize);
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
template<typename A, typename G, typename V>
void genInitWUVarCode(const BackendBase &backend, EnvironmentExternalBase &env, G &group,
                      const std::string &stride, unsigned int batchSize,
                      V genSynapseVariableRowInitFn)
{
    A adaptor(group.getArchetype());
    for (const auto &var : adaptor.getDefs()) {
        // If this variable has any initialisation code and doesn't require a kernel
        const auto resolvedType = var.type.resolve(group.getTypeContext());
        const auto &varInit = adaptor.getInitialisers().at(var.name);
        const auto *snippet = varInit.getSnippet();
        if(!snippet->getCode().empty() && !snippet->requiresKernel()) {
            CodeStream::Scope b(env.getStream());

            // Substitute in parameters and derived parameters for initialising variables
            EnvironmentGroupMergedField<G> varEnv(env, group);
            varEnv.addVarInitParams<A>(&G::isVarInitParamHeterogeneous);
            varEnv.addVarInitDerivedParams<A>(&G::isVarInitDerivedParamHeterogeneous);
            varEnv.addExtraGlobalParams(snippet->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);

            // Add field for variable itself
            varEnv.addField(resolvedType.createPointer(), "_value", var.name,
                            [&backend, var](const auto &g, size_t) 
                            { 
                                return backend.getDeviceVarPrefix() + var.name + g.getName(); 
                            });

            // Generate target-specific code to initialise variable
            genSynapseVariableRowInitFn(varEnv,
                [&group, &resolvedType, &stride, &var, batchSize, snippet]
                (EnvironmentExternalBase &env)
                { 
                    // Generate initial value into temporary variable
                    EnvironmentGroupMergedField<G> varInitEnv(env, group);
                    varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                    varInitEnv.add(resolvedType, "value", "initVal");
                        
                    // Pretty print variable initialisation code
                    Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                    prettyPrintStatements(snippet->getCode(), group.getTypeContext(), varInitEnv, errorHandler);

                    // Fill value across all batches
                    genVariableFill(varInitEnv, "value", "initVal", "id_syn", stride,
                                    getVarAccessDuplication(var.access), batchSize);
                });
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::CurrentSource
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::CurrentSource::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                    NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    genInitNeuronVarCode<CurrentSourceVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "CS" + std::to_string(getIndex()), 
        "num_neurons", 0, modelMerged.getModel().getBatchSize());
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::CurrentSource::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateVarInitParamHash<CurrentSource, CurrentSourceVarAdapter>(&CurrentSource::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<CurrentSource, CurrentSourceVarAdapter>(&CurrentSource::isVarInitParamReferenced, hash);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::CurrentSource::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &cs){ return cs.getVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::CurrentSource::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &cs){ return cs.getVarInitialisers().at(varName).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::CurrentSource::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::InSynPSM
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynPSM::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                               NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    const std::string fieldSuffix =  "InSyn" + std::to_string(getIndex());

    // Create environment for group
    EnvironmentGroupMergedField<InSynPSM, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Add field for InSyn and zero
    groupEnv.addField(getScalarType().createPointer(), "_out_post", "outPost",
                      [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPost" + g.getFusedPSVarSuffix(); });
    backend.genVariableInit(env, "num_neurons", "id",
        [&modelMerged] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, "_out_post", modelMerged.scalarExpr(0.0), 
                            "id", "num_neurons", VarAccessDuplication::DUPLICATE, 
                            modelMerged.getModel().getBatchSize());

        });

    // If dendritic delays are required
    if(getArchetype().isDendriticDelayRequired()) {
        // Add field for dendritic delay buffer and zero
        groupEnv.addField(getScalarType().createPointer(), "_den_delay", "denDelay",
                          [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelay" + g.getFusedPSVarSuffix(); });
        backend.genVariableInit(env, "num_neurons", "id",
            [&modelMerged, this](EnvironmentExternalBase &varEnv)
            {
                genVariableFill(varEnv, "_den_delay", modelMerged.scalarExpr(0.0),
                                "id", "num_neurons", VarAccessDuplication::DUPLICATE, 
                                modelMerged.getModel().getBatchSize(),
                                true, getArchetype().getMaxDendriticDelayTimesteps());
            });

        // Add field for dendritic delay pointer and zero
        groupEnv.addField(Type::Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr",
                          [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelayPtr" + g.getFusedPSVarSuffix(); });
        backend.genPopVariableInit(env,
            [](EnvironmentExternalBase &varEnv)
            {
                varEnv.getStream() << "*" << varEnv["_den_delay_ptr"] << " = 0;" << std::endl;
            });
    }

    genInitNeuronVarCode<SynapsePSMVarAdapter, NeuronInitGroupMerged>(
        backend, groupEnv, *this, ng, fieldSuffix, "num_neurons", 0, modelMerged.getModel().getBatchSize());
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynPSM::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateVarInitParamHash<InSynPSM, SynapsePSMVarAdapter>(&InSynPSM::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<InSynPSM, SynapsePSMVarAdapter>(&InSynPSM::isVarInitParamReferenced, hash);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynPSM::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getPSVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynPSM::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getPSVarInitialisers().at(varName).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynPSM::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getPSVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynPreOutput
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynPreOutput::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());

    // Create environment for group
    EnvironmentGroupMergedField<OutSynPreOutput, NeuronInitGroupMerged> groupEnv(env, *this, ng);

    // Add 
    groupEnv.addField(getScalarType().createPointer(), "_out_pre", "outPre",
                      [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPre" + g.getFusedPreOutputSuffix(); });
    backend.genVariableInit(env, "num_neurons", "id",
                            [&modelMerged] (EnvironmentExternalBase &varEnv)
                            {
                                genVariableFill(varEnv, "_out_pre", modelMerged.scalarExpr(0.0),
                                                "id", "num_neurons", VarAccessDuplication::DUPLICATE, 
                                                modelMerged.getModel().getBatchSize());
                            });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::InSynWUMPostVars
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynWUMPostVars::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    genInitNeuronVarCode<SynapseWUPostVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "InSynWUMPost" + std::to_string(getIndex()), "num_neurons", 0, modelMerged.getModel().getBatchSize());
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynWUMPostVars::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateVarInitParamHash<InSynWUMPostVars, SynapseWUPostVarAdapter>(&InSynWUMPostVars::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<InSynWUMPostVars, SynapseWUPostVarAdapter>(&InSynWUMPostVars::isVarInitParamReferenced, hash);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynWUMPostVars::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getWUPostVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynWUMPostVars::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getWUPostVarInitialisers().at(varName).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynWUMPostVars::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getWUPostVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynWUMPreVars
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynWUMPreVars::generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                                                       NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged)
{
    genInitNeuronVarCode<SynapseWUPreVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, ng, "OutSynWUMPre" + std::to_string(getIndex()), "num_neurons", 0, modelMerged.getModel().getBatchSize());
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynWUMPreVars::updateHash(boost::uuids::detail::sha1 &hash) const
{
    updateVarInitParamHash<OutSynWUMPreVars, SynapseWUPreVarAdapter>(&OutSynWUMPreVars::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<OutSynWUMPreVars, SynapseWUPreVarAdapter>(&OutSynWUMPreVars::isVarInitParamReferenced, hash);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::OutSynWUMPreVars::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getWUPreVarInitialisers().at(varName).getParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::OutSynWUMPreVars::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName,
                                     [varName](const auto &sg){ return sg.getWUPreVarInitialisers().at(varName).getDerivedParams(); });
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::OutSynWUMPreVars::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getWUPreVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronInitGroupMerged::name = "NeuronInit";
//----------------------------------------------------------------------------
NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                             const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, groups)
{
    // Build vector of vectors containing each child group's merged in syns, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, typeContext,
                             &NeuronGroupInternal::getFusedPSMInSyn,
                             &SynapseGroupInternal::getPSInitHashDigest );

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, typeContext,
                             &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             &SynapseGroupInternal::getPreOutputInitHashDigest );

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, typeContext,
                             &NeuronGroupInternal::getCurrentSources,
                             &CurrentSourceInternal::getInitHashDigest );

    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic weight update model variable, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostVarGroups, typeContext,
                             &NeuronGroupInternal::getFusedInSynWithPostVars,
                             &SynapseGroupInternal::getWUPostInitHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic weight update model variables, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreVarGroups, typeContext,
                             &NeuronGroupInternal::getFusedOutSynWithPreVars,
                             &SynapseGroupInternal::getWUPreInitHashDigest); 
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    /// Update hash with each group's neuron count
    updateHash([](const NeuronGroupInternal &g) { return g.getNumNeurons(); }, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with each group's variable initialisation parameters and derived parameters
    updateVarInitParamHash<NeuronInitGroupMerged, NeuronVarAdapter>(&NeuronInitGroupMerged::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<NeuronInitGroupMerged, NeuronVarAdapter>(&NeuronInitGroupMerged::isVarInitParamReferenced, hash);
    
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
void NeuronInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    const auto &model = modelMerged.getModel();

    // Create environment for group
    EnvironmentGroupMergedField<NeuronInitGroupMerged> groupEnv(env, *this);

    // Initialise spike counts
    genInitSpikeCount(backend, groupEnv, false, model.getBatchSize());
    genInitSpikeCount(backend, groupEnv, true, model.getBatchSize());

    // Initialise spikes
    genInitSpikes(backend, groupEnv, false,  model.getBatchSize());
    genInitSpikes(backend, groupEnv, true,  model.getBatchSize());

    // Initialize spike times
    if(getArchetype().isSpikeTimeRequired()) {
        genInitSpikeTime(backend, groupEnv, "sT",  model.getBatchSize());
    }

    // Initialize previous spike times
    if(getArchetype().isPrevSpikeTimeRequired()) {
        genInitSpikeTime( backend, groupEnv,  "prevST",  model.getBatchSize());
    }
               
    // Initialize spike-like-event times
    if(getArchetype().isSpikeEventTimeRequired()) {
        genInitSpikeTime(backend, groupEnv, "seT",  model.getBatchSize());
    }

    // Initialize previous spike-like-event times
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        genInitSpikeTime(backend, groupEnv, "prevSET",  model.getBatchSize());
    }
       
    // If neuron group requires delays
    if(getArchetype().isDelayRequired()) {
        // Add spike queue pointer field and zero
        groupEnv.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                          [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getName(); });
        backend.genPopVariableInit(env,
            [](EnvironmentExternalBase &varEnv)
            {
                varEnv.getStream() << "*" << varEnv["_spk_que_ptr"] << " = 0;" << std::endl;
            });
    }

    // Initialise neuron variables
    genInitNeuronVarCode<NeuronVarAdapter>(backend, env, *this, "", "num_neurons", 0, modelMerged.getModel().getBatchSize());

    // Generate initialisation code for child groups
    for (auto &cs : m_MergedCurrentSourceGroups) {
        cs.generate(backend, env, *this, modelMerged);
    }
    for(auto &sg : m_MergedInSynPSMGroups) {
        sg.generate(backend, env, *this, modelMerged);
    }
    for (auto &sg : m_MergedOutSynPreOutputGroups) {
        sg.generate(backend, env, *this, modelMerged);
    }  
    for (auto &sg : m_MergedOutSynWUMPreVarGroups) {
        sg.generate(backend, env, *this, modelMerged);
    }
    for (auto &sg : m_MergedInSynWUMPostVarGroups) {
        sg.generate(backend, env, *this, modelMerged);
    }
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isParamValueHeterogeneous(paramName,
                                      [varName](const NeuronGroupInternal &sg) { return sg.getVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isParamValueHeterogeneous(paramName,
                                      [varName](const NeuronGroupInternal &sg){ return sg.getVarInitialisers().at(varName).getDerivedParams(); }));
}
//--------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikeCount(const BackendBase &backend, EnvironmentExternalBase &env, 
                                              bool spikeEvent, unsigned int batchSize)
{
    // Is initialisation required at all
    const bool required = spikeEvent ? getArchetype().isSpikeEventRequired() : true;
    if(required) {
        // Add spike count field
        const std::string suffix = spikeEvent ? "Evnt" : "";
        EnvironmentGroupMergedField<NeuronInitGroupMerged> spikeCountEnv(env, *this);
        spikeCountEnv.addField(Type::Uint32.createPointer(), "_spk_cnt", "spkCnt" + suffix,
                               [&backend, &suffix](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCnt" + suffix + g.getName(); });

        // Generate variable initialisation code
        backend.genPopVariableInit(env,
            [batchSize, spikeEvent, this] (EnvironmentExternalBase &spikeCountEnv)
            {
                // Is delay required
                const bool delayRequired = spikeEvent ?
                    getArchetype().isDelayRequired() :
                    (getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genScalarFill(spikeCountEnv, "_spk_cnt", "0", VarAccessDuplication::DUPLICATE, batchSize, delayRequired, getArchetype().getNumDelaySlots());
            });
    }

}
//--------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikes(const BackendBase &backend, EnvironmentExternalBase &env, 
                                          bool spikeEvent, unsigned int batchSize)
{
    // Is initialisation required at all
    const bool required = spikeEvent ? getArchetype().isSpikeEventRequired() : true;
    if(required) {
        // Add spike count field
        const std::string suffix = spikeEvent ? "Evnt" : "";
        EnvironmentGroupMergedField<NeuronInitGroupMerged> spikeEnv(env, *this);
        spikeEnv.addField(Type::Uint32.createPointer(), "_spk", "spk" + suffix,
                          [&backend, &suffix](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpk" + suffix + g.getName(); });


        // Generate variable initialisation code
        backend.genVariableInit(spikeEnv, "num_neurons", "id",
            [batchSize, spikeEvent, this] (EnvironmentExternalBase &varEnv)
            {
   
                // Is delay required
                const bool delayRequired = spikeEvent ?
                    getArchetype().isDelayRequired() :
                    (getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genVariableFill(varEnv, "_spk", "0", "id", "num_neurons", 
                                VarAccessDuplication::DUPLICATE, batchSize, delayRequired, getArchetype().getNumDelaySlots());
            });
    }
}
//------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikeTime(const BackendBase &backend, EnvironmentExternalBase &env,
                                             const std::string &varName, unsigned int batchSize)
{
    // Add spike time field
    EnvironmentGroupMergedField<NeuronInitGroupMerged> timeEnv(env, *this);
    timeEnv.addField(getTimeType().createPointer(), "_time", varName,
                     [&backend, varName](const auto &g, size_t) { return backend.getDeviceVarPrefix() + varName + g.getName(); });


    // Generate variable initialisation code
    backend.genVariableInit(env, "num_neurons", "id",
        [batchSize, varName, this] (EnvironmentExternalBase &varEnv)
        {
            genVariableFill(varEnv, varName, "-TIME_MAX", "id", "num_neurons", VarAccessDuplication::DUPLICATE, 
                            batchSize, getArchetype().isDelayRequired(), getArchetype().getNumDelaySlots());
        });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseInitGroupMerged::name = "SynapseInit";
//----------------------------------------------------------------------------
void SynapseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Create environment for group
    EnvironmentGroupMergedField<SynapseInitGroupMerged> groupEnv(env, *this);

    // If model is batched and has kernel weights
    const bool kernel = (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL);
    if (kernel && modelMerged.getModel().getBatchSize() > 1) {
        // Loop through kernel dimensions and multiply together to calculate batch stride
        // **TODO** dependency for add
        std::ostringstream batchStrideInit;
        batchStrideInit << "const unsigned int batchStride = ";
        const auto &kernelSize = getArchetype().getKernelSize();
        for (size_t i = 0; i < kernelSize.size(); i++) {
            batchStrideInit << getKernelSize(i);

            if (i != (kernelSize.size() - 1)) {
                batchStrideInit << " * ";
            }
        }
        batchStrideInit << ";" << std::endl;
        groupEnv.add(Type::Uint32.addConst(), "_batch_stride", "batchStride",
                     {groupEnv.addInitialiser(batchStrideInit.str())});
    }

    
    // If we're using non-kernel weights, generate loop over source neurons
    if (!kernel) {
        groupEnv.getStream() << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
        groupEnv.getStream() << CodeStream::OB(1);    
        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
    }

    // Generate initialisation code
    const std::string stride = kernel ? groupEnv["_batch_stride"] : groupEnv["num_pre"] + " * " + groupEnv["_row_stride"];
    genInitWUVarCode<SynapseWUVarAdapter>(backend, groupEnv, *this, stride, modelMerged.getModel().getBatchSize(),
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
void SynapseSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    genInitWUVarCode<SynapseWUVarAdapter>(backend, env, *this, env["num_pre"] + " * " + env["_row_stride"], modelMerged.getModel().getBatchSize(),
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
void SynapseConnectivityInitGroupMerged::generateSparseRowInit(const BackendBase &backend, EnvironmentExternalBase &env)
{
    genInitConnectivity(backend, env, true);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseColumnInit(const BackendBase &backend, EnvironmentExternalBase &env)
{
    genInitConnectivity(backend, env, false);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateKernelInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Create environment for group
    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(env, *this);

    // Add substitution
    // **TODO** dependencies on kernel fields
    groupEnv.add(Type::Uint32, "id_kernel", "kernelInd", 
                 {groupEnv.addInitialiser("const unsigned int kernelInd = " + getKernelIndex(groupEnv) + ";")});

    // Initialise single (hence empty lambda function) synapse variable
    genInitWUVarCode<SynapseWUVarAdapter>(backend, groupEnv, *this, groupEnv["num_pre"] + " * " + groupEnv["_row_stride"], modelMerged.getModel().getBatchSize(),
                                          [](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
                                          {
                                              handler(varInitEnv);
                                          });
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::genInitConnectivity(const BackendBase &backend, EnvironmentExternalBase &env, bool rowNotColumns)
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    const auto *snippet = connectInit.getSnippet();

    // Create environment for group
    EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> groupEnv(env, *this);

    // Add substitution for end function 
    // **TODO** remove
    groupEnv.add(Type::ResolvedType::createFunction(Type::Void, {}), rowNotColumns ? "endRow" : "endCol", "break;");

    // Substitute in parameters and derived parameters for initialising variables
    groupEnv.addConnectInitParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                  &SynapseConnectivityInitGroupMerged::isSparseConnectivityInitParamHeterogeneous);
    groupEnv.addConnectInitDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                         &SynapseConnectivityInitGroupMerged::isSparseConnectivityInitDerivedParamHeterogeneous);
    groupEnv.addExtraGlobalParams(snippet->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", "");


    // Initialise state variables and loop on generated code to initialise sparse connectivity
    groupEnv.getStream() << "// Build sparse connectivity" << std::endl;
    const auto stateVars = rowNotColumns ? snippet->getRowBuildStateVars() : snippet->getColBuildStateVars();
    const std::string context = rowNotColumns ? "row" : "column";
    for(const auto &a : stateVars) {
       
        groupEnv.getStream() << a.type.resolve(getTypeContext()).getName() << " " << a.name << " = ";

        Transpiler::ErrorHandler errorHandler("Connectivity init " + context + " build state var" + std::to_string(getIndex()));
        prettyPrintExpression(a.value, getTypeContext(), groupEnv, errorHandler);
        
        groupEnv.getStream() << ";" << std::endl;
    }
    groupEnv.getStream() << "while(true)";
    {
        CodeStream::Scope b(groupEnv.getStream());

        Transpiler::ErrorHandler errorHandler("Connectivity init " + context + " build" + std::to_string(getIndex()));
        prettyPrintStatements(rowNotColumns ? snippet->getRowBuildCode() : snippet->getColBuildCode(), getTypeContext(), groupEnv, errorHandler);
    }
}


// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//-------------------------------------------------------------------------
void SynapseConnectivityHostInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
 
    CodeStream::Scope b(env.getStream());
    env.getStream() << "// merged synapse connectivity host init group " << getIndex() << std::endl;
    env.getStream() << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(env.getStream());

        // Get reference to group
        env.getStream() << "const auto *group = &mergedSynapseConnectivityHostInitGroup" << getIndex() << "[g]; " << std::endl;
        
        // Create environment for group
        EnvironmentGroupMergedField<SynapseConnectivityHostInitGroupMerged> groupEnv(env, *this);
        const auto &connectInit = getArchetype().getConnectivityInitialiser();

        // If matrix type is procedural then initialized connectivity init snippet will potentially be used with multiple threads per spike. 
        // Otherwise it will only ever be used for initialization which uses one thread per row
        const size_t numThreads = (getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) ? getArchetype().getNumThreadsPerSpike() : 1;

        // Create substitutions
        groupEnv.addField(Type::Uint32.addConst(), "num_pre",
                          Type::Uint32, "numSrcNeurons", 
                          [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        groupEnv.addField(Type::Uint32.addConst(), "num_post",
                          Type::Uint32, "numTrgNeurons", 
                          [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
        groupEnv.add(Type::Uint32.addConst(), "num_threads", std::to_string(numThreads));

        groupEnv.addConnectInitParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                      &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);
        groupEnv.addConnectInitDerivedParams("", &SynapseGroupInternal::getConnectivityInitialiser,
                                             &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

        /*const auto &pointerToPointerToEGP = e.type.resolve(getTypeContext()).createPointer().createPointer();
        addField(pointerToPointerToEGP, e.name,
                 [e](const SynapseGroupInternal &g, size_t) { return "&" + e.name + g.getName(); },
                 GroupMergedFieldType::HOST_DYNAMIC);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(pointerToPointerToEGP, backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getDeviceVarPrefix() + e.name + g.getName();
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
        if(!backend.getHostVarPrefix().empty()) {
            addField(pointerToPointerToEGP, backend.getHostVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getHostVarPrefix() + e.name + g.getName();
                     },
                     GroupMergedFieldType::DYNAMIC);
        }*/
        // Loop through EGPs
        for(const auto &egp : connectInit.getSnippet()->getExtraGlobalParams()) {
            // If EGP is located on the host
            const auto loc = getArchetype().getSparseConnectivityExtraGlobalParamLocation(egp.name);
            if(loc & VarLocation::HOST) {
                const auto resolvedType = egp.type.resolve(getTypeContext());
                assert(!resolvedType.isPointer());
                const auto pointerType = resolvedType.createPointer();
                const auto pointerToPointerType = pointerType.createPointer();

                // Add field for host pointer
                // **NOTE** use [0] to dereference on access to obta
                groupEnv.addField(pointerType, egp.name,
                                  pointerToPointerType, egp.name,
                                  [egp](const auto &g, size_t) { return "&" + egp.name + g.getName(); },
                                  "[0]", GroupMergedFieldType::HOST_DYNAMIC);

                // If backend requires seperate device variables, add additional (private) field)
                if(!backend.getDeviceVarPrefix().empty()) {
                    groupEnv.addField(pointerToPointerType, "_" + backend.getDeviceVarPrefix() + egp.name,
                                      backend.getDeviceVarPrefix() + egp.name,
                                      [egp](const auto &g, size_t) { return "&" + egp.name + g.getName(); },
                                      "", GroupMergedFieldType::DYNAMIC);
                }

                // If backend requires seperate host variables, add additional (private) field)
                if(!backend.getHostVarPrefix().empty()) {
                    groupEnv.addField(pointerToPointerType, "_" + backend.getHostVarPrefix() + egp.name,
                                      backend.getHostVarPrefix() + egp.name,
                                      [egp, &backend](const SynapseGroupInternal &g, size_t)
                                      {
                                          return "&" + backend.getHostVarPrefix() + egp.name + g.getName();
                                      },
                                      "", GroupMergedFieldType::DYNAMIC);
                }

                // Generate code to allocate this EGP with count specified by $(0)
                // **NOTE** we generate these with a pointer type as the fields are pointer to pointer
                std::stringstream allocStream;
                const auto &pointerToEGP = resolvedType.createPointer();
                CodeGenerator::CodeStream alloc(allocStream);
                backend.genVariableDynamicAllocation(alloc, 
                                                     pointerToEGP, egp.name,
                                                     loc, "$(0)", "group->");

                // Add substitution
                groupEnv.add(Type::ResolvedType::createFunction(Type::Void, {Type::Uint32}), "allocate" + egp.name, allocStream.str());

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genVariableDynamicPush(push, 
                                               pointerToEGP, egp.name,
                                               loc, "$(0)", "group->");


                // Add substitution
                groupEnv.add(Type::ResolvedType::createFunction(Type::Void, {Type::Uint32}), "push" + egp.name, pushStream.str());
            }
        }
        Transpiler::ErrorHandler errorHandler("Connectivity host init" + std::to_string(getIndex()));
        prettyPrintStatements(connectInit.getSnippet()->getHostInitCode(), getTypeContext(), groupEnv, errorHandler);
    }
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg){ return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isSparseConnectivityInitParamReferenced(const std::string &paramName) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
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

    // Update hash with size of custom update
    updateHash([](const CustomUpdateInternal &cg) { return cg.getSize(); }, hash);

    return hash.get_digest();
}
// ----------------------------------------------------------------------------
void CustomUpdateInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Initialise custom update variables
    genInitNeuronVarCode<CustomUpdateVarAdapter>(backend, env, *this, "", "size", 1, 
                                                 getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1);        
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
void CustomWUUpdateInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> groupEnv(env, *this);

    const bool kernel = (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL);
    if(kernel) {
        // Loop through kernel size dimensions
        for (size_t d = 0; d < getArchetype().getSynapseGroup()->getKernelSize().size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(d)) {
                groupEnv.addField(Type::Uint32, "_kernel_size_" + std::to_string(d), "kernelSize" + std::to_string(d),
                                  [d](const auto &g, size_t) { return std::to_string(g.getSynapseGroup()->getKernelSize().at(d)); });
            }
        }

        if(modelMerged.getModel().getBatchSize() > 1) {
            // Loop through kernel dimensions and multiply together to calculate batch stride
            std::ostringstream batchStrideInit;
            batchStrideInit << "const unsigned int batchStride = ";
            const auto &kernelSize = getArchetype().getSynapseGroup()->getKernelSize();
            for (size_t i = 0; i < kernelSize.size(); i++) {
                batchStrideInit << getKernelSize(i);

                if (i != (kernelSize.size() - 1)) {
                    batchStrideInit << " * ";
                }
            }
            batchStrideInit << ";" << std::endl;
            groupEnv.add(Type::Uint32.addConst(), "_batch_stride", "batchStride",
                         {groupEnv.addInitialiser(batchStrideInit.str())});
        }
    }
    else {
        groupEnv.addField(Type::Uint32.addConst(), "num_pre",
                          Type::Uint32, "numSrcNeurons", 
                          [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
        groupEnv.addField(Type::Uint32.addConst(), "num_post",
                          Type::Uint32, "numTrgNeurons", 
                          [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });
        groupEnv.addField(Type::Uint32, "_row_stride", "rowStride", 
                          [&backend](const auto &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });
      

        groupEnv.getStream() << "for(unsigned int i = 0; i < " << groupEnv["num_pre"] << "; i++)";
        groupEnv.getStream() << CodeStream::OB(3);
        groupEnv.add(Type::Uint32.addConst(), "id_pre", "i");
    }
 
    // Loop through rows
    const std::string stride = kernel ? groupEnv["_batch_stride"] : groupEnv["num_pre"] + " * " + groupEnv["_row_stride"];
    genInitWUVarCode<CustomUpdateVarAdapter>(
        backend, groupEnv, *this, stride, getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
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
void CustomWUUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Create environment for group
    EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> groupEnv(env, *this);

    /* addField(Uint32, "rowStride",
             [&backend](const auto &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });

    addField(Uint32, "numSrcNeurons",
             [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField(Uint32, "numTrgNeurons",
             [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });

    addField(Uint32.createPointer(), "rowLength", 
             [&backend](const auto &cg, size_t) 
             { 
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "rowLength" + sg->getName();
             });
    addField(getArchetype().getSynapseGroup()->getSparseIndType().createPointer(), "ind", 
             [&backend](const auto &cg, size_t) 
             { 
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "ind" + sg->getName();
             });*/

    genInitWUVarCode<CustomUpdateVarAdapter>(backend, groupEnv, *this, groupEnv["num_pre"] + " * "  + groupEnv["_row_stride"],
                                             getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
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

    // Update hash with size of custom update
    updateHash([](const auto &cg) 
               { 
                   return cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); 
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePreInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
     // Create environment for group
     EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> groupEnv(env, *this);

     groupEnv.addField(Type::Uint32.addConst(), "size", 
                       Type::Uint32, "size",
                       [](const auto &c, size_t) 
                       { 
                           return std::to_string(c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
                       });

    // Initialise presynaptic custom connectivity update variables
    genInitNeuronVarCode<CustomConnectivityUpdatePreVarAdapter>(backend, groupEnv, *this, "", "size", 0, modelMerged.getModel().getBatchSize());
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

    // Update hash with size of custom update
    updateHash([](const auto &cg)
               {
                   return cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePostInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Create environment for group
    EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> groupEnv(env, *this);

    groupEnv.addField(Type::Uint32.addConst(), "size", 
                      Type::Uint32, "size",
                      [](const auto &c, size_t) 
                      { 
                          return std::to_string(c.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); 
                      });

    // Initialise presynaptic custom connectivity update variables
    genInitNeuronVarCode<CustomConnectivityUpdatePostVarAdapter>(backend, groupEnv, *this, "", "size", 0, modelMerged.getModel().getBatchSize());
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
void CustomConnectivityUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged)
{
    // Create environment for group
    EnvironmentGroupMergedField<CustomConnectivityUpdateSparseInitGroupMerged> groupEnv(env, *this);

    groupEnv.addField(Type::Uint32.addConst(), "num_pre",
                      Type::Uint32, "numSrcNeurons", 
                      [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    groupEnv.addField(Type::Uint32.addConst(), "num_post",
                      Type::Uint32, "numTrgNeurons", 
                      [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });
    groupEnv.addField(Type::Uint32, "_row_stride", "rowStride", 
                      [&backend](const auto &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });
                        
    groupEnv.addField(Type::Uint32.createPointer(), "_row_length", "rowLength",
                      [&backend](const CustomConnectivityUpdateInternal &cg, size_t)
                      {
                          return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName();
                      });
    groupEnv.addField(getArchetype().getSynapseGroup()->getSparseIndType().createPointer(), "_ind", "ind",
                      [&backend](const auto &cg, size_t)
                      {
                          return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName();
                      });

    // Initialise custom connectivity update variables
    genInitWUVarCode<CustomConnectivityUpdateVarAdapter>(
        backend, groupEnv, *this, groupEnv["num_pre"] + " * " + groupEnv["_row_stride"], 1,
        [&backend](EnvironmentExternalBase &varInitEnv, BackendBase::HandlerEnv handler)
        {
            return backend.genSparseSynapseVariableRowInit(varInitEnv, handler);
        });
}
