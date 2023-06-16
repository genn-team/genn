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
template<typename A, typename G, typename F>
void genInitNeuronVarCode(const BackendBase &backend, EnvironmentExternalBase &env,
                          G &group, F &fieldGroup, const std::string &fieldSuffix, 
                          const std::string &count, size_t numDelaySlots, unsigned int batchSize)
{
    A adaptor(groupMerged.getArchetype());
    for (const auto &var : adaptor.getDefs()) {
        // If there is any initialisation code
        const auto resolvedType = var.type.resolve(group.getTypeContext());
        const auto &varInit = adaptor.getInitialisers().at(var.name);
        const auto *snippet = adaptor.getInitialisers().at(var.name).getSnippet();
        if (!snippet->getCode().empty()) {
            CodeStream::Scope b(env.getStream());

            EnvironmentGroupMergedField<G, F> varEnv(env, group, fieldGroup);

            // Substitute in parameters and derived parameters for initialising variables
            varEnv.addVarInitParams<A>(&G::isVarInitParamHeterogeneous, fieldSuffix);
            varEnv.addVarInitDerivedParams<A>(&G::isVarInitDerivedParamHeterogeneous, fieldSuffix);
            varEnv.addExtraGlobalParams(snippet->getExtraGlobalParameters(), backend.getDeviceVarPrefix(), var.name, fieldSuffix);

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
                    [&adaptor, &fieldSuffix, &group, &resolvedType, &var, batchSize, numDelaySlots, snippet]
                    (EnvironmentExternalBase &varInitEnv)
                    {
                        // Generate initial value into temporary variable
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                        prettyPrintStatements(snippet->getCode(), group.getTypeContext(), varInitEnv, errorHandler);
                        
                        // Fill value across all delay slots and batches
                        genScalarFill(varInitEnv, "_value", "initVal", getVarAccessDuplication(var.access),
                                      batchSize, adaptor.isVarDelayed(var.name), numDelaySlots);
                    });
            }
            // Otherwise
            else {
                backend.genVariableInit(
                    varEnvs, count, "id",
                    [&adaptor, &fieldSuffix, &group, &var, &resolvedType, batchSize, count, numDelaySlots]
                    (EnvironmentExternal &varInitEnv)
                    {
                        // Generate initial value into temporary variable
                        varInitEnv.getStream() << resolvedType.getName() << " initVal;" << std::endl;
                        varInitEnv.add(resolvedType, "value", "initVal");
                        
                        // Pretty print variable initialisation code
                        Transpiler::ErrorHandler errorHandler("Variable '" + var.name + "' init code" + std::to_string(group.getIndex()));
                        prettyPrintStatements(snippet->getCode(), group.getTypeContext(), varInitEnv, errorHandler);

                        // Fill value across all delay slots and batches
                        genVariableFill(varInitEnv(), "_value", "initVal", "id", count,
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
    genInitNeuronVarCode(backend, env, group, group, fieldSuffix, count, numDelaySlots, batchSize);
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
template<typename P, typename D, typename G>
void genInitWUVarCode(CodeStream &os, const ModelSpecMerged &modelMerged, const Substitutions &popSubs, 
                      const Models::Base::VarVec &vars, const std::unordered_map<std::string, Models::VarInit> &varInitialisers, 
                      const std::string &stride, const size_t groupIndex, unsigned int batchSize,
                      P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn, G genSynapseVariableRowInitFn)
{
    for (const auto &var : vars) {
        const auto &varInit = varInitialisers.at(var.name);

        // If this variable has any initialisation code and doesn't require a kernel
        if(!varInit.getSnippet()->getCode().empty() && !varInit.getSnippet()->requiresKernel()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            genSynapseVariableRowInitFn(os, popSubs,
                [&var, &varInit, &stride, &modelMerged, batchSize, groupIndex, isParamHeterogeneousFn, isDerivedParamHeterogeneousFn]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                      [&var, isParamHeterogeneousFn](const std::string &p) { return isParamHeterogeneousFn(var.name, p); },
                                                      "", "group->", var.name);
                    varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                      [&var, isDerivedParamHeterogeneousFn](const std::string &p) { return isDerivedParamHeterogeneousFn(var.name, p); },
                                                      "", "group->", var.name);
                    varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                                   "", "group->", var.name);

                    // Generate initial value into temporary variable
                    os << var.type.resolve(modelMerged.getTypeContext()).getName() << " initVal;" << std::endl;
                    varSubs.addVarSubstitution("value", "initVal");
                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.applyCheckUnreplaced(code, "initVar : merged" + var.name + std::to_string(groupIndex));
                    //code = ensureFtype(code, scalarType);
                    os << code << std::endl;

                    // Fill value across all batches
                    genVariableFill(os,  var.name, "initVal", "id_syn", stride,
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
    genInitNeuronVarCode<CurrentSourceVarAdapter, CurrentSource, NeuronInitGroupMerged>(
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

    genInitNeuronVarCode<SynapsePSMVarAdapter, InSynPSM, NeuronInitGroupMerged>(
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
    genInitNeuronVarCode<SynapseWUPostVarAdapter, InSynWUMPostVars, NeuronInitGroupMerged>(
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
    genInitNeuronVarCode<SynapseWUPreVarAdapter, OutSynWUMPreVars, NeuronInitGroupMerged>(
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
:   NeuronGroupMergedBase(index, typeContext, backend, groups)
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
       
    // If neuron group requires delays, zero spike queue pointer
    if(getArchetype().isDelayRequired()) {
        backend.genPopVariableInit(env,
            [](CodeStream &os, Substitutions &)
            {
                os << "*group->spkQuePtr = 0;" << std::endl;
            });
    }

    // Initialise neuron variables
    genInitNeuronVarCode<NeuronVarAdapter, NeuronInitGroupMerged>(
        backend, env, *this, "", "num_neurons", 0, modelMerged.getModel().getBatchSize());

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
void SynapseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // If model is batched and has kernel weights
    const bool kernel = (getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL);
    if (kernel && modelMerged.getModel().getBatchSize() > 1) {
        // Loop through kernel dimensions and multiply together to calculate batch stride
        os << "const unsigned int batchStride = ";
        const auto &kernelSize = getArchetype().getKernelSize();
        for (size_t i = 0; i < kernelSize.size(); i++) {
            os << getKernelSize(i);

            if (i != (kernelSize.size() - 1)) {
                os << " * ";
            }
        }
        os << ";" << std::endl;;
    }

    
    // If we're using non-kernel weights, generate loop over source neurons
    if (!kernel) {
        os << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
        os << CodeStream::OB(1);    
        popSubs.addVarSubstitution("id_pre", "i");
    }

    // Generate initialisation code
    const std::string stride = kernel ? "batchStride" : "group->numSrcNeurons * group->rowStride";
    genInitWUVarCode(os, modelMerged, popSubs, getArchetype().getWUModel()->getVars(),
                     getArchetype().getWUVarInitialisers(), stride, getIndex(), modelMerged.getModel().getBatchSize(),
                     [this](const std::string &v, const std::string &p) { return isWUVarInitParamHeterogeneous(v, p); },
                     [this](const std::string &v, const std::string &p) { return isWUVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend, kernel, this](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         if (kernel) {
                             backend.genKernelSynapseVariableInit(os, *this, kernelSubs, handler);
                         }
                         else {
                             backend.genDenseSynapseVariableRowInit(os, kernelSubs, handler);
                         }
                     });

    // If we're using non-kernel weights, close loop
    if (!kernel) {
        os << CodeStream::CB(1);
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseSparseInitGroupMerged::name = "SynapseSparseInit";
//----------------------------------------------------------------------------
void SynapseSparseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitWUVarCode(os, modelMerged, popSubs, getArchetype().getWUModel()->getVars(),
                     getArchetype().getWUVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(), modelMerged.getModel().getBatchSize(),
                     [this](const std::string &v, const std::string &p) { return isWUVarInitParamHeterogeneous(v, p); },
                     [this](const std::string &v, const std::string &p) { return isWUVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler); 
                     });
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityInitGroupMerged::name = "SynapseConnectivityInit";
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseRowInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &, Substitutions &popSubs) const
{
    genInitConnectivity(os, popSubs, true);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseColumnInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &, Substitutions &popSubs) const
{
    genInitConnectivity(os, popSubs, false);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateKernelInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Generate kernel index and add to substitutions
    os << "const unsigned int kernelInd = ";
    genKernelIndex(os, popSubs);
    os << ";" << std::endl;
    popSubs.addVarSubstitution("id_kernel", "kernelInd");

    for(const auto &var : getArchetype().getWUModel()->getVars()) {
        const auto &varInit = getArchetype().getWUVarInitialisers().at(var.name);

        // If this variable require a kernel
        if(varInit.getSnippet()->requiresKernel()) {
            CodeStream::Scope b(os);

            popSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                              [&var, this](const std::string &p) { return isWUVarInitParamHeterogeneous(var.name, p); },
                                              "", "group->", var.name);
            popSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                            [&var, this](const std::string &p) { return isWUVarInitDerivedParamHeterogeneous(var.name, p); },
                                            "", "group->", var.name);
            popSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                            "", "group->", var.name);

            // Generate initial value into temporary variable
            os << var.type.resolve(getTypeContext()).getName() << " initVal;" << std::endl;
            popSubs.addVarSubstitution("value", "initVal");
            std::string code = varInit.getSnippet()->getCode();
            //popSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(sg.getIndex()));
            popSubs.apply(code);
            //code = ensureFtype(code, modelMerged.getModel().getPrecision());
            os << code << std::endl;

            // Fill value across all batches
            genVariableFill(os,  var.name, "initVal", popSubs["id_syn"], "group->numSrcNeurons * group->rowStride", 
                            getVarAccessDuplication(var.access), modelMerged.getModel().getBatchSize());
        }
    }
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::genInitConnectivity(CodeStream &os, Substitutions &popSubs, bool rowNotColumns) const
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    const auto *snippet = connectInit.getSnippet();

    // Add substitutions
    popSubs.addFuncSubstitution(rowNotColumns ? "endRow" : "endCol", 0, "break");
    popSubs.addParamValueSubstitution(snippet->getParamNames(), connectInit.getParams(),
                                      [this](const std::string &p) { return isSparseConnectivityInitParamHeterogeneous(p);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(snippet->getDerivedParams(), connectInit.getDerivedParams(),
                                    [this](const std::string &p) { return isSparseConnectivityInitDerivedParamHeterogeneous(p);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(snippet->getExtraGlobalParams(), "", "group->");

    // Initialise state variables and loop on generated code to initialise sparse connectivity
    os << "// Build sparse connectivity" << std::endl;
    const auto stateVars = rowNotColumns ? snippet->getRowBuildStateVars() : snippet->getColBuildStateVars();
    for(const auto &a : stateVars) {
        // Apply substitutions to value
        std::string value = a.value;
        popSubs.applyCheckUnreplaced(value, "initSparseConnectivity state var : merged" + std::to_string(getIndex()));
        //value = ensureFtype(value, ftype);

        os << a.type.resolve(getTypeContext()).getName() << " " << a.name << " = " << value << ";" << std::endl;
    }
    os << "while(true)";
    {
        CodeStream::Scope b(os);

        // Apply substitutions to row build code
        std::string code = rowNotColumns ? snippet->getRowBuildCode() : snippet->getColBuildCode();
        popSubs.addVarNameSubstitution(stateVars);
        popSubs.applyCheckUnreplaced(code, "initSparseConnectivity : merged" + std::to_string(getIndex()));
        //code = ensureFtype(code, ftype);

        // Write out code
        os << code << std::endl;
    }
}


// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//------------------------------------------------------------------------
SynapseConnectivityHostInitGroupMerged::SynapseConnectivityHostInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    // **TODO** these could be generic
    addField(Uint32, "numSrcNeurons",
             [](const auto &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField(Uint32, "numTrgNeurons",
             [](const auto &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    addField(Uint32, "rowStride",
             [&backend](const auto &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

    // Add heterogeneous connectivity initialiser model parameters
    addHeterogeneousParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getConnectivityInitialiser().getSnippet()->getParamNames(), "",
        [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous);

    // Add heterogeneous connectivity initialiser derived parameters
    addHeterogeneousDerivedParams<SynapseConnectivityHostInitGroupMerged>(
        getArchetype().getConnectivityInitialiser().getSnippet()->getDerivedParams(), "",
        [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); },
        &SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous);

    // Add EGP pointers to struct for both host and device EGPs if they are seperate
    const auto egps = getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
    for(const auto &e : egps) {
        const auto &pointerToPointerToEGP = e.type.resolve(getTypeContext()).createPointer().createPointer();
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
        }
    }
}
//-------------------------------------------------------------------------
void SynapseConnectivityHostInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged) const
{
    CodeStream::Scope b(os);
    os << "// merged synapse connectivity host init group " << getIndex() << std::endl;
    os << "for(unsigned int g = 0; g < " << getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(os);

        // Get reference to group
        os << "const auto *group = &mergedSynapseConnectivityHostInitGroup" << getIndex() << "[g]; " << std::endl;

        const auto &connectInit = getArchetype().getConnectivityInitialiser();

        // If matrix type is procedural then initialized connectivity init snippet will potentially be used with multiple threads per spike. 
        // Otherwise it will only ever be used for initialization which uses one thread per row
        const size_t numThreads = (getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) ? getArchetype().getNumThreadsPerSpike() : 1;

        // Create substitutions
        Substitutions subs;
        subs.addVarSubstitution("rng", "hostRNG");
        subs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        subs.addVarSubstitution("num_post", "group->numTrgNeurons");
        subs.addVarSubstitution("num_threads", std::to_string(numThreads));
        subs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "*group->");
        subs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                       [this](const std::string &p) { return isConnectivityInitParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                     [this](const std::string &p) { return isConnectivityInitDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        for(const auto &egp : connectInit.getSnippet()->getExtraGlobalParams()) {
            // If EGP is located on the host
            const auto loc = getArchetype().getSparseConnectivityExtraGlobalParamLocation(egp.name);
            if(loc & VarLocation::HOST) {
                // Generate code to allocate this EGP with count specified by $(0)
                // **NOTE** we generate these with a pointer type as the fields are pointer to pointer
                std::stringstream allocStream;
                const auto &pointerToEGP = egp.type.resolve(getTypeContext()).createPointer();
                CodeGenerator::CodeStream alloc(allocStream);
                backend.genVariableDynamicAllocation(alloc, 
                                                     pointerToEGP, egp.name,
                                                     loc, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("allocate" + egp.name, 1, allocStream.str());

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genVariableDynamicPush(push, 
                                               pointerToEGP, egp.name,
                                               loc, "$(0)", "group->");


                // Add substitution
                subs.addFuncSubstitution("push" + egp.name, 1, pushStream.str());
            }
        }
        std::string code = connectInit.getSnippet()->getHostInitCode();
        subs.applyCheckUnreplaced(code, "hostInitSparseConnectivity : merged" + std::to_string(getIndex()));
        //code = ensureFtype(code, modelMerged.getModel().getPrecision());

        // Write out code
        os << code << std::endl;
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
CustomUpdateInitGroupMerged::CustomUpdateInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                         const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateInternal, CustomUpdateVarAdapter>(index, typeContext, backend, groups)
{
    addField(Type::Uint32, "size",
             [](const auto &c, size_t) { return std::to_string(c.getSize()); });
}
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
void CustomUpdateInitGroupMerged::generateInit(const BackendBase &backend, EnvironmentExternal &env, const ModelSpecMerged &modelMerged) const
{
    // Initialise custom update variables
    genInitNeuronVarCode<CustomUpdateVarAdapter>(backend, env, *this, m_VarInitASTs, ""), 
                                                  "size", 1, getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1);
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateInitGroupMerged::name = "CustomWUUpdateInit";
//----------------------------------------------------------------------------
CustomWUUpdateInitGroupMerged::CustomWUUpdateInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                             const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal, CustomUpdateVarAdapter>(index, typeContext, backend, groups)
{
    using namespace Type;

    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // Loop through kernel size dimensions
        for (size_t d = 0; d < getArchetype().getSynapseGroup()->getKernelSize().size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(d)) {
                addField(Uint32, "kernelSize" + std::to_string(d),
                         [d](const auto &g, size_t) { return std::to_string(g.getSynapseGroup()->getKernelSize().at(d)); });
            }
        }
    }
    else {
        addField(Uint32, "rowStride",
                 [&backend](const auto &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });
        addField(Uint32, "numSrcNeurons",
                 [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
        addField(Uint32, "numTrgNeurons",
                 [](const auto &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });
    }
}
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
void CustomWUUpdateInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const bool kernel = (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL);
    if(kernel && modelMerged.getModel().getBatchSize() > 1) {
        // Loop through kernel dimensions and multiply together to calculate batch stride
        os << "const unsigned int batchStride = ";
        const auto &kernelSize = getArchetype().getSynapseGroup()->getKernelSize();
        for (size_t i = 0; i < kernelSize.size(); i++) {
            os << getKernelSize(i);

            if (i != (kernelSize.size() - 1)) {
                os << " * ";
            }
        }
        os << ";" << std::endl;
    }
    
    if(!kernel) {
        os << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
        os << CodeStream::OB(3);
        popSubs.addVarSubstitution("id_pre", "i");
    }
 
    // Loop through rows
    const std::string stride = kernel ? "batchStride" : "group->numSrcNeurons * group->rowStride";
    genInitWUVarCode(os, modelMerged, popSubs, getArchetype().getCustomUpdateModel()->getVars(),
                    getArchetype().getVarInitialisers(), stride, getIndex(),
                    getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                    [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                    [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); },
                    [&backend, kernel, this](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                    {
                        if (kernel) {
                            backend.genKernelCustomUpdateVariableInit(os, *this, kernelSubs, handler);
                        }
                        else {
                            backend.genDenseSynapseVariableRowInit(os, kernelSubs, handler);
                        }
    
                    });
        
    if(!kernel) {
        os << CodeStream::CB(3);
    }
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateSparseInitGroupMerged::name = "CustomWUUpdateSparseInit";
//----------------------------------------------------------------------------
CustomWUUpdateSparseInitGroupMerged::CustomWUUpdateSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                         const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal, CustomUpdateVarAdapter>(index, typeContext, backend, groups)
{
    using namespace Type;

    addField(Uint32, "rowStride",
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
             });
}
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
void CustomWUUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitWUVarCode(os, modelMerged, popSubs, getArchetype().getCustomUpdateModel()->getVars(),
                     getArchetype().getVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(),
                     getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                     [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                     [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         return backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler); 
                     });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdatePreInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdatePreInitGroupMerged::name = "CustomConnectivityUpdatePreInit";
//----------------------------------------------------------------------------
CustomConnectivityUpdatePreInitGroupMerged::CustomConnectivityUpdatePreInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                                       const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdatePreVarAdapter>(index, typeContext, backend, groups)
{
    addField(Type::Uint32, "size",
             [](const auto &c, size_t) 
             { 
                 return std::to_string(c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
             });
    
    // If this backend initialises population RNGs on device and this group requires one for simulation
    if(backend.isPopulationRNGRequired() && getArchetype().isRowSimRNGRequired() && backend.isPopulationRNGInitialisedOnDevice()) {
        addPointerField(*backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rowRNG");
    }
}
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
void CustomConnectivityUpdatePreInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise presynaptic custom connectivity update variables
    // **TODO** adaptor
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getPreVars(), getArchetype().getPreVarInitialisers(),
                         "", "size", getIndex(), 1,
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdatePostInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdatePostInitGroupMerged::name = "CustomConnectivityUpdatePostInit";
//----------------------------------------------------------------------------
CustomConnectivityUpdatePostInitGroupMerged::CustomConnectivityUpdatePostInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdatePostVarAdapter>(index, typeContext, backend, groups)
{
    addField(Type::Uint32, "size",
             [](const auto &c, size_t)
             {
                 return std::to_string(c.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons());
             });
}
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
void CustomConnectivityUpdatePostInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise presynaptic custom connectivity update variables
    // **TODO** adapter
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getPostVars(), getArchetype().getPostVarInitialisers(),
                         "", "size", getIndex(), 1,
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateSparseInitGroupMerged::name = "CustomConnectivityUpdateSparseInit";
//----------------------------------------------------------------------------
CustomConnectivityUpdateSparseInitGroupMerged::CustomConnectivityUpdateSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                                             const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdateVarAdapter>(index, typeContext, backend, groups)
{
    using namespace Type;

    addField(Uint32, "rowStride",
             [&backend](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });

    addField(Uint32, "numSrcNeurons",
             [](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField(Uint32, "numTrgNeurons",
             [](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });

    addField(Uint32.createPointer(), "rowLength",
             [&backend](const CustomConnectivityUpdateInternal &cg, size_t)
             {
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "rowLength" + sg->getName();
             });
    addField(getArchetype().getSynapseGroup()->getSparseIndType().createPointer(), "ind",
             [&backend](const auto &cg, size_t)
             {
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "ind" + sg->getName();
             });
}
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
void CustomConnectivityUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise custom connectivity update variables
    genInitWUVarCode(os, modelMerged, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getVars(),
                     getArchetype().getVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(), 1,
                     [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                     [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         return backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler);
                     });
}
