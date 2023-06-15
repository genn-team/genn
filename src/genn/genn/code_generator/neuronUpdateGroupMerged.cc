#include "code_generator/neuronUpdateGroupMerged.h"

// GeNN code generator includes
#include "code_generator/standardLibrary.h"
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

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::CurrentSource
//----------------------------------------------------------------------------
// **TODO**
// * field suffix (string) and value suffix (function to get suffix from group) common to everything in group - GroupMerged fields?
// * without nasty combined groups, getParams and getDerivedParams functions can use pointers to members
// * pre and post neuron stuff in synapse update group merged can also be child classes
NeuronUpdateGroupMerged::CurrentSource::CurrentSource(size_t index, const Type::TypeContext &typeContext, Transpiler::TypeChecker::EnvironmentBase &enclosingEnv,
                                                      const BackendBase &backend, const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups)
:   GroupMerged<CurrentSourceInternal>(index, typeContext, groups)
{
    /*const std::string suffix =  "CS" + std::to_string(getIndex());

    // Create type environment
    GroupMergedTypeEnvironment<CurrentSource> typeEnvironment(*this, &enclosingEnv);

    // Add heterogeneous parameters
    const auto *cm = getArchetype().getCurrentSourceModel();
    typeEnvironment.defineHeterogeneousParams(cm->getParamNames(), suffix,
                                              &CurrentSourceInternal::getParams,
                                              &CurrentSource::isParamHeterogeneous);

    // Add heterogeneous derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams(cm->getDerivedParams(), suffix,
                                                     &CurrentSourceInternal::getDerivedParams,
                                                     &CurrentSource::isDerivedParamHeterogeneous);

    // Add variables
    typeEnvironment.defineVars(cm->getVars(), backend.getDeviceVarPrefix(), suffix);

    // Add EGPs
    typeEnvironment.defineEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", suffix);*/

}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::CurrentSource::generate(const BackendBase &backend, EnvironmentExternalBase &env,
                                                      const NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged) const
{
    const std::string suffix =  "CS" + std::to_string(getIndex());
    const auto *cm = getArchetype().getCurrentSourceModel();

    // Create new substitution environment and add parameters, derived parameters and extra global parameters
    EnvironmentSubstitute envSubs(env);
    envSubs.getStream() << "// current source " << getIndex() << std::endl;
    envSubs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(), suffix,
                                     [this](const std::string &p) { return isParamHeterogeneous(p); });
    envSubs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(), suffix,
                                    [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
    envSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), suffix);

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CurrentSourceVarAdapter, CurrentSourceInternal> varSubs(
        getArchetype(), getTypeContext(), envSubs, "l", suffix,
        [&envSubs, &modelMerged, &ng](const std::string&, const Models::VarInit&, VarAccess a)
        {
            return ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(a), envSubs["id"]);
        });

    //currSourceSubs.addFuncSubstitution("injectCurrent", 1, "Isyn += $(0)");
    
    // Pretty print previously parsed update statements
    PrettyPrinter::print(m_InjectionStatements, varSubs, getTypeContext(), m_InjectionResolvedTypes);
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
NeuronUpdateGroupMerged::InSynPSM::InSynPSM(size_t index, const Type::TypeContext &typeContext, Transpiler::TypeChecker::EnvironmentBase &enclosingEnv,
                                            const BackendBase &backend, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "InSyn" + std::to_string(getIndex());

    // Create type environment
    /*GroupMergedTypeEnvironment<InSynPSM> typeEnvironment(*this, &enclosingEnv);

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

    // Add heterogeneous parameters
    const auto *psm = getArchetype().getPSModel();
    typeEnvironment.defineHeterogeneousParams(psm->getParamNames(), suffix,
                                              &SynapseGroupInternal::getPSParams,
                                              &InSynPSM::isParamHeterogeneous);

    // Add heterogeneous derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams(psm->getDerivedParams(), suffix,
                                                     &SynapseGroupInternal::getPSDerivedParams,
                                                     &InSynPSM::isDerivedParamHeterogeneous);

    // Add variables
    typeEnvironment.defineVars(psm->getVars(), backend.getDeviceVarPrefix(), 
                               suffix, &SynapseGroupInternal::getFusedPSVarSuffix);

    // Add EGPs
    typeEnvironment.defineEGPs(psm->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", 
                               suffix, &SynapseGroupInternal::getFusedPSVarSuffix);

    // Scan, parse and type-check decay and apply input code
    ErrorHandler errorHandler;
    std::tie(m_DecayStatements, m_DecayResolvedTypes) = scanParseAndTypeCheckStatements(psm->getDecayCode(), typeContext, 
                                                                              typeEnvironment, errorHandler);
    std::tie(m_ApplyInputStatements, m_ApplyInputResolvedTypes) = scanParseAndTypeCheckStatements(psm->getApplyInputCode(), typeContext, 
                                                                                        typeEnvironment, errorHandler);*/
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynPSM::generate(const BackendBase &backend, EnvironmentExternalBase &env,
                                                 const NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged) const
{
    const std::string suffix =  "InSyn" + std::to_string(getIndex());
    const auto *psm = getArchetype().getPSModel();

    // Create new substitution environment 
    EnvironmentSubstitute envSubs(env);

    envSubs.getStream() << "// current source " << getIndex() << std::endl;
    envSubs.getStream() << "scalar linSyn = group->inSynInSyn" << getIndex() << "[";
    envSubs.getStream() << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, envSubs["id"]);
    envSubs.getStream() << "];" << std::endl;

    // If dendritic delay is required
    if (getArchetype().isDendriticDelayRequired()) {
        // Get reference to dendritic delay buffer input for this timestep
        envSubs.getStream() << backend.getPointerPrefix() << "scalar *denDelayFront = ";
        envSubs.getStream() << "&group->denDelay" << suffix << "[(*group->denDelayPtr" << suffix << " * group->numNeurons) + ";
        envSubs.getStream() << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, envSubs["id"]);
        envSubs.getStream() << "];" << std::endl;

        // Add delayed input from buffer into inSyn
        envSubs.getStream() << "linSyn += *denDelayFront;" << std::endl;

        // Zero delay buffer slot
        envSubs.getStream() << "*denDelayFront = " << modelMerged.scalarExpr(0.0) << ";" << std::endl;
    }

    // Add parameters, derived parameters and extra global parameters to environment
    envSubs.addParamValueSubstitution(psm->getParamNames(), getArchetype().getPSParams(), suffix,
                                      [this](const std::string &p) { return isParamHeterogeneous(p); });
    envSubs.addVarValueSubstitution(psm->getDerivedParams(), getArchetype().getPSDerivedParams(), suffix,
                                    [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
    envSubs.addVarNameSubstitution(psm->getExtraGlobalParams(), suffix);
    
    // **TODO** naming convention
    envSubs.addSubstitution("inSyn", "linSyn");
        
    // Allow synapse group's PS output var to override what Isyn points to
    envSubs.addSubstitution("Isyn", getArchetype().getPSTargetVar());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<SynapsePSMVarAdapter, SynapseGroupInternal> varSubs(
        getArchetype(), getTypeContext(), envSubs, "l", suffix,
        [&envSubs, &modelMerged, &ng](const std::string&, const Models::VarInit&, VarAccess a)
        {
            return ng.getVarIndex(modelMerged.getModel().getBatchSize(), getVarAccessDuplication(a), envSubs["id"]);
        });

    // Pretty print previously parsed update statements
    PrettyPrinter::print(m_ApplyInputStatements, varSubs, getTypeContext(), m_ApplyInputResolvedTypes);
    PrettyPrinter::print(m_DecayStatements, varSubs, getTypeContext(), m_DecayResolvedTypes);

    // Write back linSyn
    varSubs.getStream() << "group->inSyn" << suffix << "[";
    varSubs.getStream() << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, envSubs["id"]);
    varSubs.getStream() << "] = linSyn;" << std::endl;
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
NeuronUpdateGroupMerged::OutSynPreOutput::OutSynPreOutput(size_t index, const Type::TypeContext &typeContext, Transpiler::TypeChecker::EnvironmentBase&,
                                                          const BackendBase &backend, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());

    addField(getScalarType().createPointer(), "revInSyn" + suffix,
             [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "revInSyn" + g.getFusedPreOutputSuffix(); });
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynPreOutput::generate(EnvironmentExternal &env, const NeuronUpdateGroupMerged &ng, 
                                                        const ModelSpecMerged &modelMerged) const
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());
     
    env.getStream() << getArchetype().getPreTargetVar() << " += ";
    env.getStream() << "group->revInSyn" << suffix << "[";
    env.getStream() << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, env["id"]);
    env.getStream() << "];" << std::endl;
    env.getStream() << "group->revInSyn" << suffix << "[";
    env.getStream() << ng.getVarIndex(modelMerged.getModel().getBatchSize(), VarAccessDuplication::DUPLICATE, env["id"]);
    env.getStream() << "] = " << modelMerged.scalarExpr(0.0) << ";" << std::endl;
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
//----------------------------------------------------------------------------
NeuronUpdateGroupMerged::InSynWUMPostCode::InSynWUMPostCode(size_t index, const Type::TypeContext &typeContext, Transpiler::TypeChecker::EnvironmentBase &enclosingEnv,
                                                            const BackendBase &backend, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());

    // Create type environment
    GroupMergedTypeEnvironment<InSynWUMPostCode> typeEnvironment(*this, &enclosingEnv);

    // Add heterogeneous parameters
    const auto *wum = getArchetype().getWUModel();
    typeEnvironment.defineHeterogeneousParams(wum->getParamNames(), suffix,
                                              &SynapseGroupInternal::getWUParams,
                                              &InSynWUMPostCode::isParamHeterogeneous);

    // Add heterogeneous derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams(wum->getDerivedParams(), suffix,
                                                     &SynapseGroupInternal::getWUDerivedParams,
                                                     &InSynWUMPostCode::isDerivedParamHeterogeneous);

    // Add variables
    typeEnvironment.defineVars(wum->getPostVars(), backend.getDeviceVarPrefix(), 
                               suffix, &SynapseGroupInternal::getFusedWUPostVarSuffix);

    // Add EGPs
    typeEnvironment.defineEGPs(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", 
                               suffix, &SynapseGroupInternal::getFusedWUPostVarSuffix);

    // Scan, parse and type-check dynamics and spike code
    ErrorHandler errorHandler;
    std::tie(m_DynamicsStatements, m_DynamicsResolvedTypes) = scanParseAndTypeCheckStatements(wum->getPostDynamicsCode(), typeContext, 
                                                                                    typeEnvironment, errorHandler);
    std::tie(m_SpikeStatements, m_SpikeResolvedTypes) = scanParseAndTypeCheckStatements(wum->getPostSpikeCode(), typeContext, 
                                                                              typeEnvironment, errorHandler);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::generate(const BackendBase &backend, EnvironmentExternal &env, const NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, bool dynamicsNotSpike) const
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUModel();

    const unsigned int batchSize = modelMerged.getModel().getBatchSize();

    // If there are any statements to execute here
    const auto &statements = dynamicsNotSpike ? m_DynamicsStatements : m_SpikeStatements;
    const auto &resolvedTypes = dynamicsNotSpike ? m_DynamicsResolvedTypes : m_SpikeResolvedTypes;
    if(!statements.empty()) {
        // Create new substitution environment and add parameters, derived parameters and extra global parameters
        EnvironmentSubstitute envSubs(env);
        envSubs.getStream() << "// postsynaptic weight update " << getIndex() << std::endl;
        envSubs.addParamValueSubstitution(wum->getParamNames(), getArchetype().getWUParams(), suffix,
                                         [this](const std::string &p) { return isParamHeterogeneous(p); });
        envSubs.addVarValueSubstitution(wum->getDerivedParams(), getArchetype().getWUDerivedParams(), suffix,
                                        [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
        envSubs.addVarNameSubstitution(wum->getExtraGlobalParams(), suffix);

        // Create an environment which caches variables in local variables if they are accessed
        const bool delayed = (getArchetype().getBackPropDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, SynapseGroupInternal> varSubs(
            getArchetype(), getTypeContext(), envSubs, "l", suffix,
            [batchSize, delayed, &envSubs, &ng](const std::string&, const Models::VarInit&, VarAccess a)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), envSubs["id"]);
            },
            [batchSize, delayed, &envSubs, &ng](const std::string&, const Models::VarInit&, VarAccess a)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(a), envSubs["id"]);
            });

        /*neuronSubstitutionsInSynapticCode(subs, &ng.getArchetype(), "", "_post", "", "", "", dynamicsNotSpike,
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

        // Pretty print previously parsed statements
        PrettyPrinter::print(statements, varSubs, getTypeContext(), resolvedTypes);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::InSynWUMPostCode::genCopyDelayedVars(EnvironmentExternal &env, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged) const
{
    // If this group has a delay and no postsynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());
    if(getArchetype().getBackPropDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPostDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPostVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                env.getStream() << "group->" << v.name << suffix << "[";
                env.getStream() << ng.getWriteVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), env["id"]);
                env.getStream() << "] = ";

                env.getStream() << "group->" << v.name << suffix << "[";
                env.getStream() << ng.getReadVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), env["id"]);
                env.getStream() << "];" << std::endl;
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
NeuronUpdateGroupMerged::OutSynWUMPreCode::OutSynWUMPreCode(size_t index, const Type::TypeContext &typeContext, Transpiler::TypeChecker::EnvironmentBase &enclosingEnv,
                                                            const BackendBase &backend, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());

    // Create type environment
    GroupMergedTypeEnvironment<OutSynWUMPreCode> typeEnvironment(*this, &enclosingEnv);

    // Add heterogeneous parameters
    const auto *wum = getArchetype().getWUModel();
    typeEnvironment.defineHeterogeneousParams(wum->getParamNames(), suffix,
                                              &SynapseGroupInternal::getWUParams,
                                              &OutSynWUMPreCode::isParamHeterogeneous);

    // Add heterogeneous derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams(wum->getDerivedParams(), suffix,
                                                     &SynapseGroupInternal::getWUDerivedParams,
                                                     &OutSynWUMPreCode::isDerivedParamHeterogeneous);

    // Add variables
    typeEnvironment.defineVars(wum->getPreVars(), backend.getDeviceVarPrefix(), 
                               suffix, &SynapseGroupInternal::getFusedWUPreVarSuffix);

    // Add EGPs
    typeEnvironment.defineEGPs(wum->getExtraGlobalParams(), backend.getDeviceVarPrefix(), "", 
                               suffix, &SynapseGroupInternal::getFusedWUPreVarSuffix);

    // Scan, parse and type-check dynamics and spike code
    ErrorHandler errorHandler;
    std::tie(m_DynamicsStatements, m_DynamicsResolvedTypes) = scanParseAndTypeCheckStatements(wum->getPreDynamicsCode(), typeContext, 
                                                                                    typeEnvironment, errorHandler);
    std::tie(m_SpikeStatements, m_SpikeResolvedTypes) = scanParseAndTypeCheckStatements(wum->getPreSpikeCode(), typeContext, 
                                                                              typeEnvironment, errorHandler);
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::generate(const BackendBase &backend, EnvironmentExternal &env, const NeuronUpdateGroupMerged &ng,
                                                         const ModelSpecMerged &modelMerged, bool dynamicsNotSpike) const
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    const auto *wum = getArchetype().getWUModel();
    const unsigned int batchSize = modelMerged.getModel().getBatchSize();
   
    // If there are any statements to executre here
    const auto &statements = dynamicsNotSpike ? m_DynamicsStatements : m_SpikeStatements;
    const auto &resolvedTypes = dynamicsNotSpike ? m_DynamicsResolvedTypes : m_SpikeResolvedTypes;
    
    // If there are any statements to execute here
    if(!statements.empty()) {
        // Create new substitution environment and add parameters, derived parameters and extra global parameters
        EnvironmentSubstitute envSubs(env);
        envSubs.getStream() << "// presynaptic weight update " << getIndex() << std::endl;
        envSubs.addParamValueSubstitution(wum->getParamNames(), getArchetype().getWUParams(), suffix,
                                         [this](const std::string &p) { return isParamHeterogeneous(p); });
        envSubs.addVarValueSubstitution(wum->getDerivedParams(), getArchetype().getWUDerivedParams(), suffix,
                                        [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
        envSubs.addVarNameSubstitution(wum->getExtraGlobalParams(), suffix);

        // Create an environment which caches variables in local variables if they are accessed
        const bool delayed = (getArchetype().getDelaySteps() != NO_DELAY);
        EnvironmentLocalVarCache<SynapseWUPostVarAdapter, SynapseGroupInternal> varSubs(
            getArchetype(), getTypeContext(), envSubs, "l", suffix,
            [batchSize, delayed, &envSubs, &ng](const std::string&, const Models::VarInit&, VarAccess a)
            {
                return ng.getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), envSubs["id"]);
            },
            [batchSize, delayed, &envSubs, &ng](const std::string&, const Models::VarInit&, VarAccess a)
            {
                return ng.getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(a), envSubs["id"]);
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

        // Pretty print previously parsed statements
        PrettyPrinter::print(statements, varSubs, getTypeContext(), resolvedTypes);
    }
}
//----------------------------------------------------------------------------
void NeuronUpdateGroupMerged::OutSynWUMPreCode::genCopyDelayedVars(EnvironmentExternal &env, const NeuronUpdateGroupMerged &ng,
                                                                   const ModelSpecMerged &modelMerged) const
{
    // If this group has a delay and no presynaptic dynamics (which will already perform this copying)
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());
    if(getArchetype().getDelaySteps() != NO_DELAY && getArchetype().getWUModel()->getPreDynamicsCode().empty()) {
        // Loop through variables and copy between read and write delay slots
        for(const auto &v : getArchetype().getWUModel()->getPreVars()) {
            if(v.access & VarAccessMode::READ_WRITE) {
                env.getStream() << "group->" << v.name << suffix << "[";
                env.getStream() << ng.getWriteVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), env["id"]);
                env.getStream() << "] = ";

                env.getStream() << "group->" << v.name << suffix << "[";
                env.getStream() << ng.getReadVarIndex(true, modelMerged.getModel().getBatchSize(), getVarAccessDuplication(v.access), env["id"]);
                env.getStream() << "];" << std::endl;
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
NeuronUpdateGroupMerged::NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                                 const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, typeContext, backend, groups)
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
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, typeContext, typeEnvironment, backend,
                             &NeuronGroupInternal::getFusedPSMInSyn, &SynapseGroupInternal::getPSHashDigest);

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, typeContext, typeEnvironment, backend, 
                             &NeuronGroupInternal::getFusedPreOutputOutSyn, &SynapseGroupInternal::getPreOutputHashDigest);

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, typeContext, typeEnvironment, backend,
                             &NeuronGroupInternal::getCurrentSources, &CurrentSourceInternal::getHashDigest);


    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostCodeGroups, typeContext, typeEnvironment, backend,
                             &NeuronGroupInternal::getFusedInSynWithPostCode, &SynapseGroupInternal::getWUPostHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic synaptic updates, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreCodeGroups, typeContext, typeEnvironment, backend, 
                             &NeuronGroupInternal::getFusedOutSynWithPreCode, &SynapseGroupInternal::getWUPreHashDigest);
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
                                                   BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent) const
{
    const ModelSpecInternal &model = modelMerged.getModel();
    const unsigned int batchSize = model.getBatchSize();
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
 
    EnvironmentGroupMergedField<NeuronUpdateGroupMerged> neuronEnv(env, *this);

    // Add field for spike recording
    neuronEnv.add(Type::Uint32.createPointer(), "_record_spk", "recordSpk",
                  [&backend](const auto &ng, size_t) 
                  { 
                      return backend.getDeviceVarPrefix() + "recordSpk" + ng.getName(); 
                  }, 
                  "", GroupMergedFieldType::DYNAMIC);
   
    // Add field for spike event recording
    neuronEnv.add(Type::Uint32.createPointer(), "_record_spk_event", "recordSpkEvent",
                  [&backend](const auto &ng, size_t)
                  {
                      return backend.getDeviceVarPrefix() + "recordSpkEvent" + ng.getName(); 
                  },
                  "", GroupMergedFieldType::DYNAMIC);

    // Add default input variable
    neuronEnv.add(modelMerged.getModel().getPrecision(), "Isyn", "Isyn",
                  {neuronEnv.addInitialiser("scalar Isyn = 0;")});

    // **NOTE** arbitrary code in param value to be deprecated
    for (const auto &v : nm->getAdditionalInputVars()) {
        const auto resolvedType = v.type.resolve(getTypeContext());
        neuronEnv.add(resolvedType, v.name, v.name,
                      {neuronEnv.addInitialiser(resolvedType.getName() + " " + v.name + " = " + v.value + ";")});
    }

    // Substitute parameter and derived parameter names
    neuronEnv.addParams(nm->getParamNames(), "", &NeuronGroupInternal::getParams, &NeuronUpdateGroupMerged::isParamHeterogeneous);
    neuronEnv.addDerivedParams(nm->getDerivedParams(), "", &NeuronGroupInternal::getDerivedParams, &NeuronUpdateGroupMerged::isDerivedParamHeterogeneous);
    neuronEnv.addEGPs<NeuronEGPAdapter>(backend.getDeviceVarPrefix());
    
    // Substitute spike times
    const std::string spikeTimeReadIndex = getReadVarIndex(getArchetype().isDelayRequired(), batchSize, VarAccessDuplication::DUPLICATE, neuronEnv["id"]);
    neuronEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "sT", "lsT", 
                  {neuronEnv.addInitialiser("const timepoint lsT = " + neuronEnv["_spk_time"] + "[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "prev_sT", "lprevST", 
                  {neuronEnv.addInitialiser("const timepoint lprevST = " + neuronEnv["_prev_spk_time"] + "[" + spikeTimeReadIndex + "];")});
    neuronEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "seT", "lseT", 
                  {neuronEnv.addInitialiser("const timepoint lseT = " + neuronEnv["_spk_evnt_time"] + "[" + spikeTimeReadIndex+ "];")});
    neuronEnv.add(modelMerged.getModel().getTimePrecision().addConst(), "prev_seT", "lprevSET", 
                  {neuronEnv.addInitialiser("const timepoint lprevSET = " + neuronEnv["_prev_spk_evnt_time"] + "[" + spikeTimeReadIndex + "];")});

    // Create an environment which caches variables in local variables if they are accessed
    // **NOTE** we do this right at the top so that local copies can be used by child groups
    EnvironmentLocalVarCache<NeuronVarAdapter, NeuronGroupInternal> neuronVarEnv(
        getArchetype(), getTypeContext(), neuronEnv, "l", "",
        [batchSize, &neuronEnv, this](const std::string &varName, const Models::VarInit&, VarAccess a)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getReadVarIndex(delayed, batchSize, getVarAccessDuplication(a), neuronEnv["id"]) ;
        },
        [batchSize, &neuronEnv, this](const std::string &varName, const Models::VarInit&, VarAccess a)
        {
            const bool delayed = (getArchetype().isVarQueueRequired(varName) && getArchetype().isDelayRequired());
            return getWriteVarIndex(delayed, batchSize, getVarAccessDuplication(a), neuronEnv["id"]) ;
        });


    // Loop through incoming synapse groups
    for(const auto &sg : getMergedInSynPSMGroups()) {
        CodeStream::Scope b(env.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged);
    }

    // Loop through outgoing synapse groups with presynaptic output
    for (const auto &sg : getMergedOutSynPreOutputGroups()) {
        CodeStream::Scope b(env.getStream());
        sg.generate(neuronVarEnv, *this, modelMerged);
    }
 
    // Loop through all of neuron group's current sources
    for (const auto &cs : getMergedCurrentSourceGroups()) {
        CodeStream::Scope b(env.getStream());
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
            PrettyPrinter::print(m_ThresholdConditionExpression, neuronVarEnv, getTypeContext(), m_ThresholdConditionResolvedTypes);
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
    PrettyPrinter::print(m_SimStatements, neuronVarEnv, getTypeContext(), m_SimResolvedTypes);

    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(neuronVarEnv.getStream());
        sg.generate(backend, neuronVarEnv, *this, modelMerged, true);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (const auto &sg : getMergedInSynWUMPostCodeGroups()) {
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
    if (m_ThresholdConditionExpression) {
        neuronVarEnv.getStream() << "// test for and register a true spike" << std::endl;
        neuronVarEnv.getStream() << "if ((";
        PrettyPrinter::print(m_ThresholdConditionExpression, neuronVarEnv, getTypeContext(), m_ThresholdConditionResolvedTypes);
        neuronVarEnv.getStream() << ")";
        if (nm->isAutoRefractoryRequired()) {
            neuronVarEnv.getStream() << " && !oldSpike";
        }
        neuronVarEnv.getStream() << ")";
        {
            CodeStream::Scope b(neuronVarEnv.getStream());
            genEmitTrueSpike(neuronVarEnv, *this);

            // add after-spike reset if provided
            if (!m_ResetStatements.empty()) {
                neuronVarEnv.getStream() << "// spike reset code" << std::endl;
                PrettyPrinter::print(m_ResetStatements, neuronVarEnv, getTypeContext(), m_ResetResolvedTypes);
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
                    neuronVarEnv.getStream() << "group->sT[";
                    neuronVarEnv.getStream() << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, neuronVarEnv["id"]);
                    neuronVarEnv.getStream()  << "] = " << neuronVarEnv["sT"] << ";" << std::endl;
                }

                // If previous spike times are required, copy times from register
                if(getArchetype().isPrevSpikeTimeRequired()) {
                    neuronVarEnv.getStream() << "group->prevST[";
                    neuronVarEnv.getStream() << getWriteVarIndex(true, batchSize, VarAccessDuplication::DUPLICATE, neuronVarEnv["id"]);
                    neuronVarEnv.getStream() << "] = " << neuronVarEnv["prev_sT"] << ";" << std::endl;
                }

                // Loop through outgoing synapse groups with some sort of presynaptic code
                for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
                    sg.genCopyDelayedVars(neuronVarEnv, *this, modelMerged);
                }

                // Loop through incoming synapse groups with some sort of presynaptic code
                for (const auto &sg : getMergedInSynWUMPostCodeGroups()) {
                    sg.genCopyDelayedVars(neuronVarEnv, *this, modelMerged);
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
void NeuronUpdateGroupMerged::generateWUVarUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const
{
    // Generate var update for outgoing synaptic populations with presynaptic update code
    for (const auto &sg : getMergedOutSynWUMPreCodeGroups()) {
        CodeStream::Scope b(env.getStream());
        sg.generate(backend, env, *this, modelMerged, false);
    }

    // Generate var update for incoming synaptic populations with postsynaptic code
    for (const auto &sg : getMergedInSynWUMPostCodeGroups()) {
        CodeStream::Scope b(env.getStream());
        sg.generate(backend, env, *this, modelMerged, false);
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
