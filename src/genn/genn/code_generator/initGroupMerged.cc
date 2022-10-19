#include "code_generator/initGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void genVariableFill(CodeStream &os, const std::string &fieldName, const std::string &value, const std::string &idx, const std::string &stride,
                     VarAccessDuplication varDuplication, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDuplication == VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        os << "group->" << fieldName << "[" << idx << "] = " << value << ";" << std::endl;
    }
    // Otherwise
    else {
        os << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(os);
            os << "group->" << fieldName << "[(d * " << stride << ") + " << idx << "] = " << value << ";" << std::endl;
        }
    }
}
//--------------------------------------------------------------------------
void genScalarFill(CodeStream &os, const std::string &fieldName, const std::string &value,
                   VarAccessDuplication varDuplication, unsigned int batchSize, bool delay = false, unsigned int numDelaySlots = 1)
{
    // Determine number of values to fill in each thread
    const unsigned int numValues = ((varDuplication == VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

    // If there's only one, don't generate a loop
    if(numValues == 1) {
        os << "group->" << fieldName << "[0] = " << value << ";" << std::endl;
    }
    // Otherwise
    else {
        os << "for(unsigned int d = 0; d < " << numValues << "; d++)";
        {
            CodeStream::Scope b(os);
            os << "group->" << fieldName << "[d] = " << value << ";" << std::endl;
        }
    }
}
//------------------------------------------------------------------------
template<typename Q, typename P, typename D>
void genInitNeuronVarCode(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::vector<Models::VarInit> &varInitialisers, 
                          const std::string &fieldSuffix, const std::string &countMember, 
                          size_t numDelaySlots, const size_t groupIndex, const std::string &ftype, unsigned int batchSize,
                          Q isVarQueueRequired, P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn)
{
    const std::string count = "group->" + countMember;
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = varInitialisers.at(k);

        // If this variable has any initialisation code
        if (!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            Substitutions varSubs(&popSubs);

            // Substitute in parameters and derived parameters for initialising variables
            varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                              [k, isParamHeterogeneousFn](size_t p) { return isParamHeterogeneousFn(k, p); },
                                              "", "group->", vars[k].name + fieldSuffix);
            varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                            [k, isDerivedParamHeterogeneousFn](size_t p) { return isDerivedParamHeterogeneousFn(k, p); },
                                            "", "group->", vars[k].name + fieldSuffix);
            varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                           "", "group->", vars[k].name + fieldSuffix);

            // If variable is shared between neurons
            if (getVarAccessDuplication(vars[k].access) == VarAccessDuplication::SHARED_NEURON) {
                backend.genPopVariableInit(
                    os, varSubs,
                    [&vars, &varInit, &fieldSuffix, &ftype, batchSize, groupIndex, k, numDelaySlots, isVarQueueRequired]
                    (CodeStream &os, Substitutions &varInitSubs)
                    {
                        // Generate initial value into temporary variable
                        os << vars[k].type << " initVal;" << std::endl;
                        varInitSubs.addVarSubstitution("value", "initVal");
                        std::string code = varInit.getSnippet()->getCode();
                        varInitSubs.applyCheckUnreplaced(code, "initVar : " + vars[k].name + "merged" + std::to_string(groupIndex));
                        code = ensureFtype(code, ftype);
                        os << code << std::endl;

                        // Fill value across all delay slots and batches
                        genScalarFill(os, vars[k].name + fieldSuffix, "initVal", getVarAccessDuplication(vars[k].access),
                                      batchSize, isVarQueueRequired(k), numDelaySlots);
                    });
            }
            // Otherwise
            else {
                backend.genVariableInit(
                    os, count, "id", varSubs,
                    [&vars, &varInit, &fieldSuffix, &ftype, batchSize, groupIndex, k, count, numDelaySlots, isVarQueueRequired]
                    (CodeStream &os, Substitutions &varInitSubs)
                    {
                        // Generate initial value into temporary variable
                        os << vars[k].type << " initVal;" << std::endl;
                        varInitSubs.addVarSubstitution("value", "initVal");
                        std::string code = varInit.getSnippet()->getCode();
                        varInitSubs.applyCheckUnreplaced(code, "initVar : " + vars[k].name + "merged" + std::to_string(groupIndex));
                        code = ensureFtype(code, ftype);
                        os << code << std::endl;

                        // Fill value across all delay slots and batches
                        genVariableFill(os, vars[k].name + fieldSuffix, "initVal", varInitSubs["id"], count,
                                        getVarAccessDuplication(vars[k].access), batchSize, isVarQueueRequired(k), numDelaySlots);
                    });
            }
        }
            
    }
}
//------------------------------------------------------------------------
template<typename P, typename D>
void genInitNeuronVarCode(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::vector<Models::VarInit> &varInitialisers, 
                          const std::string &fieldSuffix, const std::string &countMember, const size_t groupIndex, 
                          const std::string &ftype, unsigned int batchSize, P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn)
{
    genInitNeuronVarCode(os, backend, popSubs, vars, varInitialisers, fieldSuffix, countMember, 0, groupIndex, ftype, batchSize,
                         [](size_t){ return false; }, 
                         isParamHeterogeneousFn,
                         isDerivedParamHeterogeneousFn);
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
template<typename P, typename D, typename G>
void genInitWUVarCode(CodeStream &os, const Substitutions &popSubs, 
                      const Models::Base::VarVec &vars, const std::vector<Models::VarInit> &varInitialisers, 
                      const std::string &stride, const size_t groupIndex, const std::string &ftype, unsigned int batchSize,
                      P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn, G genSynapseVariableRowInitFn)
{
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = varInitialisers.at(k);

        // If this variable has any initialisation code and doesn't require a kernel
        if(!varInit.getSnippet()->getCode().empty() && !varInit.getSnippet()->requiresKernel()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            genSynapseVariableRowInitFn(os, popSubs,
                [&vars, &varInit, &ftype, &stride, batchSize, k, groupIndex, isParamHeterogeneousFn, isDerivedParamHeterogeneousFn]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                      [k, isParamHeterogeneousFn](size_t p) { return isParamHeterogeneousFn(k, p); },
                                                      "", "group->", vars[k].name);
                    varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                      [k, isDerivedParamHeterogeneousFn](size_t p) { return isDerivedParamHeterogeneousFn(k, p); },
                                                      "", "group->", vars[k].name);
                    varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                                   "", "group->", vars[k].name);

                    // Generate initial value into temporary variable
                    os << vars[k].type << " initVal;" << std::endl;
                    varSubs.addVarSubstitution("value", "initVal");
                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(groupIndex));
                    code = ensureFtype(code, ftype);
                    os << code << std::endl;

                    // Fill value across all batches
                    genVariableFill(os,  vars[k].name, "initVal", varSubs["id_syn"], stride,
                                    getVarAccessDuplication(vars[k].access), batchSize);
                });
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronInitGroupMerged::name = "NeuronInit";
//----------------------------------------------------------------------------
NeuronInitGroupMerged::NeuronInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                             const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   NeuronGroupMergedBase(index, precision, timePrecision, backend, true, groups)
{
    // Build vector of vectors containing each child group's incoming 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedInSynWithPostVars, &NeuronGroupInternal::getFusedInSynWithPostVars,
                             &SynapseGroupInternal::getWUPostInitHashDigest);

    // Build vector of vectors containing each child group's outgoing 
    // synapse groups, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_SortedOutSynWithPreVars, &NeuronGroupInternal::getFusedOutSynWithPreVars,
                             &SynapseGroupInternal::getWUPreInitHashDigest);

    // Generate struct fields for incoming synapse groups with postsynaptic variables
    generateWUVar(backend, "WUPost", m_SortedInSynWithPostVars,
                  &WeightUpdateModels::Base::getPostVars, &SynapseGroupInternal::getWUPostVarInitialisers,
                  &NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPostVarSuffix);


    // Generate struct fields for outgoing synapse groups
    generateWUVar(backend, "WUPre", m_SortedOutSynWithPreVars,
                  &WeightUpdateModels::Base::getPreVars, &SynapseGroupInternal::getWUPreVarInitialisers,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous,
                  &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous,
                  &SynapseGroupInternal::getFusedWUPreVarSuffix);
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type NeuronInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic neuron group data
    updateBaseHash(true, hash);

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getInitHashDigest(), hash);

    // Update hash with each group's variable initialisation parameters and derived parameters
    updateVarInitParamHash<NeuronInitGroupMerged, NeuronVarAdapter>(&NeuronInitGroupMerged::isVarInitParamReferenced, hash);
    updateVarInitDerivedParamHash<NeuronInitGroupMerged, NeuronVarAdapter>(&NeuronInitGroupMerged::isVarInitDerivedParamReferenced, hash);
    
    // Loop through child incoming synapse groups with postsynaptic variables
    for(size_t c = 0; c < getSortedArchetypeInSynWithPostVars().size(); c++) {
        const auto *sg = getSortedArchetypeInSynWithPostVars().at(c);

        // Loop through variables and update hash with variable initialisation parameters and derived parameters
        const auto &varInit = sg->getWUPostVarInitialisers();
        for(size_t v = 0; v < varInit.size(); v++) {
            updateChildVarInitParamsHash<SynapseWUPostVarAdapter, NeuronInitGroupMerged>(
                m_SortedInSynWithPostVars, c, v, &NeuronInitGroupMerged::isInSynWUMVarInitParamReferenced, hash);
            updateChildVarInitDerivedParamsHash<SynapseWUPostVarAdapter, NeuronInitGroupMerged>(
                m_SortedInSynWithPostVars, c, v, &NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamReferenced, hash);
        }
    }

    // Loop through child outgoing synapse groups with presynaptic variables
    for(size_t c = 0; c < getSortedArchetypeOutSynWithPreVars().size(); c++) {
        const auto *sg = getSortedArchetypeOutSynWithPreVars().at(c);

        // Loop through variables and update hash with variable initialisation parameters and derived parameters
        const auto &varInit = sg->getWUPreVarInitialisers();
        for(size_t v = 0; v < varInit.size(); v++) {
            updateChildVarInitParamsHash<SynapseWUPreVarAdapter, NeuronInitGroupMerged>(
                m_SortedOutSynWithPreVars, c, v, &NeuronInitGroupMerged::isOutSynWUMVarInitParamReferenced, hash);
            updateChildVarInitDerivedParamsHash<SynapseWUPreVarAdapter, NeuronInitGroupMerged>(
                m_SortedOutSynWithPreVars, c, v, &NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamReferenced, hash);
        }
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const auto &model = modelMerged.getModel();

    // Initialise spike counts
    genInitSpikeCount(os, backend, popSubs, false, model.getBatchSize());
    genInitSpikeCount(os, backend, popSubs, true, model.getBatchSize());

    // Initialise spikes
    genInitSpikes(os, backend, popSubs, false,  model.getBatchSize());
    genInitSpikes(os, backend, popSubs, true,  model.getBatchSize());

    // Initialize spike times
    if(getArchetype().isSpikeTimeRequired()) {
        genInitSpikeTime(os, backend, popSubs, "sT",  model.getBatchSize());
    }

    // Initialize previous spike times
    if(getArchetype().isPrevSpikeTimeRequired()) {
        genInitSpikeTime(os, backend, popSubs,  "prevST",  model.getBatchSize());
    }
               
    // Initialize spike-like-event times
    if(getArchetype().isSpikeEventTimeRequired()) {
        genInitSpikeTime(os, backend, popSubs, "seT",  model.getBatchSize());
    }

    // Initialize previous spike-like-event times
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        genInitSpikeTime(os, backend, popSubs, "prevSET",  model.getBatchSize());
    }
       
    // If neuron group requires delays, zero spike queue pointer
    if(getArchetype().isDelayRequired()) {
        backend.genPopVariableInit(os, popSubs,
            [](CodeStream &os, Substitutions &)
            {
                os << "*group->spkQuePtr = 0;" << std::endl;
            });
    }

    // Initialise neuron variables
    genInitNeuronVarCode(os, backend, popSubs, getArchetype().getNeuronModel()->getVars(), getArchetype().getVarInitialisers(), 
                         "", "numNeurons", getArchetype().getNumDelaySlots(), getIndex(), model.getPrecision(), model.getBatchSize(),
                         [this](size_t i){ return getArchetype().isVarQueueRequired(i); },
                         [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); });

    // Loop through incoming synaptic populations
    for(size_t i = 0; i < getSortedArchetypeMergedInSyns().size(); i++) {
        CodeStream::Scope b(os);

        const auto *sg = getSortedArchetypeMergedInSyns().at(i);

        // Zero InSyn
        backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
            [&model, i] (CodeStream &os, Substitutions &varSubs)
            {
                genVariableFill(os, "inSynInSyn" + std::to_string(i), model.scalarExpr(0.0), 
                                varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, model.getBatchSize());

            });

        // If dendritic delays are required
        if(sg->isDendriticDelayRequired()) {
            // Zero dendritic delay buffer
            backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
                [&model, sg, i](CodeStream &os, Substitutions &varSubs)
                {
                    genVariableFill(os, "denDelayInSyn" + std::to_string(i), model.scalarExpr(0.0),
                                    varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, model.getBatchSize(),
                                    true, sg->getMaxDendriticDelayTimesteps());
                });

            // Zero dendritic delay pointer
            backend.genPopVariableInit(os, popSubs,
                [i](CodeStream &os, Substitutions &)
                {
                    os << "*group->denDelayPtrInSyn" << i << " = 0;" << std::endl;
                });
        }

        // If postsynaptic model variables should be individual
        if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
            genInitNeuronVarCode(os, backend, popSubs, sg->getPSModel()->getVars(), sg->getPSVarInitialisers(),
                                 "InSyn" + std::to_string(i), "numNeurons", i, model.getPrecision(),  model.getBatchSize(),
                                 [i, this](size_t v, size_t p) { return isPSMVarInitParamHeterogeneous(i, v, p); },
                                 [i, this](size_t v, size_t p) { return isPSMVarInitDerivedParamHeterogeneous(i, v, p); });
        }
    }

    // Loop through incoming synaptic populations with postsynaptic variables
    // **NOTE** number of delay slots is based on the target neuron (for simplicity) but whether delay is required is based on the synapse group
    for(size_t i = 0; i < getSortedArchetypeInSynWithPostVars().size(); i++) {
        const auto *sg = getSortedArchetypeInSynWithPostVars().at(i);
        genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPostVars(), sg->getWUPostVarInitialisers(),
                             "WUPost" + std::to_string(i), "numNeurons", sg->getTrgNeuronGroup()->getNumDelaySlots(),
                             i, model.getPrecision(),  model.getBatchSize(),
                             [&sg](size_t){ return (sg->getBackPropDelaySteps() != NO_DELAY); },
                             [i, this](size_t v, size_t p) { return isInSynWUMVarInitParamHeterogeneous(i, v, p); },
                             [i, this](size_t v, size_t p) { return isInSynWUMVarInitDerivedParamHeterogeneous(i, v, p); });
    }

    // Loop through outgoing synaptic populations with presynaptic variables
    // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
    for(size_t i = 0; i < getSortedArchetypeOutSynWithPreVars().size(); i++) {
        const auto *sg = getSortedArchetypeOutSynWithPreVars().at(i);
        genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPreVars(), sg->getWUPreVarInitialisers(),
                             "WUPre" + std::to_string(i), "numNeurons", sg->getSrcNeuronGroup()->getNumDelaySlots(),
                             i, model.getPrecision(),  model.getBatchSize(),
                             [&sg](size_t){ return (sg->getDelaySteps() != NO_DELAY); },
                             [i, this](size_t v, size_t p) { return isOutSynWUMVarInitParamHeterogeneous(i, v, p); },
                             [i, this](size_t v, size_t p) { return isOutSynWUMVarInitDerivedParamHeterogeneous(i, v, p); });
    }

    // Loop through outgoing synaptic populations with presynaptic output
    for(size_t i = 0; i < getSortedArchetypeMergedPreOutputOutSyns().size(); i++) {
        // Zero revInSynOutSyn
        backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
                                [&model, i] (CodeStream &os, Substitutions &varSubs)
                                {
                                    genVariableFill(os, "revInSynOutSyn" + std::to_string(i), model.scalarExpr(0.0),
                                                    varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, model.getBatchSize());
                                });
    }
    
    // Loop through current sources
    os << "// current source variables" << std::endl;
    for(size_t i = 0; i < getSortedArchetypeCurrentSources().size(); i++) {
        const auto *cs = getSortedArchetypeCurrentSources().at(i);

        genInitNeuronVarCode(os, backend, popSubs, cs->getCurrentSourceModel()->getVars(), cs->getVarInitialisers(),
                             "CS" + std::to_string(i), "numNeurons", i, model.getPrecision(),  model.getBatchSize(),
                             [i, this](size_t v, size_t p) { return isCurrentSourceVarInitParamHeterogeneous(i, v, p); },
                             [i, this](size_t v, size_t p) { return isCurrentSourceVarInitDerivedParamHeterogeneous(i, v, p); });
    }
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isInSynWUMVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isInSynWUMVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedInSynWithPostVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPostVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isOutSynWUMVarInitParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    return (isOutSynWUMVarInitDerivedParamReferenced(childIndex, varIndex, paramIndex) &&
            isChildParamValueHeterogeneous(childIndex, paramIndex, m_SortedOutSynWithPreVars,
                                           [varIndex](const SynapseGroupInternal *s) { return s->getWUPreVarInitialisers().at(varIndex).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::generateWUVar(const BackendBase &backend,
                                          const std::string &fieldPrefixStem,
                                          const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                                          Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                                          const std::vector<Models::VarInit> &(SynapseGroupInternal:: *getVarInitialiserFn)(void) const,
                                          bool(NeuronInitGroupMerged::*isParamHeterogeneousFn)(size_t, size_t, size_t) const,
                                          bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneousFn)(size_t, size_t, size_t) const,
                                          const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const)
{
    // Loop through synapse groups
    const auto &archetypeSyns = sortedSyn.front();
    for(size_t i = 0; i < archetypeSyns.size(); i++) {
        const auto *sg = archetypeSyns.at(i);

        // Loop through variables
        const auto vars = (sg->getWUModel()->*getVars)();
        const auto &varInit = (sg->*getVarInitialiserFn)();
        for(size_t v = 0; v < vars.size(); v++) {
            // Add pointers to state variable
            const auto var = vars.at(v);
            if(!varInit.at(v).getSnippet()->getCode().empty()) {
                assert(!Utils::isTypePointer(var.type));
                addField(var.type + "*", var.name + fieldPrefixStem + std::to_string(i),
                         [i, var, &backend, &sortedSyn, getFusedVarSuffix](const NeuronGroupInternal &, size_t groupIndex)
                         {
                             const std::string &varMergeSuffix = (sortedSyn.at(groupIndex).at(i)->*getFusedVarSuffix)();
                             return backend.getDeviceVarPrefix() + var.name + varMergeSuffix;
                         });
            }

            // Also add any heterogeneous, derived or extra global parameters required for initializers
            const auto *varInitSnippet = varInit.at(v).getSnippet();
            addHeterogeneousChildVarInitParams<NeuronInitGroupMerged>(varInitSnippet->getParamNames(), sortedSyn, i, v, var.name + fieldPrefixStem,
                                                                      isParamHeterogeneousFn, getVarInitialiserFn);
            addHeterogeneousChildVarInitDerivedParams<NeuronInitGroupMerged>(varInitSnippet->getDerivedParams(), sortedSyn, i, v, var.name + fieldPrefixStem,
                                                                             isDerivedParamHeterogeneousFn, getVarInitialiserFn);
            addChildEGPs(varInitSnippet->getExtraGlobalParams(), i, backend.getDeviceVarPrefix(), var.name + fieldPrefixStem,
                         [var, &sortedSyn](size_t groupIndex, size_t childIndex)
                         {
                             return var.name + sortedSyn.at(groupIndex).at(childIndex)->getName();
                         });
        }
    }
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isInSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeInSynWithPostVars().at(childIndex)->getWUPostVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::isOutSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const
{
    const auto *varInitSnippet = getSortedArchetypeOutSynWithPreVars().at(childIndex)->getWUPreVarInitialisers().at(varIndex).getSnippet();
    const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
}
//--------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikeCount(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs, 
                                              bool spikeEvent, unsigned int batchSize) const
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genPopVariableInit(os, popSubs,
            [batchSize, spikeEvent, this] (CodeStream &os, Substitutions &)
            {
                // Get variable name
                const char *spikeCntName = spikeEvent ? "spkCntEvnt" : "spkCnt";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    getArchetype().isDelayRequired() :
                    (getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genScalarFill(os, spikeCntName, "0", VarAccessDuplication::DUPLICATE, batchSize, delayRequired, getArchetype().getNumDelaySlots());
            });
    }

}
//--------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikes(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs, 
                                          bool spikeEvent, unsigned int batchSize) const
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
            [batchSize, spikeEvent, this] (CodeStream &os, Substitutions &varSubs)
            {
                // Get variable name
                const char *spikeName = spikeEvent ? "spkEvnt" : "spk";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    getArchetype().isDelayRequired() :
                    (getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genVariableFill(os, spikeName, "0", varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                                batchSize, delayRequired, getArchetype().getNumDelaySlots());
            });
    }
}
//------------------------------------------------------------------------
void NeuronInitGroupMerged::genInitSpikeTime(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                                             const std::string &varName, unsigned int batchSize) const
{
    // Generate variable initialisation code
    backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
        [batchSize, varName, this] (CodeStream &os, Substitutions &varSubs)
        {
            genVariableFill(os, varName, "-TIME_MAX", varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                            batchSize, getArchetype().isDelayRequired(), getArchetype().getNumDelaySlots());
            
        });
}

//----------------------------------------------------------------------------
// CodeGenerator::SynapseInitGroupMerged
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
    genInitWUVarCode(os, popSubs, getArchetype().getWUModel()->getVars(),
                     getArchetype().getWUVarInitialisers(), stride, getIndex(),
                     modelMerged.getModel().getPrecision(), modelMerged.getModel().getBatchSize(),
                     [this](size_t v, size_t p) { return isWUVarInitParamHeterogeneous(v, p); },
                     [this](size_t v, size_t p) { return isWUVarInitDerivedParamHeterogeneous(v, p); },
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
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseSparseInitGroupMerged::name = "SynapseSparseInit";
//----------------------------------------------------------------------------
void SynapseSparseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitWUVarCode(os, popSubs, getArchetype().getWUModel()->getVars(),
                     getArchetype().getWUVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(),
                     modelMerged.getModel().getPrecision(), modelMerged.getModel().getBatchSize(),
                     [this](size_t v, size_t p) { return isWUVarInitParamHeterogeneous(v, p); },
                     [this](size_t v, size_t p) { return isWUVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler); 
                     });
}

// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityInitGroupMerged::name = "SynapseConnectivityInit";
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseRowInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitConnectivity(os, popSubs, modelMerged.getModel().getPrecision(), true);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateSparseColumnInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitConnectivity(os, popSubs, modelMerged.getModel().getPrecision(), false);
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::generateKernelInit(const BackendBase&, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Generate kernel index and add to substitutions
    os << "const unsigned int kernelInd = ";
    genKernelIndex(os, popSubs);
    os << ";" << std::endl;
    popSubs.addVarSubstitution("id_kernel", "kernelInd");

    const auto vars = getArchetype().getWUModel()->getVars();
    for(size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = getArchetype().getWUVarInitialisers().at(k);

        // If this variable require a kernel
        if(varInit.getSnippet()->requiresKernel()) {
            CodeStream::Scope b(os);

            popSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                              [k, this](size_t p) { return isWUVarInitParamHeterogeneous(k, p); },
                                              "", "group->", vars[k].name);
            popSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                            [k, this](size_t p) { return isWUVarInitDerivedParamHeterogeneous(k, p); },
                                            "", "group->", vars[k].name);
            popSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                            "", "group->", vars[k].name);

            // Generate initial value into temporary variable
            os << vars[k].type << " initVal;" << std::endl;
            popSubs.addVarSubstitution("value", "initVal");
            std::string code = varInit.getSnippet()->getCode();
            //popSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(sg.getIndex()));
            popSubs.apply(code);
            code = ensureFtype(code, modelMerged.getModel().getPrecision());
            os << code << std::endl;

            // Fill value across all batches
            genVariableFill(os,  vars[k].name, "initVal", popSubs["id_syn"], "group->numSrcNeurons * group->rowStride", 
                            getVarAccessDuplication(vars[k].access), modelMerged.getModel().getBatchSize());
        }
    }
}
//----------------------------------------------------------------------------
void SynapseConnectivityInitGroupMerged::genInitConnectivity(CodeStream &os, Substitutions &popSubs, const std::string &ftype, bool rowNotColumns) const
{
    const auto &connectInit = getArchetype().getConnectivityInitialiser();
    const auto *snippet = connectInit.getSnippet();

    // Add substitutions
    popSubs.addFuncSubstitution(rowNotColumns ? "endRow" : "endCol", 0, "break");
    popSubs.addParamValueSubstitution(snippet->getParamNames(), connectInit.getParams(),
                                      [this](size_t i) { return isSparseConnectivityInitParamHeterogeneous(i);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(snippet->getDerivedParams(), connectInit.getDerivedParams(),
                                    [this](size_t i) { return isSparseConnectivityInitDerivedParamHeterogeneous(i);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(snippet->getExtraGlobalParams(), "", "group->");

    // Initialise state variables and loop on generated code to initialise sparse connectivity
    os << "// Build sparse connectivity" << std::endl;
    const auto stateVars = rowNotColumns ? snippet->getRowBuildStateVars() : snippet->getColBuildStateVars();
    for(const auto &a : stateVars) {
        // Apply substitutions to value
        std::string value = a.value;
        popSubs.applyCheckUnreplaced(value, "initSparseConnectivity state var : merged" + std::to_string(getIndex()));
        value = ensureFtype(value, ftype);

        os << a.type << " " << a.name << " = " << value << ";" << std::endl;
    }
    os << "while(true)";
    {
        CodeStream::Scope b(os);

        // Apply substitutions to row build code
        std::string code = rowNotColumns ? snippet->getRowBuildCode() : snippet->getColBuildCode();
        popSubs.addVarNameSubstitution(stateVars);
        popSubs.applyCheckUnreplaced(code, "initSparseConnectivity : merged" + std::to_string(getIndex()));
        code = ensureFtype(code, ftype);

        // Write out code
        os << code << std::endl;
    }
}


// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
const std::string SynapseConnectivityHostInitGroupMerged::name = "SynapseConnectivityHostInit";
//------------------------------------------------------------------------
SynapseConnectivityHostInitGroupMerged::SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string&, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, precision, groups)
{
    // **TODO** these could be generic
    addField("unsigned int", "numSrcNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "rowStride",
             [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });

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
        addField(e.type + "*", e.name,
                 [e](const SynapseGroupInternal &g, size_t) { return "&" + e.name + g.getName(); },
                 FieldType::Host);

        if(!backend.getDeviceVarPrefix().empty()) {
            addField(e.type + "*", backend.getDeviceVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getDeviceVarPrefix() + e.name + g.getName();
                     });
        }
        if(!backend.getHostVarPrefix().empty()) {
            addField(e.type + "*", backend.getHostVarPrefix() + e.name,
                     [e, &backend](const SynapseGroupInternal &g, size_t)
                     {
                         return "&" + backend.getHostVarPrefix() + e.name + g.getName();
                     });
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
                                       [this](size_t p) { return isConnectivityInitParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                     [this](size_t p) { return isConnectivityInitDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        const auto egps = connectInit.getSnippet()->getExtraGlobalParams();
        for(size_t i = 0; i < egps.size(); i++) {
            const auto loc = getArchetype().getSparseConnectivityExtraGlobalParamLocation(i);
            // If EGP is a pointer and located on the host
            if(Utils::isTypePointer(egps[i].type) && (loc & VarLocation::HOST)) {
                // Generate code to allocate this EGP with count specified by $(0)
                std::stringstream allocStream;
                CodeGenerator::CodeStream alloc(allocStream);
                backend.genExtraGlobalParamAllocation(alloc, egps[i].type + "*", egps[i].name,
                                                      loc, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("allocate" + egps[i].name, 1, allocStream.str());

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genExtraGlobalParamPush(push, egps[i].type + "*", egps[i].name,
                                                loc, "$(0)", "group->");


                // Add substitution
                subs.addFuncSubstitution("push" + egps[i].name, 1, pushStream.str());
            }
        }
        std::string code = connectInit.getSnippet()->getHostInitCode();
        subs.applyCheckUnreplaced(code, "hostInitSparseConnectivity : merged" + std::to_string(getIndex()));
        code = ensureFtype(code, modelMerged.getModel().getPrecision());

        // Write out code
        os << code << std::endl;
    }
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitParamHeterogeneous(size_t paramIndex) const
{
    return (isSparseConnectivityInitParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg){ return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const
{
    return (isSparseConnectivityInitDerivedParamReferenced(paramIndex) &&
            isParamValueHeterogeneous(paramIndex, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isSparseConnectivityInitParamReferenced(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getParamNames().at(paramIndex);
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseConnectivityHostInitGroupMerged::isSparseConnectivityInitDerivedParamReferenced(size_t paramIndex) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *connectInitSnippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const std::string paramName = connectInitSnippet->getDerivedParams().at(paramIndex).name;
    return isParamReferenced({connectInitSnippet->getHostInitCode()}, paramName);
}

// ----------------------------------------------------------------------------
// CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateInitGroupMerged::name = "CustomUpdateInit";
//----------------------------------------------------------------------------
CustomUpdateInitGroupMerged::CustomUpdateInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                         const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateInternal, CustomUpdateVarAdapter>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });
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
void CustomUpdateInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise custom update variables
    genInitNeuronVarCode(os, backend, popSubs, getArchetype().getCustomUpdateModel()->getVars(), getArchetype().getVarInitialisers(),
                        "", "size", getIndex(), modelMerged.getModel().getPrecision(), getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                        [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                        [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); });
}

// ----------------------------------------------------------------------------
// CustomWUUpdateInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateInitGroupMerged::name = "CustomWUUpdateInit";
//----------------------------------------------------------------------------
CustomWUUpdateInitGroupMerged::CustomWUUpdateInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                             const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal, CustomUpdateVarAdapter>(index, precision, backend, groups)
{
    if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // Loop through kernel size dimensions
        for (size_t d = 0; d < getArchetype().getSynapseGroup()->getKernelSize().size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(d)) {
                addField("unsigned int", "kernelSize" + std::to_string(d),
                        [d](const CustomUpdateWUInternal &g, size_t) { return std::to_string(g.getSynapseGroup()->getKernelSize().at(d)); });
            }
        }
    }
    else {
        addField("unsigned int", "rowStride",
                [&backend](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });
        addField("unsigned int", "numSrcNeurons",
                [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
        addField("unsigned int", "numTrgNeurons",
                [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });
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
        updateHash([](const CustomUpdateWUInternal &g) { return g.getSynapseGroup()->getKernelSize(); }, hash);
    }
    // Otherwise, update hash with sizes of pre and postsynaptic neuron groups
    else {
        updateHash([](const CustomUpdateWUInternal &cg) 
                   {
                       return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
                   }, hash);

        updateHash([](const CustomUpdateWUInternal &cg) 
                   {
                       return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
                   }, hash);


        updateHash([](const CustomUpdateWUInternal &cg)
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
    genInitWUVarCode(os, popSubs, getArchetype().getCustomUpdateModel()->getVars(),
                    getArchetype().getVarInitialisers(), stride, getIndex(),
                    modelMerged.getModel().getPrecision(), getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                    [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                    [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); },
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
// CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateSparseInitGroupMerged::name = "CustomWUUpdateSparseInit";
//----------------------------------------------------------------------------
CustomWUUpdateSparseInitGroupMerged::CustomWUUpdateSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                         const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal, CustomUpdateVarAdapter>(index, precision, backend, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });

    addField("unsigned int", "numSrcNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const CustomUpdateWUInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });

    addField("unsigned int*", "rowLength", 
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
             { 
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "rowLength" + sg->getName();
             });
    addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind", 
             [&backend](const CustomUpdateWUInternal &cg, size_t) 
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
    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const CustomUpdateWUInternal& cg)
               {
                   return cg.getSynapseGroup()->getMaxConnections();
               }, hash);

    return hash.get_digest();
}
// ----------------------------------------------------------------------------
void CustomWUUpdateSparseInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    genInitWUVarCode(os, popSubs, getArchetype().getCustomUpdateModel()->getVars(),
                     getArchetype().getVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(),
                     modelMerged.getModel().getPrecision(), getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                     [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                     [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); },
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
CustomConnectivityUpdatePreInitGroupMerged::CustomConnectivityUpdatePreInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                                       const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdatePreVarAdapter>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [](const CustomConnectivityUpdateInternal &c, size_t) 
             { 
                 return std::to_string(c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
             });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdatePreInitGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with generic custom update init data
    updateBaseHash(hash);

    // Update hash with size of custom update
    updateHash([](const CustomConnectivityUpdateInternal &cg) 
               { 
                   return cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); 
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePreInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise presynaptic custom connectivity update variables
    genInitNeuronVarCode(os, backend, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getPreVars(), getArchetype().getPreVarInitialisers(),
                         "", "size", getIndex(), modelMerged.getModel().getPrecision(), 1,
                         [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdatePostInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdatePostInitGroupMerged::name = "CustomConnectivityUpdatePostInit";
//----------------------------------------------------------------------------
CustomConnectivityUpdatePostInitGroupMerged::CustomConnectivityUpdatePostInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdatePostVarAdapter>(index, precision, backend, groups)
{
    addField("unsigned int", "size",
             [](const CustomConnectivityUpdateInternal &c, size_t)
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
    updateHash([](const CustomConnectivityUpdateInternal &cg)
               {
                   return cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdatePostInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise presynaptic custom connectivity update variables
    genInitNeuronVarCode(os, backend, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getPostVars(), getArchetype().getPostVarInitialisers(),
                         "", "size", getIndex(), modelMerged.getModel().getPrecision(), 1,
                         [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); });
}

// ----------------------------------------------------------------------------
// CustomConnectivityUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
const std::string CustomConnectivityUpdateSparseInitGroupMerged::name = "CustomConnectivityUpdateSparseInit";
//----------------------------------------------------------------------------
CustomConnectivityUpdateSparseInitGroupMerged::CustomConnectivityUpdateSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                                                                         const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups)
:   CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal, CustomConnectivityUpdateVarAdapter>(index, precision, backend, groups)
{
    addField("unsigned int", "rowStride",
             [&backend](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(*cg.getSynapseGroup())); });

    addField("unsigned int", "numSrcNeurons",
             [](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); });
    addField("unsigned int", "numTrgNeurons",
             [](const CustomConnectivityUpdateInternal &cg, size_t) { return std::to_string(cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons()); });

    addField("unsigned int*", "rowLength",
             [&backend](const CustomConnectivityUpdateInternal &cg, size_t)
             {
                 const SynapseGroupInternal *sg = cg.getSynapseGroup();
                 return backend.getDeviceVarPrefix() + "rowLength" + sg->getName();
             });
    addField(getArchetype().getSynapseGroup()->getSparseIndType() + "*", "ind",
             [&backend](const CustomConnectivityUpdateInternal &cg, size_t)
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
    genInitWUVarCode(os, popSubs, getArchetype().getCustomConnectivityUpdateModel()->getVars(),
                     getArchetype().getVarInitialisers(), "group->numSrcNeurons * group->rowStride", getIndex(),
                     modelMerged.getModel().getPrecision(), false,
                     [this](size_t v, size_t p) { return isVarInitParamHeterogeneous(v, p); },
                     [this](size_t v, size_t p) { return isVarInitDerivedParamHeterogeneous(v, p); },
                     [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                     {
                         return backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler);
                     });
}