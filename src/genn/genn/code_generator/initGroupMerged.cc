#include "code_generator/initGroupMerged.h"

// GeNN code generator includes
#include "code_generator/modelSpecMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

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
void genInitNeuronVarCode(CodeStream &os, const ModelSpecMerged &modelMerged, const BackendBase &backend, const Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::unordered_map<std::string, Models::VarInit> &varInitialisers, 
                          const std::string &fieldSuffix, const std::string &countMember, 
                          size_t numDelaySlots, const size_t groupIndex, unsigned int batchSize,
                          Q isVarQueueRequired, P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn)
{
    const std::string count = "group->" + countMember;
    for (const auto &var : vars) {
        const auto &varInit = varInitialisers.at(var.name);

        // If this variable has any initialisation code
        if (!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            Substitutions varSubs(&popSubs);

            // Substitute in parameters and derived parameters for initialising variables
            varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                              [&var, isParamHeterogeneousFn](const std::string &p) { return isParamHeterogeneousFn(var.name, p); },
                                              "", "group->", var.name + fieldSuffix);
            varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                            [&var, isDerivedParamHeterogeneousFn](const std::string &p) { return isDerivedParamHeterogeneousFn(var.name, p); },
                                            "", "group->", var.name + fieldSuffix);
            varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                           "", "group->", var.name + fieldSuffix);

            // If variable is shared between neurons
            if (getVarAccessDuplication(var.access) == VarAccessDuplication::SHARED_NEURON) {
                backend.genPopVariableInit(
                    os, varSubs,
                    [&var, &varInit, &fieldSuffix, &modelMerged, batchSize, groupIndex, numDelaySlots, isVarQueueRequired]
                    (CodeStream &os, Substitutions &varInitSubs)
                    {
                        // Generate initial value into temporary variable
                        os << var.type.resolve(modelMerged.getTypeContext()).getName() << " initVal;" << std::endl;
                        varInitSubs.addVarSubstitution("value", "initVal");
                        std::string code = varInit.getSnippet()->getCode();
                        varInitSubs.applyCheckUnreplaced(code, "initVar : " + var.name + "merged" + std::to_string(groupIndex));
                        //code = ensureFtype(code, scalarType);
                        os << code << std::endl;

                        // Fill value across all delay slots and batches
                        genScalarFill(os, var.name + fieldSuffix, "initVal", getVarAccessDuplication(var.access),
                                      batchSize, isVarQueueRequired(var.name), numDelaySlots);
                    });
            }
            // Otherwise
            else {
                backend.genVariableInit(
                    os, count, "id", varSubs,
                    [&var, &varInit, &modelMerged, &fieldSuffix, batchSize, groupIndex, count, numDelaySlots, isVarQueueRequired]
                    (CodeStream &os, Substitutions &varInitSubs)
                    {
                        // Generate initial value into temporary variable
                        os << var.type.resolve(modelMerged.getTypeContext()).getName() << " initVal;" << std::endl;
                        varInitSubs.addVarSubstitution("value", "initVal");
                        std::string code = varInit.getSnippet()->getCode();
                        varInitSubs.applyCheckUnreplaced(code, "initVar : " + var.name + "merged" + std::to_string(groupIndex));
                        //code = ensureFtype(code, ftype);
                        os << code << std::endl;

                        // Fill value across all delay slots and batches
                        genVariableFill(os, var.name + fieldSuffix, "initVal", varInitSubs["id"], count,
                                        getVarAccessDuplication(var.access), batchSize, isVarQueueRequired(var.name), numDelaySlots);
                    });
            }
        }
            
    }
}
//------------------------------------------------------------------------
template<typename P, typename D>
void genInitNeuronVarCode(CodeStream &os, const ModelSpecMerged &modelMerged, const BackendBase &backend, const Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::unordered_map<std::string, Models::VarInit> &varInitialisers, 
                          const std::string &fieldSuffix, const std::string &countMember, const size_t groupIndex, 
                          unsigned int batchSize, P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn)
{
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, vars, varInitialisers, fieldSuffix, countMember, 0, groupIndex, batchSize,
                         [](const std::string&){ return false; }, 
                         isParamHeterogeneousFn,
                         isDerivedParamHeterogeneousFn);
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
                    genVariableFill(os,  var.name, "initVal", varSubs["id_syn"], stride,
                                    getVarAccessDuplication(var.access), batchSize);
                });
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::CurrentSource
//----------------------------------------------------------------------------
NeuronInitGroupMerged::CurrentSource::CurrentSource(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                    const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups)
:   GroupMerged<CurrentSourceInternal>(index, typeContext, groups)
{
    const std::string suffix =  "CS" + std::to_string(getIndex());

    // Loop through variables
    // **TODO** adaptor
    const auto &varInit = getArchetype().getVarInitialisers();
    for(const auto &var : getArchetype().getCurrentSourceModel()->getVars()) {
        // Add pointers to state variable
        if(!varInit.at(var.name).getSnippet()->getCode().empty()) {
            addPointerField(var.type, var.name + suffix, 
                            backend.getDeviceVarPrefix() + var.name);
        }

        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<CurrentSource, CurrentSourceVarAdapter>(
            &CurrentSource::isVarInitParamHeterogeneous, suffix);
        addHeterogeneousVarInitDerivedParams<CurrentSource, CurrentSourceVarAdapter>(
            &CurrentSource::isVarInitDerivedParamHeterogeneous, suffix);

        // Add extra global parameters
        for(const auto &e : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
            addField(e.type.resolve(getTypeContext()).createPointer(), e.name + var.name + suffix,
                     [&backend, e, suffix, var](const auto &g, size_t)
                     { 
                         return backend.getDeviceVarPrefix() + e.name + var.name + g.getName(); 
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::CurrentSource::generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                                                    const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "CS" + std::to_string(getIndex());

    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getCurrentSourceModel()->getVars(), getArchetype().getVarInitialisers(),
                         suffix, "numNeurons", getIndex(), modelMerged.getModel().getBatchSize(),
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
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
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &cs){ return cs.getVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::CurrentSource::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &cs){ return cs.getVarInitialisers().at(varName).getDerivedParams(); }));
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
NeuronInitGroupMerged::InSynPSM::InSynPSM(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
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

    // Loop through variables
    // **TODO** adaptor
    const auto &varInit = getArchetype().getPSVarInitialisers();
    for(const auto &var : getArchetype().getPSModel()->getVars()) {
        // Add pointers to state variable
        if(!varInit.at(var.name).getSnippet()->getCode().empty()) {
            addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                     [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedPSVarSuffix(); });
        }

        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<InSynPSM, SynapsePSMVarAdapter>(
            &InSynPSM::isVarInitParamHeterogeneous, suffix);
        addHeterogeneousVarInitDerivedParams<InSynPSM, SynapsePSMVarAdapter>(
            &InSynPSM::isVarInitDerivedParamHeterogeneous, suffix);

        // Add extra global parameters
        for(const auto &e : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
            addField(e.type.resolve(getTypeContext()).createPointer(), e.name + var.name + suffix,
                     [&backend, e, suffix, var](const auto &g, size_t)
                     { 
                         return backend.getDeviceVarPrefix() + e.name + var.name + g.getFusedPSVarSuffix(); 
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynPSM::generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                                               const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "InSyn" + std::to_string(getIndex());

    // Zero InSyn
    backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
        [&modelMerged, &suffix] (CodeStream &os, Substitutions &varSubs)
        {
            genVariableFill(os, "inSyn" + suffix, modelMerged.scalarExpr(0.0), 
                            varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                            modelMerged.getModel().getBatchSize());

        });

    // If dendritic delays are required
    if(getArchetype().isDendriticDelayRequired()) {
        // Zero dendritic delay buffer
        backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
            [&modelMerged, &suffix, this](CodeStream &os, Substitutions &varSubs)
            {
                genVariableFill(os, "denDelay" + suffix, modelMerged.scalarExpr(0.0),
                                varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                                modelMerged.getModel().getBatchSize(),
                                true, getArchetype().getMaxDendriticDelayTimesteps());
            });

        // Zero dendritic delay pointer
        backend.genPopVariableInit(os, popSubs,
            [&suffix](CodeStream &os, Substitutions &)
            {
                os << "*group->denDelayPtr" << suffix << " = 0;" << std::endl;
            });
    }

    // **TODO** adaptor
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getPSModel()->getVars(), getArchetype().getPSVarInitialisers(),
                         suffix, "numNeurons", getIndex(), modelMerged.getModel().getBatchSize(),
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
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
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getPSVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynPSM::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getPSVarInitialisers().at(varName).getDerivedParams(); }));
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
NeuronInitGroupMerged::OutSynPreOutput::OutSynPreOutput(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                        const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());

    addField(getScalarType().createPointer(), "revInSyn" + suffix,
             [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "revInSyn" + g.getFusedPreOutputSuffix(); });
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynPreOutput::generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                                                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "OutSyn" + std::to_string(getIndex());

    backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
                                [&modelMerged, suffix] (CodeStream &os, Substitutions &varSubs)
                                {
                                    genVariableFill(os, "revInSyn" + suffix, modelMerged.scalarExpr(0.0),
                                                    varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                                                    modelMerged.getModel().getBatchSize());
                                });
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged::InSynWUMPostVars
//----------------------------------------------------------------------------
NeuronInitGroupMerged::InSynWUMPostVars::InSynWUMPostVars(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                          const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());

    // Loop through variables
    // **TODO** adaptor
    const auto &varInit = getArchetype().getWUPostVarInitialisers();
    for(const auto &var : getArchetype().getWUModel()->getPostVars()) {
        // Add pointers to state variable
        if(!varInit.at(var.name).getSnippet()->getCode().empty()) {
            addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                     [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedWUPostVarSuffix(); });
        }

        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<InSynWUMPostVars, SynapseWUPostVarAdapter>(
            &InSynWUMPostVars::isVarInitParamHeterogeneous, suffix);
        addHeterogeneousVarInitDerivedParams<InSynWUMPostVars, SynapseWUPostVarAdapter>(
            &InSynWUMPostVars::isVarInitDerivedParamHeterogeneous, suffix);

        // Add extra global parameters
        for(const auto &e : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
            addField(e.type.resolve(getTypeContext()).createPointer(), e.name + var.name + suffix,
                     [&backend, e, suffix, var](const auto &g, size_t)
                     { 
                         return backend.getDeviceVarPrefix() + e.name + var.name + g.getFusedWUPostVarSuffix(); 
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::InSynWUMPostVars::generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                                                       const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "InSynWUMPost" + std::to_string(getIndex());

    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getWUModel()->getPostVars(), getArchetype().getWUPostVarInitialisers(),
                         suffix, "numNeurons", getArchetype().getTrgNeuronGroup()->getNumDelaySlots(), getIndex(), modelMerged.getModel().getBatchSize(),
                         [this](const std::string&){ return (getArchetype().getBackPropDelaySteps() != NO_DELAY); },
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
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
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getWUPostVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::InSynWUMPostVars::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getWUPostVarInitialisers().at(varName).getDerivedParams(); }));
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
NeuronInitGroupMerged::OutSynWUMPreVars::OutSynWUMPreVars(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                          const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
:   GroupMerged<SynapseGroupInternal>(index, typeContext, groups)
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());

    // Loop through variables
    // **TODO** adaptor
    const auto &varInit = getArchetype().getWUPreVarInitialisers();
    for(const auto &var : getArchetype().getWUModel()->getPreVars()) {
        // Add pointers to state variable
        if(!varInit.at(var.name).getSnippet()->getCode().empty()) {
            addField(var.type.resolve(getTypeContext()).createPointer(), var.name + suffix,
                     [&backend, var](const auto &g, size_t) { return backend.getDeviceVarPrefix() + var.name + g.getFusedWUPreVarSuffix(); });
        }

        // Add heterogeneous var init parameters
        addHeterogeneousVarInitParams<OutSynWUMPreVars, SynapseWUPreVarAdapter>(
            &OutSynWUMPreVars::isVarInitParamHeterogeneous, suffix);
        addHeterogeneousVarInitDerivedParams<OutSynWUMPreVars, SynapseWUPreVarAdapter>(
            &OutSynWUMPreVars::isVarInitDerivedParamHeterogeneous, suffix);

        // Add extra global parameters
        for(const auto &e : varInit.at(var.name).getSnippet()->getExtraGlobalParams()) {
            addField(e.type.resolve(getTypeContext()).createPointer(), e.name + var.name + suffix,
                     [&backend, e, suffix, var](const auto &g, size_t)
                     { 
                         return backend.getDeviceVarPrefix() + e.name + var.name + g.getFusedWUPreVarSuffix(); 
                     },
                     GroupMergedFieldType::DYNAMIC);
        }
    }
}
//----------------------------------------------------------------------------
void NeuronInitGroupMerged::OutSynWUMPreVars::generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                                                       const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    const std::string suffix =  "OutSynWUMPre" + std::to_string(getIndex());

    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getWUModel()->getPreVars(), getArchetype().getWUPreVarInitialisers(),
                         suffix, "numNeurons", getArchetype().getSrcNeuronGroup()->getNumDelaySlots(), getIndex(), modelMerged.getModel().getBatchSize(),
                         [this](const std::string&){ return (getArchetype().getDelaySteps() != NO_DELAY); },
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
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
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getWUPreVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool NeuronInitGroupMerged::OutSynWUMPreVars::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName,
                                      [varName](const auto &sg){ return sg.getWUPreVarInitialisers().at(varName).getDerivedParams(); }));
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
    orderNeuronGroupChildren(m_MergedInSynPSMGroups, typeContext, backend,
                             &NeuronGroupInternal::getFusedPSMInSyn,
                             &SynapseGroupInternal::getPSInitHashDigest );

    // Build vector of vectors containing each child group's merged out syns with pre output, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynPreOutputGroups, typeContext, backend, 
                             &NeuronGroupInternal::getFusedPreOutputOutSyn,
                             &SynapseGroupInternal::getPreOutputInitHashDigest );

    // Build vector of vectors containing each child group's current sources, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedCurrentSourceGroups, typeContext, backend,
                             &NeuronGroupInternal::getCurrentSources,
                             &CurrentSourceInternal::getInitHashDigest );


    // Build vector of vectors containing each child group's incoming synapse groups
    // with postsynaptic weight update model variable, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedInSynWUMPostVarGroups, typeContext, backend,
                             &NeuronGroupInternal::getFusedInSynWithPostVars,
                             &SynapseGroupInternal::getWUPostInitHashDigest);

    // Build vector of vectors containing each child group's outgoing synapse groups
    // with presynaptic weight update model variables, ordered to match those of the archetype group
    orderNeuronGroupChildren(m_MergedOutSynWUMPreVarGroups, typeContext, backend, 
                             &NeuronGroupInternal::getFusedOutSynWithPreVars,
                             &SynapseGroupInternal::getWUPreInitHashDigest);


    if(backend.isPopulationRNGRequired() && getArchetype().isSimRNGRequired() 
       && backend.isPopulationRNGInitialisedOnDevice()) 
    {
        addPointerField(*backend.getMergedGroupSimRNGType(), "rng", backend.getDeviceVarPrefix() + "rng");
    }

    // Loop through variables
    const NeuronModels::Base *nm = getArchetype().getNeuronModel();
    const auto vars = nm->getVars();
    const auto &varInit = getArchetype().getVarInitialisers();
    for(const auto &var : vars) {
        // If we're not initialising or if there is initialization code for this variable
        if(!varInit.at(var.name).getSnippet()->getCode().empty()) {
            addPointerField(var.type, var.name, 
                            backend.getDeviceVarPrefix() + var.name);
        }

        // Add any var init EGPs to structure
        addEGPs(varInit.at(var.name).getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);
    }

    // Add heterogeneous var init parameters
    addHeterogeneousVarInitParams<NeuronGroupMergedBase, NeuronVarAdapter>(
        &NeuronGroupMergedBase::isVarInitParamHeterogeneous);

    addHeterogeneousVarInitDerivedParams<NeuronGroupMergedBase, NeuronVarAdapter>(
        &NeuronGroupMergedBase::isVarInitDerivedParamHeterogeneous);
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
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getNeuronModel()->getVars(), getArchetype().getVarInitialisers(), 
                         "", "numNeurons", getArchetype().getNumDelaySlots(), getIndex(), model.getBatchSize(),
                         [this](const std::string &v){ return getArchetype().isVarQueueRequired(v); },
                         [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                         [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
 
    // Generate initialisation code for child groups
    for (const auto &cs : getMergedCurrentSourceGroups()) {
        cs.generate(backend, os, *this, modelMerged, popSubs);
    }
    for(const auto &sg : getMergedInSynPSMGroups()) {
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }
    for (const auto &sg : getMergedOutSynPreOutputGroups()) {
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }  
    for (const auto &sg : getMergedOutSynWUMPreVarGroups()) {
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }
    for (const auto &sg : getMergedInSynWUMPostVarGroups()) {
        sg.generate(backend, os, *this, modelMerged, popSubs);
    }
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
void CustomUpdateInitGroupMerged::generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const
{
    // Initialise custom update variables
    genInitNeuronVarCode(os, modelMerged, backend, popSubs, getArchetype().getCustomUpdateModel()->getVars(), getArchetype().getVarInitialisers(),
                        "", "size", getIndex(), getArchetype().isBatched() ? modelMerged.getModel().getBatchSize() : 1,
                        [this](const std::string &v, const std::string &p) { return isVarInitParamHeterogeneous(v, p); },
                        [this](const std::string &v, const std::string &p) { return isVarInitDerivedParamHeterogeneous(v, p); });
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
