#include "code_generator/generateInit.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"

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
    const unsigned int numValues = ((varDuplication & VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

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
    const unsigned int numValues = ((varDuplication & VarAccessDuplication::SHARED) ? 1 : batchSize) * ((delay ? numDelaySlots : 1));

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
//--------------------------------------------------------------------------
void genInitSpikeCount(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs, 
                       const NeuronInitGroupMerged &ng, bool spikeEvent, unsigned int batchSize)
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genPopVariableInit(os, popSubs,
            [&ng, batchSize, spikeEvent] (CodeStream &os, Substitutions &)
            {
                // Get variable name
                const char *spikeCntName = spikeEvent ? "spkCntEvnt" : "spkCnt";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.getArchetype().isDelayRequired() :
                    (ng.getArchetype().isTrueSpikeRequired() && ng.getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genScalarFill(os, spikeCntName, "0", VarAccessDuplication::DUPLICATE, batchSize, delayRequired, ng.getArchetype().getNumDelaySlots());
            });
    }

}
//--------------------------------------------------------------------------
void genInitSpikes(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs, 
                   const NeuronInitGroupMerged &ng, bool spikeEvent, unsigned int batchSize)
{
    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
            [&ng, batchSize, spikeEvent] (CodeStream &os, Substitutions &varSubs)
            {
                // Get variable name
                const char *spikeName = spikeEvent ? "spkEvnt" : "spk";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.getArchetype().isDelayRequired() :
                    (ng.getArchetype().isTrueSpikeRequired() && ng.getArchetype().isDelayRequired());

                // Zero across all delay slots and batches
                genVariableFill(os, spikeName, "0", varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                                batchSize, delayRequired, ng.getArchetype().getNumDelaySlots());
            });
    }
}
//------------------------------------------------------------------------
void genInitSpikeTime(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                      const NeuronInitGroupMerged &ng, const std::string &varName, unsigned int batchSize)
{
    // Generate variable initialisation code
    backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
        [batchSize, varName, &ng] (CodeStream &os, Substitutions &varSubs)
        {
            genVariableFill(os, varName, "-TIME_MAX", varSubs["id"], "group->numNeurons", VarAccessDuplication::DUPLICATE, 
                            batchSize, ng.getArchetype().isDelayRequired(), ng.getArchetype().getNumDelaySlots());
            
        });
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
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genVariableInit(os, count, "id", popSubs,
                [&vars, &varInit, &fieldSuffix, &ftype, batchSize, groupIndex, k, count, isVarQueueRequired, isParamHeterogeneousFn, isDerivedParamHeterogeneousFn, numDelaySlots]
                (CodeStream &os, Substitutions &varSubs)
                {
                    // Substitute in parameters and derived parameters for initialising variables
                    varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                      [k, isParamHeterogeneousFn](size_t p) { return isParamHeterogeneousFn(k, p); },
                                                      "", "group->", vars[k].name + fieldSuffix);
                    varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                    [k, isDerivedParamHeterogeneousFn](size_t p) { return isDerivedParamHeterogeneousFn(k, p); },
                                                    "", "group->", vars[k].name + fieldSuffix);
                    varSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                                   "", "group->", vars[k].name + fieldSuffix);

                    // Generate initial value into temporary variable
                    os << vars[k].type << " initVal;" << std::endl;
                    varSubs.addVarSubstitution("value", "initVal");
                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.applyCheckUnreplaced(code, "initVar : " + vars[k].name + "merged" + std::to_string(groupIndex));
                    code = ensureFtype(code, ftype);
                    os << code << std::endl;
                    
                    // Fill value across all delay slots and batches
                    genVariableFill(os,  vars[k].name + fieldSuffix, "initVal", varSubs["id"], count, 
                                    getVarAccessDuplication(vars[k].access), batchSize, isVarQueueRequired(k), numDelaySlots);
                });
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
                      const size_t groupIndex, const std::string &ftype, unsigned int batchSize,
                      P isParamHeterogeneousFn, D isDerivedParamHeterogeneousFn, G genSynapseVariableRowInitFn)
{
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = varInitialisers.at(k);

        // If this variable has any initialisation code and doesn't require a kernel
        if(!varInit.getSnippet()->getCode().empty() && !varInit.getSnippet()->requiresKernel()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            genSynapseVariableRowInitFn(os, popSubs,
                [&vars, &varInit, &ftype, batchSize, k, groupIndex, isParamHeterogeneousFn, isDerivedParamHeterogeneousFn]
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
                    genVariableFill(os,  vars[k].name, "initVal", varSubs["id_syn"], "group->numSrcNeurons * group->rowStride", 
                                    getVarAccessDuplication(vars[k].access), batchSize);
                });
        }
    }
}
//------------------------------------------------------------------------
// Generate either row or column connectivity init code
void genInitConnectivity(CodeStream &os, Substitutions &popSubs, const SynapseConnectivityInitGroupMerged &sg,
                         const std::string &ftype, bool rowNotColumns)
{
    const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();
    const auto *snippet = connectInit.getSnippet();

    // Add substitutions
    popSubs.addFuncSubstitution(rowNotColumns ? "endRow" : "endCol", 0, "break");
    popSubs.addParamValueSubstitution(snippet->getParamNames(), connectInit.getParams(),
                                      [&sg](size_t i) { return sg.isConnectivityInitParamHeterogeneous(i);  },
                                      "", "group->");
    popSubs.addVarValueSubstitution(snippet->getDerivedParams(), connectInit.getDerivedParams(),
                                    [&sg](size_t i) { return sg.isConnectivityInitDerivedParamHeterogeneous(i);  },
                                    "", "group->");
    popSubs.addVarNameSubstitution(snippet->getExtraGlobalParams(), "", "group->");

    // Initialise state variables and loop on generated code to initialise sparse connectivity
    os << "// Build sparse connectivity" << std::endl;
    const auto stateVars = rowNotColumns ? snippet->getRowBuildStateVars() : snippet->getColBuildStateVars();
    for(const auto &a : stateVars) {
        // Apply substitutions to value
        std::string value = a.value;
        popSubs.applyCheckUnreplaced(value, "initSparseConnectivity state var : merged" + std::to_string(sg.getIndex()));

        os << a.type << " " << a.name << " = " << value << ";" << std::endl;
    }
    os << "while(true)";
    {
        CodeStream::Scope b(os);

        // Apply substitutions to row build code
        std::string code = rowNotColumns ? snippet->getRowBuildCode() : snippet->getColBuildCode();
        popSubs.addVarNameSubstitution(stateVars);
        popSubs.applyCheckUnreplaced(code, "initSparseConnectivity : merged" + std::to_string(sg.getIndex()));
        code = ensureFtype(code, ftype);

        // Write out code
        os << code << std::endl;
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateInit(CodeStream &os, BackendBase::MemorySpaces &memorySpaces,
                                 const ModelSpecMerged &modelMerged, const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;

    // Generate functions to push merged synapse group structures
    const ModelSpecInternal &model = modelMerged.getModel();

    backend.genInit(os, modelMerged, memorySpaces,
        // Preamble handler
        [&modelMerged, &backend](CodeStream &os)
        {
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedNeuronInitGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomUpdateInitGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedCustomWUUpdateDenseInitGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedSynapseDenseInitGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedSynapseConnectivityInitGroups(), backend);
            modelMerged.genMergedGroupPush(os, modelMerged.getMergedSynapseSparseInitGroups(), backend);
        },
        // Local neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronInitGroupMerged &ng, Substitutions &popSubs)
        {
            // Initialise spike counts
            genInitSpikeCount(os, backend, popSubs, ng, false, model.getBatchSize());
            genInitSpikeCount(os, backend, popSubs, ng, true, model.getBatchSize());

            // Initialise spikes
            genInitSpikes(os, backend, popSubs, ng, false,  model.getBatchSize());
            genInitSpikes(os, backend, popSubs, ng, true,  model.getBatchSize());

            // Initialize spike times
            if(ng.getArchetype().isSpikeTimeRequired()) {
                genInitSpikeTime(os, backend, popSubs, ng, "sT",  model.getBatchSize());
            }

            // Initialize previous spike times
            if(ng.getArchetype().isPrevSpikeTimeRequired()) {
                genInitSpikeTime(os, backend, popSubs, ng, "prevST",  model.getBatchSize());
            }
               
            // Initialize spike-like-event times
            if(ng.getArchetype().isSpikeEventTimeRequired()) {
                genInitSpikeTime(os, backend, popSubs, ng, "seT",  model.getBatchSize());
            }

            // Initialize previous spike-like-event times
            if(ng.getArchetype().isPrevSpikeEventTimeRequired()) {
                genInitSpikeTime(os, backend, popSubs, ng, "prevSET",  model.getBatchSize());
            }
       
            // If neuron group requires delays, zero spike queue pointer
            if(ng.getArchetype().isDelayRequired()) {
                backend.genPopVariableInit(os, popSubs,
                    [](CodeStream &os, Substitutions &)
                    {
                        os << "*group->spkQuePtr = 0;" << std::endl;
                    });
            }

            // Initialise neuron variables
            genInitNeuronVarCode(os, backend, popSubs, ng.getArchetype().getNeuronModel()->getVars(), ng.getArchetype().getVarInitialisers(), 
                                 "", "numNeurons", ng.getArchetype().getNumDelaySlots(), ng.getIndex(), model.getPrecision(), model.getBatchSize(),
                                 [&ng](size_t i){ return ng.getArchetype().isVarQueueRequired(i); },
                                 [&ng](size_t v, size_t p) { return ng.isVarInitParamHeterogeneous(v, p); },
                                 [&ng](size_t v, size_t p) { return ng.isVarInitDerivedParamHeterogeneous(v, p); });

            // Loop through incoming synaptic populations
            for(size_t i = 0; i < ng.getArchetype().getMergedInSyn().size(); i++) {
                CodeStream::Scope b(os);

                const auto *sg = ng.getArchetype().getMergedInSyn()[i];

                // If this synapse group's input variable should be initialised on device
                // Generate target-specific code to initialise variable
                backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
                    [&model, i] (CodeStream &os, Substitutions &varSubs)
                    {
                        os << "group->inSynInSyn" << i << "[" << varSubs["id"] << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    });

                // If dendritic delays are required
                if(sg->isDendriticDelayRequired()) {
                    backend.genVariableInit(os, "group->numNeurons", "id", popSubs,
                        [&model, sg, i](CodeStream &os, Substitutions &varSubs)
                        {
                            os << "for (unsigned int d = 0; d < " << sg->getMaxDendriticDelayTimesteps() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(d * group->numNeurons) + " + varSubs["id"];
                                os << "group->denDelayInSyn" << i << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
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
                                         [&ng, i](size_t v, size_t p) { return ng.isPSMVarInitParamHeterogeneous(i, v, p); },
                                         [&ng, i](size_t v, size_t p) { return ng.isPSMVarInitDerivedParamHeterogeneous(i, v, p); });
                }
            }

            // Loop through incoming synaptic populations with postsynaptic variables
            // **NOTE** number of delay slots is based on the target neuron (for simplicity) but whether delay is required is based on the synapse group
            const auto inSynWithPostVars = ng.getArchetype().getInSynWithPostVars();
            for(size_t i = 0; i < inSynWithPostVars.size(); i++) {
                const auto *sg = inSynWithPostVars.at(i);
                genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPostVars(), sg->getWUPostVarInitialisers(),
                                     "WUPost" + std::to_string(i), "numNeurons", sg->getTrgNeuronGroup()->getNumDelaySlots(),
                                     i, model.getPrecision(),  model.getBatchSize(),
                                     [&sg](size_t){ return (sg->getBackPropDelaySteps() != NO_DELAY); },
                                     [&ng, i](size_t v, size_t p) { return ng.isInSynWUMVarInitParamHeterogeneous(i, v, p); },
                                     [&ng, i](size_t v, size_t p) { return ng.isInSynWUMVarInitDerivedParamHeterogeneous(i, v, p); });
            }

            // Loop through outgoing synaptic populations with presynaptic variables
            // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
            const auto outSynWithPostVars = ng.getArchetype().getOutSynWithPreVars();
            for(size_t i = 0; i < outSynWithPostVars.size(); i++) {
                const auto *sg = outSynWithPostVars.at(i);
                genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPreVars(), sg->getWUPreVarInitialisers(),
                                     "WUPre" + std::to_string(i), "numNeurons", sg->getSrcNeuronGroup()->getNumDelaySlots(),
                                     i, model.getPrecision(),  model.getBatchSize(),
                                     [&sg](size_t){ return (sg->getDelaySteps() != NO_DELAY); },
                                     [&ng, i](size_t v, size_t p) { return ng.isOutSynWUMVarInitParamHeterogeneous(i, v, p); },
                                     [&ng, i](size_t v, size_t p) { return ng.isOutSynWUMVarInitDerivedParamHeterogeneous(i, v, p); });
            }

            // Loop through current sources
            os << "// current source variables" << std::endl;
            for(size_t i = 0; i < ng.getArchetype().getCurrentSources().size(); i++) {
                const auto *cs = ng.getArchetype().getCurrentSources()[i];

                genInitNeuronVarCode(os, backend, popSubs, cs->getCurrentSourceModel()->getVars(), cs->getVarInitialisers(),
                                     "CS" + std::to_string(i), "numNeurons", i, model.getPrecision(),  model.getBatchSize(),
                                     [&ng, i](size_t v, size_t p) { return ng.isCurrentSourceVarInitParamHeterogeneous(i, v, p); },
                                     [&ng, i](size_t v, size_t p) { return ng.isCurrentSourceVarInitDerivedParamHeterogeneous(i, v, p); });
            }
        },
        // Custom update group initialisation
        [&backend, &model](CodeStream &os, const CustomUpdateInitGroupMerged &cg, Substitutions &popSubs)
        {
            // Initialise custom update variables
            genInitNeuronVarCode(os, backend, popSubs, cg.getArchetype().getCustomUpdateModel()->getVars(), cg.getArchetype().getVarInitialisers(),
                                 "", "size", cg.getIndex(), model.getPrecision(), cg.getArchetype().isBatched() ? model.getBatchSize() : 1,
                                 [&cg](size_t v, size_t p) { return cg.isVarInitParamHeterogeneous(v, p); },
                                 [&cg](size_t v, size_t p) { return cg.isVarInitDerivedParamHeterogeneous(v, p); });
        },
        // Custom WU update dense variable initialisation
        [&backend, &model](CodeStream &os, const CustomWUUpdateDenseInitGroupMerged &cg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
            {
                CodeStream::Scope b(os);
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, popSubs, cg.getArchetype().getCustomUpdateModel()->getVars(),
                                 cg.getArchetype().getVarInitialisers(), cg.getIndex(),
                                 model.getPrecision(), cg.getArchetype().isBatched() ? model.getBatchSize() : 1,
                                 [&cg](size_t v, size_t p) { return cg.isVarInitParamHeterogeneous(v, p); },
                                 [&cg](size_t v, size_t p) { return cg.isVarInitDerivedParamHeterogeneous(v, p); },
                                 [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                                 {
                                     return backend.genDenseSynapseVariableRowInit(os, kernelSubs, handler); 
                                 });

            }
        },
        // Dense synaptic matrix variable initialisation
        [&backend, &model](CodeStream &os, const SynapseDenseInitGroupMerged &sg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < group->numSrcNeurons; i++)";
            {
                CodeStream::Scope b(os);
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, popSubs, sg.getArchetype().getWUModel()->getVars(),
                                 sg.getArchetype().getWUVarInitialisers(), sg.getIndex(),
                                 model.getPrecision(), model.getBatchSize(),
                                 [&sg](size_t v, size_t p) { return sg.isWUVarInitParamHeterogeneous(v, p); },
                                 [&sg](size_t v, size_t p) { return sg.isWUVarInitDerivedParamHeterogeneous(v, p); },
                                 [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                                 {
                                     return backend.genDenseSynapseVariableRowInit(os, kernelSubs, handler); 
                                 });
            }
        },
        // Sparse synaptic matrix row connectivity initialisation
        [&model](CodeStream &os, const SynapseConnectivityInitGroupMerged &sg, Substitutions &popSubs)
        {
            genInitConnectivity(os, popSubs, sg, model.getPrecision(), true);
        },
        // Sparse synaptic matrix column connectivity initialisation
        [&model](CodeStream &os, const SynapseConnectivityInitGroupMerged &sg, Substitutions &popSubs)
        {
            genInitConnectivity(os, popSubs, sg, model.getPrecision(), false);
        },
        // Kernel matrix var initialisation
        [&backend, &model](CodeStream &os, const SynapseConnectivityInitGroupMerged &sg, Substitutions &popSubs)
        {
            // Generate kernel index and add to substitutions
            os << "const unsigned int kernelInd = ";
            genKernelIndex(os, popSubs, sg);
            os << ";" << std::endl;
            popSubs.addVarSubstitution("id_kernel", "kernelInd");

            const auto vars = sg.getArchetype().getWUModel()->getVars();
            for(size_t k = 0; k < vars.size(); k++) {
                const auto &varInit = sg.getArchetype().getWUVarInitialisers().at(k);

                // If this variable require a kernel
                if(varInit.getSnippet()->requiresKernel()) {
                    CodeStream::Scope b(os);

                    popSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                      [k, &sg](size_t p) { return sg.isWUVarInitParamHeterogeneous(k, p); },
                                                      "", "group->", vars[k].name);
                    popSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                    [k, &sg](size_t p) { return sg.isWUVarInitDerivedParamHeterogeneous(k, p); },
                                                    "", "group->", vars[k].name);
                    popSubs.addVarNameSubstitution(varInit.getSnippet()->getExtraGlobalParams(),
                                                    "", "group->", vars[k].name);

                    // Generate initial value into temporary variable
                    os << vars[k].type << " initVal;" << std::endl;
                    popSubs.addVarSubstitution("value", "initVal");
                    std::string code = varInit.getSnippet()->getCode();
                    //popSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(sg.getIndex()));
                    popSubs.apply(code);
                    code = ensureFtype(code, model.getPrecision());
                    os << code << std::endl;

                    // Fill value across all batches
                    genVariableFill(os,  vars[k].name, "initVal", popSubs["id_syn"], "group->numSrcNeurons * group->rowStride", 
                                    getVarAccessDuplication(vars[k].access), model.getBatchSize());
                }
            }
        },
        // Sparse synaptic matrix var initialisation
        [&backend, &model](CodeStream &os, const SynapseSparseInitGroupMerged &sg, Substitutions &popSubs)
        {
            genInitWUVarCode(os, popSubs, sg.getArchetype().getWUModel()->getVars(),
                             sg.getArchetype().getWUVarInitialisers(), sg.getIndex(),
                             model.getPrecision(), model.getBatchSize(),
                             [&sg](size_t v, size_t p) { return sg.isWUVarInitParamHeterogeneous(v, p); },
                             [&sg](size_t v, size_t p) { return sg.isWUVarInitDerivedParamHeterogeneous(v, p); },
                             [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                             {
                                 return backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler); 
                             });
        },
        // Custom WU update sparse variable initialisation
        [&backend, &model](CodeStream &os, const CustomWUUpdateSparseInitGroupMerged &cg, Substitutions &popSubs)
        {
            genInitWUVarCode(os, popSubs, cg.getArchetype().getCustomUpdateModel()->getVars(),
                             cg.getArchetype().getVarInitialisers(), cg.getIndex(),
                             model.getPrecision(), cg.getArchetype().isBatched() ? model.getBatchSize() : 1,
                             [&cg](size_t v, size_t p) { return cg.isVarInitParamHeterogeneous(v, p); },
                             [&cg](size_t v, size_t p) { return cg.isVarInitDerivedParamHeterogeneous(v, p); },
                             [&backend](CodeStream &os, const Substitutions &kernelSubs, BackendBase::Handler handler)
                             {
                                 return backend.genSparseSynapseVariableRowInit(os, kernelSubs, handler); 
                             });
        },
        // Initialise push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<NeuronInitGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<CustomUpdateInitGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<CustomWUUpdateDenseInitGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<SynapseDenseInitGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<SynapseConnectivityInitGroupMerged>(os, backend);
        },
        // Initialise sparse push EGP handler
        [&backend, &modelMerged](CodeStream &os)
        {
            modelMerged.genScalarEGPPush<SynapseSparseInitGroupMerged>(os, backend);
            modelMerged.genScalarEGPPush<CustomWUUpdateSparseInitGroupMerged>(os, backend);
        });
}
