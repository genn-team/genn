#include "code_generator/generateInit.h"

// Standard C++ includes
#include <string>

// GeNN includes
#include "models.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void genInitSpikeCount(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                       const CodeGenerator::Substitutions &popSubs, const CodeGenerator::NeuronGroupMerged &ng, bool spikeEvent)
{
    using namespace CodeGenerator;

    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genPopVariableInit(os, popSubs,
            [&backend, &ng, spikeEvent] (CodeStream &os, Substitutions &)
            {
                // Get variable name
                const char *spikeCntName = spikeEvent ? "spkCntEvnt" : "spkCnt";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.getArchetype().isDelayRequired() :
                    (ng.getArchetype().isTrueSpikeRequired() && ng.getArchetype().isDelayRequired());

                if(delayRequired) {
                    os << "for (unsigned int d = 0; d < " << ng.getArchetype().getNumDelaySlots() << "; d++)";
                    {
                        CodeStream::Scope b(os);
                        os << "group." << spikeCntName << "[d] = 0;" << std::endl;
                    }
                }
                else {
                    os << "group." << spikeCntName << "[0] = 0;" << std::endl;
                }
            });
    }

}
//--------------------------------------------------------------------------
void genInitSpikes(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                   const CodeGenerator::Substitutions &popSubs, const CodeGenerator::NeuronGroupMerged &ng, bool spikeEvent)
{
    using namespace CodeGenerator;

    // Is initialisation required at all
    const bool initRequired = spikeEvent ? ng.getArchetype().isSpikeEventRequired() : true;
    if(initRequired) {
        // Generate variable initialisation code
        backend.genVariableInit(os, "group.numNeurons", "id", popSubs,
            [&backend, &ng, spikeEvent] (CodeStream &os, Substitutions &varSubs)
            {
                // Get variable name
                const char *spikeName = spikeEvent ? "spkEvnt" : "spk";

                // Is delay required
                const bool delayRequired = spikeEvent ?
                    ng.getArchetype().isDelayRequired() :
                    (ng.getArchetype().isTrueSpikeRequired() && ng.getArchetype().isDelayRequired());

                if(delayRequired) {
                    os << "for (unsigned int d = 0; d < " << ng.getArchetype().getNumDelaySlots() << "; d++)";
                    {
                        CodeStream::Scope b(os);
                        os << "group." << spikeName << "[(d * group.numNeurons) + " + varSubs["id"] + "] = 0;" << std::endl;
                    }
                }
                else {
                    os << "group." << spikeName << "[" << varSubs["id"] << "] = 0;" << std::endl;
                }
            });
    }
}
//------------------------------------------------------------------------
template<typename I, typename Q>
void genInitNeuronVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend, const CodeGenerator::Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::string &fieldSuffix, const std::string &countMember, 
                          size_t numDelaySlots, const size_t groupIndex, const std::string &ftype,
                          I getVarInitialiser, Q isVarQueueRequired)
{
    using namespace CodeGenerator;

    const std::string count = "group." + countMember;
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = getVarInitialiser(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genVariableInit(os, count, "id", popSubs,
                [&backend, &vars, &varInit, &fieldSuffix, &ftype, groupIndex, k, count, isVarQueueRequired, numDelaySlots]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams());
                    varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams());

                    // If variable requires a queue
                    if (isVarQueueRequired(k)) {
                        // Generate initial value into temporary variable
                        os << vars[k].type << " initVal;" << std::endl;
                        varSubs.addVarSubstitution("value", "initVal");


                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.applyCheckUnreplaced(code, "initVar : " + vars[k].name + "merged" + std::to_string(groupIndex));
                        code = ensureFtype(code, ftype);
                        os << code << std::endl;

                        // Copy this into all delay slots
                        os << "for (unsigned int d = 0; d < " << numDelaySlots << "; d++)";
                        {
                            CodeStream::Scope b(os);
                            os << "group." + vars[k].name << fieldSuffix << "[(d * " << count << ") + " + varSubs["id"] + "] = initVal;" << std::endl;
                        }
                    }
                    else {
                        varSubs.addVarSubstitution("value", "group." + vars[k].name + fieldSuffix + "[" + varSubs["id"] + "]");

                        std::string code = varInit.getSnippet()->getCode();
                        varSubs.applyCheckUnreplaced(code, "initVar : " + vars[k].name + "merged" + std::to_string(groupIndex));
                        code = ensureFtype(code, ftype);
                        os << code << std::endl;
                    }
                });
        }
    }
}
//------------------------------------------------------------------------
template<typename I>
void genInitNeuronVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend, const CodeGenerator::Substitutions &popSubs,
                          const Models::Base::VarVec &vars, const std::string &fieldSuffix, const std::string &countMember, 
                          const size_t groupIndex, const std::string &ftype, I getVarInitialiser)
{
    genInitNeuronVarCode(os, backend, popSubs, vars, fieldSuffix, countMember, 0, groupIndex, ftype,
                         getVarInitialiser,
                         [](size_t){ return false; });
}
//------------------------------------------------------------------------
// Initialise one row of weight update model variables
void genInitWUVarCode(CodeGenerator::CodeStream &os, const CodeGenerator::BackendBase &backend,
                      const CodeGenerator::Substitutions &popSubs, const CodeGenerator::SynapseGroupMerged &sg, const std::string &ftype)
{
    using namespace CodeGenerator;

    const auto vars = sg.getArchetype().getWUModel()->getVars();
    for (size_t k = 0; k < vars.size(); k++) {
        const auto &varInit = sg.getArchetype().getWUVarInitialisers().at(k);

        // If this variable has any initialisation code
        if(!varInit.getSnippet()->getCode().empty()) {
            CodeStream::Scope b(os);

            // Generate target-specific code to initialise variable
            backend.genSynapseVariableRowInit(os, sg, popSubs,
                [&backend, &vars, &varInit, &sg, &ftype, k]
                (CodeStream &os, Substitutions &varSubs)
                {
                    varSubs.addVarSubstitution("value", "group." + vars[k].name + "[" + varSubs["id_syn"] +  "]");
                    varSubs.addParamValueSubstitution(varInit.getSnippet()->getParamNames(), varInit.getParams(),
                                                      [k, &sg](size_t p) { return sg.isWUVarInitParamHeterogeneous(k, p); },
                                                      "", "group.", vars[k].name);
                    varSubs.addVarValueSubstitution(varInit.getSnippet()->getDerivedParams(), varInit.getDerivedParams(),
                                                      [k, &sg](size_t p) { return sg.isWUVarInitDerivedParamHeterogeneous(k, p); },
                                                      "", "group.", vars[k].name);

                    std::string code = varInit.getSnippet()->getCode();
                    varSubs.applyCheckUnreplaced(code, "initVar : merged" + vars[k].name + std::to_string(sg.getIndex()));
                    code = ensureFtype(code, ftype);
                    os << code << std::endl;
                });
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
void CodeGenerator::generateInit(CodeStream &os, const MergedEGPMap &mergedEGPs, const ModelSpecMerged &modelMerged,
                                 const BackendBase &backend)
{
    os << "#include \"definitionsInternal.h\"" << std::endl;

    // Generate functions to push merged synapse group structures
    const ModelSpecInternal &model = modelMerged.getModel();
    genMergedGroupPush(os, modelMerged.getMergedNeuronInitGroups(), mergedEGPs, "NeuronInit", backend);
    genMergedGroupPush(os, modelMerged.getMergedSynapseDenseInitGroups(), mergedEGPs, "SynapseDenseInit", backend);
    genMergedGroupPush(os, modelMerged.getMergedSynapseConnectivityInitGroups(), mergedEGPs, "SynapseConnectivityInit", backend);
    genMergedGroupPush(os, modelMerged.getMergedSynapseSparseInitGroups(), mergedEGPs, "SynapseSparseInit", backend);

    backend.genInit(os, modelMerged,
        // Local neuron group initialisation
        [&backend, &model](CodeStream &os, const NeuronGroupMerged &ng, Substitutions &popSubs)
        {
            // Initialise spike counts
            genInitSpikeCount(os, backend, popSubs, ng, false);
            genInitSpikeCount(os, backend, popSubs, ng, true);

            // Initialise spikes
            genInitSpikes(os, backend, popSubs, ng, false);
            genInitSpikes(os, backend, popSubs, ng, true);

            // If spike times are required
            if(ng.getArchetype().isSpikeTimeRequired()) {
                // Generate variable initialisation code
                backend.genVariableInit(os, "group.numNeurons", "id", popSubs,
                    [&backend, &ng] (CodeStream &os, Substitutions &varSubs)
                    {
                        // Is delay required
                        if(ng.getArchetype().isDelayRequired()) {
                            os << "for (unsigned int d = 0; d < " << ng.getArchetype().getNumDelaySlots() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                os << "group.sT[(d * group.numNeurons) + " + varSubs["id"] + "] = -TIME_MAX;" << std::endl;
                            }
                        }
                        else {
                            os << "group.sT[" << varSubs["id"] << "] = -TIME_MAX;" << std::endl;
                        }
                    });
            }

            // Initialise neuron variables
            genInitNeuronVarCode(os, backend, popSubs, ng.getArchetype().getNeuronModel()->getVars(), "", "numNeurons",
                                 ng.getArchetype().getNumDelaySlots(), ng.getIndex(), model.getPrecision(),
                                 [&ng](size_t i){ return ng.getArchetype().getVarInitialisers().at(i); },
                                 [&ng](size_t i){ return ng.getArchetype().isVarQueueRequired(i); });

            // Loop through incoming synaptic populations
            for(size_t i = 0; i < ng.getArchetype().getMergedInSyn().size(); i++) {
                CodeStream::Scope b(os);

                const auto *sg = ng.getArchetype().getMergedInSyn()[i].first;

                // If this synapse group's input variable should be initialised on device
                // Generate target-specific code to initialise variable
                backend.genVariableInit(os, "group.numNeurons", "id", popSubs,
                    [&backend, &model, sg, i] (CodeStream &os, Substitutions &varSubs)
                    {
                        os << "group.inSynInSyn" << i << "[" << varSubs["id"] << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                    });

                // If dendritic delays are required
                if(sg->isDendriticDelayRequired()) {
                    backend.genVariableInit(os, "group.numNeurons", "id", popSubs,
                        [&backend, &model, sg, i](CodeStream &os, Substitutions &varSubs)
                        {
                            os << "for (unsigned int d = 0; d < " << sg->getMaxDendriticDelayTimesteps() << "; d++)";
                            {
                                CodeStream::Scope b(os);
                                const std::string denDelayIndex = "(d * group.numNeurons) + " + varSubs["id"];
                                os << "group.denDelayInSyn" << i << "[" << denDelayIndex << "] = " << model.scalarExpr(0.0) << ";" << std::endl;
                            }
                        });
                }

                // If postsynaptic model variables should be individual
                if(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    genInitNeuronVarCode(os, backend, popSubs, sg->getPSModel()->getVars(), 
                                         "InSyn" + std::to_string(i), "numNeurons",
                                         i, model.getPrecision(),
                                         [sg](size_t i){ return sg->getPSVarInitialisers().at(i); });
                }
            }

            // Loop through incoming synaptic populations with postsynaptic update code
            const auto inSynWithPostCode = ng.getArchetype().getInSynWithPostCode();
            for(size_t i = 0; i < inSynWithPostCode.size(); i++) {
                const auto *sg = inSynWithPostCode[i];
                genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPostVars(),
                                     "WUPost" + std::to_string(i), "numNeurons", sg->getTrgNeuronGroup()->getNumDelaySlots(),
                                     i, model.getPrecision(),
                                     [&sg](size_t i){ return sg->getWUPostVarInitialisers().at(i); },
                                     [&sg](size_t){ return (sg->getBackPropDelaySteps() != NO_DELAY); });
            }

            // Loop through outgoing synaptic populations with presynaptic update code
            const auto outSynWithPreCode = ng.getArchetype().getOutSynWithPreCode();
            for(size_t i = 0; i < outSynWithPreCode.size(); i++) {
                const auto *sg = outSynWithPreCode[i];
                // **NOTE** number of delay slots is based on the source neuron (for simplicity) but whether delay is required is based on the synapse group
                genInitNeuronVarCode(os, backend, popSubs, sg->getWUModel()->getPreVars(),
                                     "WUPre" + std::to_string(i), "numNeurons", sg->getSrcNeuronGroup()->getNumDelaySlots(),
                                     i, model.getPrecision(),
                                     [&sg](size_t i){ return sg->getWUPreVarInitialisers().at(i); },
                                     [&sg](size_t){ return (sg->getDelaySteps() != NO_DELAY); });
            }

            // Loop through current sources
            os << "// current source variables" << std::endl;
            for(size_t i = 0; i < ng.getArchetype().getCurrentSources().size(); i++) {
                const auto *cs = ng.getArchetype().getCurrentSources()[i];

                genInitNeuronVarCode(os, backend, popSubs, cs->getCurrentSourceModel()->getVars(), 
                                     "CS" + std::to_string(i), "numNeurons",
                                     i, model.getPrecision(),
                                     [cs](size_t i){ return cs->getVarInitialisers().at(i); });
            }
        },
        // Dense syanptic matrix variable initialisation
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
        {
            // Loop through rows
            os << "for(unsigned int i = 0; i < group.numSrcNeurons; i++)";
            {
                CodeStream::Scope b(os);
                popSubs.addVarSubstitution("id_pre", "i");
                genInitWUVarCode(os, backend, popSubs, sg, model.getPrecision());

            }
        },
        // Sparse synaptic matrix connectivity initialisation
        [&model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
        {
            const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();

            // Add substitutions
            popSubs.addFuncSubstitution("endRow", 0, "break");
            popSubs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams());
            popSubs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams());
            popSubs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "group.");

            // Initialise row building state variables and loop on generated code to initialise sparse connectivity
            os << "// Build sparse connectivity" << std::endl;
            for(const auto &a : connectInit.getSnippet()->getRowBuildStateVars()) {
                // Apply substitutions to value
                std::string value = a.value;
                popSubs.applyCheckUnreplaced(value, "initSparseConnectivity row build state var : merged" + std::to_string(sg.getIndex()));

                os << a.type << " " << a.name << " = " << value << ";" << std::endl;
            }
            os << "while(true)";
            {
                CodeStream::Scope b(os);

                // Apply substitutions to row build code
                std::string code = connectInit.getSnippet()->getRowBuildCode();
                popSubs.addVarNameSubstitution(connectInit.getSnippet()->getRowBuildStateVars());
                popSubs.applyCheckUnreplaced(code, "initSparseConnectivity : merged" + std::to_string(sg.getIndex()));
                code = ensureFtype(code, model.getPrecision());

                // Write out code
                os << code << std::endl;
            }
        },
        // Sparse synaptic matrix var initialisation
        [&backend, &model](CodeStream &os, const SynapseGroupMerged &sg, Substitutions &popSubs)
        {
            genInitWUVarCode(os, backend, popSubs, sg, model.getPrecision());
        },
        // Initialise push EGP handler
        [&backend, &mergedEGPs](CodeStream &os)
        {
            genScalarEGPPush(os, mergedEGPs, "NeuronInit", backend);
            genScalarEGPPush(os, mergedEGPs, "SynapseDenseInit", backend);
            genScalarEGPPush(os, mergedEGPs, "SynapseConnectivityInit", backend);
        },
        // Initialise sparse push EGP handler
        [&backend, &mergedEGPs](CodeStream &os)
        {
            genScalarEGPPush(os, mergedEGPs, "SynapseSparseInit", backend);
        });
}
