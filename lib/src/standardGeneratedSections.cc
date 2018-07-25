#include "standardGeneratedSections.h"

// GeNN includes
#include "codeStream.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardGeneratedSections
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronOutputInit(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix)
{
    if (ng.isDelayRequired()) { // with delay
        // **NOTE** only device spike queue pointers should be reset here
        if(!devPrefix.empty()) {
            os << devPrefix << "spkQuePtr" << ng.getName() << " = (" << devPrefix << "spkQuePtr" << ng.getName() << " + 1) % " << ng.getNumDelaySlots() << ";" << std::endl;
        }

        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << std::endl;
        }
        if (ng.isTrueSpikeRequired()) {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << std::endl;
        }
        else {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
        }
    }
    else { // no delay
        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[0] = 0;" << std::endl;
        }
        os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << std::endl;
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronLocalVarInit(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID)
{
    for(const auto &v : nmVars.container) {
        os << v.second << " l" << v.first << " = ";
        os << devPrefix << v.first << ng.getName() << "[";
        if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
            os << "(delaySlot * " << ng.getNumNeurons() << ") + ";
        }
        os << localID << "];" << std::endl;
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronLocalVarWrite(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID)
{
    // store the defined parts of the neuron state into the global state variables dd_V etc
   for(const auto &v : nmVars.container) {
        if (ng.isVarQueueRequired(v.first)) {
            os << devPrefix << v.first << ng.getName() << "[" << ng.getQueueOffset(devPrefix) << localID << "] = l" << v.first << ";" << std::endl;
        }
        else {
            os << devPrefix << v.first << ng.getName() << "[" << localID << "] = l" << v.first << ";" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronSpikeEventTest(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Create local variable
    os << "bool spikeLikeEvent = false;" << std::endl;

    // Loop through outgoing synapse populations that will contribute to event condition code
    for(const auto &spkEventCond : ng.getSpikeEventCondition()) {
        // Replace of parameters, derived parameters and extraglobalsynapse parameters
        string eCode = spkEventCond.first;

        // code substitutions ----
        substitute(eCode, "$(id)", "n");
        StandardSubstitutions::neuronSpikeEventCondition(eCode, ng, nmVars, nmExtraGlobalParams,
                                                         functions, ftype, rng);

        // Open scope for spike-like event test
        os << CodeStream::OB(31);

        // Use synapse population support code namespace if required
        if (!spkEventCond.second.empty()) {
            os << " using namespace " << spkEventCond.second << ";" << std::endl;
        }

        // Combine this event threshold test with
        os << "spikeLikeEvent |= (" << eCode << ");" << std::endl;

        // Close scope for spike-like event test
        os << CodeStream::CB(31);
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronCurrentInjection(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &localID,
    const std::vector<FunctionTemplate> functions,
    const std::string &ftype,
    const std::string &rng)
{
    if (ng.isInjected()) {
        os << "// add injected current from sources" << std::endl;
        const auto css = ng.getCurrentSources();
        for (const auto *cs : css)
        {
            const auto* csm = cs->getCurrentSourceModel();
            VarNameIterCtx csVars(csm->getVars());
            DerivedParamNameIterCtx csDerivedParams(csm->getDerivedParams());
            ExtraGlobalParamNameIterCtx csExtraGlobalParams(csm->getExtraGlobalParams());
            os << "// current source " << cs->getName() << std::endl;
            if (!csm->getInjectionCode().empty()){
                string iCode = csm->getInjectionCode();
                substitute(iCode, "$(id)", localID);
                StandardSubstitutions::currentSourceInjection(iCode, cs,
                                    csVars, csDerivedParams, csExtraGlobalParams,
                                    functions, ftype, rng,
                                    ng.getName());
                os << iCode << std::endl;
            }
        }
    }
}
void StandardGeneratedSections::neuronCurrentInjectionLocalVarDeclare(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix,
    const std::string &localID)
{
    if (ng.isInjected()) {
        os << "// pull injected current variables in a coalesced access" << std::endl;
        const auto css = ng.getCurrentSources();
        for (const auto *cs : css)
        {
            const auto* csm = cs->getCurrentSourceModel();
            VarNameIterCtx csVars(csm->getVars());
            // store the defined parts of the neuron state into the global state variables dd_V etc
            for(const auto &v : csVars.container) {
                os <<  v.second << " l" << v.first << cs->getName() << " = ";
                os << devPrefix << v.first << cs->getName() << ng.getName() << "[" << localID << "];" << std::endl;
            }
        }
    }
}
