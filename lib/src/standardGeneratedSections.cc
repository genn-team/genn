#include "standardGeneratedSections.h"

// GeNN includes
#include "CodeHelper.h"
#include "modelSpec.h"

//----------------------------------------------------------------------------
// StandardGeneratedSections
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronOutputInit(
    std::ostream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix)
{
    if (ng.isDelayRequired()) { // with delay
        os << devPrefix << "spkQuePtr" << ng.getName() << " = (" << devPrefix << "spkQuePtr" << ng.getName() << " + 1) % " << ng.getNumDelaySlots() << ";" << ENDL;
        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << ENDL;
        }
        if (ng.isTrueSpikeRequired()) {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[" << devPrefix << "spkQuePtr" << ng.getName() << "] = 0;" << ENDL;
        }
        else {
            os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << ENDL;
        }
    }
    else { // no delay
        if (ng.isSpikeEventRequired()) {
            os << devPrefix << "glbSpkCntEvnt" << ng.getName() << "[0] = 0;" << ENDL;
        }
        os << devPrefix << "glbSpkCnt" << ng.getName() << "[0] = 0;" << ENDL;
    }
}

void StandardGeneratedSections::neuronLocalVarInit(
    std::ostream &os,
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
        os << localID << "];" << ENDL;
    }
}

void StandardGeneratedSections::neuronLocalVarWrite(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const std::string &devPrefix,
    const std::string &localID)
{
    // store the defined parts of the neuron state into the global state variables dd_V etc
   for(const auto &v : nmVars.container) {
        if (ng.isVarQueueRequired(v.first)) {
            os << devPrefix << v.first << ng.getName() << "[" << ng.getQueueOffset(devPrefix) << localID << "] = l" << v.first << ";" << ENDL;
        }
        else {
            os << devPrefix << v.first << ng.getName() << "[" << localID << "] = l" << v.first << ";" << ENDL;
        }
    }
}

void StandardGeneratedSections::neuronSpikeEventTest(
    std::ostream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &,
    const std::string &ftype)
{
    // Create local variable
    os << "bool spikeLikeEvent = false;" << ENDL;

    // Loop through outgoing synapse populations that will contribute to event condition code
    for(const auto &spkEventCond : ng.getSpikeEventCondition()) {
        // Replace of parameters, derived parameters and extraglobalsynapse parameters
        string eCode = spkEventCond.first;

        // code substitutions ----
        substitute(eCode, "$(id)", "n");
        StandardSubstitutions::neuronSpikeEventCondition(eCode, ng, nmVars, nmExtraGlobalParams, ftype);

        // Open scope for spike-like event test
        os << OB(31);

        // Use synapse population support code namespace if required
        if (!spkEventCond.second.empty()) {
            os << " using namespace " << spkEventCond.second << ";" << ENDL;
        }

        // Combine this event threshold test with
        os << "spikeLikeEvent |= (" << eCode << ");" << ENDL;

        // Close scope for spike-like event test
        os << CB(31);
    }
}