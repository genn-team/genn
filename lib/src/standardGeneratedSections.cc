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
    const std::string &localID,
    const std::string &ttype)
{
    for(const auto &v : nmVars.container) {
        os << v.second << " l" << v.first << " = ";
        os << devPrefix << v.first << ng.getName() << "[";
        if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
            os << "readDelayOffset + ";
        }
        os << localID << "];" << std::endl;
    }
    
    // Also read spike time into local variable
    if(ng.isSpikeTimeRequired()) {
        os << ttype << " lsT = " << devPrefix << "sT" << ng.getName() << "[";
        if (ng.isDelayRequired()) {
            os << "readDelayOffset + ";
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
        os << devPrefix << v.first << ng.getName() << "[";
        if (ng.isVarQueueRequired(v.first) && ng.isDelayRequired()) {
             os << "writeDelayOffset + ";
        }
        os << localID <<  "] = l" << v.first << ";" << std::endl;
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronSpikeEventTest(
    CodeStream &os,
    const NeuronGroup &ng,
    const VarNameIterCtx &nmVars,
    const ExtraGlobalParamNameIterCtx &nmExtraGlobalParams,
    const std::string &,
    const std::vector<FunctionTemplate> &functions,
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
        {
            CodeStream::Scope b(os);

            // Use synapse population support code namespace if required
            if (!spkEventCond.second.empty()) {
                os << " using namespace " << spkEventCond.second << ";" << std::endl;
            }

            // Combine this event threshold test with
            os << "spikeLikeEvent |= (" << eCode << ");" << std::endl;
        }
    }
}
//----------------------------------------------------------------------------
void StandardGeneratedSections::neuronCurrentInjection(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix,
    const std::string &localID,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype,
    const std::string &rng)
{
    // Loop through all of neuron group's current sources
    for (const auto *cs : ng.getCurrentSources())
    {
        os << "// current source " << cs->getName() << std::endl;
        CodeStream::Scope b(os);

        const auto* csm = cs->getCurrentSourceModel();
        VarNameIterCtx csVars(csm->getVars());
        DerivedParamNameIterCtx csDerivedParams(csm->getDerivedParams());
        ExtraGlobalParamNameIterCtx csExtraGlobalParams(csm->getExtraGlobalParams());

        // Read current source variables into registers
        for(const auto &v : csVars.container) {
            os <<  v.second << " l" << v.first << " = " << devPrefix << v.first << cs->getName() << "[" << localID << "];" << std::endl;
        }

        string iCode = csm->getInjectionCode();
        substitute(iCode, "$(id)", localID);
        StandardSubstitutions::currentSourceInjection(iCode, cs,
                            csVars, csDerivedParams, csExtraGlobalParams,
                            functions, ftype, rng);
        os << iCode << std::endl;

        // Write updated variables back to global memory
        for(const auto &v : csVars.container) {
             os << devPrefix << v.first << cs->getName() << "[" << localID << "] = l" << v.first << ";" << std::endl;
        }
    }
}

void StandardGeneratedSections::weightUpdatePreSpike(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix,
    const std::string &localID,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype)
{
    // Loop through outgoing synaptic populations
    for(const auto *sg : ng.getOutSyn()) {
        // If weight update model has any presynaptic update code
        if(!sg->getWUModel()->getPreSpikeCode().empty()) {
            CodeStream::Scope b(os);
            os << "// perform presynaptic update required for " << sg->getName() << std::endl;

            // Fetch presynaptic variables from global memory
            for(const auto &v : sg->getWUModel()->getPreVars()) {
                os << v.second << " l" << v.first << " = ";
                os << devPrefix << v.first << sg->getName() << "[";
                if (sg->getDelaySteps() != NO_DELAY) {
                    os << "readDelayOffset + ";
                }
                os << localID << "];" << std::endl;
            }

            // Perform standard substitutions
            string pCode = sg->getWUModel()->getPreSpikeCode();
            StandardSubstitutions::weightUpdatePreSpike(pCode, sg, localID, devPrefix, functions, ftype);
            os << pCode;
            
            // Write back presynaptic variables into global memory
            for(const auto &v : sg->getWUModel()->getPreVars()) {
                os << devPrefix << v.first << sg->getName() << "[";
                if (sg->getDelaySteps() != NO_DELAY) {
                    os << "writeDelayOffset + ";
                }
                os << localID <<  "] = l" << v.first << ";" << std::endl;
            }
        }
    }
}

void StandardGeneratedSections::weightUpdatePostSpike(
    CodeStream &os,
    const NeuronGroup &ng,
    const std::string &devPrefix,
    const std::string &localID,
    const std::vector<FunctionTemplate> &functions,
    const std::string &ftype)
{
    // Loop through incoming synaptic populations
    for(const auto *sg : ng.getInSyn()) {
        // If weight update model has any postsynaptic update code
        if(!sg->getWUModel()->getPostSpikeCode().empty()) {
            CodeStream::Scope b(os);
            os << "// perform postsynaptic update required for " << sg->getName() << std::endl;

            // Fetch postsynaptic variables from global memory
            for(const auto &v : sg->getWUModel()->getPostVars()) {
                os << v.second << " l" << v.first << " = ";
                os << devPrefix << v.first << sg->getName() << "[";
                if (sg->getBackPropDelaySteps() != NO_DELAY) {
                    os << "readDelayOffset + ";
                }
                os << localID << "];" << std::endl;
            }

            // Perform standard substitutions
            string pCode = sg->getWUModel()->getPostSpikeCode();
            StandardSubstitutions::weightUpdatePostSpike(pCode, sg, localID, devPrefix, functions, ftype);
            os << pCode;

            // Write back presynaptic variables into global memory
            for(const auto &v : sg->getWUModel()->getPostVars()) {
                os << devPrefix << v.first << sg->getName() << "[";
                if (sg->getBackPropDelaySteps() != NO_DELAY) {
                    os << "writeDelayOffset + ";
                }
                os << localID <<  "] = l" << v.first << ";" << std::endl;
            }
        }
    }
}
