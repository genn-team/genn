/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateCPU.cc 

  \brief Functions for generating code that will run the neuron and synapse simulations on the CPU. Part of the code generation section.

*/
//--------------------------------------------------------------------------

#include "generateCPU.h"
#include "global.h"
#include "utils.h"
#include "codeGenUtils.h"
#include "standardGeneratedSections.h"
#include "standardSubstitutions.h"
#include "codeStream.h"

#include <algorithm>
#include <typeinfo>

//-------------------------------------------------------------------------
// Anonymous namespace
//-------------------------------------------------------------------------
namespace
{
//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic
  spikes or spike type events
*/
//-------------------------------------------------------------------------
void generate_process_presynaptic_events_code_CPU(
    CodeStream &os, //!< output stream for code
    const string &sgName,
    const SynapseGroup &sg,
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    bool evnt = postfix == "Evnt";

    if ((evnt && sg.isSpikeEventRequired()) || (!evnt && sg.isTrueSpikeRequired())) {
        const auto *wu = sg.getWUModel();

        // Detect spike events or spikes and do the update
        os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << std::endl;
        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "for (unsigned int i = 0; i < glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[delaySlot]; i++)";
        }
        else {
            os << "for (unsigned int i = 0; i < glbSpkCnt" << postfix << sg.getSrcNeuronGroup()->getName() << "[0]; i++)";
        }
        {
            CodeStream::Scope b(os);

            os << "const unsigned int ipre = glbSpk" << postfix << sg.getSrcNeuronGroup()->getName() << "[" << sg.getOffsetPre() << "i];" << std::endl;

            if (sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                if(sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                    os << "const unsigned int npost = C" << sgName << ".indInG[ipre + 1] - C" << sgName << ".indInG[ipre];" << std::endl;
                }
                else {
                    os << "const unsigned int npost = C" << sgName << ".rowLength[ipre];" << std::endl;
                }
                os << "for (unsigned int j = 0; j < npost; j++)";
            }
            // Otherwise (DENSE or BITMASK)
            else {
                os << "for (unsigned int ipost = 0; ipost < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; ipost++)";
            }
            {
                CodeStream::Scope b(os);
                if(sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                    os << "const unsigned int ipost = C" << sgName << ".ind[C" << sgName << ".indInG[ipre] + j];" << std::endl;
                }
                else if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                    // **TODO** seperate stride from max connections
                    os << "const unsigned int ipost = C" << sgName << ".ind[(ipre * " << sg.getMaxConnections() << ") + j];" << std::endl;
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "const unsigned int gid = (ipre * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + ipost);" << std::endl;
                }

                if (!wu->getSimSupportCode().empty()) {
                    os << " using namespace " << sgName << "_weightupdate_simCode;" << std::endl;
                }

                // Create iteration context to iterate over the variables; derived and extra global parameters
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                VarNameIterCtx wuVars(wu->getVars());

                if (evnt) {
                    os << "if ";
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        os << "((B(gp" << sgName << "[gid / 32], gid & 31)) && ";
                    }

                    // code substitutions ----
                    string eCode = wu->getEventThresholdConditionCode();
                    substitute(eCode, "$(id)", "n");
                    substitute(eCode, "$(t)", "t");
                    StandardSubstitutions::weightUpdateThresholdCondition(eCode, sg,
                                                                        wuDerivedParams, wuExtraGlobalParams,
                                                                        "ipre", "ipost", "",
                                                                        cpuFunctions, ftype);

                    // end code substitutions ----
                    os << "(" << eCode << ")";

                    if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                        os << ")";
                    }
                    os << CodeStream::OB(2041);
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << "if (B(gp" << sgName << "[gid / 32], gid & 31))" << CodeStream::OB(2041);
                }



                // Code substitutions ----------------------------------------------------------------------------------
                string wCode = evnt ? wu->getEventCode() : wu->getSimCode();

                if(sg.isDendriticDelayRequired()) {
                    functionSubstitute(wCode, "addToInSynDelay", 2, "denDelay" + sgName + "[" + sg.getDendriticDelayOffset("", "$(1)") + "ipost] += $(0)");
                }
                else {
                    os << ftype << " addtoinSyn;" << std::endl;
                    substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                    substitute(wCode, "$(inSyn)", "inSyn" + sgName + "[ipost]");
                }

                substitute(wCode, "$(t)", "t");
                if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    if (sg.getMatrixType() & SynapseMatrixConnectivity::YALE) {
                        name_substitutions(wCode, "", wuVars.nameBegin, wuVars.nameEnd,
                                           sgName + "[C" + sgName + ".indInG[ipre] + j]");
                    }
                    else if(sg.getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                        // **TODO** seperate stride from max connections
                        name_substitutions(wCode, "", wuVars.nameBegin, wuVars.nameEnd,
                                           sgName + "[(ipre * " + to_string(sg.getMaxConnections()) + ") + j]");
                    }
                    else {
                        name_substitutions(wCode, "", wuVars.nameBegin, wuVars.nameEnd,
                                           sgName + "[ipre * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + " + ipost]");
                    }
                }


                StandardSubstitutions::weightUpdateSim(wCode, sg,
                                                    wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                    "ipre", "ipost", "", cpuFunctions, ftype);
                // end Code substitutions -------------------------------------------------------------------------
                os << wCode << std::endl;

                if (evnt) {
                    os << CodeStream::CB(2041); // end if (eCode)
                }
                else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                    os << CodeStream::CB(2041); // end if (B(gp" << sgName << "[gid / 32], gid
                }
            }
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
/*!
  \brief Function that generates the code of the function the will simulate all neurons on the CPU.
*/
//--------------------------------------------------------------------------

void genNeuronFunction(const NNmodel &model, //!< Model description
                       const string &path) //!< Path for code generation
{
    // Open a file output stream for writing synapse function
    ofstream fs;
    string name = model.getGeneratedCodePath(path, "neuronFnct.cc");
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // write header content
    writeHeader(os);
    os << std::endl;

    // compiler/include control (include once)
    os << "#ifndef _" << model.getName() << "_neuronFnct_cc" << std::endl;
    os << "#define _" << model.getName() << "_neuronFnct_cc" << std::endl;
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file neuronFnct.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName();
    os << " containing the the equivalent of neuron kernel function for the CPU-only version." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    os << "// include the support codes provided by the user for neuron or synaptic models" << std::endl;
    os << "#include \"support_code.h\"" << std::endl << std::endl;

    // function header
    os << "void calcNeuronsCPU(" << model.getPrecision() << " t)";
    {
        CodeStream::Scope b(os);

        // function code
        for(const auto &n : model.getLocalNeuronGroups()) {
            os << "// neuron group " << n.first << std::endl;
            {
                CodeStream::Scope b(os);

                // increment spike queue pointer and reset spike count
                StandardGeneratedSections::neuronOutputInit(os, n.second, "");

                if (n.second.isVarQueueRequired() && n.second.isDelayRequired()) {
                    os << "unsigned int delaySlot = (spkQuePtr" << n.first;
                    os << " + " << (n.second.getNumDelaySlots() - 1);
                    os << ") % " << n.second.getNumDelaySlots() << ";" << std::endl;
                }
                os << std::endl;

                os << "for (int n = 0; n < " <<  n.second.getNumNeurons() << "; n++)";
                {
                    CodeStream::Scope b(os);

                    // Get neuron model associated with this group
                    auto nm = n.second.getNeuronModel();

                    // Create iteration context to iterate over the variables; derived and extra global parameters
                    VarNameIterCtx nmVars(nm->getVars());
                    DerivedParamNameIterCtx nmDerivedParams(nm->getDerivedParams());
                    ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->getExtraGlobalParams());

                    // Generate code to copy neuron state into local variable
                    StandardGeneratedSections::neuronLocalVarInit(os, n.second, nmVars, "", "n");

                    if ((nm->getSimCode().find("$(sT)") != string::npos)
                        || (nm->getThresholdConditionCode().find("$(sT)") != string::npos)
                        || (nm->getResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
                        os << model.getPrecision() << " lsT= sT" <<  n.first << "[";
                        if (n.second.isDelayRequired()) {
                            os << "(delaySlot * " << n.second.getNumNeurons() << ") + ";
                        }
                        os << "n];" << std::endl;
                    }
                    os << std::endl;

                    if (n.second.getInSyn().size() > 0 || (nm->getSimCode().find("Isyn") != string::npos)) {
                        os << model.getPrecision() << " Isyn = 0;" << std::endl;
                    }

                    // Initialise any additional input variables supported by neuron model
                    for(const auto &a : nm->getAdditionalInputVars()) {
                        os << a.second.first << " " << a.first << " = " << a.second.second << ";" << std::endl;
                    }

                    for(const auto *sg : n.second.getInSyn()) {
                        const auto *psm = sg->getPSModel();

                        // If dendritic delay is required
                        if(sg->isDendriticDelayRequired()) {
                            // Get reference to dendritic delay buffer input for this timestep
                            os << model.getPrecision() << " &denDelayFront" << sg->getName() << " = denDelay" + sg->getName() + "[" + sg->getDendriticDelayOffset("") + "n];" << std::endl;

                            // Add delayed input from buffer into inSyn
                            os << "inSyn" + sg->getName() + "[n] += denDelayFront" << sg->getName() << ";" << std::endl;

                            // Zero delay buffer slot
                            os << "denDelayFront" << sg->getName() << " = " << model.scalarExpr(0.0) << ";" << std::endl;
                        }

                        if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                            for(const auto &v : psm->getVars()) {
                                os << v.second << " lps" << v.first << sg->getName();
                                os << " = " <<  v.first << sg->getName() << "[n];" << std::endl;
                            }
                        }

                        // Apply substitutions to current converter code
                        string psCode = psm->getApplyInputCode();
                        substitute(psCode, "$(id)", "n");
                        substitute(psCode, "$(inSyn)", "inSyn" + sg->getName() + "[n]");
                        StandardSubstitutions::postSynapseApplyInput(psCode, sg, n.second,
                            nmVars, nmDerivedParams, nmExtraGlobalParams, cpuFunctions, model.getPrecision(), "rng");

                        if (!psm->getSupportCode().empty()) {
                            os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                        }
                        os << psCode << std::endl;
                        if (!psm->getSupportCode().empty()) {
                            os << CodeStream::CB(29) << " // namespace bracket closed" << std::endl;
                        }
                    }

                    if (!nm->getSupportCode().empty()) {
                        os << " using namespace " << n.first << "_neuron;" << std::endl;
                    }

                    string thCode = nm->getThresholdConditionCode();
                    if (thCode.empty()) { // no condition provided
                        cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << n.first << "\" was provided. There will be no spikes detected in this population!" << endl;
                    }
                    else {
                        os << "// test whether spike condition was fulfilled previously" << std::endl;
                        substitute(thCode, "$(id)", "n");
                        StandardSubstitutions::neuronThresholdCondition(thCode, n.second,
                                                                        nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                                        cpuFunctions, model.getPrecision(), "rng");
                        if (GENN_PREFERENCES::autoRefractory) {
                            os << "bool oldSpike= (" << thCode << ");" << std::endl;
                        }
                    }

                    os << "// calculate membrane potential" << std::endl;
                    string sCode = nm->getSimCode();
                    substitute(sCode, "$(id)", "n");
                    StandardSubstitutions::neuronSim(sCode, n.second,
                                                    nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                    cpuFunctions, model.getPrecision(), "rng");
                    if (nm->isPoisson()) {
                        substitute(sCode, "lrate", "rates" + n.first + "[n + offset" + n.first + "]");
                    }
                    os << sCode << std::endl;

                    string queueOffset = n.second.getQueueOffset("");

                    // look for spike type events first.
                    if (n.second.isSpikeEventRequired()) {
                        // Generate spike event test
                        StandardGeneratedSections::neuronSpikeEventTest(os, n.second,
                                                                        nmVars, nmExtraGlobalParams, "n",
                                                                        cpuFunctions, model.getPrecision(), "rng");

                        os << "// register a spike-like event" << std::endl;
                        os << "if (spikeLikeEvent)";
                        {
                            CodeStream::Scope b(os);
                            os << "glbSpkEvnt" << n.first << "[" << queueOffset << "glbSpkCntEvnt" << n.first;
                            if (n.second.isDelayRequired()) { // WITH DELAY
                                os << "[spkQuePtr" << n.first << "]++] = n;" << std::endl;
                            }
                            else { // NO DELAY
                                os << "[0]++] = n;" << std::endl;
                            }
                        }
                    }

                    // test for true spikes if condition is provided
                    if (!thCode.empty()) {
                        os << "// test for and register a true spike" << std::endl;
                        if (GENN_PREFERENCES::autoRefractory) {
                            os << "if ((" << thCode << ") && !(oldSpike))";
                        }
                        else {
                            os << "if (" << thCode << ")";
                        }
                        {
                            CodeStream::Scope b(os);

                            string queueOffsetTrueSpk = n.second.isTrueSpikeRequired() ? queueOffset : "";
                            os << "glbSpk" << n.first << "[" << queueOffsetTrueSpk << "glbSpkCnt" << n.first;
                            if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) { // WITH DELAY
                                os << "[spkQuePtr" << n.first << "]++] = n;" << std::endl;
                            }
                            else { // NO DELAY
                                os << "[0]++] = n;" << std::endl;
                            }
                            if (n.second.isSpikeTimeRequired()) {
                                os << "sT" << n.first << "[" << queueOffset << "n] = t;" << std::endl;
                            }

                            // add after-spike reset if provided
                            if (!nm->getResetCode().empty()) {
                                string rCode = nm->getResetCode();
                                substitute(rCode, "$(id)", "n");
                                StandardSubstitutions::neuronReset(rCode, n.second,
                                                                nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                                cpuFunctions, model.getPrecision(), "rng");
                                os << "// spike reset code" << std::endl;
                                os << rCode << std::endl;
                            }
                        }
                    }

                    // store the defined parts of the neuron state into the global state variables V etc
                    StandardGeneratedSections::neuronLocalVarWrite(os, n.second, nmVars, "", "n");

                    for(const auto *sg : n.second.getInSyn()) {
                        const auto *psm = sg->getPSModel();

                        string pdCode = psm->getDecayCode();
                        substitute(pdCode, "$(id)", "n");
                        substitute(pdCode, "$(inSyn)", "inSyn" + sg->getName() + "[n]");
                        StandardSubstitutions::postSynapseDecay(pdCode, sg, n.second,
                                                                nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                                cpuFunctions, model.getPrecision(), "rng");
                        os << "// the post-synaptic dynamics" << std::endl;
                        if (!psm->getSupportCode().empty()) {
                            os << CodeStream::OB(29) << " using namespace " << sg->getName() << "_postsyn;" << std::endl;
                        }
                        os << pdCode << std::endl;
                        if (!psm->getSupportCode().empty()) {
                            os << CodeStream::CB(29) << " // namespace bracket closed" << endl;
                        }
                        for (const auto &v : psm->getVars()) {
                            os << v.first << sg->getName() << "[n]" << " = lps" << v.first << sg->getName() << ";" << std::endl;
                        }
                    }
                }
            }
            os << std::endl;
        }
    }
    os << "#endif" << std::endl;
    fs.close();
} 

//--------------------------------------------------------------------------
/*!
  \brief Function that generates code that will simulate all synapses of the model on the CPU.
*/
//--------------------------------------------------------------------------

void genSynapseFunction(const NNmodel &model, //!< Model description
                        const string &path) //!< Path for code generation
{
    // Open a file output stream for writing synapse function
    ofstream fs;
    string name = model.getGeneratedCodePath(path, "synapseFnct.cc");
    fs.open(name.c_str());

    // Attach this to a code stream
    CodeStream os(fs);

    // write header content
    writeHeader(os);
    os << std::endl;

    // compiler/include control (include once)
    os << "#ifndef _" << model.getName() << "_synapseFnct_cc" << std::endl;
    os << "#define _" << model.getName() << "_synapseFnct_cc" << std::endl;
    os << std::endl;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << std::endl;
    os << "/*! \\file synapseFnct.cc" << std::endl << std::endl;
    os << "\\brief File generated from GeNN for the model " << model.getName() << " containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version." << std::endl;
    os << "*/" << std::endl;
    os << "//-------------------------------------------------------------------------" << std::endl << std::endl;

    if (!model.getSynapseDynamicsGroups().empty()) {
        // synapse dynamics function
        os << "void calcSynapseDynamicsCPU(" << model.getPrecision() << " t)";
        {
            CodeStream::Scope b(os);
            os << model.getPrecision() << " addtoinSyn;" << std::endl;
            os << std::endl;
            os << "// execute internal synapse dynamics if any" << std::endl;

            for(const auto &s : model.getSynapseDynamicsGroups())
            {
                const SynapseGroup *sg = model.findSynapseGroup(s.first);
                const auto *wu = sg->getWUModel();

                // there is some internal synapse dynamics
                if (!wu->getSynapseDynamicsCode().empty()) {

                    os << "// synapse group " << s.first << std::endl;
                    {
                        CodeStream::Scope b(os);

                        if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                            os << "unsigned int delaySlot = (spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                            os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                            os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                        }

                        if (!wu->getSynapseDynamicsSuppportCode().empty()) {
                            os << "using namespace " << s.first << "_weightupdate_synapseDynamics;" << std::endl;
                        }

                        // Create iteration context to iterate over the variables and derived parameters
                        DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                        ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                        VarNameIterCtx wuVars(wu->getVars());

                        string SDcode= wu->getSynapseDynamicsCode();
                        substitute(SDcode, "$(t)", "t");

                        if (sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                            os << "for (int n= 0; n < C" << s.first << ".connN; n++)";
                            {
                                CodeStream::Scope b(os);
                                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                    // name substitute synapse var names in synapseDynamics code
                                    name_substitutions(SDcode, "", wuVars.nameBegin, wuVars.nameEnd, s.first + "[n]");
                                }

                                const std::string postIdx = "C" + s.first + ".ind[n]";
                                if(sg->isDendriticDelayRequired()) {
                                    functionSubstitute(SDcode, "addtoinSynDelay", 2, "denDelay" + s.first + "[" + sg->getDendriticDelayOffset("", "$(1)") + postIdx + "] += $(0)");
                                }
                                else {
                                    substitute(SDcode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                                    substitute(SDcode, "$(inSyn)", "inSyn" + s.first + "[" + postIdx + "]");
                                }

                                StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                            "C" + s.first + ".preInd[n]", postIdx, "",
                                                                            cpuFunctions, model.getPrecision());
                                os << SDcode << std::endl;
                            }
                        }
                        else if(sg->getMatrixType() & SynapseMatrixConnectivity::RAGGED) {
                            os << "for (int i = 0; i < " <<  sg->getSrcNeuronGroup()->getNumNeurons() << "; i++)";
                            {
                                CodeStream::Scope b(os);
                                os << "for (int j = 0; j < C" << s.first << ".rowLength[i]; j++)";
                                {
                                    CodeStream::Scope b(os);

                                    // Calculate index of synapse in arrays
                                    os << "const int n = (i * " + std::to_string(sg->getMaxConnections()) + ") + j;" << std::endl;

                                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                        // name substitute synapse var names in synapseDynamics code
                                        // **TODO** seperate stride from max connections
                                        name_substitutions(SDcode, "", wuVars.nameBegin, wuVars.nameEnd, s.first + "[n]");
                                    }

                                    const std::string postIdx = "C" + s.first + ".ind[n]";
                                    if(sg->isDendriticDelayRequired()) {
                                        functionSubstitute(SDcode, "addtoinSynDelay", 2, "denDelay" + s.first + "[" + sg->getDendriticDelayOffset("", "$(1)") + postIdx + "] += $(0)");
                                    }
                                    else {
                                        substitute(SDcode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                                        substitute(SDcode, "$(inSyn)", "inSyn" + s.first + "[" + postIdx + "]");
                                    }

                                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                                "i", postIdx, "", cpuFunctions, model.getPrecision());
                                    os << SDcode << std::endl;
                                }
                            }
                        }
                        else {
                            os << "for (int i = 0; i < " <<  sg->getSrcNeuronGroup()->getNumNeurons() << "; i++)";
                            {
                                CodeStream::Scope b(os);
                                os << "for (int j = 0; j < " <<  sg->getTrgNeuronGroup()->getNumNeurons() << "; j++)";
                                {
                                    CodeStream::Scope b(os);
                                    os << "// loop through all synapses" << endl;
                                    // substitute initial values as constants for synapse var names in synapseDynamics code
                                    if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                                        name_substitutions(SDcode, "", wuVars.nameBegin, wuVars.nameEnd,
                                                        s.first + "[(i * " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + j]");
                                    }

                                    if(sg->isDendriticDelayRequired()) {
                                        functionSubstitute(SDcode, "addToInSynDelay", 2, "denDelay" + s.first + "[" + sg->getDendriticDelayOffset("", "$(1)") + "j] += $(0)");
                                    }
                                    else {
                                        substitute(SDcode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
                                        substitute(SDcode, "$(inSyn)", "inSyn" + s.first + "[j]");
                                    }

                                    StandardSubstitutions::weightUpdateDynamics(SDcode, sg, wuVars, wuDerivedParams, wuExtraGlobalParams,
                                                                                "i","j", "", cpuFunctions, model.getPrecision());
                                    os << SDcode << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // synapse function header
    os << "void calcSynapsesCPU(" << model.getPrecision() << " t)";
    {
        CodeStream::Scope b(os);
        os << std::endl;

        for(const auto &s : model.getLocalSynapseGroups()) {
            os << "// synapse group " << s.first << std::endl;
            {
                CodeStream::Scope b(os);
                if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
                    os << "const unsigned int delaySlot = (spkQuePtr" << s.second.getSrcNeuronGroup()->getName();
                    os << " + " << (s.second.getSrcNeuronGroup()->getNumDelaySlots() - s.second.getDelaySteps());
                    os << ") % " << s.second.getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                }

                // generate the code for processing spike-like events
                if (s.second.isSpikeEventRequired()) {
                    generate_process_presynaptic_events_code_CPU(os, s.first, s.second, "Evnt", model.getPrecision());
                }

                // generate the code for processing true spike events
                if (s.second.isTrueSpikeRequired()) {
                    generate_process_presynaptic_events_code_CPU(os, s.first, s.second, "", model.getPrecision());
                }
            }
            os << std::endl;
        }
    }
    os << std::endl;


    //////////////////////////////////////////////////////////////
    // function for learning synapses, post-synaptic spikes

    if (!model.getSynapsePostLearnGroups().empty()) {

        os << "void learnSynapsesPostHost(" << model.getPrecision() << " t)";
        {
            CodeStream::Scope b(os);

            os << "unsigned int ipost;" << std::endl;
            os << "unsigned int ipre;" << std::endl;
            os << "unsigned int lSpk;" << std::endl;

            // If any synapse groups have sparse connectivity
            if(any_of(begin(model.getLocalSynapseGroups()), end(model.getLocalSynapseGroups()),
                [](const NNmodel::SynapseGroupValueType &s)
                {
                    return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE);

                }))
            {
                os << "unsigned int npre;" << std::endl;
            }
            os << std::endl;

            for(const auto &s : model.getSynapsePostLearnGroups())
            {
                const SynapseGroup *sg = model.findSynapseGroup(s.first);
                const auto *wu = sg->getWUModel();
                const bool sparse = sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE;

                // Create iteration context to iterate over the variables; derived and extra global parameters
                DerivedParamNameIterCtx wuDerivedParams(wu->getDerivedParams());
                ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->getExtraGlobalParams());
                VarNameIterCtx wuVars(wu->getVars());

                // NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM

                os << "// synapse group " << s.first << std::endl;
                {
                    CodeStream::Scope b(os);

                    if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                        os << "unsigned int delaySlot = (spkQuePtr" << sg->getSrcNeuronGroup()->getName();
                        os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                        os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << std::endl;
                    }

                    if (!wu->getLearnPostSupportCode().empty()) {
                        os << "using namespace " << s.first << "_weightupdate_simLearnPost;" << std::endl;
                    }

                    if (sg->getTrgNeuronGroup()->isDelayRequired() && sg->getTrgNeuronGroup()->isTrueSpikeRequired()) {
                        os << "for (ipost = 0; ipost < glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[spkQuePtr" << sg->getTrgNeuronGroup()->getName() << "]; ipost++)";
                    }
                    else {
                        os << "for (ipost = 0; ipost < glbSpkCnt" << sg->getTrgNeuronGroup()->getName() << "[0]; ipost++)";
                    }
                    {
                        CodeStream::Scope b(os);

                        const string offsetTrueSpkPost = sg->getTrgNeuronGroup()->isTrueSpikeRequired()
                            ? sg->getTrgNeuronGroup()->getQueueOffset("")
                            : "";
                        os << "lSpk = glbSpk" << sg->getTrgNeuronGroup()->getName() << "[" << offsetTrueSpkPost << "ipost];" << std::endl;

                        if (sparse) {
                            if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                os << "npre = C" << s.first << ".revIndInG[lSpk + 1] - C" << s.first << ".revIndInG[lSpk];" << std::endl;
                            }
                            else {
                                os << "npre = C" << s.first << ".colLength[lSpk];" << std::endl;
                            }
                            os << "for (int l = 0; l < npre; l++)";
                        }
                        else {
                            os << "for (ipre = 0; ipre < " << sg->getSrcNeuronGroup()->getNumNeurons() << "; ipre++)";
                        }
                        {
                            CodeStream::Scope b(os);
                            if(sparse) {
                                if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                    os << "ipre = C" << s.first << ".revIndInG[lSpk] + l;" << std::endl;
                                }
                                else {
                                    os << "ipre = (lSpk * " << sg->getMaxSourceConnections() << ") + l;" << std::endl;
                                }
                            }

                            string code = wu->getLearnPostCode();
                            substitute(code, "$(t)", "t");
                            // Code substitutions ----------------------------------------------------------------------------------
                            std::string preIndex;
                            if (sparse) {
                                name_substitutions(code, "", wuVars.nameBegin, wuVars.nameEnd,
                                                   s.first + "[C" + s.first + ".remap[ipre]]");
                                if(sg->getMatrixType() & SynapseMatrixConnectivity::YALE) {
                                    preIndex = "C" + s.first + ".revInd[ipre]";
                                }
                                else {
                                    preIndex = "(C" + s.first + ".remap[ipre] / " + to_string(sg->getMaxConnections()) + ")";
                                }
                            }
                            else { // DENSE
                                name_substitutions(code, "", wuVars.nameBegin, wuVars.nameEnd,
                                                s.first + "[lSpk + " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + " * ipre]");
                                
                                preIndex = "ipre";
                            }
                            StandardSubstitutions::weightUpdatePostLearn(code, sg,
                                                                        wuDerivedParams, wuExtraGlobalParams,
                                                                        preIndex, "lSpk", "", cpuFunctions, model.getPrecision());

                            // end Code substitutions -------------------------------------------------------------------------
                            os << code << std::endl;
                        }
                    }
                }
            }
        }
    }
    os << std::endl;


    os << "#endif" << std::endl;
    fs.close();

//  cout << "exiting genSynapseFunction" << endl;
}
