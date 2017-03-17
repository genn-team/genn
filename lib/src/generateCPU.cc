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
#include "standardSubstitutions.h"
#include "CodeHelper.h"

#include <algorithm>
#include <typeinfo>

//--------------------------------------------------------------------------
/*!
  \brief Function that generates the code of the function the will simulate all neurons on the CPU.
*/
//--------------------------------------------------------------------------

void genNeuronFunction(const NNmodel &model, //!< Model description
                       const string &path //!< Path for code generation
    )
{
    ofstream os;

    string name = path + "/" + model.name + "_CODE/neuronFnct.cc";
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_neuronFnct_cc" << ENDL;
    os << "#define _" << model.name << "_neuronFnct_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file neuronFnct.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name;
    os << " containing the the equivalent of neuron kernel function for the CPU-only version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    os << "// include the support codes provided by the user for neuron or synaptic models" << ENDL;
    os << "#include \"support_code.h\"" << ENDL << ENDL; 

    // function header
    os << "void calcNeuronsCPU(" << model.ftype << " t)" << ENDL;
    os << OB(51);

    // function code
    for(auto &n : model.getNeuronGroups()) {
        string queueOffset = n.second.isDelayRequired()
            ? "(spkQuePtr" + n.first + " * " + to_string(n.second.getNumNeurons()) + ") + "
            : "";
        string queueOffsetTrueSpk = (n.second.isTrueSpikeRequired() ? queueOffset : "");

        os << "// neuron group " << n.first << ENDL;
        os << OB(55);

        // increment spike queue pointer and reset spike count
        if (n.second.isDelayRequired()) { // with delay
            os << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << ENDL;
            if (n.second.isSpikeEventRequired()) {
                os << "glbSpkCntEvnt" << n.first << "[spkQuePtr" << n.first << "] = 0;" << ENDL;
            }
            if (n.second.isTrueSpikeRequired()) {
                os << "glbSpkCnt" << n.first << "[spkQuePtr" << n.first << "] = 0;" << ENDL;
            }
            else {
                os << "glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
            }
        }
        else { // no delay
            if (n.second.isSpikeEventRequired()) {
                os << "glbSpkCntEvnt" << n.first << "[0] = 0;" << ENDL;
            }
            os << "glbSpkCnt" << n.first << "[0] = 0;" << ENDL;
        }
        if (n.second.isVarQueueRequired() && n.second.isDelayRequired()) {
            os << "unsigned int delaySlot = (spkQuePtr" << n.first;
            os << " + " << (n.second.getNumDelaySlots() - 1);
            os << ") % " << n.second.getNumDelaySlots() << ";" << ENDL;
        }
        os << ENDL;

        os << "for (int n = 0; n < " <<  n.second.getNumNeurons() << "; n++)" << OB(10);

        // Get neuron model associated with this group
        auto nm = n.second.getNeuronModel();

        // Create iteration context to iterate over the variables; derived and extra global parameters
        VarNameIterCtx nmVars(nm->GetVars());
        DerivedParamNameIterCtx nmDerivedParams(nm->GetDerivedParams());
        ExtraGlobalParamNameIterCtx nmExtraGlobalParams(nm->GetExtraGlobalParams());

        for (size_t k = 0; k < nmVars.container.size(); k++) {

            os << nmVars.container[k].second << " l" << nmVars.container[k].first << " = ";
            os << nmVars.container[k].first << n.first << "[";
            if (n.second.isVarQueueRequired(k) && n.second.isDelayRequired()) {
                os << "(delaySlot * " << n.second.getNumNeurons() << ") + ";
            }
            os << "n];" << ENDL;
        }
        if ((nm->GetSimCode().find("$(sT)") != string::npos)
            || (nm->GetThresholdConditionCode().find("$(sT)") != string::npos)
            || (nm->GetResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
            os << model.ftype << " lsT= sT" <<  n.first << "[";
            if (n.second.isDelayRequired()) {
                os << "(delaySlot * " << n.second.getNumNeurons() << ") + ";
            }
            os << "n];" << ENDL;
        }
        os << ENDL;

        if ((n.second.getInSyn().size() > 0) || (nm->GetSimCode().find("Isyn") != string::npos)) {
            os << model.ftype << " Isyn = 0;" << ENDL;
        }


        for(const auto &sName : n.second.getInSyn()) {
            const SynapseGroup *sg = model.findSynapseGroup(sName);
            const auto *psm = sg->getPSModel();

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                for(const auto &v : psm->GetVars()) {
                    os << v.second << " lps" << v.first << sName;
                    os << " = " <<  v.first << sName << "[n];" << ENDL;
                }
            }

            // Apply substitutions to current converter code
            string psCode = psm->GetCurrentConverterCode();
            substitute(psCode, "$(id)", "n");
            substitute(psCode, "$(inSyn)", "inSyn" + sName + "[n]");
            StandardSubstitutions::postSynapseCurrentConverter(psCode, sName, sg, n.first, n.second,
                nmVars, nmDerivedParams, nmExtraGlobalParams, model.ftype);

            if (!psm->GetSupportCode().empty()) {
                os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;
            }
            os << "Isyn += ";
            os << psCode << ";" << ENDL;
            if (!psm->GetSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << ENDL;
            }
        }

        if (!nm->GetSupportCode().empty()) {
            os << " using namespace " << n.first << "_neuron;" << ENDL;
        }

        string thCode = nm->GetThresholdConditionCode();
        if (thCode.empty()) { // no condition provided
            cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*nm).name() << " used for population \"" << n.first << "\" was provided. There will be no spikes detected in this population!" << endl;
        }
        else {
            os << "// test whether spike condition was fulfilled previously" << ENDL;
            substitute(thCode, "$(id)", "n");
            StandardSubstitutions::neuronThresholdCondition(thCode, n.first, n.second,
                                                            nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                            model.ftype);
            if (GENN_PREFERENCES::autoRefractory) {
                os << "bool oldSpike= (" << thCode << ");" << ENDL;
            }
        }

        os << "// calculate membrane potential" << ENDL;
        string sCode = nm->GetSimCode();
        substitute(sCode, "$(id)", "n");
        StandardSubstitutions::neuronSim(sCode, n.first, n.second,
                                         nmVars, nmDerivedParams, nmExtraGlobalParams,
                                         model.ftype);
        if (nm->IsPoisson()) {
            substitute(sCode, "lrate", "rates" + n.first + "[n + offset" + n.first + "]");
        }
        os << sCode << ENDL;

        // look for spike type events first.
        if (n.second.isSpikeEventRequired()) {
            // Create local variable
            os << "bool spikeLikeEvent = false;" << ENDL;

            // Loop through outgoing synapse populations that will contribute to event condition code
            for(const auto &spkEventCond : n.second.getSpikeEventCondition()) {
                // Replace of parameters, derived parameters and extraglobalsynapse parameters
                string eCode = spkEventCond.first;

                // code substitutions ----
                substitute(eCode, "$(id)", "n");
                StandardSubstitutions::neuronSpikeEventCondition(eCode, n.first,
                                                                 nmVars, nmExtraGlobalParams,
                                                                 model.ftype);
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

            os << "// register a spike-like event" << ENDL;
            os << "if (spikeLikeEvent)" << OB(30);
            os << "glbSpkEvnt" << n.first << "[" << queueOffset << "glbSpkCntEvnt" << n.first;
            if (n.second.isDelayRequired()) { // WITH DELAY
                os << "[spkQuePtr" << n.first << "]++] = n;" << ENDL;
            }
            else { // NO DELAY
                os << "[0]++] = n;" << ENDL;
            }
            os << CB(30);
        }

        // test for true spikes if condition is provided
        if (!thCode.empty()) {
            os << "// test for and register a true spike" << ENDL;
            if (GENN_PREFERENCES::autoRefractory) {
              os << "if ((" << thCode << ") && !(oldSpike))" << OB(40);
            }
            else{
              os << "if (" << thCode << ") " << OB(40);
            }
            os << "glbSpk" << n.first << "[" << queueOffsetTrueSpk << "glbSpkCnt" << n.first;
            if (n.second.isDelayRequired() && n.second.isTrueSpikeRequired()) { // WITH DELAY
                os << "[spkQuePtr" << n.first << "]++] = n;" << ENDL;
            }
            else { // NO DELAY
                os << "[0]++] = n;" << ENDL;
            }
            if (n.second.isSpikeTimeRequired()) {
                os << "sT" << n.first << "[" << queueOffset << "n] = t;" << ENDL;
            }

            // add after-spike reset if provided
            if (!nm->GetResetCode().empty()) {
                string rCode = nm->GetResetCode();
                substitute(rCode, "$(id)", "n");
                StandardSubstitutions::neuronReset(sCode, n.first, n.second,
                                                   nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                   model.ftype);
                os << "// spike reset code" << ENDL;
                os << rCode << ENDL;
            }
            os << CB(40);
        }

        // store the defined parts of the neuron state into the global state variables V etc
        for (size_t k = 0; k < nmVars.container.size(); k++) {
            if (n.second.isVarQueueRequired(k)) {
                os << nmVars.container[k].first << n.first << "[" << queueOffset << "n] = l" << nmVars.container[k].first << ";" << ENDL;
            }
            else {
                os << nmVars.container[k].first << n.first << "[n] = l" << nmVars.container[k].first << ";" << ENDL;
            }
        }

         for(const auto &sName : n.second.getInSyn()) {
            const SynapseGroup *sg = model.findSynapseGroup(sName);
            const auto *psm = sg->getPSModel();
            string pdCode = psm->GetDecayCode();
            substitute(pdCode, "$(id)", "n");

            StandardSubstitutions::postSynapseDecay(pdCode, sName, sg, n.first, n.second,
                                                    nmVars, nmDerivedParams, nmExtraGlobalParams,
                                                    model.ftype);
            os << "// the post-synaptic dynamics" << ENDL;
            if (!psm->GetSupportCode().empty()) {
                os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;
            }
            os << pdCode << ENDL;
            if (!psm->GetSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << endl;
            }
            for (const auto &v : psm->GetVars()) {
                os << v.first << sName << "[n]" << " = lps" << v.first << sName << ";" << ENDL;
            }
        }
        os << CB(10);
        os << CB(55);
        os << ENDL;
    }
    os << CB(51) << ENDL;
    os << "#endif" << ENDL;
    os.close();
} 


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic 
  spikes or spike type events
*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code_CPU(
    ostream &os, //!< output stream for code
    const string &sgName,
    const SynapseGroup &sg,
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype)
{
    bool evnt = postfix == "Evnt";
    int UIntSz = sizeof(unsigned int) * 8;
    int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f);

    if ((evnt && sg.isSpikeEventRequired()) || (!evnt && sg.isTrueSpikeRequired())) {
        const auto *wu = sg.getWUModel();
        const bool sparse = sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE;

        string offsetPre = sg.getSrcNeuronGroup()->isDelayRequired()
            ? "(delaySlot * " + to_string(sg.getSrcNeuronGroup()->getNumNeurons()) + ") + "
            : "";

        string offsetPost = sg.getTrgNeuronGroup()->isDelayRequired()
            ? "(spkQuePtr" + sg.getTrgNeuronGroupName() + " * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + ") + "
            : "";

        // Detect spike events or spikes and do the update
        os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << ENDL;
        if (sg.getSrcNeuronGroup()->isDelayRequired()) {
            os << "for (int i = 0; i < glbSpkCnt" << postfix << sg.getSrcNeuronGroupName() << "[delaySlot]; i++)" << OB(201);
        }
        else {
            os << "for (int i = 0; i < glbSpkCnt" << postfix << sg.getSrcNeuronGroupName() << "[0]; i++)" << OB(201);
        }

        os << "ipre = glbSpk" << postfix << sg.getSrcNeuronGroupName() << "[" << offsetPre << "i];" << ENDL;

        if (sparse) { // SPARSE
            os << "npost = C" << sgName << ".indInG[ipre + 1] - C" << sgName << ".indInG[ipre];" << ENDL;
            os << "for (int j = 0; j < npost; j++)" << OB(202);
            os << "ipost = C" << sgName << ".ind[C" << sgName << ".indInG[ipre] + j];" << ENDL;
        }
        else { // DENSE
            os << "for (ipost = 0; ipost < " << sg.getTrgNeuronGroup()->getNumNeurons() << "; ipost++)" << OB(202);
        }

        if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << "unsigned int gid = (ipre * " << sg.getTrgNeuronGroup()->getNumNeurons() << " + ipost);" << ENDL;
        }

        if (!wu->GetSimSupportCode().empty()) {
            os << " using namespace " << sgName << "_weightupdate_simCode;" << ENDL;
        }

        // Create iteration context to iterate over the variables; derived and extra global parameters
        DerivedParamNameIterCtx wuDerivedParams(wu->GetDerivedParams());
        ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->GetExtraGlobalParams());
        VarNameIterCtx wuVars(wu->GetVars());

        if (evnt) {
            // code substitutions ----
            string eCode = wu->GetEventThresholdConditionCode();
            substitute(eCode, "$(id)", "n");
            substitute(eCode, "$(t)", "t");
            value_substitutions(eCode, wu->GetParamNames(), sg.getWUParams());
            value_substitutions(eCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
            name_substitutions(eCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sgName);
            neuron_substitutions_in_synaptic_code(eCode, &sg, offsetPre, offsetPost,
                                                  "ipre", "ipost", "");
            eCode= ensureFtype(eCode, ftype);
            checkUnreplacedVariables(eCode, "evntThreshold");
            // end code substitutions ----

            if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                os << "if ((B(gp" << sgName << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1;
                os << ")) && (" << eCode << "))" << OB(2041);
            }
            else {
                os << "if (" << eCode << ")" << OB(2041);
            }
        }
        else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << "if (B(gp" << sgName << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << OB(2041);
        }

        // Code substitutions ----------------------------------------------------------------------------------
        string wCode = evnt ? wu->GetEventCode() : wu->GetSimCode();
        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
        substitute(wCode, "$(t)", "t");
        if (sparse) { // SPARSE
            if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                name_substitutions(wCode, "", wuVars.nameBegin, wuVars.nameEnd,
                                   sgName + "[C" + sgName + ".indInG[ipre] + j]");
            }
            else {
                value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUInitVals());
            }
        }
        else { // DENSE
            if (sg.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                name_substitutions(wCode, "", wuVars.nameBegin, wuVars.nameEnd,
                                    sgName + "[ipre * " + to_string(sg.getTrgNeuronGroup()->getNumNeurons()) + " + ipost]");
            }
            else {
                value_substitutions(wCode, wuVars.nameBegin, wuVars.nameEnd, sg.getWUInitVals());
            }
        }
        substitute(wCode, "$(inSyn)", "inSyn" + sgName + "[ipost]");
        value_substitutions(wCode, wu->GetParamNames(), sg.getWUParams());
        value_substitutions(wCode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg.getWUDerivedParams());
        name_substitutions(wCode, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, sgName);
        substitute(wCode, "$(addtoinSyn)", "addtoinSyn");
        neuron_substitutions_in_synaptic_code(wCode, &sg, offsetPre, offsetPost,
                                              "ipre", "ipost", "");
        wCode= ensureFtype(wCode, ftype);
        checkUnreplacedVariables(wCode, "simCode"+postfix);
        // end Code substitutions -------------------------------------------------------------------------
        os << wCode << ENDL;

        if (evnt) {
            os << CB(2041); // end if (eCode)
        }
        else if (sg.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            os << CB(2041); // end if (B(gp" << sgName << "[gid >> " << logUIntSz << "], gid
        }
        os << CB(202);
        os << CB(201);
    }
}


//--------------------------------------------------------------------------
/*!
  \brief Function that generates code that will simulate all synapses of the model on the CPU.
*/
//--------------------------------------------------------------------------

void genSynapseFunction(const NNmodel &model, //!< Model description
                        const string &path //!< Path for code generation
    )
{
    ofstream os;

//    cout << "entering genSynapseFunction" << endl;
    string name = path + "/" + model.name + "_CODE/synapseFnct.cc";
    os.open(name.c_str());

    // write header content
    writeHeader(os);
    os << ENDL;

    // compiler/include control (include once)
    os << "#ifndef _" << model.name << "_synapseFnct_cc" << ENDL;
    os << "#define _" << model.name << "_synapseFnct_cc" << ENDL;
    os << ENDL;

    // write doxygen comment
    os << "//-------------------------------------------------------------------------" << ENDL;
    os << "/*! \\file synapseFnct.cc" << ENDL << ENDL;
    os << "\\brief File generated from GeNN for the model " << model.name << " containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version." << ENDL;
    os << "*/" << ENDL;
    os << "//-------------------------------------------------------------------------" << ENDL << ENDL;

    // synapse dynamics function
    os << "void calcSynapseDynamicsCPU(" << model.ftype << " t)" << ENDL;
    os << OB(1000);
    os << "// execute internal synapse dynamics if any" << ENDL;

    for(const auto &s : model.getSynapseDynamicsGroups())
    {
        const SynapseGroup *sg = model.findSynapseGroup(s.first);
        const auto *wu = sg->getWUModel();

        string offsetPre = sg->getSrcNeuronGroup()->isDelayRequired()
            ? "(delaySlot * " + to_string(sg->getSrcNeuronGroup()->getNumNeurons()) + ") + "
            : "";
        string offsetPost = sg->getTrgNeuronGroup()->isDelayRequired()
            ? "(spkQuePtr" + sg->getTrgNeuronGroupName() +" * " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + "
            : "";

        // there is some internal synapse dynamics
        if (!wu->GetSynapseDynamicsCode().empty()) {

            os << "// synapse group " << s.first << ENDL;
            os << OB(1005);

            if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                os << "unsigned int delaySlot = (spkQuePtr" << sg->getSrcNeuronGroupName();
                os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
            }

            if (!wu->GetSynapseDynamicsSuppportCode().empty()) {
                os << "using namespace " << s.first << "_weightupdate_synapseDynamics;" << ENDL;
            }

            // Create iteration context to iterate over the variables and derived parameters
            DerivedParamNameIterCtx wuDerivedParams(wu->GetDerivedParams());
            VarNameIterCtx wuVars(wu->GetVars());

            string SDcode= wu->GetSynapseDynamicsCode();
            substitute(SDcode, "$(t)", "t");
            if (sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) { // SPARSE
                os << "for (int n= 0; n < C" << s.first << ".connN; n++)" << OB(24) << ENDL;
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    // name substitute synapse var names in synapseDynamics code
                    name_substitutions(SDcode, "", wuVars.nameBegin, wuVars.nameEnd, s.first + "[n]");
                }
                else {
                    // substitute initial values as constants for synapse var names in synapseDynamics code
                    value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUInitVals());
                }
                // substitute parameter values for parameters in synapseDynamics code
                value_substitutions(SDcode, wu->GetParamNames(), sg->getWUParams());
                // substitute values for derived parameters in synapseDynamics code
                value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
                neuron_substitutions_in_synaptic_code(SDcode, sg, offsetPre, offsetPost,
                                                      "C" + s.first + ".preInd[n]", "C" +s.first + ".ind[n]", "");
                SDcode= ensureFtype(SDcode, model.ftype);
                checkUnreplacedVariables(SDcode, "synapseDynamics");
                os << SDcode << ENDL;
                os << CB(24);
            }
            else { // DENSE
                os << "for (int i = 0; i < " <<  sg->getSrcNeuronGroup()->getNumNeurons() << "; i++)" << OB(25);
                os << "for (int j = 0; j < " <<  sg->getTrgNeuronGroup()->getNumNeurons() << "; j++)" << OB(26);
                os << "// loop through all synapses" << endl;
                // substitute initial values as constants for synapse var names in synapseDynamics code
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                    name_substitutions(SDcode, "", wuVars.nameBegin, wuVars.nameEnd,
                                       s.first + "[i*" + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + "+j]");
                }
                else {
                    // substitute initial values as constants for synapse var names in synapseDynamics code
                    value_substitutions(SDcode, wuVars.nameBegin, wuVars.nameEnd, sg->getWUInitVals());
                }
                // substitute parameter values for parameters in synapseDynamics code
                value_substitutions(SDcode, wu->GetParamNames(), sg->getWUParams());
                // substitute values for derived parameters in synapseDynamics code
                value_substitutions(SDcode, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
                neuron_substitutions_in_synaptic_code(SDcode, sg, offsetPre, offsetPost,
                                                      "i", "j", "");
                SDcode= ensureFtype(SDcode, model.ftype);
                checkUnreplacedVariables(SDcode, "synapseDynamics");
                os << SDcode << ENDL;
                os << CB(26);
                os << CB(25);
            }
            os << CB(1005);
        }
    }
    os << CB(1000);

    // synapse function header
    os << "void calcSynapsesCPU(" << model.ftype << " t)" << ENDL;

    // synapse function code
    os << OB(1001);

    os << "unsigned int ipost;" << ENDL;
    os << "unsigned int ipre;" << ENDL;
    for(const auto &s : model.getSynapseGroups()) {
        if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            os << "unsigned int npost;" << ENDL;
            break;
        }
    }
    os << model.ftype << " addtoinSyn;" << ENDL;  
    os << ENDL;

   for(const auto &s : model.getSynapseGroups()) {
        os << "// synapse group " << s.first << ENDL;
        os << OB(1006);

        if (s.second.getSrcNeuronGroup()->isDelayRequired()) {
            os << "unsigned int delaySlot = (spkQuePtr" << s.second.getSrcNeuronGroupName();
            os << " + " << (s.second.getSrcNeuronGroup()->getNumDelaySlots() - s.second.getDelaySteps());
            os << ") % " << s.second.getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
        }

        // generate the code for processing spike-like events
        if (s.second.isSpikeEventRequired()) {
            generate_process_presynaptic_events_code_CPU(os, s.first, s.second, "Evnt", model.ftype);
        }

        // generate the code for processing true spike events
        if (s.second.isTrueSpikeRequired()) {
            generate_process_presynaptic_events_code_CPU(os, s.first, s.second, "", model.ftype);
        }

        os << CB(1006);
        os << ENDL;
    }
    os << CB(1001);
    os << ENDL;


    //////////////////////////////////////////////////////////////
    // function for learning synapses, post-synaptic spikes

    if (!model.getSynapsePostLearnGroups().empty()) {

        os << "void learnSynapsesPostHost(" << model.ftype << " t)" << ENDL;
        os << OB(811);

        os << "unsigned int ipost;" << ENDL;
        os << "unsigned int ipre;" << ENDL;
        os << "unsigned int lSpk;" << ENDL;

        // If any synapse groups have sparse connectivity
        if(any_of(begin(model.getSynapseGroups()), end(model.getSynapseGroups()),
            [](const std::pair<string, SynapseGroup> &s)
            {
                return (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE);

            }))
        {
            os << "unsigned int npre;" << ENDL;
        }
        os << ENDL;

        for(const auto &s : model.getSynapsePostLearnGroups())
        {
            const SynapseGroup *sg = model.findSynapseGroup(s.first);
            const auto *wu = sg->getWUModel();
            const bool sparse = sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE;

            string offsetPre = sg->getSrcNeuronGroup()->isDelayRequired()
                ? "(delaySlot * " + to_string(sg->getSrcNeuronGroup()->getNumNeurons()) + ") + "
                : "";
            string offsetTrueSpkPre = sg->getSrcNeuronGroup()->isTrueSpikeRequired() ? offsetPre : "";

            string offsetPost = sg->getTrgNeuronGroup()->isDelayRequired()
                ? "(spkQuePtr" + sg->getTrgNeuronGroupName() + " * " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + ") + "
                : "";
            string offsetTrueSpkPost = sg->getTrgNeuronGroup()->isTrueSpikeRequired() ? offsetPost : "";

            // Create iteration context to iterate over the variables; derived and extra global parameters
            DerivedParamNameIterCtx wuDerivedParams(wu->GetDerivedParams());
            ExtraGlobalParamNameIterCtx wuExtraGlobalParams(wu->GetExtraGlobalParams());
            VarNameIterCtx wuVars(wu->GetVars());

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM

            os << "// synapse group " << s.first << ENDL;
            os << OB(950);

            if (sg->getSrcNeuronGroup()->isDelayRequired()) {
                os << "unsigned int delaySlot = (spkQuePtr" << sg->getSrcNeuronGroupName();
                os << " + " << (sg->getSrcNeuronGroup()->getNumDelaySlots() - sg->getDelaySteps());
                os << ") % " << sg->getSrcNeuronGroup()->getNumDelaySlots() << ";" << ENDL;
            }

            if (!wu->GetLearnPostSupportCode().empty()) {
                os << "using namespace " << s.first << "_weightupdate_simLearnPost;" << ENDL;
            }

            if (sg->getTrgNeuronGroup()->isDelayRequired() && sg->getTrgNeuronGroup()->isTrueSpikeRequired()) {
                os << "for (ipost = 0; ipost < glbSpkCnt" << sg->getTrgNeuronGroupName() << "[spkQuePtr" << sg->getTrgNeuronGroupName() << "]; ipost++)" << OB(910);
            }
            else {
                os << "for (ipost = 0; ipost < glbSpkCnt" << sg->getTrgNeuronGroupName() << "[0]; ipost++)" << OB(910);
            }

            os << "lSpk = glbSpk" << sg->getTrgNeuronGroupName() << "[" << offsetTrueSpkPost << "ipost];" << ENDL;

            if (sparse) { // SPARSE
                // TODO: THIS NEEDS CHECKING AND FUNCTIONAL C.POST* ARRAYS
                os << "npre = C" << s.first << ".revIndInG[lSpk + 1] - C" << s.first << ".revIndInG[lSpk];" << ENDL;
                os << "for (int l = 0; l < npre; l++)" << OB(121);
                os << "ipre = C" << s.first << ".revIndInG[lSpk] + l;" << ENDL;
            }
            else { // DENSE
                os << "for (ipre = 0; ipre < " << sg->getSrcNeuronGroup()->getNumNeurons() << "; ipre++)" << OB(121);
            }

            string code = wu->GetLearnPostCode();
            substitute(code, "$(t)", "t");
            // Code substitutions ----------------------------------------------------------------------------------
            if (sparse) { // SPARSE
                name_substitutions(code, "", wuVars.nameBegin, wuVars.nameEnd,
                                   s.first + "[C" + s.first + ".remap[ipre]]");
            }
            else { // DENSE
                name_substitutions(code, "", wuVars.nameBegin, wuVars.nameEnd,
                                   s.first + "[lSpk + " + to_string(sg->getTrgNeuronGroup()->getNumNeurons()) + " * ipre]");
            }
            value_substitutions(code, wu->GetParamNames(), sg->getWUParams());
            value_substitutions(code, wuDerivedParams.nameBegin, wuDerivedParams.nameEnd, sg->getWUDerivedParams());
            name_substitutions(code, "", wuExtraGlobalParams.nameBegin, wuExtraGlobalParams.nameEnd, s.first);

            // presynaptic neuron variables and parameters
            if (sparse) { // SPARSE
                neuron_substitutions_in_synaptic_code(code, sg, offsetPre, offsetPost,
                                                      "C" + s.first + ".revInd[ipre]", "lSpk", "");
            }
            else { // DENSE
                neuron_substitutions_in_synaptic_code(code, sg, offsetPre, offsetPost,
                                                      "ipre", "lSpk", "");
            }
            code= ensureFtype(code, model.ftype);
            checkUnreplacedVariables(code, "simLearnPost");
            // end Code substitutions -------------------------------------------------------------------------
            os << code << ENDL;

            os << CB(121);
            os << CB(910);
            os << CB(950);
        }
        os << CB(811);
    }
    os << ENDL;


    os << "#endif" << ENDL;
    os.close();

//  cout << "exiting genSynapseFunction" << endl;
}
