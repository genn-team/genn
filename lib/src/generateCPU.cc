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
    string s;
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
    for (unsigned int i = 0; i < model.neuronGrpN; i++) {
        string queueOffset = (model.neuronDelaySlots[i] > 1 ? "(spkQuePtr" + model.neuronName[i] + " * " + to_string(model.neuronN[i]) + ") + " : "");
        string queueOffsetTrueSpk = (model.neuronNeedTrueSpk[i] ? queueOffset : "");

        os << "// neuron group " << model.neuronName[i] << ENDL;
        os << OB(55);

        // increment spike queue pointer and reset spike count
        if (model.neuronDelaySlots[i] > 1) { // with delay
            os << "spkQuePtr" << model.neuronName[i] << " = (spkQuePtr" << model.neuronName[i] << " + 1) % " << model.neuronDelaySlots[i] << ";" << ENDL;
            if (model.neuronNeedSpkEvnt[i]) {
                os << "glbSpkCntEvnt" << model.neuronName[i] << "[spkQuePtr" << model.neuronName[i] << "] = 0;" << ENDL;
            }
            if (model.neuronNeedTrueSpk[i]) {
                os << "glbSpkCnt" << model.neuronName[i] << "[spkQuePtr" << model.neuronName[i] << "] = 0;" << ENDL;
            }
            else {
                os << "glbSpkCnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
            }
        }
        else { // no delay
            if (model.neuronNeedSpkEvnt[i]) {
                os << "glbSpkCntEvnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
            }
            os << "glbSpkCnt" << model.neuronName[i] << "[0] = 0;" << ENDL;
        }
        vector<bool> varNeedQueue = model.neuronVarNeedQueue[i];
        if ((find(varNeedQueue.begin(), varNeedQueue.end(), true) != varNeedQueue.end()) && (model.neuronDelaySlots[i] > 1)) {
            os << "unsigned int delaySlot = (spkQuePtr" << model.neuronName[i];
            os << " + " << (model.neuronDelaySlots[i] - 1);
            os << ") % " << model.neuronDelaySlots[i] << ";" << ENDL;
        }
        os << ENDL;

        os << "for (int n = 0; n < " <<  model.neuronN[i] << "; n++)" << OB(10);

        // Get neuron model associated with this group
        auto neuronModel = model.neuronModel[i];

        // Create iterators to iterate over the names of the neuron model's initial values
        auto neuronModelVars = neuronModel->GetVars();
        auto neuronModelVarNameBegin = GetPairKeyConstIter(neuronModelVars.cbegin());
        auto neuronModelVarNameEnd = GetPairKeyConstIter(neuronModelVars.cend());

        // Create iterators to iterate over the names of the neuron model's derived parameters
        auto neuronModelDerivedParams = neuronModel->GetDerivedParams();
        auto neuronModelDerivedParamNameBegin= GetPairKeyConstIter(neuronModelDerivedParams.cbegin());
        auto neuronModelDerivedParamNameEnd = GetPairKeyConstIter(neuronModelDerivedParams.cend());

        // Create iterators to iterate over the names of the neuron model's extra global parameters
        auto neuronModelExtraGlobalParams = neuronModel->GetExtraGlobalParams();
        auto neuronModelExtraGlobalParamsNameBegin = GetPairKeyConstIter(neuronModelExtraGlobalParams.cbegin());
        auto neuronModelExtraGlobalParamsNameEnd = GetPairKeyConstIter(neuronModelExtraGlobalParams.cend());
        for (size_t k = 0; k < neuronModelVars.size(); k++) {

            os << neuronModelVars[k].second << " l" << neuronModelVars[k].first << " = ";
            os << neuronModelVars[k].first << model.neuronName[i] << "[";
            if ((model.neuronVarNeedQueue[i][k]) && (model.neuronDelaySlots[i] > 1)) {
                os << "(delaySlot * " << model.neuronN[i] << ") + ";
            }
            os << "n];" << ENDL;
        }
        if ((neuronModel->GetSimCode().find("$(sT)") != string::npos)
            || (neuronModel->GetThresholdConditionCode().find("$(sT)") != string::npos)
            || (neuronModel->GetResetCode().find("$(sT)") != string::npos)) { // load sT into local variable
            os << model.ftype << " lsT= sT" <<  model.neuronName[i] << "[";
            if (model.neuronDelaySlots[i] > 1) {
                os << "(delaySlot * " << model.neuronN[i] << ") + ";
            }
            os << "n];" << ENDL;
        }
        os << ENDL;

        if ((model.inSyn[i].size() > 0) || (neuronModel->GetSimCode().find("Isyn") != string::npos)) {
            os << model.ftype << " Isyn = 0;" << ENDL;
        }
        for (size_t j = 0; j < model.inSyn[i].size(); j++) {
            unsigned int synPopID = model.inSyn[i][j]; // number of (post)synapse group
            const auto *psm = model.postSynapseModel[synPopID];
            const string &sName = model.synapseName[synPopID];


            if (model.synapseGType[synPopID] == INDIVIDUALG) {
                for(const auto &v : psm->GetVars()) {
                    os << v.second << " lps" << v.first << sName;
                    os << " = " <<  v.first << sName << "[n];" << ENDL;
                }
            }

            string psCode = psm->GetCurrentConverterCode();
            substitute(psCode, "$(id)", "n");
            substitute(psCode, "$(t)", "t");
            substitute(psCode, "$(inSyn)", "inSyn" + sName + "[n]");

            name_substitutions(psCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "");
            value_substitutions(psCode, neuronModel->GetParamNames(), model.neuronPara[i]);
            value_substitutions(psCode, neuronModelDerivedParamNameBegin, neuronModelDerivedParamNameEnd, model.dnp[i]);

            // Create iterators to iterate over the names of the postsynaptic model's initial values
            auto psmVars = psm->GetVars();
            auto psmVarNameBegin = GetPairKeyConstIter(psmVars.cbegin());
            auto psmVarNameEnd = GetPairKeyConstIter(psmVars.cend());

            if (model.synapseGType[synPopID] == INDIVIDUALG) {
                name_substitutions(psCode, "lps", psmVarNameBegin, psmVarNameEnd, sName);
            }
            else {
                value_substitutions(psCode, psmVarNameBegin, psmVarNameEnd, model.postSynIni[synPopID]);
            }
            value_substitutions(psCode, psm->GetParamNames(), model.postSynapsePara[synPopID]);

            // Create iterators to iterate over the names of the postsynaptic model's derived parameters
            auto psmDerivedParams = psm->GetDerivedParams();
            value_substitutions(psCode, GetPairKeyConstIter(psmDerivedParams.cbegin()),
                                GetPairKeyConstIter(psmDerivedParams.cend()), model.dpsp[synPopID]);
            name_substitutions(psCode, "", neuronModelExtraGlobalParamsNameBegin, neuronModelExtraGlobalParamsNameEnd, model.neuronName[i]);
            psCode= ensureFtype(psCode, model.ftype);
            checkUnreplacedVariables(psCode, "postSyntoCurrent");
            if (!psm->GetSupportCode().empty()) {
                os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;
            }
            os << "Isyn += ";
            os << psCode << ";" << ENDL;
            if (!psm->GetSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << ENDL;
            }
        }

	    if (!neuronModel->GetSupportCode().empty()) {
	        os << " using namespace " << model.neuronName[i] << "_neuron;" << ENDL;
	    }
      
        string thCode= neuronModel->GetThresholdConditionCode();
        if (thCode.empty()) { // no condition provided
            cerr << "Warning: No thresholdConditionCode for neuron type " << typeid(*neuronModel).name() << " used for population \"" << model.neuronName[i] << "\" was provided. There will be no spikes detected in this population!" << endl;
        }
        else {
            os << "// test whether spike condition was fulfilled previously" << ENDL;
            substitute(thCode, "$(id)", "n");
            substitute(thCode, "$(t)", "t");
            name_substitutions(thCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "");
            substitute(thCode, "$(sT)", "lsT");
            value_substitutions(thCode, neuronModel->GetParamNames(), model.neuronPara[i]);
            value_substitutions(thCode, neuronModelDerivedParamNameBegin, neuronModelDerivedParamNameEnd, model.dnp[i]);
            substitute(thCode, "$(Isyn)", "Isyn");
            thCode = ensureFtype(thCode, model.ftype);
            checkUnreplacedVariables(thCode, "thresholdConditionCode");
            if (GENN_PREFERENCES::autoRefractory) {
                os << "bool oldSpike= (" << thCode << ");" << ENDL;
            }
        }

        os << "// calculate membrane potential" << ENDL;
        string sCode = neuronModel->GetSimCode();
        substitute(sCode, "$(id)", "n");
        substitute(sCode, "$(t)", "t");
        name_substitutions(sCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "");
        value_substitutions(sCode, neuronModel->GetParamNames(), model.neuronPara[i]);
        value_substitutions(sCode, neuronModelDerivedParamNameBegin, neuronModelDerivedParamNameEnd, model.dnp[i]);
        name_substitutions(sCode, "", neuronModelExtraGlobalParamsNameBegin, neuronModelExtraGlobalParamsNameEnd, model.neuronName[i]);
        if (neuronModel->IsPoisson()) {
            substitute(sCode, "lrate", "rates" + model.neuronName[i] + "[n + offset" + model.neuronName[i] + "]");
        }
        substitute(sCode, "$(Isyn)", "Isyn");
        substitute(sCode, "$(sT)", "lsT");
        sCode= ensureFtype(sCode, model.ftype);
        checkUnreplacedVariables(sCode,"neuron simCode");
        os << sCode << ENDL;

        // look for spike type events first.
        if (model.neuronNeedSpkEvnt[i]) {
            // Create local variable
            os << "bool spikeLikeEvent = false;" << ENDL;

            // Loop through outgoing synapse populations that will contribute to event condition code
            for(const auto &spkEventCond : model.neuronSpkEvntCondition[i]) {
                // Replace of parameters, derived parameters and extraglobalsynapse parameters
                string eCode = spkEventCond.first;
                
                // code substitutions ----
                substitute(eCode, "$(id)", "n");
                substitute(eCode, "$(t)", "t");
                name_substitutions(eCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "", "_pre");
                name_substitutions(eCode, "", neuronModelExtraGlobalParamsNameBegin, neuronModelExtraGlobalParamsNameEnd, model.neuronName[i]);
                eCode = ensureFtype(eCode, model.ftype);
                checkUnreplacedVariables(eCode, "neuronSpkEvntCondition");

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
            os << "glbSpkEvnt" << model.neuronName[i] << "[" << queueOffset << "glbSpkCntEvnt" << model.neuronName[i];
            if (model.neuronDelaySlots[i] > 1) { // WITH DELAY
                os << "[spkQuePtr" << model.neuronName[i] << "]++] = n;" << ENDL;
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
            os << "glbSpk" << model.neuronName[i] << "[" << queueOffsetTrueSpk << "glbSpkCnt" << model.neuronName[i];
            if ((model.neuronDelaySlots[i] > 1) && (model.neuronNeedTrueSpk[i])) { // WITH DELAY
                os << "[spkQuePtr" << model.neuronName[i] << "]++] = n;" << ENDL;
            }
            else { // NO DELAY
                os << "[0]++] = n;" << ENDL;
            }
            if (model.neuronNeedSt[i]) {
                os << "sT" << model.neuronName[i] << "[" << queueOffset << "n] = t;" << ENDL;
            }

            // add after-spike reset if provided
            if (!neuronModel->GetResetCode().empty()) {
                string rCode = neuronModel->GetResetCode();
                substitute(rCode, "$(id)", "n");
                substitute(rCode, "$(t)", "t");
                name_substitutions(rCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "");
                value_substitutions(rCode, neuronModel->GetParamNames(), model.neuronPara[i]);
                value_substitutions(rCode, neuronModelDerivedParamNameBegin, neuronModelDerivedParamNameEnd, model.dnp[i]);
                substitute(rCode, "$(Isyn)", "Isyn");
                substitute(rCode, "$(sT)", "lsT");
                os << "// spike reset code" << ENDL;
                name_substitutions(rCode, "", neuronModelExtraGlobalParamsNameBegin, neuronModelExtraGlobalParamsNameEnd, model.neuronName[i]);
                rCode= ensureFtype(rCode, model.ftype);
                checkUnreplacedVariables(rCode, "resetCode");
                os << rCode << ENDL;
            }
            os << CB(40);
        }

        // store the defined parts of the neuron state into the global state variables V etc
        for (size_t k = 0; k < neuronModelVars.size(); k++) {
            if (model.neuronVarNeedQueue[i][k]) {
                os << neuronModelVars[k].first << model.neuronName[i] << "[" << queueOffset << "n] = l" << neuronModelVars[k].first << ";" << ENDL;
            }
            else {
                os << neuronModelVars[k].first << model.neuronName[i] << "[n] = l" << neuronModelVars[k].first << ";" << ENDL;
            }
        }

        for (size_t j = 0; j < model.inSyn[i].size(); j++) {
            const auto *psm= model.postSynapseModel[model.inSyn[i][j]];
            string sName= model.synapseName[model.inSyn[i][j]];
            string pdCode = psm->GetDecayCode();
            substitute(pdCode, "$(id)", "n");
            substitute(pdCode, "$(t)", "t");
            substitute(pdCode, "$(inSyn)", "inSyn" + sName + "[n]");

            auto psmVars = psm->GetVars();
            name_substitutions(pdCode, "lps", GetPairKeyConstIter(psmVars.cbegin()),
                               GetPairKeyConstIter(psmVars.cend()), sName);
            value_substitutions(pdCode, psm->GetParamNames(), model.postSynapsePara[model.inSyn[i][j]]);

            auto psmDerivedParams = psm->GetDerivedParams();
            value_substitutions(pdCode, GetPairKeyConstIter(psmDerivedParams.cbegin()),
                                GetPairKeyConstIter(psmDerivedParams.cend()), model.dpsp[model.inSyn[i][j]]);
            name_substitutions(pdCode, "l", neuronModelVarNameBegin, neuronModelVarNameEnd, "");
            value_substitutions(pdCode, neuronModel->GetParamNames(), model.neuronPara[i]);
            value_substitutions(pdCode, neuronModelDerivedParamNameBegin, neuronModelDerivedParamNameEnd, model.dnp[i]);
            os << "// the post-synaptic dynamics" << ENDL;
            pdCode= ensureFtype(pdCode, model.ftype);
            checkUnreplacedVariables(pdCode, "postSynDecay");
            if (!psm->GetSupportCode().empty()) {
                os << OB(29) << " using namespace " << sName << "_postsyn;" << ENDL;
            }
            os << pdCode << ENDL;
            if (!psm->GetSupportCode().empty()) {
                os << CB(29) << " // namespace bracket closed" << endl;
            }
            for (const auto &v : psmVars) {
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
    const NNmodel &model, //!< the neuronal network model to generate code for
    unsigned int src, //!< the number of the src neuron population
    unsigned int trg, //!< the number of the target neuron population
    int i, //!< the index of the synapse group being processed
    const string &postfix //!< whether to generate code for true spikes or spike type events
    )
{
    bool evnt = postfix == "Evnt";
    int UIntSz = sizeof(unsigned int) * 8;
    int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f);

    if ((evnt && model.synapseUsesSpikeEvents[i]) || (!evnt && model.synapseUsesTrueSpikes[i])) {
        const auto *wu = model.synapseModel[i];
        bool sparse = model.synapseConnType[i] == SPARSE;

        bool delayPre = model.neuronDelaySlots[src] > 1;
        string offsetPre = delayPre ? ("(delaySlot * " + to_string(model.neuronN[src]) + ") + ") : "";

        bool delayPost = model.neuronDelaySlots[trg] > 1;
        string offsetPost = delayPost ? ("(spkQuePtr" + model.neuronName[trg] + " * " + to_string(model.neuronN[trg]) + ") + ") : "";

        // Detect spike events or spikes and do the update
        os << "// process presynaptic events: " << (evnt ? "Spike type events" : "True Spikes") << ENDL;
        if (delayPre) {
            os << "for (int i = 0; i < glbSpkCnt" << postfix << model.neuronName[src] << "[delaySlot]; i++)" << OB(201);
        }
        else {
            os << "for (int i = 0; i < glbSpkCnt" << postfix << model.neuronName[src] << "[0]; i++)" << OB(201);
        }

        os << "ipre = glbSpk" << postfix << model.neuronName[src] << "[" << offsetPre << "i];" << ENDL;

        if (sparse) { // SPARSE
            os << "npost = C" << model.synapseName[i] << ".indInG[ipre + 1] - C" << model.synapseName[i] << ".indInG[ipre];" << ENDL;
            os << "for (int j = 0; j < npost; j++)" << OB(202);
            os << "ipost = C" << model.synapseName[i] << ".ind[C" << model.synapseName[i] << ".indInG[ipre] + j];" << ENDL;
        }
        else { // DENSE
            os << "for (ipost = 0; ipost < " << model.neuronN[trg] << "; ipost++)" << OB(202);
        }

        if (model.synapseGType[i] == INDIVIDUALID) {
            os << "unsigned int gid = (ipre * " << model.neuronN[trg] << " + ipost);" << ENDL;
        }

        if (!wu->GetSimSupportCode().empty()) {
            os << " using namespace " << model.synapseName[i] << "_weightupdate_simCode;" << ENDL;
        }

        // Create iterators to iterate over the names of the weight update model's derived parameters
        auto wuDerivedParams = wu->GetDerivedParams();
        auto wuDerivedParamNameBegin= GetPairKeyConstIter(wuDerivedParams.cbegin());
        auto wuDerivedParamNameEnd = GetPairKeyConstIter(wuDerivedParams.cend());

        // Create iterators to iterate over the names of the weight update model's extra global parameters
        auto wuExtraGlobalParams = wu->GetExtraGlobalParams();
        auto wuExtraGlobalParamsNameBegin = GetPairKeyConstIter(wuExtraGlobalParams.cbegin());
        auto wuExtraGlobalParamsNameEnd = GetPairKeyConstIter(wuExtraGlobalParams.cend());

        // Create iterators to iterate over the names of the weight update model's initial values
        auto wuVars = wu->GetVars();
        auto wuVarNameBegin = GetPairKeyConstIter(wuVars.cbegin());
        auto wuVarNameEnd = GetPairKeyConstIter(wuVars.cend());

        if (evnt) {
            // code substitutions ----
            string eCode = wu->GetEventThresholdConditionCode();
            substitute(eCode, "$(id)", "n");
            substitute(eCode, "$(t)", "t");
            value_substitutions(eCode, wu->GetParamNames(), model.synapsePara[i]);
            value_substitutions(eCode, wuDerivedParamNameBegin, wuDerivedParamNameEnd, model.dsp_w[i]);
            name_substitutions(eCode, "", wuExtraGlobalParamsNameBegin, wuExtraGlobalParamsNameEnd, model.synapseName[i]);
            neuron_substitutions_in_synaptic_code(eCode, model, src, trg, offsetPre, offsetPost, "ipre", "ipost", "");
            eCode= ensureFtype(eCode, model.ftype);
            checkUnreplacedVariables(eCode, "evntThreshold");
            // end code substitutions ----

            if (model.synapseGType[i] == INDIVIDUALID) {
                os << "if ((B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1;
                os << ")) && (" << eCode << "))" << OB(2041);
            }
            else {
                os << "if (" << eCode << ")" << OB(2041);
            }
        }
        else if (model.synapseGType[i] == INDIVIDUALID) {
            os << "if (B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid & " << UIntSz - 1 << "))" << OB(2041);
        }

        // Code substitutions ----------------------------------------------------------------------------------
        string wCode = evnt ? wu->GetEventCode() : wu->GetSimCode();
        substitute(wCode, "$(updatelinsyn)", "$(inSyn) += $(addtoinSyn)");
        substitute(wCode, "$(t)", "t");
        if (sparse) { // SPARSE
            if (model.synapseGType[i] == INDIVIDUALG) {
                name_substitutions(wCode, "", wuVarNameBegin, wuVarNameEnd, model.synapseName[i] + "[C" + model.synapseName[i] + ".indInG[ipre] + j]");
            }
            else {
                value_substitutions(wCode, wuVarNameBegin, wuVarNameEnd, model.synapseIni[i]);
            }
        }
        else { // DENSE
            if (model.synapseGType[i] == INDIVIDUALG) {
                name_substitutions(wCode, "", wuVarNameBegin, wuVarNameEnd, model.synapseName[i] + "[ipre * " + to_string(model.neuronN[trg]) + " + ipost]");
            }
            else {
                value_substitutions(wCode, wuVarNameBegin, wuVarNameEnd, model.synapseIni[i]);
            }
        }
        substitute(wCode, "$(inSyn)", "inSyn" + model.synapseName[i] + "[ipost]");
        value_substitutions(wCode, wu->GetParamNames(), model.synapsePara[i]);
        value_substitutions(wCode, wuDerivedParamNameBegin, wuDerivedParamNameEnd, model.dsp_w[i]);
        name_substitutions(wCode, "", wuExtraGlobalParamsNameBegin, wuExtraGlobalParamsNameEnd, model.synapseName[i]);
        substitute(wCode, "$(addtoinSyn)", "addtoinSyn");
        neuron_substitutions_in_synaptic_code(wCode, model, src, trg, offsetPre, offsetPost, "ipre", "ipost", "");
        wCode= ensureFtype(wCode, model.ftype);
        checkUnreplacedVariables(wCode, "simCode"+postfix);
        // end Code substitutions -------------------------------------------------------------------------
        os << wCode << ENDL;

        if (evnt) {
            os << CB(2041); // end if (eCode)
        }
        else if (model.synapseGType[i] == INDIVIDUALID) {
            os << CB(2041); // end if (B(gp" << model.synapseName[i] << "[gid >> " << logUIntSz << "], gid
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

    for (unsigned int i = 0; i < model.synDynGroups; i++) {
        unsigned int k= model.synDynGrp[i];
        unsigned int src= model.synapseSource[k];
        unsigned int trg= model.synapseTarget[k];
        const auto *wu = model.synapseModel[k];
        string synapseName= model.synapseName[k];
        unsigned int srcno= model.neuronN[src];
        unsigned int trgno= model.neuronN[trg];
        bool delayPre = model.neuronDelaySlots[src] > 1;
        bool delayPost = model.neuronDelaySlots[trg] > 1;
        string offsetPre = (delayPre ? "(delaySlot * " + to_string(model.neuronN[src]) + ") + " : "");
        string offsetPost = (delayPost ? "(spkQuePtr" + model.neuronName[trg] +" * " + to_string(model.neuronN[trg]) + ") + " : "");

        // there is some internal synapse dynamics
        if (!wu->GetSynapseDynamicsCode().empty()) {

            os << "// synapse group " << synapseName << ENDL;
            os << OB(1005);

            if (model.neuronDelaySlots[src] > 1) {
                os << "unsigned int delaySlot = (spkQuePtr" << model.neuronName[src];
                os << " + " << (model.neuronDelaySlots[src] - model.synapseDelay[k]);
                os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
            }

            if (!wu->GetSynapseDynamicsSuppportCode().empty()) {
                os << "using namespace " << synapseName << "_weightupdate_synapseDynamics;" << ENDL;
            }

            // Create iterators to iterate over the names of the weight update model's derived parameters
            auto wuDerivedParams = wu->GetDerivedParams();
            auto wuDerivedParamNameBegin= GetPairKeyConstIter(wuDerivedParams.cbegin());
            auto wuDerivedParamNameEnd = GetPairKeyConstIter(wuDerivedParams.cend());

            // Create iterators to iterate over the names of the weight update model's initial values
            auto wuVars = wu->GetVars();
            auto wuVarNameBegin = GetPairKeyConstIter(wuVars.cbegin());
            auto wuVarNameEnd = GetPairKeyConstIter(wuVars.cend());

            string SDcode= wu->GetSynapseDynamicsCode();
            substitute(SDcode, "$(t)", "t");
            if (model.synapseConnType[k] == SPARSE) { // SPARSE
                os << "for (int n= 0; n < C" << synapseName << ".connN; n++)" << OB(24) << ENDL;
                if (model.synapseGType[k] == INDIVIDUALG) {
                    // name substitute synapse var names in synapseDynamics code
                    name_substitutions(SDcode, "", wuVarNameBegin, wuVarNameEnd, synapseName + "[n]");
                }
                else {
                    // substitute initial values as constants for synapse var names in synapseDynamics code
                    value_substitutions(SDcode, wuVarNameBegin, wuVarNameEnd, model.synapseIni[k]);
                }
                // substitute parameter values for parameters in synapseDynamics code
                value_substitutions(SDcode, wu->GetParamNames(), model.synapsePara[k]);
                // substitute values for derived parameters in synapseDynamics code
                value_substitutions(SDcode, wuDerivedParamNameBegin, wuDerivedParamNameEnd, model.dsp_w[k]);
                neuron_substitutions_in_synaptic_code(SDcode, model, src, trg, offsetPre, offsetPost,
                                                      "C" + synapseName + ".preInd[n]", "C" +synapseName + ".ind[n]", "");
                SDcode= ensureFtype(SDcode, model.ftype);
                checkUnreplacedVariables(SDcode, "synapseDynamics");
                os << SDcode << ENDL;
                os << CB(24);
            }
            else { // DENSE
                os << "for (int i = 0; i < " <<  srcno << "; i++)" << OB(25);
                os << "for (int j = 0; j < " <<  trgno << "; j++)" << OB(26);
                os << "// loop through all synapses" << endl;
                // substitute initial values as constants for synapse var names in synapseDynamics code
                if (model.synapseGType[k] == INDIVIDUALG) {
                    name_substitutions(SDcode, "", wuVarNameBegin, wuVarNameEnd, synapseName + "[i*" + to_string(trgno) + "+j]");
                }
                else {
                    // substitute initial values as constants for synapse var names in synapseDynamics code
                    value_substitutions(SDcode, wuVarNameBegin, wuVarNameEnd, model.synapseIni[k]);
                }
                // substitute parameter values for parameters in synapseDynamics code
                value_substitutions(SDcode, wu->GetParamNames(), model.synapsePara[k]);
                // substitute values for derived parameters in synapseDynamics code
                value_substitutions(SDcode, wuDerivedParamNameBegin, wuDerivedParamNameEnd, model.dsp_w[k]);
                neuron_substitutions_in_synaptic_code(SDcode, model, src, trg, offsetPre, offsetPost, "i", "j", "");
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
    for (unsigned int i = 0; i < model.synapseGrpN; i++) {
        if (model.synapseConnType[i] == SPARSE) {
            os << "unsigned int npost;" << ENDL;
            break;
        }
    }
    os << model.ftype << " addtoinSyn;" << ENDL;  
    os << ENDL;

    for (unsigned int i = 0; i < model.synapseGrpN; i++) {
        unsigned int src = model.synapseSource[i];
        unsigned int trg = model.synapseTarget[i];

        os << "// synapse group " << model.synapseName[i] << ENDL;
        os << OB(1006);

        if (model.neuronDelaySlots[src] > 1) {
            os << "unsigned int delaySlot = (spkQuePtr" << model.neuronName[src];
            os << " + " << (model.neuronDelaySlots[src] - model.synapseDelay[i]);
            os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
        }

        // generate the code for processing spike-like events
        if (model.synapseUsesSpikeEvents[i]) {
            generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, "Evnt");
        }

        // generate the code for processing true spike events
        if (model.synapseUsesTrueSpikes[i]) {
            generate_process_presynaptic_events_code_CPU(os, model, src, trg, i, "");
        }

        os << CB(1006);
        os << ENDL;
    }
    os << CB(1001);
    os << ENDL;


    //////////////////////////////////////////////////////////////
    // function for learning synapses, post-synaptic spikes

    if (model.lrnGroups > 0) {

        os << "void learnSynapsesPostHost(" << model.ftype << " t)" << ENDL;
        os << OB(811);

        os << "unsigned int ipost;" << ENDL;
        os << "unsigned int ipre;" << ENDL;
        os << "unsigned int lSpk;" << ENDL;
        for (unsigned int i = 0; i < model.synapseGrpN; i++) {
            if (model.synapseConnType[i] == SPARSE) {
                os << "unsigned int npre;" << ENDL;
                break;
            }
        }
        os << ENDL;

        for (unsigned int i = 0; i < model.lrnGroups; i++) {
            unsigned int k = model.lrnSynGrp[i];
            unsigned int src = model.synapseSource[k];
            unsigned int trg = model.synapseTarget[k];
            const auto *wu = model.synapseModel[i];
            bool sparse = model.synapseConnType[k] == SPARSE;

            bool delayPre = model.neuronDelaySlots[src] > 1;
            string offsetPre = (delayPre ? "(delaySlot * " + to_string(model.neuronN[src]) + ") + " : "");
            string offsetTrueSpkPre = (model.neuronNeedTrueSpk[src] ? offsetPre : "");

            bool delayPost = model.neuronDelaySlots[trg] > 1;
            string offsetPost = (delayPost ? "(spkQuePtr" + model.neuronName[trg] + " * " + to_string(model.neuronN[trg]) + ") + " : "");
            string offsetTrueSpkPost = (model.neuronNeedTrueSpk[trg] ? offsetPost : "");

            // Create iterators to iterate over the names of the weight update model's derived parameters
            auto wuDerivedParams = wu->GetDerivedParams();
            auto wuDerivedParamNameBegin= GetPairKeyConstIter(wuDerivedParams.cbegin());
            auto wuDerivedParamNameEnd = GetPairKeyConstIter(wuDerivedParams.cend());

            // Create iterators to iterate over the names of the weight update model's extra global parameters
            auto wuExtraGlobalParams = wu->GetExtraGlobalParams();
            auto wuExtraGlobalParamsNameBegin = GetPairKeyConstIter(wuExtraGlobalParams.cbegin());
            auto wuExtraGlobalParamsNameEnd = GetPairKeyConstIter(wuExtraGlobalParams.cend());

            // Create iterators to iterate over the names of the weight update model's initial values
            auto wuVars = wu->GetVars();
            auto wuVarNameBegin = GetPairKeyConstIter(wuVars.cbegin());
            auto wuVarNameEnd = GetPairKeyConstIter(wuVars.cend());

// NOTE: WE DO NOT USE THE AXONAL DELAY FOR BACKWARDS PROPAGATION - WE CAN TALK ABOUT BACKWARDS DELAYS IF WE WANT THEM

            os << "// synapse group " << model.synapseName[k] << ENDL;
            os << OB(950);

            if (delayPre) {
                os << "unsigned int delaySlot = (spkQuePtr" << model.neuronName[src];
                os << " + " << (model.neuronDelaySlots[src] - model.synapseDelay[k]);
                os << ") % " << model.neuronDelaySlots[src] << ";" << ENDL;
            }

            if (!wu->GetLearnPostSupportCode().empty()) {
                os << "using namespace " << model.synapseName[k] << "_weightupdate_simLearnPost;" << ENDL;
            }

            if (delayPost && model.neuronNeedTrueSpk[trg]) {
                os << "for (ipost = 0; ipost < glbSpkCnt" << model.neuronName[trg] << "[spkQuePtr" << model.neuronName[trg] << "]; ipost++)" << OB(910);
            }
            else {
                os << "for (ipost = 0; ipost < glbSpkCnt" << model.neuronName[trg] << "[0]; ipost++)" << OB(910);
            }

            os << "lSpk = glbSpk" << model.neuronName[trg] << "[" << offsetTrueSpkPost << "ipost];" << ENDL;

            if (sparse) { // SPARSE
                // TODO: THIS NEEDS CHECKING AND FUNCTIONAL C.POST* ARRAYS
                os << "npre = C" << model.synapseName[k] << ".revIndInG[lSpk + 1] - C" << model.synapseName[k] << ".revIndInG[lSpk];" << ENDL;
                os << "for (int l = 0; l < npre; l++)" << OB(121);
                os << "ipre = C" << model.synapseName[k] << ".revIndInG[lSpk] + l;" << ENDL;
            }
            else { // DENSE
                os << "for (ipre = 0; ipre < " << model.neuronN[src] << "; ipre++)" << OB(121);
            }

            string code = wu->GetLearnPostCode();
            substitute(code, "$(t)", "t");
            // Code substitutions ----------------------------------------------------------------------------------
            if (sparse) { // SPARSE
                name_substitutions(code, "", wuVarNameBegin, wuVarNameEnd, model.synapseName[k] + "[C" + model.synapseName[k] + ".remap[ipre]]");
            }
            else { // DENSE
                name_substitutions(code, "", wuVarNameBegin, wuVarNameEnd, model.synapseName[k] + "[lSpk + " + to_string(model.neuronN[trg]) + " * ipre]");
            }
            value_substitutions(code, wu->GetParamNames(), model.synapsePara[k]);
            value_substitutions(code, wuDerivedParamNameBegin, wuDerivedParamNameEnd, model.dsp_w[k]);
            name_substitutions(code, "", wuExtraGlobalParamsNameBegin, wuExtraGlobalParamsNameEnd, model.synapseName[k]);

            // presynaptic neuron variables and parameters
            if (sparse) { // SPARSE
                neuron_substitutions_in_synaptic_code(code, model, src, trg, offsetPre, offsetPost, "C" + model.synapseName[k] + ".revInd[ipre]", "lSpk", "");
            }
            else { // DENSE
                neuron_substitutions_in_synaptic_code(code, model, src, trg, offsetPre, offsetPost, "ipre", "lSpk", "");
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
