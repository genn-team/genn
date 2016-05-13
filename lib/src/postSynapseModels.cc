
#ifndef POSTSYNAPSEMODELS_CC
#define POSTSYNAPSEMODELS_CC

#include "postSynapseModels.h"
#include "stringUtils.h"

// Post-Synapse Types
vector<postSynModel> postSynModels; //!< Global C++ vector containing all post-synaptic update model descriptions
unsigned int EXPDECAY; //default - exponential decay
unsigned int IZHIKEVICH_PS; //empty postsynaptic rule for the Izhikevich model.


//--------------------------------------------------------------------------
/*! \brief Function that prepares the standard post-synaptic models, including their variables, parameters, dependent parameters and code strings.
 */
//--------------------------------------------------------------------------

void preparePostSynModels()
{
    postSynModel ps;

    // 0: Exponential decay
    ps.varNames.clear();
    ps.varTypes.clear();
    ps.pNames.clear();
    ps.dpNames.clear(); 
    ps.pNames.push_back("tau"); 
    ps.pNames.push_back("E");  
    ps.dpNames.push_back("expDecay");
    ps.postSynDecay= "$(inSyn)*=$(expDecay);\n";
    ps.postSyntoCurrent= "$(inSyn)*($(E)-$(V))";
    ps.dps = new expDecayDp;
    postSynModels.push_back(ps);
    EXPDECAY= postSynModels.size()-1;


    // 1: IZHIKEVICH MODEL (NO POSTSYN RULE)
    ps.varNames.clear();
    ps.varTypes.clear();
    ps.pNames.clear();
    ps.dpNames.clear(); 
    ps.postSynDecay= "";
    ps.postSyntoCurrent= "$(inSyn); $(inSyn)= 0";
    postSynModels.push_back(ps);
    IZHIKEVICH_PS= postSynModels.size()-1;


#include "extra_postsynapses.h"
}

#endif // POSTSYNAPSEMODELS_CC
