
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


    // Add new postsynapse type - Exp Decay: 
    ps.varNames.clear();
    ps.varTypes.clear();
    ps.varNames.push_back("g_PS");
    ps.varTypes.push_back("float");
    ps.pNames.clear();
    ps.pNames.push_back("tau_syn_PS");
    ps.pNames.push_back("E_PS");
    ps.dpNames.clear();
    ps.postSyntoCurrent = " \
  0; \n \
     	float Isyn_NB = 0; \n \
     	 { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n";
    ps.postSynDecay = " \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0;";
    postSynModels.push_back(ps);


    // Add new postsynapse type - Exp Decay: 
    ps.varNames.clear();
    ps.varTypes.clear();
    ps.varNames.push_back("g_PS");
    ps.varTypes.push_back("float");
    ps.pNames.clear();
    ps.pNames.push_back("tau_syn_PS");
    ps.pNames.push_back("E_PS");
    ps.dpNames.clear();
    ps.postSyntoCurrent = " \
  0; \n \
      { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n";
    ps.postSynDecay = " \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0;";
    postSynModels.push_back(ps);


    // Add new postsynapse type - Exp Decay: 
    ps.varNames.clear();
    ps.varTypes.clear();
    ps.varNames.push_back("g_PS");
    ps.varTypes.push_back("float");
    ps.pNames.clear();
    ps.pNames.push_back("tau_syn_PS");
    ps.pNames.push_back("E_PS");
    ps.dpNames.clear();
    ps.postSyntoCurrent = " \
  0; \n \
     	float Isyn_NB = 0; \n \
     	 { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n";
    ps.postSynDecay = " \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0;";
    postSynModels.push_back(ps);
}
