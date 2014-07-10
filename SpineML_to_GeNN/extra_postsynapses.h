
			
  // Add new postsynapse type - Exp Decay: 
  ps.varNames.clear();
  ps.varTypes.clear();
  
  ps.varNames.push_back(tS("g_PS"));
  ps.varTypes.push_back(tS("float"));
  ps.pNames.clear();
  
  ps.pNames.push_back(tS("tau_syn_PS"));
  ps.pNames.push_back(tS("E_PS"));
  ps.dpNames.clear();

  ps.postSyntoCurrent = tS(" \
  0; \n \
     	float Isyn_NB = 0; \n \
     	 { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n \
");
  	 
	ps.postSynDecay = tS(" \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0; \
  	");

  postSynModels.push_back(ps);

	
			
  // Add new postsynapse type - Exp Decay: 
  ps.varNames.clear();
  ps.varTypes.clear();
  
  ps.varNames.push_back(tS("g_PS"));
  ps.varTypes.push_back(tS("float"));
  ps.pNames.clear();
  
  ps.pNames.push_back(tS("tau_syn_PS"));
  ps.pNames.push_back(tS("E_PS"));
  ps.dpNames.clear();

  ps.postSyntoCurrent = tS(" \
  0; \n \
      { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n \
");
  	 
	ps.postSynDecay = tS(" \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0; \
  	");

  postSynModels.push_back(ps);

	
			
  // Add new postsynapse type - Exp Decay: 
  ps.varNames.clear();
  ps.varTypes.clear();
  
  ps.varNames.push_back(tS("g_PS"));
  ps.varTypes.push_back(tS("float"));
  ps.pNames.clear();
  
  ps.pNames.push_back(tS("tau_syn_PS"));
  ps.pNames.push_back(tS("E_PS"));
  ps.dpNames.clear();

  ps.postSyntoCurrent = tS(" \
  0; \n \
     	float Isyn_NB = 0; \n \
     	 { \n \
    	float v_PS = lV_NB; \n \
     	 float g_in_PS = $(inSyn); \
$(g_PS) = $(g_PS)+g_in_PS; \n \
Isyn_NB += ($(g_PS)*($(E_PS)-v_PS)); \n \
	  } \n \
");
  	 
	ps.postSynDecay = tS(" \
  	 $(g_PS) += (-$(g_PS)/$(tau_syn_PS))*DT; \n \
	 		$(inSyn) = 0; \
  	");

  postSynModels.push_back(ps);

	