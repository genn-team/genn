

  // Add new neuron type - LIntF: 
  n.varNames.clear();
  n.varTypes.clear();
  
  n.varNames.push_back("V");
  n.varTypes.push_back("float");
  n.varNames.push_back("V_NB");
  n.varTypes.push_back("float");
  n.varNames.push_back("tSpike_NB");
  n.varTypes.push_back("float");
  n.varNames.push_back("__regime_val");
  n.varTypes.push_back("int");
  n.pNames.clear();
  
  n.pNames.push_back("VReset_NB");
  n.pNames.push_back("VThresh_NB");
  n.pNames.push_back("tRefrac_NB");
  n.pNames.push_back("VRest_NB");
  n.pNames.push_back("TAUm_NB");
  n.pNames.push_back("Cm_NB");
  n.dpNames.clear();

  n.simCode = " \
           $(V) = -1000000; \
           if ($(__regime_val)==1) { \n \
$(V_NB) += (Isyn_NB/$(Cm_NB)+($(VRest_NB)-$(V_NB))/$(TAUm_NB))*DT; \n \
                 if ($(V_NB)>$(VThresh_NB)) { \n \
$(V_NB) = $(VReset_NB); \n \
$(tSpike_NB) = t; \n \
                $(V) = 100000; \
$(__regime_val) = 2; \n \
} \n \
} \n \
if ($(__regime_val)==2) { \n \
if (t-$(tSpike_NB) > $(tRefrac_NB)) { \n \
$(__regime_val) = 1; \n \
} \n \
} \n \
";

  nModels.push_back(n);



  // Add new neuron type - regular spike: 
  n.varNames.clear();
  n.varTypes.clear();
  
  n.varNames.push_back("V");
  n.varTypes.push_back("float");
  n.varNames.push_back("count_t_NB");
  n.varTypes.push_back("float");
  n.varNames.push_back("__regime_val");
  n.varTypes.push_back("int");
  n.pNames.clear();
  
  n.pNames.push_back("max_t_NB");
  n.dpNames.clear();

  n.simCode = " \
           $(V) = -1000000; \
           if ($(__regime_val)==1) { \n \
$(count_t_NB) += (1)*DT; \n \
                 if ($(count_t_NB) > $(max_t_NB)-0.0001) { \n \
$(count_t_NB) = 0; \n \
                $(V) = 100000; \
$(__regime_val) = 1; \n \
} \n \
} \n \
";

  nModels.push_back(n);



  // Add new neuron type - LInt: 
  n.varNames.clear();
  n.varTypes.clear();
  
  n.varNames.push_back("V");
  n.varTypes.push_back("float");
  n.varNames.push_back("V_NB");
  n.varTypes.push_back("float");
  n.varNames.push_back("tSpike_NB");
  n.varTypes.push_back("float");
  n.varNames.push_back("__regime_val");
  n.varTypes.push_back("int");
  n.pNames.clear();
  
  n.pNames.push_back("VReset_NB");
  n.pNames.push_back("VThresh_NB");
  n.pNames.push_back("tRefrac_NB");
  n.pNames.push_back("VRest_NB");
  n.pNames.push_back("TAUm_NB");
  n.pNames.push_back("Cm_NB");
  n.dpNames.clear();

  n.simCode = " \
           $(V) = -1000000; \
           if ($(__regime_val)==1) { \n \
$(V_NB) += (Isyn_NB/$(Cm_NB)+($(VRest_NB)-$(V_NB))/$(TAUm_NB))*DT; \n \
                 if ($(V_NB)>$(VThresh_NB)) { \n \
$(V_NB) = $(VReset_NB); \n \
$(tSpike_NB) = t; \n \
                $(V) = 100000; \
$(__regime_val) = 2; \n \
} \n \
} \n \
if ($(__regime_val)==2) { \n \
if (t-$(tSpike_NB) > $(tRefrac_NB)) { \n \
$(__regime_val) = 1; \n \
} \n \
} \n \
";

  nModels.push_back(n);

