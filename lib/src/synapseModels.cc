
// Synapse Types
vector<weightUpdateModel> weightUpdateModels; //!< Global C++ vector containing all weightupdate model descriptions
unsigned int NSYNAPSE; //!< Variable attaching  the name NSYNAPSE to the non-learning synapse
unsigned int NGRADSYNAPSE; //!< Variable attaching  the name NGRADSYNAPSE to the graded synapse wrt the presynaptic voltage
unsigned int LEARN1SYNAPSE; //!< Variable attaching  the name LEARN1SYNAPSE to the the primitive STDP model for learning


//--------------------------------------------------------------------------
/*! \brief Function that prepares the standard (pre) synaptic models, including their variables, parameters, dependent parameters and code strings.
 */
//--------------------------------------------------------------------------

void prepareWeightUpdateModels()
{
    weightUpdateModel wu;

    // NSYNAPSE weightupdate model: "normal" pulse coupling synapse
    wu.varNames.clear();
    wu.varTypes.clear();
    wu.varNames.push_back("g");
    wu.varTypes.push_back("scalar");
    wu.pNames.clear();
    wu.dpNames.clear();
    // code for presynaptic spike:
    wu.simCode = "  $(addtoinSyn) = $(g);\n\
  $(updatelinsyn);\n";
    weightUpdateModels.push_back(wu);
    NSYNAPSE= weightUpdateModels.size()-1;
    

    // NGRADSYNAPSE weightupdate model: "normal" graded synapse
    wu.varNames.clear();
    wu.varTypes.clear();
    wu.varNames.push_back("g");
    wu.varTypes.push_back("scalar");
    wu.pNames.clear();
    wu.pNames.push_back("Epre"); 
    wu.pNames.push_back("Vslope"); 
    wu.dpNames.clear();
    // code for presynaptic spike event 
    wu.simCodeEvnt = "$(addtoinSyn) = $(g) * tanh(($(V_pre) - $(Epre)) / $(Vslope))* DT;\n\
    if ($(addtoinSyn) < 0) $(addtoinSyn) = 0.0;\n\
    $(updatelinsyn);\n";
    // definition of presynaptic spike event 
    wu.evntThreshold = "$(V_pre) > $(Epre)";
    weightUpdateModels.push_back(wu);
    NGRADSYNAPSE= weightUpdateModels.size()-1; 


    // LEARN1SYNAPSE weightupdate model: "normal" synapse with a type of STDP
    wu.varNames.clear();
    wu.varTypes.clear();
    wu.varNames.push_back("g");
    wu.varTypes.push_back("scalar");
    wu.varNames.push_back("gRaw"); 
    wu.varTypes.push_back("scalar");
    wu.pNames.clear();
    wu.pNames.push_back("tLrn");  //0
    wu.pNames.push_back("tChng"); //1
    wu.pNames.push_back("tDecay"); //2
    wu.pNames.push_back("tPunish10"); //3
    wu.pNames.push_back("tPunish01"); //4
    wu.pNames.push_back("gMax"); //5
    wu.pNames.push_back("gMid"); //6
    wu.pNames.push_back("gSlope"); //7
    wu.pNames.push_back("tauShift"); //8
    wu.pNames.push_back("gSyn0"); //9
    wu.dpNames.clear(); 
    wu.dpNames.push_back("lim0");
    wu.dpNames.push_back("lim1");
    wu.dpNames.push_back("slope0");
    wu.dpNames.push_back("slope1");
    wu.dpNames.push_back("off0");
    wu.dpNames.push_back("off1");
    wu.dpNames.push_back("off2");
    // code for presynaptic spike
    wu.simCode = "$(addtoinSyn) = $(g);\n\
  $(updatelinsyn); \n				\
  scalar dt = $(sT_post) - $(t) - ($(tauShift)); \n	\
  scalar dg = 0;\n				\
  if (dt > $(lim0))  \n				\
      dg = -($(off0)) ; \n			\
  else if (dt > 0)  \n			\
      dg = $(slope0) * dt + ($(off1)); \n\
  else if (dt > $(lim1))  \n			\
      dg = $(slope1) * dt + ($(off1)); \n\
  else dg = - ($(off2)) ; \n\
  $(gRaw) += dg; \n\
  $(g)=$(gMax)/2 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n";
    wu.dps = new pwSTDP;
    // code for post-synaptic spike 
    wu.simLearnPost = "scalar dt = $(t) - ($(sT_pre)) - ($(tauShift)); \n\
  scalar dg =0; \n\
  if (dt > $(lim0))  \n\
      dg = -($(off0)) ; \n \
  else if (dt > 0)  \n\
      dg = $(slope0) * dt + ($(off1)); \n\
  else if (dt > $(lim1))  \n\
      dg = $(slope1) * dt + ($(off1)); \n\
  else dg = -($(off2)) ; \n\
  $(gRaw) += dg; \n\
  $(g)=$(gMax)/2.0 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n";
    wu.needPreSt= TRUE;
    wu.needPostSt= TRUE;
    weightUpdateModels.push_back(wu);
    LEARN1SYNAPSE= weightUpdateModels.size()-1; 
}
