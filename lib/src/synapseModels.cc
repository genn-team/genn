
#ifndef SYNAPSEMODELS_CC
#define SYNAPSEMODELS_CC

#include "synapseModels.h"
#include "stringUtils.h"


//--------------------------------------------------------------------------
/*! \brief Constructor for weightUpdateModel objects
 */
//--------------------------------------------------------------------------

weightUpdateModel::weightUpdateModel()
{
    dps = NULL;
    needPreSt = false;
    needPostSt = false;
}


//--------------------------------------------------------------------------
/*! \brief Destructor for weightUpdateModel objects
 */
//--------------------------------------------------------------------------

weightUpdateModel::~weightUpdateModel() {}


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
    // NSYNAPSE weightupdate model: "normal" pulse coupling synapse
    weightUpdateModel wuN;
    wuN.varNames.clear();
    wuN.varTypes.clear();
    wuN.varNames.push_back("g");
    wuN.varTypes.push_back("scalar");
    wuN.pNames.clear();
    wuN.dpNames.clear();
    // code for presynaptic spike:
    wuN.simCode = "  $(addtoinSyn) = $(g);\n\
  $(updatelinsyn);\n";
    weightUpdateModels.push_back(wuN);
    NSYNAPSE= weightUpdateModels.size()-1;
    

    // NGRADSYNAPSE weightupdate model: "normal" graded synapse
    weightUpdateModel wuG;
    wuG.varNames.clear();
    wuG.varTypes.clear();
    wuG.varNames.push_back("g");
    wuG.varTypes.push_back("scalar");
    wuG.pNames.clear();
    wuG.pNames.push_back("Epre"); 
    wuG.pNames.push_back("Vslope"); 
    wuG.dpNames.clear();
    // code for presynaptic spike event 
    wuG.simCodeEvnt = "$(addtoinSyn) = $(g) * tanh(($(V_pre) - $(Epre)) / $(Vslope))* DT;\n\
    if ($(addtoinSyn) < 0) $(addtoinSyn) = 0.0;\n\
    $(updatelinsyn);\n";
    // definition of presynaptic spike event 
    wuG.evntThreshold = "$(V_pre) > $(Epre)";
    weightUpdateModels.push_back(wuG);
    NGRADSYNAPSE= weightUpdateModels.size()-1; 


    // LEARN1SYNAPSE weightupdate model: "normal" synapse with a type of STDP
    weightUpdateModel wuL;
    wuL.varNames.clear();
    wuL.varTypes.clear();
    wuL.varNames.push_back("g");
    wuL.varTypes.push_back("scalar");
    wuL.varNames.push_back("gRaw"); 
    wuL.varTypes.push_back("scalar");
    wuL.pNames.clear();
    wuL.pNames.push_back("tLrn");  //0
    wuL.pNames.push_back("tChng"); //1
    wuL.pNames.push_back("tDecay"); //2
    wuL.pNames.push_back("tPunish10"); //3
    wuL.pNames.push_back("tPunish01"); //4
    wuL.pNames.push_back("gMax"); //5
    wuL.pNames.push_back("gMid"); //6
    wuL.pNames.push_back("gSlope"); //7
    wuL.pNames.push_back("tauShift"); //8
    wuL.pNames.push_back("gSyn0"); //9
    wuL.dpNames.clear(); 
    wuL.dpNames.push_back("lim0");
    wuL.dpNames.push_back("lim1");
    wuL.dpNames.push_back("slope0");
    wuL.dpNames.push_back("slope1");
    wuL.dpNames.push_back("off0");
    wuL.dpNames.push_back("off1");
    wuL.dpNames.push_back("off2");
    // code for presynaptic spike
    wuL.simCode = "$(addtoinSyn) = $(g);\n\
  $(updatelinsyn); \n                                \
  scalar dt = $(sT_post) - $(t) - ($(tauShift)); \n        \
  scalar dg = 0;\n                                \
  if (dt > $(lim0))  \n                                \
      dg = -($(off0)) ; \n                        \
  else if (dt > 0)  \n                        \
      dg = $(slope0) * dt + ($(off1)); \n\
  else if (dt > $(lim1))  \n                        \
      dg = $(slope1) * dt + ($(off1)); \n\
  else dg = - ($(off2)) ; \n\
  $(gRaw) += dg; \n\
  $(g)=$(gMax)/2 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n";
    wuL.dps = new pwSTDP;
    // code for post-synaptic spike 
    wuL.simLearnPost = "scalar dt = $(t) - ($(sT_pre)) - ($(tauShift)); \n\
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
    wuL.needPreSt= true;
    wuL.needPostSt= true;
    weightUpdateModels.push_back(wuL);
    LEARN1SYNAPSE= weightUpdateModels.size()-1; 


#include "extra_weightupdates.h"
}

#endif // SYNAPSEMODELS_CC
