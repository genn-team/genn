
class synapseDP
{
public:
    virtual double calculateDerivedParameter(int index, vector<double> pars, double dt = 1.0) { return -1; }
};

/*! \brief Structure to hold the information that defines a weightupdate model (a model of how spikes affect synaptic (and/or) (mostly) post-synaptic neuron variables. It also allows to define changes in response to post-synaptic spikes/spike-like events.
 */

class weightUpdateModel
{
public:
    string simCode; //!< \brief Simulation code that is used for true spikes (only one time step after spike detection)
    string simCodeEvnt; //!< \brief Simulation code that is used for spike events (all the instances where event threshold condition is met)
    string simLearnPost; //!< \brief Simulation code which is used in the learnSynapsesPost kernel/function, where postsynaptic neuron spikes before the presynaptic neuron in the STDP window.
    string evntThreshold; //!< \brief Simulation code for spike event detection.
    string synapseDynamics; //!< \brief Simulation code for synapse dynamics independent of spike detection
    string simCode_supportCode; //!< \brief Support code is made available within the synapse kernel definition file and is meant to contain user defined device functions that are used in the neuron codes. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions; note that this support code is available to simCode, evntThreshold and simCodeEvnt
    string simLearnPost_supportCode; //!< \brief Support code is made available within the synapse kernel definition file and is meant to contain user defined device functions that are used in the neuron codes. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions
    string synapseDynamics_supportCode; //!< \brief Support code is made available within the synapse kernel definition file and is meant to contain user defined device functions that are used in the neuron codes. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions
    vector<string> varNames; //!< \brief Names of the variables in the postsynaptic model
    vector<string> varTypes; //!< \brief Types of the variable named above, e.g. "float". Names and types are matched by their order of occurrence in the vector.
    vector<string> pNames; //!< \brief Names of (independent) parameters of the model. 
    vector<string> dpNames; //!< \brief Names of dependent parameters of the model. 

    vector<string> extraGlobalSynapseKernelParameters; //!< Additional parameter in the neuron kernel; it is translated to a population specific name but otherwise assumed to be one parameter per population rather than per synapse.

    vector<string> extraGlobalSynapseKernelParameterTypes; //!< Additional parameters in the neuron kernel; they are translated to a population specific name but otherwise assumed to be one parameter per population rather than per synapse.
    synapseDP *dps;
    bool needPreSt; //!< \brief Whether presynaptic spike times are needed or not
    bool needPostSt; //!< \brief Whether postsynaptic spike times are needed or not
};


//--------------------------------------------------------------------------
/*! This class defines derived parameters for the learn1synapse standard 
    weightupdate model
*/
//--------------------------------------------------------------------------

class pwSTDP : public synapseDP //!TODO This class definition may be code-generated in a future release
{
public:
    double calculateDerivedParameter(int index, vector<double> pars, 
				    double dt)
    {		
	switch (index) {
	case 0:
	    return lim0(pars, dt);
	case 1:
	    return lim1(pars, dt);
	case 2:
	    return slope0(pars, dt);
	case 3:
	    return slope1(pars, dt);
	case 4:
	    return off0(pars, dt);
	case 5:
	    return off1(pars, dt);
	case 6:
	    return off2(pars, dt);
	}
	return -1;
    }
    
    double lim0(vector<double> pars, double dt) {
	return (1/pars[4] + 1/pars[1]) * pars[0] / (2/pars[1]);
    }
    double lim1(vector<double> pars, double dt) {
	return -((1/pars[3] + 1/pars[1]) * pars[0] / (2/pars[1]));
    }
    double slope0(vector<double> pars, double dt) {
	return -2*pars[5]/(pars[1]*pars[0]); 
    }
    double slope1(vector<double> pars, double dt) {
	return -1*slope0(pars, dt);
    }
    double off0(vector<double> pars, double dt) {
	return pars[5]/pars[4];
    }
    double off1(vector<double> pars, double dt) {
	return pars[5]/pars[1];
    }
    double off2(vector<double> pars, double dt) {
	return pars[5]/pars[3];
    }
};


vector<weightUpdateModel> weightUpdateModels; //!< Global C++ vector containing all weightupdate model descriptions

//--------------------------------------------------------------------------
/*! \brief Function that prepares the standard (pre) synaptic models, including their variables, parameters, dependent parameters and code strings.
 */
//--------------------------------------------------------------------------

void prepareWeightUpdateModels()
{
    weightUpdateModel wuN, wuG, wuL;
    
    // NSYNAPSE weightupdate model: "normal" pulse coupling synapse
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
    wuL.needPreSt= TRUE;
    wuL.needPostSt= TRUE;
    weightUpdateModels.push_back(wuL);
    LEARN1SYNAPSE= weightUpdateModels.size()-1; 

#include "extra_weightupdates.h"
}
