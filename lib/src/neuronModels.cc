
#ifndef NEURONMODELS_CC
#define NEURONMODELS_CC

#include "neuronModels.h"

// Neuron Types
vector<neuronModel> nModels; //!< Global C++ vector containing all neuron model descriptions
unsigned int MAPNEURON; //!< variable attaching the name "MAPNEURON" 
unsigned int POISSONNEURON; //!< variable attaching the name "POISSONNEURON" 
unsigned int TRAUBMILES_FAST; //!< variable attaching the name "TRAUBMILES_FAST" 
unsigned int TRAUBMILES_ALTERNATIVE; //!< variable attaching the name "TRAUBMILES_ALTERNATIVE" 
unsigned int TRAUBMILES_SAFE; //!< variable attaching the name "TRAUBMILES_SAFE" 
unsigned int TRAUBMILES; //!< variable attaching the name "TRAUBMILES" 
unsigned int TRAUBMILES_PSTEP;//!< variable attaching the name "TRAUBMILES_PSTEP" 
unsigned int IZHIKEVICH; //!< variable attaching the name "IZHIKEVICH" 
unsigned int IZHIKEVICH_V; //!< variable attaching the name "IZHIKEVICH_V" 
unsigned int SPIKESOURCE; //!< variable attaching the name "SPIKESOURCE"


//--------------------------------------------------------------------------
/*! \brief Function that defines standard neuron models

  The neuron models are defined and added to the C++ vector nModels that is holding all neuron model descriptions. User defined neuron models can be appended to this vector later in (a) separate function(s).
*/

void prepareStandardModels()
{
    neuronModel n;

    // Rulkov neurons
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("preV");
    n.varTypes.push_back("scalar");
    n.pNames.push_back("Vspike");
    n.pNames.push_back("alpha");
    n.pNames.push_back("y");
    n.pNames.push_back("beta");
    n.dpNames.push_back("ip0");
    n.dpNames.push_back("ip1");
    n.dpNames.push_back("ip2");
    n.simCode= "    if ($(V) <= 0) {\n\
      $(preV)= $(V);\n\
      $(V)= $(ip0)/(($(Vspike)) - $(V) - ($(beta))*$(Isyn)) +($(ip1));\n\
    }\n\
    else {\n\
      if (($(V) < $(ip2)) && ($(preV) <= 0)) {\n\
        $(preV)= $(V);\n\
        $(V)= $(ip2);\n\
      }\n\
      else {\n\
        $(preV)= $(V);\n\
        $(V)= -($(Vspike));\n\
      }\n\
    }\n";
    n.thresholdConditionCode = "$(V) >= $(ip2)";
    n.dps = new rulkovdp();
    nModels.push_back(n);
    MAPNEURON= nModels.size()-1;


    // Poisson neurons
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("seed");
    n.varTypes.push_back("uint64_t");
    n.varNames.push_back("spikeTime");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.pNames.push_back("therate");
    n.pNames.push_back("trefract");
    n.pNames.push_back("Vspike");
    n.pNames.push_back("Vrest");
    n.dpNames.clear();
    n.extraGlobalNeuronKernelParameters.push_back("rates");
    n.extraGlobalNeuronKernelParameterTypes.push_back("uint64_t *");
    n.extraGlobalNeuronKernelParameters.push_back("offset");
    n.extraGlobalNeuronKernelParameterTypes.push_back("unsigned int");
    n.simCode= "    uint64_t theRnd;\n\
    if ($(V) > $(Vrest)) {\n\
      $(V)= $(Vrest);\n\
    }\n\
    else {\n\
      if ($(t) - $(spikeTime) > ($(trefract))) {\n\
        MYRAND($(seed),theRnd);\n\
        if (theRnd < *($(rates)+$(offset)+$(id))) {\n			\
          $(V)= $(Vspike);\n\
          $(spikeTime)= $(t);\n\
        }\n\
      }\n\
    }\n";
    n.thresholdConditionCode = "$(V) >= $(Vspike)";
    n.dps= NULL;
    nModels.push_back(n);
    POISSONNEURON= nModels.size()-1;


    // Traub and Miles HH neurons TRAUBMILES_FAST - Original fast implementation, using 25 inner iterations. There are singularities in this model, which can be  easily hit in float precision.  
    n.varNames.clear();
    n.varTypes.clear();
    n.extraGlobalNeuronKernelParameters.clear();
    n.extraGlobalNeuronKernelParameterTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("m");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("h");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("n");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.pNames.push_back("gNa");
    n.pNames.push_back("ENa");
    n.pNames.push_back("gK");
    n.pNames.push_back("EK");
    n.pNames.push_back("gl");
    n.pNames.push_back("El");
    n.pNames.push_back("C");
    n.dpNames.clear();
    n.simCode= "   scalar Imem;\n\
    unsigned int mt;\n\
    scalar mdt= DT/25.0;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-$(Isyn));\n\
      scalar _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n\
      scalar _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n\
      $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*exp((-48.0-$(V))/18.0);\n\
      _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n\
      $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
      _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n\
      _b= 0.5*exp((-55.0-$(V))/40.0);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n";
    n.thresholdConditionCode = "$(V) > 0.0"; //TODO check this, to get better value
    n.dps= NULL;
    nModels.push_back(n);
    TRAUBMILES_FAST= nModels.size()-1;


    // Traub and Miles HH neurons TRAUBMILES_ALTERNATIVE - Using a workaround to avoid singularity: adding the munimum numerical value of the floating point precision used. 
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("m");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("h");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("n");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.pNames.push_back("gNa");
    n.pNames.push_back("ENa");
    n.pNames.push_back("gK");
    n.pNames.push_back("EK");
    n.pNames.push_back("gl");
    n.pNames.push_back("El");
    n.pNames.push_back("C");
    n.dpNames.clear();
    n.simCode= "   scalar Imem;\n\
    unsigned int mt;\n\
    scalar mdt= DT/25.0;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-$(Isyn));\n\
      scalar volatile _tmp= abs(exp((-52.0-$(V))/4.0)-1.0);\n\
      scalar _a= 0.32*abs(-52.0-$(V))/(_tmp+SCALAR_MIN);\n\
      _tmp= abs(exp(($(V)+25.0)/5.0)-1.0);\n\
      scalar _b= 0.28*abs($(V)+25.0)/(_tmp+SCALAR_MIN);\n\
      $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*exp((-48.0-$(V))/18.0);\n\
      _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n\
      $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
      _tmp= abs(exp((-50.0-$(V))/5.0)-1.0);\n\
      _a= 0.032*abs(-50.0-$(V))/(_tmp+SCALAR_MIN); \n\
      _b= 0.5*exp((-55.0-$(V))/40.0);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n";
    n.thresholdConditionCode = "$(V) > 0"; //TODO check this, to get better value
    n.dps= NULL;
    nModels.push_back(n);
    TRAUBMILES_ALTERNATIVE= nModels.size()-1;


    // Traub and Miles HH neurons TRAUBMILES_SAFE - Using IF statements to check if a value that a singularity would be hit. If so, value calculated by L'Hospital rule is used. TRAUBMILES method points to this model.
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("m");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("h");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("n");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.pNames.push_back("gNa");
    n.pNames.push_back("ENa");
    n.pNames.push_back("gK");
    n.pNames.push_back("EK");
    n.pNames.push_back("gl");
    n.pNames.push_back("El");
    n.pNames.push_back("C");
    n.dpNames.clear();
    n.simCode= "   scalar Imem;\n\
    unsigned int mt;\n\
    scalar mdt= DT/25.0;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-$(Isyn));\n\
      scalar _a;\n\
      if (lV == -52.0) _a= 1.28;\n\
      else _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n\
      scalar _b;\n\
      if (lV == -25.0) _b= 1.4;\n\
      else _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n\
      $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*exp((-48.0-$(V))/18.0);\n\
      _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n\
      $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
      if (lV == -50.0) _a= 0.16;\n\
      else _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n\
      _b= 0.5*exp((-55.0-$(V))/40.0);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n";
    n.thresholdConditionCode = "$(V) > 0.0"; //TODO check this, to get better value.
    n.dps= NULL;
    nModels.push_back(n);
    TRAUBMILES_SAFE= nModels.size()-1;
    TRAUBMILES= TRAUBMILES_SAFE;


    // Traub and Miles HH neurons TRAUBMILES_PSTEP - same as TRAUBMILES_SAFE but the number of inner loops can be set as a parameter.
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("m");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("h");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("n");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.pNames.push_back("gNa");
    n.pNames.push_back("ENa");
    n.pNames.push_back("gK");
    n.pNames.push_back("EK");
    n.pNames.push_back("gl");
    n.pNames.push_back("El");
    n.pNames.push_back("C");
    n.pNames.push_back("ntimes");
    n.dpNames.clear();
    n.simCode= "   scalar Imem;\n\
    unsigned int mt;\n\
    scalar mdt= DT/scalar($(ntimes));\n\
    for (mt=0; mt < $(ntimes); mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-$(Isyn));\n\
      scalar _a;\n\
      if (lV == -52.0) _a= 1.28;\n\
      else _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);\n\
      scalar _b;\n\
      if (lV == -25.0) _b= 1.4;\n\
      else _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);\n\
      $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*exp((-48.0-$(V))/18.0);\n\
      _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);\n\
      $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
      if (lV == -50.0) _a= 0.16;\n\
      else _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);\n\
      _b= 0.5*exp((-55.0-$(V))/40.0);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n";
    n.thresholdConditionCode = "$(V) > 0.0"; //TODO check this, to get better value
    n.dps= NULL;
    nModels.push_back(n);
    TRAUBMILES_PSTEP= nModels.size()-1;


    // Izhikevich neurons
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");  
    n.varNames.push_back("U");
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    //n.pNames.push_back("Vspike");
    n.pNames.push_back("a"); // time scale of U
    n.pNames.push_back("b"); // sensitivity of U
    n.pNames.push_back("c"); // after-spike reset value of V
    n.pNames.push_back("d"); // after-spike reset value of U
    n.dpNames.clear();
    //TODO: replace the resetting in the following with BRIAN-like threshold and resetting 
    n.simCode= "    if ($(V) >= 30.0){\n\
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n	\
   //  $(V)=30.0; \n\
   //}\n";
    n.thresholdConditionCode = "$(V) >= 29.99";
    /*  n.resetCode="//reset code is here\n";
	$(V)=$(c);\n\
	$(U)+=$(d);\n\
    */
    nModels.push_back(n);
    IZHIKEVICH= nModels.size()-1;


    // Izhikevich neurons with variable parameters
    n.varNames.clear();
    n.varTypes.clear();
    n.varNames.push_back("V");
    n.varTypes.push_back("scalar");  
    n.varNames.push_back("U");
    n.varTypes.push_back("scalar");
    n.varNames.push_back("a"); // time scale of U
    n.varTypes.push_back("scalar");
    n.varNames.push_back("b"); // sensitivity of U
    n.varTypes.push_back("scalar");
    n.varNames.push_back("c"); // after-spike reset value of V
    n.varTypes.push_back("scalar");
    n.varNames.push_back("d"); // after-spike reset value of U
    n.varTypes.push_back("scalar");
    n.pNames.clear();
    n.dpNames.clear(); 
    //TODO: replace the resetting in the following with BRIAN-like threshold and resetting 
    n.simCode= "    if ($(V) >= 30.0){\n\
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
    //if ($(V) > 30.0){      //keep this only for visualisation -- not really necessaary otherwise \n\
    //  $(V)=30.0; \n\
    //}\n";
    n.thresholdConditionCode = "$(V) > 29.99";
    n.dps= NULL;
    nModels.push_back(n);
    IZHIKEVICH_V= nModels.size()-1;
  

    // Spike Source ("empty" neuron that does nothing - spikes need to be copied in explicitly from host code)
    n.varNames.clear();
    n.varTypes.clear();
    n.pNames.clear();
    n.dpNames.clear(); 
    n.simCode= "";
    n.thresholdConditionCode = "0";
    n.dps= NULL;
    nModels.push_back(n);
    SPIKESOURCE= nModels.size()-1;


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
} \n";
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
} \n";
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
} \n";
    nModels.push_back(n);
}

#endif // NEURONMODELS_H
