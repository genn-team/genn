
class neuronDP
{
public:
    virtual double calculateDerivedParameter(int index, vector<double> pars, double dt = 1.0) { return -1; }
};

/*! \brief class for specifying a neuron model.
 */

class neuronModel
{
public:
    string simCode; /*!< \brief Code that defines the execution of one timestep of integration of the neuron model
		      The code will refer to $(NN) for the value of the variable with name "NN". It needs to refer to the predefined variable "ISYN", i.e. contain $(ISYN), if it is to receive input. */
    string thresholdConditionCode; /*!< \brief Code evaluating to a bool (e.g. "V > 20") that defines the condition for a true spike in the described neuron model */
    string resetCode; /*!< \brief Code that defines the reset action taken after a spike occurred. This can be empty */
    string supportCode; //!< \brief Support code is made available within the neuron kernel definition file and is meant to contain user defined device functions that are used in the neuron codes. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions
    vector<string> varNames; //!< Names of the variables in the neuron model
    vector<string> tmpVarNames; //!< never used
    vector<string> varTypes; //!< Types of the variable named above, e.g. "float". Names and types are matched by their order of occurrence in the vector.
    vector<string> tmpVarTypes; //!< never used
    vector<string> pNames; //!< Names of (independent) parameters of the model. 
    vector<string> dpNames; /*!< \brief Names of dependent parameters of the model.      
			      The dependent parameters are functions of independent parameters that enter into the neuron model. To avoid unecessary computational overhead, these parameters are calculated at compile time and inserted as explicit values into the generated code. See method NNmodel::initDerivedNeuronPara for how this is done.*/ 
    vector<string> extraGlobalNeuronKernelParameters; //!< Additional parameter in the neuron kernel; it is translated to a population specific name but otherwise assumed to be one parameter per population rather than per neuron.
    vector<string> extraGlobalNeuronKernelParameterTypes; //!< Additional parameters in the neuron kernel; they are translated to a population specific name but otherwise assumed to be one parameter per population rather than per neuron.
    neuronDP *dps; //!< \brief Derived parameters
    bool needPreSt; //!< \brief Whether presynaptic spike times are needed or not
    bool needPostSt; //!< \brief Whether postsynaptic spike times are needed or not
};


//--------------------------------------------------------------------------
//! \brief Class defining the dependent parameters of the Rulkov map neuron.
//--------------------------------------------------------------------------

class rulkovdp : public neuronDP
{
public:
    double calculateDerivedParameter(int index, vector <double> pars, double dt = 1.0) {
	switch (index) {
	case 0:
	    return ip0(pars);
	case 1:
	    return ip1(pars);
	case 2:
	    return ip2(pars);
	}
	return -1;
    }

    double ip0(vector<double> pars) {
	return pars[0]*pars[0]*pars[1];
    }
    double ip1(vector<double> pars) {
	return pars[0]*pars[2];
    }
    double ip2(vector<double> pars) {
	return pars[0]*pars[1]+pars[0]*pars[2];
    }
};


vector<neuronModel> nModels; //!< Global C++ vector containing all neuron model descriptions

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

#include "extra_neurons.h"
}
