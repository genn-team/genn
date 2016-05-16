
#ifndef NEURONMODELS_H
#define NEURONMODELS_H

#include "dpclass.h"

#include <string>
#include <vector>

using namespace std;


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
    dpclass *dps; //!< \brief Derived parameters
    bool needPreSt; //!< \brief Whether presynaptic spike times are needed or not
    bool needPostSt; //!< \brief Whether postsynaptic spike times are needed or not
};


// Neuron Types
extern vector<neuronModel> nModels; //!< Global C++ vector containing all neuron model descriptions
extern unsigned int MAPNEURON; //!< variable attaching the name "MAPNEURON" 
extern unsigned int POISSONNEURON; //!< variable attaching the name "POISSONNEURON" 
extern unsigned int TRAUBMILES_FAST; //!< variable attaching the name "TRAUBMILES_FAST" 
extern unsigned int TRAUBMILES_ALTERNATIVE; //!< variable attaching the name "TRAUBMILES_ALTERNATIVE" 
extern unsigned int TRAUBMILES_SAFE; //!< variable attaching the name "TRAUBMILES_SAFE" 
extern unsigned int TRAUBMILES; //!< variable attaching the name "TRAUBMILES" 
extern unsigned int TRAUBMILES_PSTEP;//!< variable attaching the name "TRAUBMILES_PSTEP" 
extern unsigned int IZHIKEVICH; //!< variable attaching the name "IZHIKEVICH" 
extern unsigned int IZHIKEVICH_V; //!< variable attaching the name "IZHIKEVICH_V" 
extern unsigned int SPIKESOURCE; //!< variable attaching the name "SPIKESOURCE"
const unsigned int MAXNRN = 7; // maximum number of neuron types: SpineML needs to know this


//--------------------------------------------------------------------------
//! \brief Class defining the dependent parameters of the Rulkov map neuron.
//--------------------------------------------------------------------------

class rulkovdp : public dpclass
{
public:
    double calculateDerivedParameter(int index, vector<double> pars, double dt = 1.0) {
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


//--------------------------------------------------------------------------
/*! \brief Function that defines standard neuron models

  The neuron models are defined and added to the C++ vector nModels that is holding all neuron model descriptions. User defined neuron models can be appended to this vector later in (a) separate function(s).
*/

void prepareStandardModels();

#endif // NEURONMODELS_H
