
#ifndef POSTSYNAPSEMODELS_H
#define POSTSYNAPSEMODELS_H

#include "dpclass.h"

#include <string>
#include <vector>
#include <cmath>

using namespace std;


/*! \brief Class to hold the information that defines a post-synaptic model (a model of how synapses affect post-synaptic neuron variables, classically in the form of a synaptic current). It also allows to define an equation for the dynamics that can be applied to the summed synaptic input variable "insyn".
 */

class postSynModel
{
public:
    string postSyntoCurrent; //!< \brief Code that defines how postsynaptic update is translated to current 
    string postSynDecay; //!< \brief Code that defines how postsynaptic current decays 
    string supportCode; //!< \brief Support code is made available within the neuron kernel definition file and is meant to contain user defined device functions that are used in the neuron codes. Preprocessor defines are also allowed if appropriately safeguarded against multiple definition by using ifndef; functions should be declared as "__host__ __device__" to be available for both GPU and CPU versions
    vector<string> varNames; //!< Names of the variables in the postsynaptic model
    vector<string> varTypes; //!< Types of the variable named above, e.g. "float". Names and types are matched by their order of occurrence in the vector.
    vector<string> pNames; //!< Names of (independent) parameters of the model. 
    vector<string> dpNames; //!< \brief Names of dependent parameters of the model. 
    dpclass *dps; //!< \brief Derived parameters 
};


// Post-Synapse Types
extern vector<postSynModel> postSynModels; //!< Global C++ vector containing all post-synaptic update model descriptions
extern unsigned int EXPDECAY; //default - exponential decay
extern unsigned int IZHIKEVICH_PS; //empty postsynaptic rule for the Izhikevich model.
const unsigned int MAXPOSTSYN = 2; // maximum number of postsynaptic integration: SpineML needs to know this


//--------------------------------------------------------------------------
//! \brief Class defining the dependent parameter for exponential decay.
//--------------------------------------------------------------------------

class expDecayDp : public dpclass
{
public:
    double calculateDerivedParameter(int index, vector<double> pars, double dt = 1.0) {
	switch (index) {
	case 0:
	    return expDecay(pars, dt);
	}
	return -1;
    }

    double expDecay(vector<double> pars, double dt) {
	return exp(-dt/pars[0]);
    }
};


//--------------------------------------------------------------------------
/*! \brief Function that prepares the standard post-synaptic models, including their variables, parameters, dependent parameters and code strings.
 */
//--------------------------------------------------------------------------

void preparePostSynModels();

#endif // POSTSYNAPSEMODELS_H
