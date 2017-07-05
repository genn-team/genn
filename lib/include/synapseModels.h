
#ifndef SYNAPSEMODELS_H
#define SYNAPSEMODELS_H

#include "dpclass.h"

#include <string>
#include <vector>

using namespace std;


/*! \brief Class to hold the information that defines a weightupdate model (a model of how spikes affect synaptic (and/or) (mostly) post-synaptic neuron variables. It also allows to define changes in response to post-synaptic spikes/spike-like events.
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
    dpclass *dps;
    bool needPreSt; //!< \brief Whether presynaptic spike times are needed or not
    bool needPostSt; //!< \brief Whether postsynaptic spike times are needed or not

    weightUpdateModel(); //!< \brief Constructor for weightUpdateModel objects
    ~weightUpdateModel(); //!< \brief Destructor for weightUpdateModel objects
};


// Synapse Types
extern vector<weightUpdateModel> weightUpdateModels; //!< Global C++ vector containing all weightupdate model descriptions
extern unsigned int NSYNAPSE; //!< Variable attaching  the name NSYNAPSE to the non-learning synapse
extern unsigned int NGRADSYNAPSE; //!< Variable attaching  the name NGRADSYNAPSE to the graded synapse wrt the presynaptic voltage
extern unsigned int LEARN1SYNAPSE; //!< Variable attaching  the name LEARN1SYNAPSE to the the primitive STDP model for learning
const unsigned int SYNTYPENO = 4; // maximum number of synapse types: SpineML needs to know this


//--------------------------------------------------------------------------
/*! This class defines derived parameters for the learn1synapse standard 
  weightupdate model
*/
//--------------------------------------------------------------------------

class pwSTDP : public dpclass //!TODO This class definition may be code-generated in a future release
{
public:
    double calculateDerivedParameter(int index, vector<double> pars, double = 1.0) {
        switch (index) {
        case 0:
            return lim0(pars);
        case 1:
            return lim1(pars);
        case 2:
            return slope0(pars);
        case 3:
            return slope1(pars);
        case 4:
            return off0(pars);
        case 5:
            return off1(pars);
        case 6:
            return off2(pars);
        }
        return -1;
    }

private:
    double lim0(const vector<double> &pars) {
        return (1/pars[4] + 1/pars[1]) * pars[0] / (2/pars[1]);
    }

    double lim1(const vector<double> &pars) {
        return -((1/pars[3] + 1/pars[1]) * pars[0] / (2/pars[1]));
    }

    double slope0(const vector<double> &pars) {
        return -2*pars[5]/(pars[1]*pars[0]);
    }

    double slope1(const vector<double> &pars) {
        return -1*slope0(pars);
    }

    double off0(const vector<double> &pars) {
        return pars[5]/pars[4];
    }

    double off1(const vector<double> &pars) {
        return pars[5]/pars[1];
    }

    double off2(const vector<double> &pars) {
        return pars[5]/pars[3];
    }
};


//--------------------------------------------------------------------------
/*! \brief Function that prepares the standard (pre) synaptic models, including their variables, parameters, dependent parameters and code strings.
 */
//--------------------------------------------------------------------------

void prepareWeightUpdateModels();

#endif // SYNAPSEMODELS_H
