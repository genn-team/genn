/*--------------------------------------------------------------------------
   Author: Alan Diamond
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------


This file contains the network model definition for the "Schmuker_2014_classifier" model.

-------------------------------------------------------------------------- */


/*--------------------------------------------------------------------------
 Util to extract a model parameter value from named file
 -------------------------------------------------------------------------- */
#include <stdlib.h> // atoi
#include <fstream> //ifstream
//--------------------------------------------------------------------------
using namespace std;
////number of VR generated to map the input space
static const unsigned int global_NumVR = 40;

//network sizes and parameters
#define DT 0.5  //This defines the global time step at which the simulation will run
#define DISTINCT_SAMPLES_PER_RECORDING 4 //NB: "Max levels only" uses 1 sample for each recording
#define NUM_SENSORS_RECORDED 12 //How many sensors were originally recorded
#define NUM_SENSORS_CHOSEN 4 //how many sensors were included in the (potentially subsampled) data set (1-12)
#define NUM_FEATURES (NUM_SENSORS_CHOSEN * DISTINCT_SAMPLES_PER_RECORDING) //dimensionality of data set
#define NUM_CLASSES 20 //number of classes to be classified
#define NETWORK_SCALE 10 //single parameter to scale size of populations up and down
//#define CLUST_SIZE_AN  (NETWORK_SCALE * 8) //output cluster size
//#define CLUST_SIZE_PN  (NETWORK_SCALE * 7) //projection neuron cluster size
#define CLUST_SIZE_AN  (NETWORK_SCALE * 6) //output cluster size
#define CLUST_SIZE_PN  (NETWORK_SCALE * 6) //projection neuron cluster size
#define CLUST_SIZE_RN  (NETWORK_SCALE * 6) //receptor neuron cluster size

//Synapse time constants in ms (controls how fast arriving charge drains out of synapse into post-syn. neuron)
#define SYNAPSE_TAU_RNPN 1.0
#define SYNAPSE_TAU_PNPN 5.5
#define SYNAPSE_TAU_PNAN 1.0
#define SYNAPSE_TAU_ANAN 8.0

//switchable flags
//#define USE_SPARSE_ENCODING 1

#include "modelSpec.h"
#include "modelSpec.cc"
//#include "LIFModel.cc"

/*--------------------------------------------------------------------------
 This function defines the Enose classifier based on the Schmuker_2014_classifier model
-------------------------------------------------------------------------- */

void modelDefinition(NNmodel &model) 
{

	cout << "GeNN building model with " << global_NumVR << " x VRs" << endl;
  	model.setName("Enose_classifier");
  	
  	/*--------------------------------------------------------------------------*/
  
    //DEFINE NEURON POPULATIONS ..
      
	/*--------------------------------------------------------------------------
 	RN receptor neuron Population. Clusters of Poisson neurons take rate level input from set of VR
	-------------------------------------------------------------------------- */
	
	float poissonRN_params[4]= {
  	0.1,        // 0 - firing rate
  	2.5,        // 1 - refractory period
  	20.0,       // 2 - Vspike
  	-60.0       // 3 - Vrest
	};

	float poissonRN_ini[3]= { //initial values for the neron variables
 	-60.0,        // 0 - V
  	0,           // 1 - seed
  	-10.0,       // 2 - SpikeTime
	};
	int countRN = global_NumVR * CLUST_SIZE_RN;
  	model.addNeuronPopulation("RN", countRN, POISSONNEURON, poissonRN_params,  poissonRN_ini);
  
  
	/*--------------------------------------------------------------------------
 	PN projection neuron Population. Uses MAP neuron model. 
 	Clusters of PN neurons take excitatory input 1:1 from RN clusters, 
 	whilst conducting an weak WTA among themselves
	-------------------------------------------------------------------------- */
	
	float stdMAP_params[4]= {
	60.0,          // 0 - Vspike: spike Amplitude factor
	3.0,           // 1 - alpha: "steepness / size" parameter
	-2.468,        // 2 - y: "shift / excitation" parameter
	0.0165         // 3 - beta: input sensitivity
	};

	float stdMAP_ini[2]= {
	-60.0,         // 0 - V: initial value for membrane potential
	-60.0          // 1 - preV: initial previous value
	};
	int countPN = global_NumVR * CLUST_SIZE_PN;
    model.addNeuronPopulation("PN", countPN, MAPNEURON, stdMAP_params,  stdMAP_ini);
    	
	/*--------------------------------------------------------------------------
 	AN output Association Neuron population. Uses MAP neuron model. 
 	Clusters of AN neurons, each representing an output class take excitatory input from all PN clusters, 
 	whilst conducting an strong WTA among themselves
	-------------------------------------------------------------------------- */
    int countAN = NUM_CLASSES * CLUST_SIZE_AN;
    model.addNeuronPopulation("AN", countAN, MAPNEURON, stdMAP_params,  stdMAP_ini);
    
    //DEFINE SYNAPSES ..
    
    /*--------------------------------------------------------------------------
 	Define RN to PN Synapses. These are fixed weight, excitatory. cluster-cluster 1:1 connections, with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining cluster-cluster 1:1 connections are generated and loaded in the initialisation of the classifier class
 	Note that this connectivity will move to SPARSE data structure when available
	-------------------------------------------------------------------------- */
	
	float synapsesStdExcitatory_params[3]= {
  	0.0,           		// 0 - Erev: Reversal potential (0V for excitatory synapses)
  	-20.0,         		// 1 - Epre: Presynaptic threshold potential
  	SYNAPSE_TAU_RNPN 	// 2 - tau_S: decay time constant for S [ms]
	};

#ifdef USE_SPARSE_ENCODING
	model.addSynapsePopulation("RNPN", NSYNAPSE, SPARSE, INDIVIDUALG, "RN", "PN", synapsesStdExcitatory_params);
	model.setMaxConn("RNPN", countRN * CLUST_SIZE_PN);// max is that every RN connects to every PN in the corresponding cluster
#else
	//DENSE
	model.addSynapsePopulation("RNPN", NSYNAPSE, DENSE, INDIVIDUALG, "RN", "PN", synapsesStdExcitatory_params);
#endif
	
    /*--------------------------------------------------------------------------
 	Define PN to PN Synapses. These are fixed weight, inhibitory synapses. configured as a weak WTA between clusters, with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
	-------------------------------------------------------------------------- */

	//Average inbitory synapse (created from mid point of strong and weak examples)
	float synapsesWTA_AvgInhibitory_params[4]= {
  	-92.0,        		// 0 - Erev: Reversal potential
  	-35.0,        		// 1 - Epre: Presynaptic threshold potential (strong -40, weak -30)
  	SYNAPSE_TAU_PNPN, 	// 2 - tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
  	50.0          		// 3 - Vslope: Activation slope of graded release
	};
	//model.addSynapsePopulation("PNPN", NGRADSYNAPSE, DENSE, INDIVIDUALG, "PN", "PN", synapsesWTA_AvgInhibitory_params);
	model.addSynapsePopulation("PNPN", NSYNAPSE, DENSE, INDIVIDUALG, "PN", "PN", synapsesWTA_AvgInhibitory_params);
	
	/*--------------------------------------------------------------------------
 	Define PN to AN Synapses. These are plastic, excitatory. all-all connections, but with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining connections are generated and loaded in the initialisation of the classifier class
 	Initial weight values are set randomly between upper and lower limits
 	Weights are altered on the CPU by a learning rule between time steps and revised matrix uploaded to the GPU
	-------------------------------------------------------------------------- */
    	
	float synapsesWeakExcitatory_params[3]= {
  	0.0,           		// 0 - Erev: Reversal potential (0V for excitatory synapses)
  	-20.0,         		// 1 - Epre: Presynaptic threshold potential
  	SYNAPSE_TAU_PNAN	// 2 - tau_S: decay time constant for S [ms] (was 1.0)
	};
	model.addSynapsePopulation("PNAN", NSYNAPSE, DENSE, INDIVIDUALG, "PN", "AN", synapsesWeakExcitatory_params);
	
	/*--------------------------------------------------------------------------
	Define AN to AN Synapses. These are fixed weight, inhibitory synapses. configured as a strong WTA between output class clusters, with N% connectivity (e.g. 50%)
	NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
	-------------------------------------------------------------------------- */

	float synapsesWTA_StrongInhibitory_params[4]= {
	-92.0,        		// 0 - Erev: Reversal potential
	-35.0,        		// 1 - Epre: Presynaptic threshold potential (strong -40, weak -30)
	SYNAPSE_TAU_ANAN,	// 2 - tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
	50.0          		// 3 - Vslope: Activation slope of graded release
	};
	//model.addSynapsePopulation("ANAN", NGRADSYNAPSE, DENSE, INDIVIDUALG, "AN", "AN", synapsesWTA_StrongInhibitory_params);
	model.addSynapsePopulation("ANAN", NSYNAPSE, DENSE, INDIVIDUALG, "AN", "AN", synapsesWTA_StrongInhibitory_params);


}

/*--------------------------------------------------------------------------
  END
-------------------------------------------------------------------------- */
