/*--------------------------------------------------------------------------
   Author: Alan Diamond
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------


This file contains the network model definition for the "Schmuker_2014_classifier" model.

-------------------------------------------------------------------------- */

//network sizes and parameters
#define DT 0.5  //This defines the global time step at which the simulation will run
#define NUM_VR 10 //number of VR generated to map the input space
#define NUM_FEATURES 4 //dimensionality of data set
#define NUM_CLASSES 3 //number of classes to be classified
#define NETWORK_SCALE 10 //single parameter to scale size of populations up and down
//#define CLUST_SIZE_AN  NETWORK_SCALE * 8 //output cluster size
//#define CLUST_SIZE_PN  NETWORK_SCALE * 7 //projection neuron cluster size
#define CLUST_SIZE_AN  (NETWORK_SCALE * 6) //output cluster size
#define CLUST_SIZE_PN  (NETWORK_SCALE * 6) //projection neuron cluster size
#define CLUST_SIZE_RN  (NETWORK_SCALE * 6) //receptor neuron cluster size

//Synapse time constants in ms (controls how fast arriving charge drains out of synapse into post-syn. neuron)
#define SYNAPSE_TAU_RNPN 1.0
#define SYNAPSE_TAU_PNPN 5.5
#define SYNAPSE_TAU_PNAN 1.0
#define SYNAPSE_TAU_ANAN 8.0


#include "modelSpec.h"
#include "modelSpec.cc"

/*--------------------------------------------------------------------------
 This function defines the Schmuker_2014_classifier model
-------------------------------------------------------------------------- */

void modelDefinition(NNmodel &model) 
{

	cout << "GeNN building model with " << NUM_VR << " x VRs" << endl;
    initGeNN();
    model.setPrecision(GENN_FLOAT);
    model.setName("Schmuker_2014_classifier");


  	/*--------------------------------------------------------------------------*/
  
    //DEFINE NEURON POPULATIONS ..
      
	/*--------------------------------------------------------------------------
 	RN receptor neuron Population. Clusters of Poisson neurons take rate level input from set of VR
	-------------------------------------------------------------------------- */
	
	double poissonRN_params[4]= {
  	0.1,        // 0 - firing rate
  	2.5,        // 1 - refractory period
  	20.0,       // 2 - Vspike
  	-60.0       // 3 - Vrest
	};

	double poissonRN_ini[3]= { //initial values for the neron variables
 	-60.0,        // 0 - V
  	0,           // 1 - seed
  	-10.0,       // 2 - SpikeTime
	};
	int countRN = NUM_VR * CLUST_SIZE_RN;
  	model.addNeuronPopulation("RN", countRN, POISSONNEURON, poissonRN_params,  poissonRN_ini);
  
  
	/*--------------------------------------------------------------------------
 	PN projection neuron Population. Uses MAP neuron model. 
 	Clusters of PN neurons take excitatory input 1:1 from RN clusters, 
 	whilst conducting an weak WTA among themselves
	-------------------------------------------------------------------------- */
	
	double stdMAP_params[4]= {
	60.0,          // 0 - Vspike: spike Amplitude factor
	3.0,           // 1 - alpha: "steepness / size" parameter
	-2.468,        // 2 - y: "shift / excitation" parameter
	0.0165         // 3 - beta: input sensitivity
	};

	double stdMAP_ini[2]= {
	-60.0,         // 0 - V: initial value for membrane potential
	-60.0          // 1 - preV: initial previous value
	};
	int countPN = NUM_VR * CLUST_SIZE_PN;
    model.addNeuronPopulation("PN", countPN, MAPNEURON, stdMAP_params,  stdMAP_ini);
    	
	/*--------------------------------------------------------------------------
 	AN output Association Neuron population. Uses MAP neuron model. 
 	Clusters of AN neurons, each representing an output class take excitatory input from all PN clusters, 
 	whilst conducting an strong WTA among themselves
	-------------------------------------------------------------------------- */
    int countAN = NUM_CLASSES * CLUST_SIZE_AN;
    model.addNeuronPopulation("AN", countAN, MAPNEURON, stdMAP_params,  stdMAP_ini);
    
    /*--------------------------------------------------------------------------
    DEFINE SYNAPSES
    -------------------------------------------------------------------------- */
    /* setup a synapse model NSYNAPSE_SPK_EVNT that drives from spike type events with V over a certain threshold
    */
    weightUpdateModel wuModel;
    wuModel.varNames.push_back(toString("g"));
    wuModel.varTypes.push_back(toString("scalar"));
    wuModel.pNames.push_back(toString("Epre"));
    wuModel.simCodeEvnt= toString("$(addtoinSyn) = $(g);\n\
        $(updatelinsyn);\n");
    wuModel.evntThreshold = toString("$(V_pre) > $(Epre)");
    //add to GenNN as a new weight update model
    unsigned int NSYNAPSE_SPK_EVNT = weightUpdateModels.size();
    weightUpdateModels.push_back(wuModel);


    //std shared params
	double synapsesStdExcitatory_params[1]= {-20.0};// Epre: Presynaptic threshold potential
	double initialConductanceValue[1]={0.0};
	double *postSynV = NULL;
    
    /*--------------------------------------------------------------------------
 	Define RN to PN Synapses. These are fixed weight, excitatory. cluster-cluster 1:1 connections, with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining cluster-cluster 1:1 connections are generated and loaded in the initialisation of the classifier class
 	Note that this connectivity will move to SPARSE data structure when available
	-------------------------------------------------------------------------- */
	
	double postExpSynapsePopn_RNPN[2] = {
			SYNAPSE_TAU_RNPN, 	//tau_S: decay time constant [ms]
			0.0	// Erev: Reversal potential
			};

	model.addSynapsePopulation("RNPN", NSYNAPSE_SPK_EVNT, DENSE, INDIVIDUALG,NO_DELAY, EXPDECAY, "RN", "PN", initialConductanceValue, synapsesStdExcitatory_params, postSynV,postExpSynapsePopn_RNPN);
	
    /*--------------------------------------------------------------------------
 	Define PN to PN Synapses. These are fixed weight, inhibitory synapses. configured as a weak WTA between clusters, with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
	-------------------------------------------------------------------------- */

	/*
	//Average inbitory synapse (created from mid point of strong and weak examples)
	double synapsesWTA_AvgInhibitory_params[2]= {
  	-35.0,        		// Epre: Presynaptic threshold potential (strong -40, weak -30)
  	50.0          		// Vslope: Activation slope of graded release
	};
	*/
	//Average inhibitory synapse (created from mid point of strong and weak examples)
	double synapsesWTA_AvgInhibitory_params[1]= {-35}; // Epre: Presynaptic threshold potential (strong -40, weak -30)


	double postExpSynapsePopn_PNPN[2] = {
			SYNAPSE_TAU_PNPN, 	// tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
			-92.0        		// Erev: Reversal potential
	};

	model.addSynapsePopulation("PNPN", NSYNAPSE_SPK_EVNT, DENSE, INDIVIDUALG,NO_DELAY, EXPDECAY, "PN", "PN", initialConductanceValue, synapsesWTA_AvgInhibitory_params,postSynV,postExpSynapsePopn_PNPN);
	
	/*--------------------------------------------------------------------------
 	Define PN to AN Synapses. These are plastic, excitatory. all-all connections, but with N% connectivity (e.g. 50%)
 	NB: The specific matrix entries defining connections are generated and loaded in the initialisation of the classifier class
 	Initial weight values are set randomly between upper and lower limits
 	Weights are altered on the CPU by a learning rule between time steps and revised matrix uploaded to the GPU
	-------------------------------------------------------------------------- */
    	
	double postExpSynapsePopn_PNAN[2] = {
				SYNAPSE_TAU_PNAN, 	//tau_S: decay time constant [ms]
				0.0	// Erev: Reversal potential
				};
	model.addSynapsePopulation("PNAN", NSYNAPSE_SPK_EVNT, DENSE, INDIVIDUALG,NO_DELAY, EXPDECAY, "PN", "AN", initialConductanceValue, synapsesStdExcitatory_params,postSynV,postExpSynapsePopn_PNAN);
	
	/*--------------------------------------------------------------------------
	Define AN to AN Synapses. These are fixed weight, inhibitory synapses. configured as a strong WTA between output class clusters, with N% connectivity (e.g. 50%)
	NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
	-------------------------------------------------------------------------- */

	double postExpSynapsePopn_ANAN[2] = {
			SYNAPSE_TAU_ANAN, 	// tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
			-92.0        		// Erev: Reversal potential
	};

	model.addSynapsePopulation("ANAN", NSYNAPSE_SPK_EVNT, DENSE, INDIVIDUALG,NO_DELAY, EXPDECAY, "AN", "AN", initialConductanceValue, synapsesWTA_AvgInhibitory_params,postSynV,postExpSynapsePopn_ANAN);

	//initializing learning parameters to start
	model.finalize();
}

/*--------------------------------------------------------------------------
  END
-------------------------------------------------------------------------- */
