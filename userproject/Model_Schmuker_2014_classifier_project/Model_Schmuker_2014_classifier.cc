/*--------------------------------------------------------------------------
   Author: Alan Diamond
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------


This file contains the network model definition for the "Schmuker_2014_classifier" model.

-------------------------------------------------------------------------- */
#include "parameters.h"


#include "modelSpec.h"
#include <iostream>

using namespace std;

// setup a synapse model NSYNAPSE_SPK_EVNT that drives from spike type events with V over a certain threshold
class WeightUpdateModelSpikeEvent : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(WeightUpdateModelSpikeEvent, 1, 1);

    SET_EVENT_CODE("$(addToInSyn, $(g));\n");
    SET_EVENT_THRESHOLD_CONDITION_CODE("$(V_pre) > $(Epre)");

    SET_PARAM_NAMES({"Epre"});
    SET_VARS({{"g", "scalar"}});
};
IMPLEMENT_MODEL(WeightUpdateModelSpikeEvent);


/*--------------------------------------------------------------------------
 This function defines the Schmuker_2014_classifier model
-------------------------------------------------------------------------- */
void modelDefinition(ModelSpec &model)
{

    cout << "GeNN building model with " << NUM_VR << " x VRs" << endl;
    model.setPrecision(GENN_FLOAT);
    model.setName("Schmuker_2014_classifier");

    /*--------------------------------------------------------------------------*/
  
    //DEFINE NEURON POPULATIONS ..
      
    /*--------------------------------------------------------------------------
     RN receptor neuron Population. Clusters of Poisson neurons take rate level input from set of VR
    -------------------------------------------------------------------------- */
    NeuronModels::Poisson::ParamValues poissonRN_params(
        2.5,        // 0 - refractory period
        0.5,        // 1 - spike time
        20.0,       // 2 - Vspike
        -60.0);     // 3 - Vrest

    NeuronModels::Poisson::VarValues poissonRN_ini( //initial values for the neron variables
        -60.0,       // 0 - V
        -10.0);      // 1 - SpikeTime

    int countRN = NUM_VR * CLUST_SIZE_RN;
    model.addNeuronPopulation<NeuronModels::Poisson>("RN", countRN, poissonRN_params,  poissonRN_ini);
  
  
    /*--------------------------------------------------------------------------
     PN projection neuron Population. Uses MAP neuron model.
     Clusters of PN neurons take excitatory input 1:1 from RN clusters,
     whilst conducting an weak WTA among themselves
    -------------------------------------------------------------------------- */
    NeuronModels::RulkovMap::ParamValues stdMAP_params(
        60.0,          // 0 - Vspike: spike Amplitude factor
        3.0,           // 1 - alpha: "steepness / size" parameter
        -2.468,        // 2 - y: "shift / excitation" parameter
        0.0165         // 3 - beta: input sensitivity
    );

    NeuronModels::RulkovMap::VarValues stdMAP_ini(
        -60.0,         // 0 - V: initial value for membrane potential
        -60.0          // 1 - preV: initial previous value
    );
    int countPN = NUM_VR * CLUST_SIZE_PN;
    model.addNeuronPopulation<NeuronModels::RulkovMap>("PN", countPN, stdMAP_params,  stdMAP_ini);

    /*--------------------------------------------------------------------------
     AN output Association Neuron population. Uses MAP neuron model.
     Clusters of AN neurons, each representing an output class take excitatory input from all PN clusters,
     whilst conducting an strong WTA among themselves
    -------------------------------------------------------------------------- */
    int countAN = NUM_CLASSES * CLUST_SIZE_AN;
    model.addNeuronPopulation<NeuronModels::RulkovMap>("AN", countAN, stdMAP_params,  stdMAP_ini);
    
    /*--------------------------------------------------------------------------
    DEFINE SYNAPSES
    -------------------------------------------------------------------------- */
    //std shared params
    WeightUpdateModelSpikeEvent::ParamValues synapsesStdExcitatory_params(
        -20.0 // Epre: Presynaptic threshold potential
    );
    WeightUpdateModelSpikeEvent::VarValues initialConductanceValue(uninitialisedVar());

    
    /*--------------------------------------------------------------------------
     Define RN to PN Synapses. These are fixed weight, excitatory. cluster-cluster 1:1 connections, with N% connectivity (e.g. 50%)
     NB: The specific matrix entries defining cluster-cluster 1:1 connections are generated and loaded in the initialisation of the classifier class
     Note that this connectivity will move to SPARSE data structure when available
    -------------------------------------------------------------------------- */
    PostsynapticModels::ExpCond::ParamValues postExpSynapsePopn_RNPN(
        SYNAPSE_TAU_RNPN,     //tau_S: decay time constant [ms]
        0.0    // Erev: Reversal potential
    );

    model.addSynapsePopulation<WeightUpdateModelSpikeEvent, PostsynapticModels::ExpCond>("RNPN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                         "RN", "PN",
                                                                                         synapsesStdExcitatory_params, initialConductanceValue,
                                                                                         postExpSynapsePopn_RNPN, {});

    /*--------------------------------------------------------------------------
     Define PN to PN Synapses. These are fixed weight, inhibitory synapses. configured as a weak WTA between clusters, with N% connectivity (e.g. 50%)
     NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
    -------------------------------------------------------------------------- */

    /*
    //Average inbitory synapse (created from mid point of strong and weak examples)
    WeightUpdateModelSpikeEvent::ParamValues synapsesWTA_AvgInhibitory_params(
      -35.0,                // Epre: Presynaptic threshold potential (strong -40, weak -30)
      50.0                  // Vslope: Activation slope of graded release
    );
    */
    //Average inhibitory synapse (created from mid point of strong and weak examples)
    WeightUpdateModelSpikeEvent::ParamValues synapsesWTA_AvgInhibitory_params(-35); // Epre: Presynaptic threshold potential (strong -40, weak -30)

    PostsynapticModels::ExpCond::ParamValues postExpSynapsePopn_PNPN(
        SYNAPSE_TAU_PNPN,     // tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
        -92.0                // Erev: Reversal potential
    );

    model.addSynapsePopulation<WeightUpdateModelSpikeEvent, PostsynapticModels::ExpCond>("PNPN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                         "PN", "PN",
                                                                                         synapsesWTA_AvgInhibitory_params, initialConductanceValue,
                                                                                         postExpSynapsePopn_PNPN, {});

    /*--------------------------------------------------------------------------
     Define PN to AN Synapses. These are plastic, excitatory. all-all connections, but with N% connectivity (e.g. 50%)
     NB: The specific matrix entries defining connections are generated and loaded in the initialisation of the classifier class
     Initial weight values are set randomly between upper and lower limits
     Weights are altered on the CPU by a learning rule between time steps and revised matrix uploaded to the GPU
    -------------------------------------------------------------------------- */

    PostsynapticModels::ExpCond::ParamValues postExpSynapsePopn_PNAN(
                SYNAPSE_TAU_PNAN,     //tau_S: decay time constant [ms]
                0.0    // Erev: Reversal potential
    );
    model.addSynapsePopulation<WeightUpdateModelSpikeEvent, PostsynapticModels::ExpCond>("PNAN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                         "PN", "AN",
                                                                                         synapsesStdExcitatory_params, initialConductanceValue,
                                                                                         postExpSynapsePopn_PNAN, {});

    /*--------------------------------------------------------------------------
    Define AN to AN Synapses. These are fixed weight, inhibitory synapses. configured as a strong WTA between output class clusters, with N% connectivity (e.g. 50%)
    NB: The specific matrix entries defining cluster-cluster connections are generated and loaded in the initialisation of the classifier class
    -------------------------------------------------------------------------- */

    PostsynapticModels::ExpCond::ParamValues postExpSynapsePopn_ANAN(
            SYNAPSE_TAU_ANAN,     // tau_S: decay time constant for S [ms] //may need tuning(fast/strong 3ms, slow/weak 8ms avg:5.5)
            -92.0                // Erev: Reversal potential
    );

    model.addSynapsePopulation<WeightUpdateModelSpikeEvent, PostsynapticModels::ExpCond>("ANAN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                         "AN", "AN",
                                                                                         synapsesWTA_AvgInhibitory_params, initialConductanceValue,
                                                                                         postExpSynapsePopn_ANAN, {});

}

/*--------------------------------------------------------------------------
  END
-------------------------------------------------------------------------- */
