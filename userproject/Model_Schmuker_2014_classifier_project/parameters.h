#pragma once

//network sizes and parameters
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
