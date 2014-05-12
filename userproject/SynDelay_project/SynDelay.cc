
#define DT 1.0f

#include "modelSpec.h"
#include "modelSpec.cc"


// INPUT NEURONS
//==============

float input_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,  // 0 - a
  0.2,   // 1 - b
  -65,   // 2 - c
  6      // 3 - d
};
float input_ini[2] = { // Izhikevich variables - tonic spiking
  -65,   // 0 - V
  -20    // 1 - U
};
float constInput = 4.0; // constant input to input neurons
float postExpInp[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// INTERNEURONS
//=============

float inter_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,	   // 0 - a
  0.2, 	   // 1 - b
  -65, 	   // 2 - c
  6 	   // 3 - d
};
float inter_ini[2] = { // Izhikevich variables - tonic spiking
  -65,	   // 0 - V
  -20	   // 1 - U
};
float postExpInt[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// OUTPUT NEURONS
//===============

float output_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,	   // 0 - a
  0.2, 	   // 1 - b
  -65, 	   // 2 - c
  6 	   // 3 - d
};
float output_ini[2] = { // Izhikevich variables - tonic spiking
  -65,	   // 0 - V
  -20	   // 1 - U
};
float postExpOut[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// INPUT-INTER, INPUT-OUTPUT & INTER-OUTPUT SYNAPSES
//==================================================

float synapses_p[3] = {
  0.0,     // 0 - Erev: Reversal potential
  -30.0,   // 1 - Epre: Presynaptic threshold potential
  1.0      // 2 - tau_S: decay time constant for S [ms]
};
float postSynV[0] = {
};
float strongSynG = 0.0006; // strong synapse G
float weakSynG = 0.0002;   // weak synapse G


void modelDefinition(NNmodel &model) 
{
  model.setName("SynDelay");

  model.addNeuronPopulation("Input", 500, IZHIKEVICH, input_p, input_ini);
  model.activateDirectInput("Input", CONSTINP);
  model.setConstInp("Input", constInput);
  model.addNeuronPopulation("Inter", 500, IZHIKEVICH, inter_p, inter_ini);
  model.addNeuronPopulation("Output", 500, IZHIKEVICH, output_p, output_ini);

  model.addSynapsePopulation("InputInter", NSYNAPSE, ALLTOALL, GLOBALG, 3, EXPDECAY, "Input", "Inter", synapses_p, postSynV, postExpInp);
  model.setSynapseG("InputInter", strongSynG);
  //model.addSynapsePopulation("InputOutput", NSYNAPSE, ALLTOALL, GLOBALG, 6, EXPDECAY, "Input", "Output", synapses_p, postSynV, postExpOut);
  //model.setSynapseG("InputOutput", weakSynG);
  model.addSynapsePopulation("InterOutput", NSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, EXPDECAY, "Inter", "Output", synapses_p, postSynV, postExpInt);
  model.setSynapseG("InterOutput", weakSynG);
}
