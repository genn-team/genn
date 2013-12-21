
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


// INPUT-INTER, INPUT-OUTPUT & INTER-OUTPUT SYNAPSES
//==================================================

float synapses_p[3] = {
  0.0,     // 0 - Erev: Reversal potential
  -30.0,   // 1 - Epre: Presynaptic threshold potential
  1.0      // 2 - tau_S: decay time constant for S [ms]
};
float strongSynG = 0.003; // strong synapse G
float weakSynG = 0.001;   // weak synapse G


void modelDefinition(NNmodel &model) 
{
  model.setName("SynDelay");

  model.addNeuronPopulation("Input", 100, IZHIKEVICH, input_p, input_ini);
  model.activateDirectInput("Input", CONSTINP);
  model.setConstInp("Input", constInput);
  model.addNeuronPopulation("Inter", 100, IZHIKEVICH, inter_p, inter_ini);
  model.addNeuronPopulation("Output", 100, IZHIKEVICH, output_p, output_ini);

  model.addSynapsePopulation("Input-Inter", NSYNAPSE, ALLTOALL, GLOBALG, 3, "Input", "Inter", synapses_p);
  model.setSynapseG("Input-Inter", strongSynG);
  model.addSynapsePopulation("Input-Output", NSYNAPSE, ALLTOALL, GLOBALG, 6, "Input", "Output", synapses_p);
  model.setSynapseG("Input-Output", weakSynG);
  model.addSynapsePopulation("Inter-Output", NSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, "Inter", "Output", synapses_p);
  model.setSynapseG("Inter-Output", weakSynG);
}
