
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
float synG = 0.01; // global synapse conductance for synapses


void modelDefinition(NNmodel &model) 
{
  model.setName("SynDelay");

  model.addNeuronPopulation("Input", 500, IZHIKEVICH, input_p, input_ini);
  model.activateDirectInput("Input", CONSTINP);
  model.setConstInp("Input", constInput);

  model.addNeuronPopulation("Interneuron", 500, IZHIKEVICH, inter_p, inter_ini);
  model.addSynapsePopulation("Input-Interneuron", NSYNAPSE, ALLTOALL, GLOBALG, 4, "Input", "Interneuron", synapses_p);
  model.setSynapseG("Input-Interneuron", synG);

  model.addNeuronPopulation("Output", 500, IZHIKEVICH, output_p, output_ini);
  model.addSynapsePopulation("Input-Output", NSYNAPSE, ALLTOALL, GLOBALG, 5, "Input", "Output", synapses_p);
  model.setSynapseG("Input-Output", synG);
  model.addSynapsePopulation("Interneuron-Output", NSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, "Interneuron", "Output", synapses_p);
  model.setSynapseG("Interneuron-Output", synG);
}
