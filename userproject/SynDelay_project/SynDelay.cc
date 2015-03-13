
#define DT 1.0f

#include "modelSpec.h"
#include "modelSpec.cc"


// INPUT NEURONS
//==============

double input_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,  // 0 - a
  0.2,   // 1 - b
  -65,   // 2 - c
  6      // 3 - d
};
double input_ini[2] = { // Izhikevich variables - tonic spiking
  -65,   // 0 - V
  -20    // 1 - U
};
double postExpInp[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// INTERNEURONS
//=============

double inter_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,	   // 0 - a
  0.2, 	   // 1 - b
  -65, 	   // 2 - c
  6 	   // 3 - d
};
double inter_ini[2] = { // Izhikevich variables - tonic spiking
  -65,	   // 0 - V
  -20	   // 1 - U
};
double postExpInt[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// OUTPUT NEURONS
//===============

double output_p[4] = { // Izhikevich parameters - tonic spiking
  0.02,	   // 0 - a
  0.2, 	   // 1 - b
  -65, 	   // 2 - c
  6 	   // 3 - d
};
double output_ini[2] = { // Izhikevich variables - tonic spiking
  -65,	   // 0 - V
  -20	   // 1 - U
};
double postExpOut[2] = {
  1.0,            // 0 - tau_S: decay time constant for S [ms]
  0.0		  // 1 - Erev: Reversal potential
};

// INPUT-INTER, INPUT-OUTPUT & INTER-OUTPUT SYNAPSES
//==================================================

double *synapses_p= NULL;

double inputInter_ini[1] = {
  0.06   // 0 - default synaptic conductance
};
double inputOutput_ini[1] = {
  0.03   // 0 - default synaptic conductance
};
double interOutput_ini[1] = {
  0.03   // 0 - default synaptic conductance
};
double *postSynV = NULL;


double constInput = 4.0; // constant input to input neurons


void modelDefinition(NNmodel &model) 
{
  initGeNN();
  model.setName("SynDelay");

  model.addNeuronPopulation("Input", 500, IZHIKEVICH, input_p, input_ini);
  model.activateDirectInput("Input", CONSTINP);
  model.setConstInp("Input", constInput);
  model.addNeuronPopulation("Inter", 500, IZHIKEVICH, inter_p, inter_ini);
  model.addNeuronPopulation("Output", 500, IZHIKEVICH, output_p, output_ini);

  model.addSynapsePopulation("InputInter", NSYNAPSE, DENSE, GLOBALG, 3, IZHIKEVICH_PS, "Input", "Inter", inputInter_ini, synapses_p, postSynV, postExpInp);
  model.addSynapsePopulation("InputOutput", NSYNAPSE, DENSE, GLOBALG, 6, IZHIKEVICH_PS, "Input", "Output", inputOutput_ini, synapses_p, postSynV, postExpOut);
  model.addSynapsePopulation("InterOutput", NSYNAPSE, DENSE, GLOBALG, NO_DELAY, IZHIKEVICH_PS, "Inter", "Output", interOutput_ini, synapses_p, postSynV, postExpInt);
  model.finalize();
}
