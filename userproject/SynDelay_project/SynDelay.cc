
#define DT 1.0f

#include "modelSpec.h"
#include "modelSpec.cc"

float input_p[4] = { //Izhikevich model parameters - tonic spiking
  0.02,	   // 0 - a
  0.2, 	   // 1 - b
  -65, 	   // 2 - c
  6 	   // 3 - d
};

float input_ini[2] = { //Izhikevich model initial conditions - tonic spiking
  -65,	   //0 - V
  -20	   //1 - U
};

float synapses_p[3] = {
  0.0,     // 0 - Erev: Reversal potential
  -30.0,   // 1 - Epre: Presynaptic threshold potential
  1.0      // 2 - tau_S: decay time constant for S [ms]
};

float constInput = 4.0;
float synG = 0.01;


void modelDefinition(NNmodel &model) 
{
  model.setName("SynDelay");

  model.addNeuronPopulation("Input", 500, IZHIKEVICH, input_p, input_ini);
  model.activateDirectInput("Input", CONSTINP);
  model.setConstInp("Input", constInput);

  model.addNeuronPopulation("Interneuron", 500, IZHIKEVICH, input_p, input_ini);
  model.addSynapsePopulation("Input-Interneuron", NSYNAPSE, ALLTOALL, GLOBALG, 4.0f, "Input", "Interneuron", synapses_p);
  model.setSynapseG("Input-Interneuron", synG);

  model.addNeuronPopulation("Output", 500, IZHIKEVICH, input_p, input_ini);
  model.addSynapsePopulation("Input-Output", NSYNAPSE, ALLTOALL, GLOBALG, 5.0f, "Input", "Output", synapses_p);
  model.setSynapseG("Input-Output", synG);
  model.addSynapsePopulation("Interneuron-Output", NSYNAPSE, ALLTOALL, GLOBALG, NO_DELAY, "Interneuron", "Output", synapses_p);
  model.setSynapseG("Interneuron-Output", synG);
}
