
#define DT 1.0f

#include "modelSpec.h"
#include "modelSpec.cc"

float inputIzh_p[4] = {
//Izhikevich model parameters - tonic spiking
	0.02,	// 0 - a
	0.2, 	// 1 - b
	-65, 	// 2 - c
	6 	// 3 - d
};

float inputIzh_ini[2] = {
//Izhikevich model initial conditions - tonic spiking
	-65,	//0 - V
	-20	//1 - U
};

float synInOut_p[3] = {
  0.0,           // 0 - Erev: Reversal potential
  -30.0,         // 1 - Epre: Presynaptic threshold potential
  1.0            // 2 - tau_S: decay time constant for S [ms]
};

float constInput = 4.0;
float synInOutG = 0.01;


void modelDefinition(NNmodel &model) 
{
  model.setName("SynDelay");

  model.addNeuronPopulation("inputIzh", 500, IZHIKEVICH, inputIzh_p, inputIzh_ini);
  model.activateDirectInput("inputIzh", CONSTINP);
  model.setConstInp("inputIzh", constInput);

  model.addNeuronPopulation("outputIzh", 500, IZHIKEVICH, inputIzh_p, inputIzh_ini);

  model.addSynapsePopulation("InOut", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, "inputIzh", "outputIzh", synInOut_p);
  model.setSynapseG("InOut", synInOutG);
}
