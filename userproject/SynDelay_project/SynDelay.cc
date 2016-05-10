//--------------------------------------------------------------------------
//   Author:    James Turner
//  
//   Institute: Center for Computational Neuroscience and Robotics
//              University of Sussex
//              Falmer, Brighton BN1 9QJ, UK 
//  
//   email to:  J.P.Turner@sussex.ac.uk
//  
//--------------------------------------------------------------------------

#include "modelSpec.h"


void modelDefinition(NNmodel &model) 
{
    initGeNN();
    model.setName("SynDelay");
    model.setDT(1.0);
    model.setPrecision(GENN_FLOAT);

    neuronModel n= nModels[IZHIKEVICH];
    n.pNames.push_back("I0");
    n.simCode= "if ($(V) >= 30.0) {\n\
        $(V)=$(c);\n\
        $(U)+=$(d);\n\
    }\n\
    $(V) += 0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V) += 0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT;\n\
    $(U) += $(a)*($(b)*$(V)-$(U))*DT;\n\
   //if ($(V) > 30.0) { // keep this only for visualisation -- not really necessaary otherwise\n\
   //    $(V) = 30.0;\n\
   //}\n";
    unsigned int MYIZHIKEVICH= nModels.size();
    nModels.push_back(n);

    
    // INPUT NEURONS
    //==============

    double input_p[5] = { // Izhikevich parameters - tonic spiking
	0.02,  // 0 - a
	0.2,   // 1 - b
	-65,   // 2 - c
	6,      // 3 - d
	4.0     // 4 - I0 (input current)
    };

    double input_ini[2] = { // Izhikevich variables - tonic spiking
	-65,   // 0 - V
	-20    // 1 - U
    };

    double postExpInp[2] = {
	1.0,            // 0 - tau_S: decay time constant for S [ms]
	0.0		  // 1 - Erev: Reversal potential
    };

    model.addNeuronPopulation("Input", 500, MYIZHIKEVICH, input_p, input_ini);


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

    model.addNeuronPopulation("Inter", 500, IZHIKEVICH, inter_p, inter_ini);


    // OUTPUT NEURONS
    //===============

    double output_p[5] = { // Izhikevich parameters - tonic spiking
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

    model.addNeuronPopulation("Output", 500, IZHIKEVICH, output_p, output_ini);


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

    model.addSynapsePopulation("InputInter", NSYNAPSE, DENSE, GLOBALG, 3, IZHIKEVICH_PS, "Input", "Inter", inputInter_ini, synapses_p, postSynV, postExpInp);
    model.addSynapsePopulation("InputOutput", NSYNAPSE, DENSE, GLOBALG, 6, IZHIKEVICH_PS, "Input", "Output", inputOutput_ini, synapses_p, postSynV, postExpOut);
    model.addSynapsePopulation("InterOutput", NSYNAPSE, DENSE, GLOBALG, NO_DELAY, IZHIKEVICH_PS, "Inter", "Output", interOutput_ini, synapses_p, postSynV, postExpInt);


    model.finalize();
}
