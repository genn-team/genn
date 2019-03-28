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


void modelDefinition(ModelSpec &model) 
{
#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG


    model.setName("SynDelay");
    model.setDT(1.0);
    model.setPrecision(GENN_FLOAT);
    
    // INPUT NEURONS
    //==============
    NeuronModels::Izhikevich::ParamValues input_p( // Izhikevich parameters - tonic spiking
        0.02,  // 0 - a
        0.2,   // 1 - b
        -65,   // 2 - c
        6);     // 3 - d

    NeuronModels::Izhikevich::VarValues input_ini( // Izhikevich variables - tonic spiking
        -65,   // 0 - V
        -20);  // 1 - U

    PostsynapticModels::ExpCond::ParamValues postExpInp(
        1.0,            // 0 - tau_S: decay time constant for S [ms]
        0.0);            // 1 - Erev: Reversal potential

     CurrentSourceModels::DC::ParamValues input_current_p(
        4.0);  // 0 - magnitude


    model.addNeuronPopulation<NeuronModels::Izhikevich>("Input", 500, input_p, input_ini);
    model.addCurrentSource<CurrentSourceModels::DC>("InputCurrentSource", "Input",
                                                    input_current_p, {});


    // INTERNEURONS
    //=============
    NeuronModels::Izhikevich::ParamValues inter_p( // Izhikevich parameters - tonic spiking
        0.02,      // 0 - a
        0.2,       // 1 - b
        -65,       // 2 - c
        6);    // 3 - d


    NeuronModels::Izhikevich::VarValues inter_ini( // Izhikevich variables - tonic spiking
        -65,       // 0 - V
        -20);      // 1 - U

    PostsynapticModels::ExpCond::ParamValues postExpInt(
        1.0,            // 0 - tau_S: decay time constant for S [ms]
        0.0);             // 1 - Erev: Reversal potential

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Inter", 500, inter_p, inter_ini);


    // OUTPUT NEURONS
    //===============

    NeuronModels::Izhikevich::ParamValues output_p( // Izhikevich parameters - tonic spiking
        0.02,      // 0 - a
        0.2,       // 1 - b
        -65,       // 2 - c
        6);        // 3 - d


    NeuronModels::Izhikevich::VarValues output_ini( // Izhikevich variables - tonic spiking
        -65,    // 0 - V
        -20);   // 1 - U


    model.addNeuronPopulation<NeuronModels::Izhikevich>("Output", 500, output_p, output_ini);


    // INPUT-INTER, INPUT-OUTPUT & INTER-OUTPUT SYNAPSES
    //==================================================
    WeightUpdateModels::StaticPulse::VarValues inputInter_ini(
        0.06); // 0 - default synaptic conductance

    WeightUpdateModels::StaticPulse::VarValues inputOutput_ini(
        0.03); // 0 - default synaptic conductance

    WeightUpdateModels::StaticPulse::VarValues interOutput_ini(
        0.03); // 0 - default synaptic conductance

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("InputInter", SynapseMatrixType::DENSE_GLOBALG, 3,
                                                                                               "Input", "Inter",
                                                                                               {}, inputInter_ini,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("InputOutput", SynapseMatrixType::DENSE_GLOBALG, 6,
                                                                                               "Input", "Output",
                                                                                               {}, inputOutput_ini,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("InterOutput", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                               "Inter", "Output",
                                                                                               {}, interOutput_ini,
                                                                                               {}, {});
}
