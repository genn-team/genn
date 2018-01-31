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
#include "global.h"


class MyIzhikevich : public NeuronModels::Izhikevich
{
public:
    DECLARE_MODEL(MyIzhikevich, 5, 2);

    SET_SIM_CODE(
        "if ($(V) >= 30.0) {\n"
        "    $(V)=$(c);\n"
        "    $(U)+=$(d);\n"
        "}\n"
        "$(V) += 0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT; //at two times for numerical stability\n"
        "$(V) += 0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT;\n"
        "$(U) += $(a)*($(b)*$(V)-$(U))*DT;\n"
        "//if ($(V) > 30.0) { // keep this only for visualisation -- not really necessaary otherwise\n"
        "//    $(V) = 30.0;\n"
        "//}\n");
    SET_PARAM_NAMES({"a", "b", "c", "d", "I0"});
};
IMPLEMENT_MODEL(MyIzhikevich);

void modelDefinition(NNmodel &model) 
{
    initGeNN();

#ifdef DEBUG
    GENN_PREFERENCES::debugCode = true;
#else
    GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG

    // By default we want to initialise variables on device
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    model.setName("SynDelay");
    model.setDT(1.0);
    model.setPrecision(GENN_FLOAT);
    
    // INPUT NEURONS
    //==============
    MyIzhikevich::ParamValues input_p( // Izhikevich parameters - tonic spiking
        0.02,  // 0 - a
        0.2,   // 1 - b
        -65,   // 2 - c
        6,      // 3 - d
        4.0     // 4 - I0 (input current)
                              );

    MyIzhikevich::VarValues input_ini( // Izhikevich variables - tonic spiking
        -65,   // 0 - V
        -20    // 1 - U
    );

    PostsynapticModels::ExpCond::ParamValues postExpInp(
        1.0,            // 0 - tau_S: decay time constant for S [ms]
        0.0              // 1 - Erev: Reversal potential
    );

    model.addNeuronPopulation<MyIzhikevich>("Input", 500, input_p, input_ini);


    // INTERNEURONS
    //=============
    NeuronModels::Izhikevich::ParamValues inter_p( // Izhikevich parameters - tonic spiking
        0.02,      // 0 - a
        0.2,       // 1 - b
        -65,       // 2 - c
        6      // 3 - d
    );

    NeuronModels::Izhikevich::VarValues inter_ini( // Izhikevich variables - tonic spiking
        -65,       // 0 - V
        -20        // 1 - U
    );

    PostsynapticModels::ExpCond::ParamValues postExpInt(
        1.0,            // 0 - tau_S: decay time constant for S [ms]
        0.0               // 1 - Erev: Reversal potential
    );

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Inter", 500, inter_p, inter_ini);


    // OUTPUT NEURONS
    //===============

    NeuronModels::Izhikevich::ParamValues output_p( // Izhikevich parameters - tonic spiking
        0.02,      // 0 - a
        0.2,       // 1 - b
        -65,       // 2 - c
        6          // 3 - d
    );

    NeuronModels::Izhikevich::VarValues output_ini( // Izhikevich variables - tonic spiking
        -65,    // 0 - V
        -20     // 1 - U
    );
    PostsynapticModels::ExpCond::ParamValues postExpOut(
        1.0,            // 0 - tau_S: decay time constant for S [ms]
        0.0             // 1 - Erev: Reversal potential
    );

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Output", 500, output_p, output_ini);


    // INPUT-INTER, INPUT-OUTPUT & INTER-OUTPUT SYNAPSES
    //==================================================
    WeightUpdateModels::StaticPulse::VarValues inputInter_ini(
        0.06   // 0 - default synaptic conductance
    );

    WeightUpdateModels::StaticPulse::VarValues inputOutput_ini(
        0.03   // 0 - default synaptic conductance
    );

    WeightUpdateModels::StaticPulse::VarValues interOutput_ini(
        0.03   // 0 - default synaptic conductance
    );

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>("InputInter", SynapseMatrixType::DENSE_GLOBALG, 3,
                                                                                             "Input", "Inter",
                                                                                             {}, inputInter_ini,
                                                                                             postExpInp, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>("InputOutput", SynapseMatrixType::DENSE_GLOBALG, 6,
                                                                                             "Input", "Output",
                                                                                             {}, inputOutput_ini,
                                                                                             postExpOut, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>("InterOutput", SynapseMatrixType::DENSE_GLOBALG, NO_DELAY,
                                                                                             "Inter", "Output",
                                                                                             {}, interOutput_ini,
                                                                                             postExpInt, {});


    model.finalize();
}
