/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "modelSpec.h"

#include "sizes.h"

class RandomizeSq : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(RandomizeSq, 2);

    SET_CODE(
        "const scalar random = $(gennrand_uniform);\n"
        "$(value) = $(min) + (random * random * $(scale));");

    SET_PARAM_NAMES({"min", "scale"});
};
IMPLEMENT_SNIPPET(RandomizeSq);

class Randomize : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Randomize, 2);

    SET_CODE(
        "const scalar random = $(gennrand_uniform);\n"
        "$(value) = $(min) + (random * $(scale));");

    SET_PARAM_NAMES({"min", "scale"});
};
IMPLEMENT_SNIPPET(Randomize);

//we modify the IzhikevichVariable neuron to add noise to input current
class MyIzhikevichVariableNoise : public NeuronModels::Base
{
public:
    DECLARE_MODEL(MyIzhikevichVariableNoise, 2, 6);

    SET_SIM_CODE(
        "if ($(V) >= 30.0){\n"
        "   $(V)=$(c);\n"
        "   $(U)+=$(d);\n"
        "} \n"
        "const scalar i0 = $(I0Mean) + ($(gennrand_normal) * $(I0SD));\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+i0+$(Isyn))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+i0+$(Isyn))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "//if ($(V) > 30.0){      //keep this only for visualisation -- not really necessaary otherwise \n"
        "//  $(V)=30.0; \n"
        "//}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 29.99");

    SET_PARAM_NAMES({"I0Mean", "I0SD"});

    SET_VARS({{"V","scalar"}, {"U", "scalar"},
             {"a", "scalar"}, {"b", "scalar"},
             {"c", "scalar"}, {"d", "scalar"}});
};
IMPLEMENT_MODEL(MyIzhikevichVariableNoise);

Randomize::ParamValues randomizedA(
    0.02,   // 0 - min
    0.08);  // 1 - scale

Randomize::ParamValues randomizedB(
    0.2,   // 0 - min
    -0.05);  // 1 - scale

RandomizeSq::ParamValues randomizedC(
    -65.0,  // 0 - min
    15.0);  // 1 - scale

RandomizeSq::ParamValues randomizedD(
    8.0,  // 0 - min
    -6.0    // 1 - scale
);

MyIzhikevichVariableNoise::ParamValues IzhExc_par(
    0.0,            // 0 -I0Mean
    5.0 * inputFac  // 1 - I0SD
);

MyIzhikevichVariableNoise::VarValues IzhExc_ini(
//Izhikevich model initial conditions - excitatory population
    -65.0,                              // 0 - V
    uninitialisedVar(),                 // 1 - U
    0.02,                               // 2 - a
    0.2,                                // 3 - b
    initVar<RandomizeSq>(randomizedC),  // 4 - c
    initVar<RandomizeSq>(randomizedD)); // 5 - d


MyIzhikevichVariableNoise::ParamValues IzhInh_par(
    0.0,                // 0 -I0Mean
    2.0 * inputFac);    // 1 - I0SD


MyIzhikevichVariableNoise::VarValues IzhInh_ini(
//Izhikevich model initial conditions - inhibitory population
    -65.0,                              // 0 - V
    uninitialisedVar(),                 // 1 - U
    initVar<Randomize>(randomizedA),    // 2 - a
    initVar<Randomize>(randomizedB),    // 3 - b
    -65.0,                              // 4 - c
    2.0);                               // 5 - d


// Weights are initialised from file so don't initialise
WeightUpdateModels::StaticPulse::VarValues SynIzh_ini(
    uninitialisedVar());    // default synaptic conductance



void modelDefinition(NNmodel &model) 
{
    initGeNN();

#ifdef DEBUG
    GENN_PREFERENCES::debugCode = true;
#else
    GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG

    // For this model we want to initialise variables on host as weights are not initialised anyway and
    // we need to manually initialise U parameters based on other parameters
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_HOST;
  
    model.setName("Izh_sparse");
    model.setDT(1.0);
    model.addNeuronPopulation<MyIzhikevichVariableNoise>("PExc", _NExc, IzhExc_par, IzhExc_ini);

    model.addNeuronPopulation<MyIzhikevichVariableNoise>("PInh", _NInh, IzhInh_par, IzhInh_ini);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Exc_Exc", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                               "PExc", "PExc",
                                                                                               {}, SynIzh_ini,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Exc_Inh", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                               "PExc", "PInh",
                                                                                               {}, SynIzh_ini,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Inh_Exc", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                               "PInh", "PExc",
                                                                                               {}, SynIzh_ini,
                                                                                               {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("Inh_Inh",  SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
                                                                                               "PInh", "PInh",
                                                                                               {}, SynIzh_ini,
                                                                                               {}, {});

    fprintf(stderr, "#model created.\n");

    model.setMaxConn("Exc_Exc", _NMaxConnP0);
    model.setMaxConn("Exc_Inh", _NMaxConnP1);
    model.setMaxConn("Inh_Exc", _NMaxConnP2);
    model.setMaxConn("Inh_Inh", _NMaxConnP3);

#ifdef nGPU
    cerr << "nGPU: " << nGPU << endl;
    model.setGPUDevice(nGPU);
#endif
    model.setPrecision(_FTYPE);
    model.finalize();
}
