/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "modelSpec.h"
#include "global.h"
#include "stringUtils.h"
#include <vector>
#include "sizes.h"

//we modify the IzhikevichVariable neuron to receive external inputs
class MyIzhikevichVariable : public NeuronModels::IzhikevichVariable
{
public:
    DECLARE_MODEL(MyIzhikevichVariable, 0, 7);

    SET_SIM_CODE(
        "if ($(V) >= 30.0){\n"
        "   $(V)=$(c);\n"
        "   $(U)+=$(d);\n"
        "} \n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(I0)+$(Isyn))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "//if ($(V) > 30.0){      //keep this only for visualisation -- not really necessaary otherwise \n"
        "//  $(V)=30.0; \n"
        "//}\n");

    SET_VARS({{"V","scalar"}, {"U", "scalar"},
             {"a", "scalar"}, {"b", "scalar"},
             {"c", "scalar"}, {"d", "scalar"},
             {"I0", "scalar"}});
};
IMPLEMENT_MODEL(MyIzhikevichVariable);

std::vector<unsigned int> neuronPSize;
std::vector<unsigned int> neuronVSize;
std::vector<unsigned int> synapsePSize; 

scalar meanInpExc = 5.0*inputFac; //5.0 for balanced regime
scalar meanInpInh = 2.0*inputFac; //2.0 for balanced regime

MyIzhikevichVariable::VarValues IzhExc_ini(
//Izhikevich model initial conditions - excitatory population
    -65.0,	//0 - V
    0.0,	//1 - U
    0.02,	// 2 - a
    0.2, 	// 3 - b
    -65.0, 	// 4 - c
    8.0, 	// 5 - d
    0.0     // 6 - I0
);

MyIzhikevichVariable::VarValues IzhInh_ini(
//Izhikevich model initial conditions - inhibitory population
    -65.0,	//0 - V
    0.0,	//1 - U
    0.02,	// 2 - a
    0.25, 	// 3 - b
    -65.0, 	// 4 - c
    2.0, 	// 5 - d
    0.0     // 6 - I0
);

WeightUpdateModels::StaticPulse::VarValues SynIzh_ini(
    0.0 // default synaptic conductance
);


void modelDefinition(NNmodel &model) 
{
  initGeNN();

#ifdef DEBUG
  GENN_PREFERENCES::debugCode = true;
#else
  GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG
  
  model.setName("Izh_sparse");
  model.setDT(1.0);
  model.addNeuronPopulation<MyIzhikevichVariable>("PExc", _NExc, {}, IzhExc_ini);

  model.addNeuronPopulation<MyIzhikevichVariable>("PInh", _NInh, {}, IzhInh_ini);
  
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
