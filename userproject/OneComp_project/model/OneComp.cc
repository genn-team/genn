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

//Izhikevich model parameters - tonic spiking
MyIzhikevich::ParamValues exIzh_p(
    0.02,       // 0 - a
    0.2,        // 1 - b
    -65,        // 2 - c
    6,          // 3 - d
    4.0         // 4 - I0 (input current)
);

//Izhikevich model initial conditions - tonic spiking
MyIzhikevich::VarValues exIzh_ini(
    -65,        //0 - V
    -20         //1 - U
);



void modelDefinition(NNmodel &model) 
{
    initGeNN();

#ifdef DEBUG
    GENN_PREFERENCES::debugCode = true;
#else
    GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;


    model.setName("OneComp");
    model.setDT(1.0);

    model.addNeuronPopulation<MyIzhikevich>("Izh1", _NC1, exIzh_p, exIzh_ini);
    model.setPrecision(_FTYPE);
    model.finalize();
}
