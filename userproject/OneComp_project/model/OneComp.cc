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

void modelDefinition(ModelSpec &model) 
{
#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG

#ifdef _GPU_DEVICE
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = _GPU_DEVICE;
#endif

    // Izhikevich model parameters - tonic spiking
    NeuronModels::Izhikevich::ParamValues exIzh_p(
        0.02,       // 0 - a
        0.2,        // 1 - b
        -65,        // 2 - c
        6);         // 3 - d

    // Izhikevich model initial conditions - tonic spiking
    NeuronModels::Izhikevich::VarValues exIzh_ini(
        -65,        //0 - V
        -20         //1 - U
    );

    CurrentSourceModels::DC::ParamValues exIzh_curr_p(
        4.0);  // 0 - magnitude

    model.setName("OneComp");
    model.setDT(1.0);

    model.addNeuronPopulation<NeuronModels::Izhikevich>("Izh1", _NN, exIzh_p, exIzh_ini);
    model.addCurrentSource<CurrentSourceModels::DC>("Curr1", "Izh1",
                                                    exIzh_curr_p, {});
    model.setPrecision(_FTYPE);
    model.setTiming(_TIMING);
}
