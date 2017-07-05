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
#include "sizes.h"

NeuronModels::Poisson::ParamValues myPOI_p(
//POISSON neuron parameters
  1,        // 0 - firing rate
  2.5,        // 1 - refratory period
  20.0,       // 2 - Vspike
  -60.0       // 3 - Vrest
);

NeuronModels::Poisson::VarValues myPOI_ini(
 -60.0,        // 0 - V
  0,           // 1 - seed
  -10.0       // 2 - SpikeTime
);

NeuronModels::Izhikevich::ParamValues exIzh_p(
//Izhikevich model parameters - tonic spiking
    0.02,       // 0 - a
    0.2,        // 1 - b
    -65,        // 2 - c
    6           // 3 - d
);

NeuronModels::Izhikevich::VarValues exIzh_ini(
//Izhikevich model initial conditions - tonic spiking
    -65,        //0 - V
    -20         //1 - U
);

WeightUpdateModels::StaticPulse::VarValues mySyn_ini(
  0.0 //initial values of g
);


void modelDefinition(NNmodel &model) 
{
  initGeNN();

#ifdef DEBUG
  GENN_PREFERENCES::debugCode = true;
#else
  GENN_PREFERENCES::optimizeCode = true;
#endif // DEBUG

  model.setName("PoissonIzh");
  model.setDT(1.0);
  model.addNeuronPopulation<NeuronModels::Poisson>("PN", _NPoisson, myPOI_p, myPOI_ini);
  model.addNeuronPopulation<NeuronModels::Izhikevich>("Izh1", _NIzh, exIzh_p, exIzh_ini);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("PNIzh1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                             "PN", "Izh1",
                                                                                             {}, mySyn_ini,
                                                                                             {}, {});
  //model.setSynapseG("PNIzh1", gPNIzh1);
  model.setSeed(1234);
  model.setPrecision(_FTYPE);
  model.finalize();
}
