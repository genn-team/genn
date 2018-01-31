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

NeuronModels::PoissonNew::ParamValues myPOI_p(
//POISSON neuron parameters
    20.0        // 0 - firing rate [hZ]
);

NeuronModels::PoissonNew::VarValues myPOI_ini(
    0.0        // 0 - Time to spike
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
    uninitialisedVar() // Weights are initialised in simulator so don't initialise here
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
    model.addNeuronPopulation<NeuronModels::PoissonNew>("PN", _NPoisson, myPOI_p, myPOI_ini);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Izh1", _NIzh, exIzh_p, exIzh_ini);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>("PNIzh1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
                                                                                                "PN", "Izh1",
                                                                                                {}, mySyn_ini,
                                                                                                {}, {});
    model.setSeed(1234);
    model.setPrecision(_FTYPE);
    model.finalize();
}
