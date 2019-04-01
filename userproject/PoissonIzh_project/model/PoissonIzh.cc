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

// Create variable initialisation snippet to zero all weights aside from those
// that pass a fixed probability test. Sample those from the normal distribution.
class GaussianFixedProbability : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(GaussianFixedProbability, 3);

    SET_CODE(
        "const scalar r = $(gennrand_uniform);\n"
        "if(r < $(pconn)) {\n"
        "   $(value) = $(gsynMean) + ($(gennrand_normal) * $(gsynSD));\n"
        "}\n"
        "else {\n"
        "   $(value) = 0.0;\n"
        "}\n");

    SET_PARAM_NAMES({"pconn", "gsynMean", "gsynSD"});
};
IMPLEMENT_SNIPPET(GaussianFixedProbability);

void modelDefinition(ModelSpec &model) 
{
#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG

    // POISSON neuron parameters
    NeuronModels::PoissonNew::ParamValues myPOI_p(
        20.0);      // 0 - firing rate [hZ]


    NeuronModels::PoissonNew::VarValues myPOI_ini(
        0.0);       // 0 - Time to spike


    // Izhikevich model parameters - tonic spiking
    NeuronModels::Izhikevich::ParamValues exIzh_p(
        0.02,       // 0 - a
        0.2,        // 1 - b
        -65,        // 2 - c
        6);         // 3 - d

    // Izhikevich model initial conditions - tonic spiking
    NeuronModels::Izhikevich::VarValues exIzh_ini(
        -65,        //0 - V
        -20);       //1 - U

    // Gaussiuan fixed probability var initialiser parameters
    GaussianFixedProbability::ParamValues mySyn_connectivity_p(
        _PConn,                                 // 0 -Probability of connection
        100.0f / _NPoisson * _GScale,           // 1 - GSyn mean
        100.0f / _NPoisson * _GScale / 15.0f);  // 2 - GSyn S.D.

    // Initialise weights using snippet and parameterss
    WeightUpdateModels::StaticPulse::VarValues mySyn_ini(
        initVar<GaussianFixedProbability>(mySyn_connectivity_p));

    model.setName("PoissonIzh");
    model.setDT(1.0);
    model.addNeuronPopulation<NeuronModels::PoissonNew>("PN", _NPoisson, myPOI_p, myPOI_ini);
    model.addNeuronPopulation<NeuronModels::Izhikevich>("Izh1", _NIzh, exIzh_p, exIzh_ini);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "PNIzh1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "PN", "Izh1",
        {}, mySyn_ini,
        {}, {});
    model.setSeed(1234);
    model.setPrecision(_FTYPE);
}
