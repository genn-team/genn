/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file HHVClamp.cc

\brief This file contains the model definition of HHVClamp model. It is used in both the GeNN code generation and the user side simulation code. The HHVClamp model implements a population of unconnected Hodgkin-Huxley neurons that evolve to mimick a model run on the CPU, using genetic algorithm techniques.
*/
//--------------------------------------------------------------------------

#include "modelSpec.h"
#include "HHVClampParameters.h"
#include "HHNeuronParameters.h"

class MyHH : public NeuronModels::Base
{
public:
    DECLARE_MODEL(MyHH, 0, 12);

    SET_SIM_CODE(
        "scalar Imem;\n"
        "unsigned int mt;\n"
        "scalar mdt= DT/100.0;\n"
        "scalar Icoupl;\n"
        "for (mt=0; mt < 100; mt++) {\n"
        "   Icoupl= 200.0*($(stepVG)-$(V));\n"
        "   Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n"
        "       $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n"
        "       $(gl)*($(V)-($(El)))-Icoupl);\n"
        "   scalar _a= (3.5+0.1*$(V)) / (1.0-exp(-3.5-0.1*$(V)));\n"
        "   scalar _b= 4.0*exp(-($(V)+60.0)/18.0);\n"
        "   $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n"
        "   _a= 0.07*exp(-$(V)/20.0-3.0);\n"
        "   _b= 1.0 / (exp(-3.0-0.1*$(V))+1.0);\n"
        "   $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n"
        "   _a= (-0.5-0.01*$(V)) / (exp(-5.0-0.1*$(V))-1.0);\n"
        "   _b= 0.125*exp(-($(V)+60.0)/80.0);\n"
        "   $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n"
        "   $(V)+= Imem/$(C)*mdt;\n"
        "}\n"
        "$(err)+= abs(Icoupl-$(IsynG));\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) > 100");

    SET_VARS({{"V", "scalar"},{"m", "scalar"},{"h","scalar"},{"n","scalar"},{"gNa","scalar"},
             {"ENa","scalar"},{"gK","scalar"},{"EK","scalar"},{"gl","scalar"},{"El","scalar"},
             {"C","scalar"},{"err","scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"stepVG", "scalar"}, {"IsynG", "scalar"}});
};
IMPLEMENT_MODEL(MyHH);

//--------------------------------------------------------------------------
/*! \brief This function defines the HH model with variable parameters.
 */
//--------------------------------------------------------------------------

void modelDefinition(ModelSpec &model) 
{
#ifdef DEBUG
    GENN_PREFERENCES.debugCode = true;
#else
    GENN_PREFERENCES.optimizeCode = true;
#endif // DEBUG

    model.setName("HHVClamp");
    model.setDT(0.25);
    model.setPrecision(_FTYPE);

    MyHH::VarValues myHH_ini(
        initialHHValues[0],                                 // 0 - membrane potential E
        initialHHValues[1],                                 // 1 - prob. for Na channel activation m
        initialHHValues[2],                                 // 2 - prob. for not Na channel blocking h
        initialHHValues[3],                                 // 3 - prob. for K channel activation n
        initVar<InitVarSnippet::Uniform>({1.0, 200.0}),     // 4 - gNa: Na conductance in 1/(mOhms * cm^2)
        initVar<InitVarSnippet::Uniform>({0.0, 100.0}),     // 5 - ENa: Na equi potential in mV
        initVar<InitVarSnippet::Uniform>({1.0, 100.0}),     // 6 - gK: K conductance in 1/(mOhms * cm^2)
        initVar<InitVarSnippet::Uniform>({-100.0, -20.0}),  // 7 - EK: K equi potential in mV
        initVar<InitVarSnippet::Uniform>({1.0, 50.0}),      // 8 - gl: leak conductance in 1/(mOhms * cm^2)
        initVar<InitVarSnippet::Uniform>({-100.0, -20.0}),  // 9 - El: leak equi potential in mV
        initVar<InitVarSnippet::Uniform>({1e-1, 10.0}),     // 10 - Cmem: membr. capacity density in muF/cm^2
        initialHHValues[11]);                               // 11 - error

    model.addNeuronPopulation<MyHH>("HH", NPOP, {}, myHH_ini);
}
