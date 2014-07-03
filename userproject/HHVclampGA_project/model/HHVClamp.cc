/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file MBody1.cc

\brief This file contains the model definition of the mushroom body "MBody1" model. It is used in both the GeNN code generation and the user side simulation code (class classol, file classol_sim).
*/
//--------------------------------------------------------------------------

#define DT 0.5  //!< This defines the global time step at which the simulation will run
#include "modelSpec.h"
#include "modelSpec.cc"
#include "HHVClampParameters.h"

float myHH_ini[11]= {
  -60.0,         // 0 - membrane potential E
  0.0529324,     // 1 - prob. for Na channel activation m
  0.3176767,     // 2 - prob. for not Na channel blocking h
  0.5961207,      // 3 - prob. for K channel activation n
  120.0,         // 4 - gNa: Na conductance in 1/(mOhms * cm^2)
  55.0,          // 5 - ENa: Na equi potential in mV
  36.0,          // 6 - gK: K conductance in 1/(mOhms * cm^2)
  -72.0,         // 7 - EK: K equi potential in mV
  0.3,           // 8 - gl: leak conductance in 1/(mOhms * cm^2)
  -50.0,         // 9 - El: leak equi potential in mV
  1.0            // 10 - Cmem: membr. capacity density in muF/cm^2
};

float *myHH_p= NULL;


//--------------------------------------------------------------------------
/*! \brief This function defines the HH model with variable parameters.
 */
//--------------------------------------------------------------------------

void modelDefinition(NNmodel &model) 
{
  neuronModel n;
  // HH neurons with adjustable parameters (introduced as variables)
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("m"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("h"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("n"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("gNa"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("ENa"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("gK"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("EK"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("gl"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("El"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("C"));
  n.varTypes.push_back(tS("double"));
  n.varNames.push_back(tS("err"));
  n.varTypes.push_back(tS("double"));
  n.extraGlobalNeuronKernelParameters.push_back(tS("stepVG"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("double"));
  n.extraGlobalNeuronKernelParameters.push_back(tS("IsynG"));
  n.extraGlobalNeuronKernelParameterTypes.push_back(tS("double"));

  n.simCode= tS("   double Imem;\n\
    unsigned int mt;\n\
    double mdt= DT/100.0;\n\
    for (mt=0; mt < 100; mt++) {\n\
      Isyn= 200.0*($(stepVG)-$(V));\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-Isyn);\n\
      double _a= (3.5+0.1*$(V)) / (1.0-exp(-3.5-0.1*$(V)));\n\
      double _b= 4.0*exp(-($(V)+60.0)/18.0);\n\
      $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;\n\
      _a= 0.07*exp(-$(V)/20.0-3.0);\n\
      _b= 1.0 / (exp(-3.0-0.1*$(V))+1.0);\n\
      $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;\n\
      _a= (-0.5-0.01*$(V)) / (exp(-5.0-0.1*$(V))-1.0);\n\
      _b= 0.125*exp(-($(V)+60.0)/80.0);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n\
    $(err)+= abs(Isyn-$(IsynG));\n");

  n.thresholdConditionCode = tS("$(V) > 100");//TODO check this, to get better value
  int HHV= nModels.size();
  nModels.push_back(n);

  model.setName("HHVClamp");
  model.setPrecision(DOUBLE);
  model.addNeuronPopulation("HH", NPOP, HHV, myHH_p, myHH_ini);
}
