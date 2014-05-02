
/*--------------------------------------------------------------------------
   Author/Modifier: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef _UTILS_H_
#define _UTILS_H_ //!< macro for avoiding multiple inclusion during compilation


//--------------------------------------------------------------------------
/*! \file utils.h

\brief This file contains standard utility functions provide within the NVIDIA CUDA software development toolkit (SDK). The remainder of the file contains a function that defines the standard neuron models.
*/
//--------------------------------------------------------------------------

#include <cstdlib> // for exit() and EXIT_FAIL / EXIT_SUCCESS
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>

#include "modelSpec.h"
#include "toString.h"


//--------------------------------------------------------------------------
/* \brief Macro for a "safe" output of a parameter into generated code by essentially just
   adding a bracket around the parameter value in the generated code.
 */
//--------------------------------------------------------------------------
 
#define SAVEP(X) "(" << X << ")" 


//--------------------------------------------------------------------------
/* \brief Macro for wrapping cuda runtime function calls and catching any errors that may be thrown.
 */
//--------------------------------------------------------------------------

#define CHECK_CUDA_ERRORS(call)					           \
{								      	   \
  cudaError_t error = call;						   \
  if (error != cudaSuccess)						   \
  {                                                                        \
    fprintf(stderr, "%s: %i: cuda error %i: %s\n",			   \
	    __FILE__, __LINE__, (int)error, cudaGetErrorString(error));	   \
    exit(EXIT_FAILURE);						           \
  }									   \
}

//--------------------------------------------------------------------------
/* \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &os) 
{
  string s;
  ifstream is("header.src");
  getline(is, s);
  while (is.good()) {
    os << s << endl;
    getline(is, s);
  }
  os << endl;
}


//--------------------------------------------------------------------------
//! \brief Tool for substituting strings in the neuron code strings or other templates
//--------------------------------------------------------------------------

void substitute(string &s, const string trg, const string rep)
{
  size_t found= s.find(trg);
  while (found != string::npos) {
    s.replace(found,trg.length(),rep);
    found= s.find(trg);
  }
}

//--------------------------------------------------------------------------
//! \brief Tool for determining the size of variable types on the current architecture
//--------------------------------------------------------------------------

unsigned int theSize(string type) 
{
  unsigned int size = 0;
  if (type == "int") size = sizeof(int);
  if (type == "unsigned int") size = sizeof(unsigned int);
  if (type == "float") size = sizeof(float);
  if (type == "double") size = sizeof(double);
  if (type == "long double") size = sizeof(long double);
  return size;
}


//--------------------------------------------------------------------------
/*! \brief Function that defines standard neuron models

The neuron models are defined and added to the C++ vector nModels that is holding all neuron model descriptions. User defined neuron models can be appended to this vector later in (a) separate function(s).
*/
//--------------------------------------------------------------------------
//NOTE: calcSynapses takes the first variable of each model.neuronName[src] as an argument, of type float. If you add a neuron model, keep this in mind. 

vector<neuronModel> nModels; //!< Global c++ vector containing all neuron model descriptions

class rulkovdp : public dpclass
{
public:
	float calculateDerivedParameter(int index, vector <float> pars, float dt = 1.0) {
		switch (index) {
			case 0:
			return ip0(pars);
			case 1:
			return ip1(pars);
			case 2:
			return ip2(pars);
		}
		return -1;
	}

	float ip0(vector<float> pars) {
		return pars[0]*pars[0]*pars[1];
	}
	float ip1(vector<float> pars) {
		return pars[0]*pars[2];
	}
	float ip2(vector<float> pars) {
		return pars[0]*pars[1]+pars[0]*pars[2];
	}
};

void prepareStandardModels()
{
  neuronModel n;

  //Rulkov neurons
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("preV"));
  n.varTypes.push_back(tS("float"));
  n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("alpha"));
  n.pNames.push_back(tS("y"));
  n.pNames.push_back(tS("beta"));
  n.dpNames.push_back(tS("ip0"));
  n.dpNames.push_back(tS("ip1"));
  n.dpNames.push_back(tS("ip2"));
  n.simCode= tS("    if ($(V) <= 0.0) {\n\
      $(preV)= $(V);\n\
      $(V)= $(ip0)/(($(Vspike)) - $(V) - ($(beta))*$(Isyn)) +($(ip1));\n\
    }\n\
    else {\n\
      if (($(V) < $(ip2)) && ($(preV) <= 0.0)) {\n\
        $(preV)= $(V);\n\
        $(V)= $(ip2);\n\
      }\n\
      else {\n\
        $(preV)= $(V);\n\
        $(V)= -($(Vspike));\n\
      }\n\
    }\n");

  n.thresholdConditionCode = tS("$(V) > $(ip2) - 0.01");

  n.dps = new rulkovdp();

  nModels.push_back(n);

  // Poisson neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("seed"));
  n.varTypes.push_back(tS("unsigned int"));
  n.varNames.push_back(tS("spikeTime"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.pNames.push_back(tS("therate"));
  n.pNames.push_back(tS("trefract"));
  n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("Vrest"));
  n.dpNames.clear();
  n.simCode= tS("    unsigned int theRnd;\n\
    if ($(V) > $(Vrest)) {\n\
      $(V)= $(Vrest);\n\
    }\n\
    else {\n\
      if (t - $(spikeTime) > ($(trefract))) {\n\
        RAND($(seed),theRnd);\n\
        if (theRnd < lrate) {\n\
          $(V)= $(Vspike);\n\
          $(spikeTime)= t;\n\
        }\n\
      }\n\
    }\n");

  n.thresholdConditionCode = tS("$(V) > $(Vspike) - 0.01");

  nModels.push_back(n);

// Traub and Miles HH neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("m"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("h"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("n"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.pNames.push_back(tS("gNa"));
  n.pNames.push_back(tS("ENa"));
  n.pNames.push_back(tS("gK"));
  n.pNames.push_back(tS("EK"));
  n.pNames.push_back(tS("gl"));
  n.pNames.push_back(tS("El"));
  n.pNames.push_back(tS("C"));
  n.dpNames.clear();
  n.simCode= tS("   float Imem;\n\
    unsigned int mt;\n\
    float mdt= DT/25.0f;\n\
    for (mt=0; mt < 25; mt++) {\n\
      Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+\n\
              $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+\n\
              $(gl)*($(V)-($(El)))-Isyn);\n\
      float _a= 0.32f*(-52.0f-$(V)) / (exp((-52.0f-$(V))/4.0f)-1.0f);\n\
      float _b= 0.28f*($(V)+25.0f)/(exp(($(V)+25.0f)/5.0f)-1.0f);\n\
      $(m)+= (_a*(1.0f-$(m))-_b*$(m))*mdt;\n\
      _a= 0.128*expf((-48.0f-$(V))/18.0f);\n\
      _b= 4.0f / (expf((-25.0f-$(V))/5.0f)+1.0f);\n\
      $(h)+= (_a*(1.0f-$(h))-_b*$(h))*mdt;\n\
      _a= .032f*(-50.0f-$(V)) / (expf((-50.0f-$(V))/5.0f)-1.0f); \n\
      _b= 0.5f*expf((-55.0f-$(V))/40.0f);\n\
      $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;\n\
      $(V)+= Imem/$(C)*mdt;\n\
    }\n");

  n.thresholdConditionCode = tS("$(V) > 20");//TODO check this, to get better value

  nModels.push_back(n);
  
 //Izhikevich neurons
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));  
  n.varNames.push_back(tS("U"));
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  //n.pNames.push_back(tS("Vspike"));
  n.pNames.push_back(tS("a")); // time scale of U
  n.pNames.push_back(tS("b")); // sensitivity of U
  n.pNames.push_back(tS("c")); // after-spike reset value of V
  n.pNames.push_back(tS("d")); // after-spike reset value of U
  n.dpNames.clear(); 
  //TODO: replace the resetting in the following with BRIAN-like threshold and resetting 
  n.simCode= tS("    if ($(V) >= 30){\n\
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
   // if ($(V) > 30){   //keep this only for visualisation -- not really necessaary otherwise \n\
    //  $(V)=30; \n\
   //}\n\
   ");
    
  n.thresholdConditionCode = tS("$(V) >= 29.99");

 /*  n.resetCode=tS("//reset code is here\n ");
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
  */
  nModels.push_back(n);

//Izhikevich neurons with variable parameters
  n.varNames.clear();
  n.varTypes.clear();
  n.varNames.push_back(tS("V"));
  n.varTypes.push_back(tS("float"));  
  n.varNames.push_back(tS("U"));
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("a")); // time scale of U
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("b")); // sensitivity of U
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("c")); // after-spike reset value of V
  n.varTypes.push_back(tS("float"));
  n.varNames.push_back(tS("d")); // after-spike reset value of U
  n.varTypes.push_back(tS("float"));
  n.pNames.clear();
  n.dpNames.clear(); 
  //TODO: replace the resetting in the following with BRIAN-like threshold and resetting 
  n.simCode= tS("    if ($(V) >= 30){\n\
      $(V)=$(c);\n\
		  $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn))*DT; //at two times for numerical stability\n\
    $(V)+=0.5f*(0.04f*$(V)*$(V)+5*$(V)+140-$(U)+$(Isyn))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
    //if ($(V) > 30){      //keep this only for visualisation -- not really necessaary otherwise \n\
    //  $(V)=30; \n\
    //}\n\
    ");
  n.thresholdConditionCode = tS("$(V) > 29.99");
  nModels.push_back(n);
  
  #include "extra_neurons.h"

}

class expDecayDp : public dpclass
{
public:
	float calculateDerivedParameter(int index, vector <float> pars, float dt = 1.0) {
		switch (index) {
			case 0:
			return expDecay(pars, dt);
		}
		return -1;
	}

	float expDecay(vector<float> pars, float dt) {
		return expf(-dt/pars[0]);
	}
};



vector<postSynModel> postSynModels;

void preparePostSynModels(){
  postSynModel ps;
  
  //0: Exponential decay
  ps.varNames.clear();
  ps.varTypes.clear();
  
  //ps.varNames.push_back(tS("E"));  
  //ps.varTypes.push_back(tS("float"));  
  
  ps.pNames.clear();
  ps.dpNames.clear(); 
  
  ps.pNames.push_back(tS("tau")); 
  ps.pNames.push_back(tS("E"));  
  ps.dpNames.push_back(tS("expDecay"));
  
  ps.postSynDecay=tS(" 	 $(inSyn)*=$(expDecay);\n");
  ps.postSyntoCurrent=tS("$(inSyn)*($(E)-$(V))");
  
  ps.dps = new expDecayDp;
  
  postSynModels.push_back(ps);
  
  
  //1: IZHIKEVICH MODEL (NO POSTSYN RULE)
  ps.varNames.clear();
  ps.varTypes.clear();
  
  ps.pNames.clear();
  ps.dpNames.clear(); 
  
  ps.postSynDecay=tS("");
  ps.postSyntoCurrent=tS("$(inSyn); $(inSyn)=0");
  
  postSynModels.push_back(ps);
  
  #include "extra_postsynapses.h"
}

void prepareSynapseModels(){
}
// bit tool macros
#include "numlib/simpleBit.h"

#endif  // _UTILS_H_
