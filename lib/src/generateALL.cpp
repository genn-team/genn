/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#include "global.h"
#include "utils.h"


#include "currentModel.cc"

#include "generateKernels.cc"
#include "generateRunner.cc"
#include "generateCPU.cc"

void generate_model_runner(NNmodel &model, string path)
{
  string cmd, name;
  
  #ifdef _WIN32
  cmd= toString("mkdir ")+ path + toString("\\") + model.name + toString("_CODE");
#else
  cmd= toString("mkdir -p ")+ path + toString("/") + model.name + toString("_CODE");
#endif

  cerr << cmd << endl;
  system(cmd.c_str());
  
/* TODO: change the code above to direct system calls 
  #ifdef _WIN32
  cmd= path + toString("\\") + model.name + toString("_CODE");
  _mkdir(cmd.c_str());
  #else 
  cmd= path + toString("/") + model.name + toString("_CODE");
  mkdir(cmd.c_str(), 0777); 
  #endif
*/  


  // general/ shared code for GPU and CPU versions
  name= path + toString("/") + model.name + toString("_CODE/runner.cc");
  // cerr << name.c_str() << endl;
  ofstream os(name.c_str());
  genRunner(model, os, cerr);
  os.close();

  // GPU specific code generation
  name= path + toString("/") + model.name + toString("_CODE/runnerGPU.cc");
  os.open(name.c_str());
  genRunnerGPU(model, os, cerr);
  os.close();
  
  name= path + toString("/") + model.name + toString("_CODE/neuronKrnl.cc");
  os.open(name.c_str());
  genNeuronKernel(model, os, cerr);
  os.close();
  
  name= path + toString("/") + model.name + toString("_CODE/synapseKrnl.cc");
  os.open(name.c_str());
  genSynapseKernel(model, os, cerr);
  os.close();

  // CPU specific code generation
  name= path + toString("/") + model.name + toString("_CODE/runnerCPU.cc");
  os.open(name.c_str());
  genRunnerCPU(model, os, cerr);
  os.close();

  name= path + toString("/") + model.name + toString("_CODE/neuronFnct.cc");
  os.open(name.c_str());
  genNeuronFunction(model, os, cerr);
  os.close();
  
  name= path + toString("/") + model.name + toString("_CODE/synapseFnct.cc");
  os.open(name.c_str());
  genSynapseFunction(model, os, cerr);
  os.close();
}


int main(int argc, char *argv[])
{
  if (argc != 2) {
    cerr << "usage: generateALL <target dir>" << endl;
    exit(1);
  }
  cerr << "call was ";
  for (int i= 0; i < argc; i++) {
    cerr << argv[i] << " ";
  }
  cerr << endl;

  string path= toString(argv[1]);
  prepare(cerr);
  prepareStandardModels();

  NNmodel model;
  	  
  modelDefinition(model);
  cerr << "model allocated" << endl;
	
  generate_model_runner(model, path);
  
  return 0;
}

