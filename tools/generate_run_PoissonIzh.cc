/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generate_run.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool that wraps all the other tools into one chain of tasks, including running all the gen_* tools for generating connectivity, providing the population size information through ../userproject/include/sizes.h to the model definition, running the GeNN code generation and compilation steps, executing the model and collecting some timing information. This tool is the recommended way to quickstart using Poisson-Izhikevich example in GeNN as it only requires a single command line to execute all necessary tasks.
*/ 
//--------------------------------------------------------------------------

using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else
#include <sys/stat.h> //needed for mkdir
#endif

//--------------------------------------------------------------------------
/*! \brief Template function for string conversion 
 */
//--------------------------------------------------------------------------

template<typename T>
std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 


int main(int argc, char *argv[])
{
  if (argc != 10)
  {
    cerr << "usage: generate_run_PoissonIzh <CPU=0, GPU=1> <nPoisson> <nIzh> <pConn> <gscale> <outdir> <executable name> <model name> <debug mode? (0/1)>";
    exit(1);
  }

  string cmd;
  
  int which= atoi(argv[1]);
  int nPoisson= atoi(argv[2]);
  int nIzh= atoi(argv[3]);
  float pConn= atof(argv[4]);
  float gscale= atof(argv[5]);
  string OutDir = toString(argv[6]) +"_output";  
  string execName = argv[7];
  string modelName = argv[8];
  int DBGMODE = atoi(argv[9]); //set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  
  float meangsyn= 100.0f/nPoisson*gscale;
  float gsyn_sigma= 100.0f/nPoisson*gscale/15.0f; 

  #ifdef _WIN32
  _mkdir(OutDir.c_str());
  #else 
  if (mkdir(OutDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH)==-1){
  	cerr << "Directory cannot be created. It may exist already." << endl;
  	}; 
  #endif
  
  // generate Poisson-Izhikevich synapses
  cmd= toString("$GeNNPATH/tools/gen_syns_sparse ");
  cmd+= toString(nPoisson) + toString(" ") ;
  cmd+= toString(nIzh) + toString(" ") ;
  cmd+= toString(pConn) + toString(" ") ;
  cmd+= toString(meangsyn) + toString(" ") ;
  cmd+= toString(gsyn_sigma) + toString(" ") ;
  cmd+= OutDir+ "/g"+toString(argv[8]);
  system(cmd.c_str()); 
  cout << "connectivity generation script call was:" << cmd.c_str() << endl;
  
    // generate input patterns
  cmd= toString("$GeNNPATH/tools/gen_input_fixfixfixno_struct ");
  cmd+= toString(nPoisson) + toString(" ") ;
  cmd+= toString("10 10 0.1 0.1 32768 17 ") ;
  cmd+= OutDir+ "/"+ toString(argv[8]) + toString(".inpat");
  cmd+= toString(" &> ") + OutDir+ "/"+ toString(argv[8]) + toString(".inpat.msg");
  system(cmd.c_str());

  string GeNNPath= getenv("GeNNPATH");
  cerr << GeNNPath << endl;
  string fname= GeNNPath+string("/userproject/include/sizes.h");
  ofstream os(fname.c_str());
  os << "#define _NPoisson " << nPoisson << endl;
  os << "#define _NIzh " << nIzh << endl;
  os.close();
  
  cmd= toString("cd $GeNNPATH/userproject/")+toString(modelName)+("_project && buildmodel ")+ toString(modelName)+ toString(" ") + toString(DBGMODE);

  
  cout << "Debug mode " << DBGMODE << endl;

  cout << "script call was:" << cmd.c_str() << endl;
  system(cmd.c_str());
  cmd= toString("cd $GeNNPATH/userproject/")+modelName+("_project && ");
  if(DBGMODE==1) {
		cmd+= toString("make clean debug && make debug");
  }
  else{
		cmd+= toString("make clean && make");  
  	}	
  system(cmd.c_str());

  cmd= toString("echo $GeNNOSTYPE");
  system(cmd.c_str());

  // run it!
  cout << "running test..." <<endl;
#if defined _WIN32 || defined __CYGWIN__
  //cout << "win32" <<endl;
  if(DBGMODE==1) {
	cerr << "Debugging with gdb is not possible on cl platform." << endl;
	}
	else {
  		cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[6]) + toString(" ") + toString(which);
	}

#else
  //cout << "not win" <<endl;
  //cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); ../userproject/$GeNNOSTYPE/release/classol_sim ")+  toString(argv[6]) + toString(" ") + toString(which);
   if(DBGMODE==1) {
  //debug 
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); cuda-gdb -tui --args $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/debug/")+execName + toString(" ")+  toString(argv[6]) + toString(" ") + toString(which);
  }
  else
  {
//release  
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[6]) + toString(" ") + toString(which);
  	}
#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
  
}

