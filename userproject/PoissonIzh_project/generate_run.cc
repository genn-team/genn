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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#define BUILDMODEL "buildmodel.bat"
#include <direct.h>
#include <stdlib.h>
#else // UNIX
#define BUILDMODEL "buildmodel.sh"
#include <sys/stat.h> // needed for mkdir
#endif

using namespace std;

//--------------------------------------------------------------------------
/*! \brief Template function for string conversion 
 */
//--------------------------------------------------------------------------

template<typename T> std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
 */
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc != 10)
  {
    cerr << "usage: generate_run_PoissonIzh <CPU=0, GPU=1> <nPoisson> <nIzh> <pConn> <gscale> <outdir> <executable name> <model name> <debug mode? (0/1)>";
    exit(1);
  }

  string cmd;

  int which = atoi(argv[1]);
  int nPoisson = atoi(argv[2]);
  int nIzh = atoi(argv[3]);
  float pConn = atof(argv[4]);
  float gscale = atof(argv[5]);
  string outdir = toString(argv[6]) +"_output";  
  string execName = argv[7];
  string modelName = argv[8];
  int DBGMODE = atoi(argv[9]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  
  float meangsyn = 100.0f / nPoisson * gscale;
  float gsyn_sigma = 100.0f / nPoisson * gscale / 15.0f; 

#ifdef _WIN32
  _mkdir(outdir.c_str());
#else 
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif
  
  // generate Poisson-Izhikevich synapses
  cmd = "$GENNPATH/userproject/" + modelName + "_project/gen_syns_sparse ";
  cmd += toString(nPoisson) + " ";
  cmd += toString(nIzh) + " ";
  cmd += toString(pConn) + " ";
  cmd += toString(meangsyn) + " ";
  cmd += toString(gsyn_sigma) + " ";
  cmd += outdir + "/g" + toString(argv[8]);
  system(cmd.c_str()); 
  cout << "connectivity generation script call was: " << cmd.c_str() << endl;

  // generate input patterns
  cmd = "$GENNPATH/userproject/" + modelName + "_project/gen_input_structured ";
  cmd += toString(nPoisson) + " ";
  cmd += "10 10 0.1 0.1 32768 17 ";
  cmd += outdir + "/" + toString(argv[8]) + ".inpat";
  cmd += " &> " + outdir + "/" + toString(argv[8]) + ".inpat.msg";
  system(cmd.c_str());

  string gennPath = getenv("GENNPATH");
  cerr << gennPath << endl;
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NPoisson " << nPoisson << endl;
  os << "#define _NIzh " << nIzh << endl;
  os.close();
  
  cmd = "cd $GENNPATH/userproject/" + modelName + "_project/model && " + BUILDMODEL + " " + modelName + " " + toString(DBGMODE);
  cout << "Debug mode " << DBGMODE << endl;
  cout << "script call was: " << cmd.c_str() << endl;
  system(cmd.c_str());
  cmd = "cd $GENNPATH/userproject/" + modelName + "_project/model && ";
  if (DBGMODE == 1) {
    cmd += "make clean && make debug";
  }
  else {
    cmd += "make clean && make";
  }
  system(cmd.c_str());

  // run it!
  cout << "running test..." << endl;

#if defined _WIN32
  if (DBGMODE == 1) { // debug
    cerr << "Debugging with gdb is not possible on cl platform." << endl;
  }
  else { // release
    cmd = "$GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }

#else // UNIX
  if(DBGMODE == 1) { // debug 
    cmd = "cuda-gdb -tui --args $GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }
  else { // release
    cmd = "$GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }

#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
}
