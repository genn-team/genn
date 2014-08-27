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
#include <direct.h>
#include <stdlib.h>
#else // UNIX
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
  string gennPath = getenv("GENNPATH");
  string outdir = toString(argv[6]) + "_output";  
  string execName = argv[7];
  string modelName = argv[8];
  int dbgMode = atoi(argv[9]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release

#ifdef _WIN32
  const string buildModel = "buildmodel.bat";
#else // UNIX
  const string buildModel = "buildmodel.sh";
#endif

  int which = atoi(argv[1]);
  int nPoisson = atoi(argv[2]);
  int nIzh = atoi(argv[3]);
  float pConn = atof(argv[4]);
  float gscale = atof(argv[5]);
  
  float meangsyn = 100.0f / nPoisson * gscale;
  float gsyn_sigma = 100.0f / nPoisson * gscale / 15.0f; 

#ifdef _WIN32
  _mkdir(outdir.c_str());
#else // UNIX
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif
  
  // generate Poisson-Izhikevich synapses
  cmd = gennPath + "/userproject/tools/gen_syns_sparse ";
  cmd += toString(nPoisson) + " ";
  cmd += toString(nIzh) + " ";
  cmd += toString(pConn) + " ";
  cmd += toString(meangsyn) + " ";
  cmd += toString(gsyn_sigma) + " ";
  cmd += outdir + "/g" + toString(argv[8]);
  system(cmd.c_str());
  cout << "connectivity generation script call was: " << cmd.c_str() << endl;

  // generate input patterns
  cmd = gennPath + "/userproject/tools/gen_input_structured ";
  cmd += toString(nPoisson) + " ";
  cmd += "10 10 0.1 0.1 32768 17 ";
  cmd += outdir + "/" + toString(argv[8]) + ".inpat";
  cmd += " &> " + outdir + "/" + toString(argv[8]) + ".inpat.msg";
  system(cmd.c_str());

  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NPoisson " << nPoisson << endl;
  os << "#define _NIzh " << nIzh << endl;
  os.close();

  // build it
  cmd = "cd model && " + buildModel + " " + modelName + " " + toString(dbgMode);
  system(cmd.c_str());
#ifdef _WIN32
  if (dbgMode == 1) {
    cmd = "cd model && nmake /f WINmakefile clean && nmake /f WINmakefile debug";
  }
  else {
    cmd = "cd model && nmake /f WINmakefile clean && nmake /f WINmakefile";
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cd model && make clean && make debug";
  }
  else {
    cmd = "cd model && make clean && make";
  }
#endif
  system(cmd.c_str());

  // run it!
  cout << "running test..." << endl;
#ifdef _WIN32
  if (dbgMode == 1) {
    cerr << "Debugging mode is not yet supported on Windows." << endl;
    exit(1);
  }
  else {
    cmd = "model/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }
  else {
    cmd = "model/" + execName + " " + toString(argv[6]) + " " + toString(which);
  }
#endif
  system(cmd.c_str());

  return 0;
}
