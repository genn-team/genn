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
#include <cfloat>
#include <locale>
using namespace std;

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else // UNIX
#include <sys/stat.h> // needed for mkdir
#endif

#include "command_line_processing.h"

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
 */
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc < 8)
  {
    cerr << "usage: generate_run <CPU=0, AUTO GPU=1, GPU n= \"n+2\"> <nPoisson> <nIzh> <pConn> <gscale> <outdir> <model name> <OPTIONS> \n\
Possible options: \n\
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger \n\
FTYPE=DOUBLE of FTYPE=FLOAT (default FLOAT): What floating point type to use \n\
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run \n\
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) \"CPU only\" mode." << endl;
    exit(1);
  }
  int retval;
  string cmd;
  string gennPath = getenv("GENN_PATH");
  int which = atoi(argv[1]);
  int nPoisson = atoi(argv[2]);
  int nIzh = atoi(argv[3]);
  float pConn = atof(argv[4]);
  float gscale = atof(argv[5]);
  string outdir = toString(argv[6]) + "_output";  
  string modelName = argv[7];

  int argStart= 8;
#include "parse_options.h"  // parse options
  
  float meangsyn = 100.0f / nPoisson * gscale;
  float gsyn_sigma = 100.0f / nPoisson * gscale / 15.0f; 

  // write neuron population sizes
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NPoisson " << nPoisson << endl;
  os << "#define _NIzh " << nIzh << endl;
  string tmps= tS(ftype);
  os << "#define _FTYPE " << "GENN_" << toUpper(tmps) << endl;
  os << "#define scalar " << toLower(tmps) << endl;
  if (toLower(ftype) == "double") {
      os << "#define SCALAR_MIN " << DBL_MIN << endl;
      os << "#define SCALAR_MAX " << DBL_MAX << endl;
  }
  else {
      os << "#define SCALAR_MIN " << FLT_MIN << "f" << endl;
      os << "#define SCALAR_MAX " << FLT_MAX << "f" << endl;
  } 

  os.close();

  // build it
#ifdef _WIN32
  cmd = "cd model && buildmodel.bat " + modelName + " DEBUG=" + toString(dbgMode);
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }
  cmd += " && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile ";
  if (dbgMode == 1) {
    cmd += " DEBUG=1";
  }
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }
#else // UNIX
  cmd = "cd model && buildmodel.sh " + modelName + " DEBUG=" + toString(dbgMode);
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }
  cmd += " && make clean && make";
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }
  else {
      if (dbgMode == 1) {
	  cmd += " debug";
      }  
      else {
	  cmd += " release";
      }
  }
#endif
  retval=system(cmd.c_str());
  if (retval != 0){
    cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
    cerr << "Exiting..." << endl;
    exit(1);
  }

  // create output directory
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
  cmd += outdir + "/g" + toString(argv[7]);
  retval=system(cmd.c_str());
  if (retval != 0){
    cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
    cerr << "Exiting..." << endl;
    exit(1);
  }

  // run it!
  cout << "running test..." << endl;
#ifdef _WIN32
  if (dbgMode == 1) {
    cmd = "devenv /debugexe model\\PoissonIzh_sim.exe " + toString(argv[6]) + " " + toString(which);
  }
  else {
    cmd = "model\\PoissonIzh_sim.exe " + toString(argv[6]) + " " + toString(which);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/PoissonIzh_sim " + toString(argv[6]) + " " + toString(which);
  }
  else {
    cmd = "model/PoissonIzh_sim " + toString(argv[6]) + " " + toString(which);
  }
#endif
  retval=system(cmd.c_str());
  if (retval != 0){
    cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
    cerr << "Exiting..." << endl;
    exit(1);
  }

  return 0;
}
