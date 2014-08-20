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

This file compiles to a tool that wraps all the other tools into one chain of tasks, including running all the gen_* tools for generating connectivity, providing the population size information through ../userproject/include/sizes.h to the MBody1 model definition, running the GeNN code generation and compilation steps, executing the model and collecting some timing information. This tool is the recommended way to quickstart using GeNN as it only requires a single command line to execute all necessary tasks.
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
#include <sys/stat.h> // needed for mkdir?
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
  if (argc != 7)
  {
    cerr << "usage: generate_run_1comp <CPU=0, GPU=1> <nC1> <outdir> <executable name> <model name> <debug mode? (0/1)>";
    exit(1);
  }

  int DBGMODE = atoi(argv[6]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  string cmd;
  string execName = argv[4];
  string modelName = argv[5];
  
  int which = atoi(argv[1]);
  int nC1 = atoi(argv[2]);
  string outdir = toString(argv[3]) + "_output";  
  
#ifdef _WIN32
  _mkdir(outdir.c_str());
#else // UNIX
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif
  
  string gennPath = getenv("GENNPATH");
  cerr << gennPath << endl;
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NC1 " << nC1 << endl;
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

#ifdef _WIN32
  if (DBGMODE == 1) { // debug
    cerr << "Debugging with gdb is not possible on cl platform." << endl;
  }
  else { // release
    cmd = "$GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }

#else // UNIX
  if (DBGMODE == 1) { // debug 
    cmd = "cuda-gdb -tui --args $GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }
  else { // release
    cmd = "$GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }

#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
}
