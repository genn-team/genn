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
#include <direct.h>
#include <stdlib.h>
#else // UNIX
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

  string cmd;
  string gennPath = getenv("GENNPATH");
  string outdir = toString(argv[3]) + "_output";  
  string execName = argv[4];
  string modelName = argv[5];
  int dbgMode = atoi(argv[6]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release

#ifdef _WIN32
  const string buildModel = "buildmodel.bat";
#else // UNIX
  const string buildModel = "buildmodel.sh";
#endif
  
  int which = atoi(argv[1]);
  int nC1 = atoi(argv[2]);
  
#ifdef _WIN32
  _mkdir(outdir.c_str());
#else // UNIX
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif
  
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NC1 " << nC1 << endl;
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
    cmd = "model/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args $GENNPATH/userproject/" + modelName + "_project/model/bin/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }
  else {
    cmd = "model/" + execName + " " + toString(argv[3]) + " " + toString(which);
  }
#endif
  system(cmd.c_str());

  return 0;
}
