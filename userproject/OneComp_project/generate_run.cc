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
  if (argc != 6)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <nC1> <outdir> <model name> <debug mode? (0/1)>" << endl;
    exit(1);
  }
  int retval;
  string cmd;
  string gennPath = getenv("GENN_PATH");
  string outdir = toString(argv[3]) + "_output";  
  string modelName = argv[4];
  int dbgMode = atoi(argv[5]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release

  int which = atoi(argv[1]);
  int nC1 = atoi(argv[2]);

  // write neuron population sizes
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NC1 " << nC1 << endl;
  os.close();

  // build it  
#ifdef _WIN32
  cmd = "cd model && buildmodel.bat " + modelName + " " + toString(dbgMode);
  cmd += " && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile";
  if (dbgMode == 1) {
    cmd += " DEBUG=1";
  }
#else // UNIX
  cmd = "cd model && buildmodel.sh " + modelName + " " + toString(dbgMode);
  cmd += " && make clean && make release";
  if (dbgMode == 1) {
    cmd += " debug";
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
  
  // run it!
  cout << "running test..." << endl;
#ifdef _WIN32
  if (dbgMode == 1) {
    cmd = "devenv /debugexe model\\OneComp_sim.exe " + toString(argv[3]) + " " + toString(which);
  }
  else {
    cmd = "model\\OneComp_sim.exe " + toString(argv[3]) + " " + toString(which);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/OneComp_sim " + toString(argv[3]) + " " + toString(which);
  }
  else {
    cmd = "model/OneComp_sim " + toString(argv[3]) + " " + toString(which);
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
