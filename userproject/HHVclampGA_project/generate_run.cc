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

  \brief This file is used to run the HHVclampGA model with a single command line.


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

#include "usertools.h"

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
 */
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc != 7)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <protocol> <nPop> <totalT> <outdir> <debug mode? (0/1)>" << endl;
    exit(1);
  }
  int retval;
  string cmd;
  string gennPath = getenv("GENN_PATH");
  string outDir = toString(argv[5]) + "_output";  
  int dbgMode = atoi(argv[6]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release

  int which = atoi(argv[1]);
  int protocol = atoi(argv[2]);
  int nPop = atoi(argv[3]);
  double totalT = atof(argv[4]);

  // write model parameters
  string fname = gennPath + "/userproject/include/HHVClampParameters.h";
  ofstream os(fname.c_str());
  os << "#define NPOP " << nPop << endl;
  os << "#define TOTALT " << totalT << endl;
  os.close();

  // build it
#ifdef _WIN32
  cmd= ensureCompilerEnvironmentCmd();
  cmd += " cd model && buildmodel.bat HHVClamp " + toString(dbgMode);
  cmd += " && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile";
  if (dbgMode == 1) {
    cmd += " DEBUG=1";
    cout << cmd << endl;
  }
#else // UNIX
  cmd = "cd model && buildmodel.sh HHVClamp " + toString(dbgMode);
  cmd += " && make clean && make";
  if (dbgMode == 1) {
    cmd += " debug";
  }
#endif
  cerr << cmd << endl;
  retval=system(cmd.c_str());
  if (retval != 0){
    cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
    cerr << "Exiting..." << endl;
    exit(1);
  }

  // create output directory
#ifdef _WIN32
  _mkdir(outDir.c_str());
#else // UNIX
  if (mkdir(outDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif

  // run it!
  cout << "running test..." << endl;
#ifdef _WIN32
  if (dbgMode == 1) {
    cmd = "devenv /debugexe model\\VClampGA.exe " + toString(argv[5]) + " " + toString(which) + " " + toString(protocol);
  }
  else {
    cmd = "model\\VClampGA.exe " + toString(argv[5]) + " " + toString(which) + " " + toString(protocol);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/VClampGA " + toString(argv[5]) + " " + toString(which) + " " + toString(protocol);
  }
  else {
    cmd = "model/VClampGA " + toString(argv[5]) + " " + toString(which) + " " + toString(protocol);
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
