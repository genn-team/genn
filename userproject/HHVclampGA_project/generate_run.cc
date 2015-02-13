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
#include "toString.h"

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else // UNIX
#include <sys/stat.h> // needed for mkdir
#endif

using namespace std;

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
 */
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  if (argc != 8)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <protocol> <nPop> <totalT> <outdir> <debug mode? (0/1)> <GPU choice>" << endl;
    exit(1);
  }
  int which = atoi(argv[1]);
  int protocol = atoi(argv[2]);
  int nPop = atoi(argv[3]);
  double totalT = atof(argv[4]);
  string outDir = toString(argv[5]) + "_output";  
  int dbgMode = atoi(argv[6]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  int GPU= atoi(argv[7]);
  int retval;
  string cmd;

  string GeNNPath= getenv("GeNNPATH");
  cerr << GeNNPath << endl;
  // write model parameters
  string fname = "model/HHVClampParameters.h";
  ofstream os(fname.c_str());
  os << "#define NPOP " << nPop << endl;
  os << "#define TOTALT " << totalT << endl;
  os << "#define fixGPU " << GPU << endl;
  os.close();

  // build it
#ifdef _WIN32
  cmd= "cd model && buildmodel.bat HHVClamp " + toString(dbgMode);
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
  else {
    cmd += " release";
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
  cmd= toString(argv[5]) + " " + toString(which) + " " + toString(protocol);
#ifdef _WIN32
  if (dbgMode == 1) {
    cmd = "devenv /debugexe model\\VClampGA.exe " + cmd;
  }
  else {
    cmd = "model\\VClampGA.exe " + cmd);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/VClampGA " + cmd;
  }
  else {
    cmd = "model/VClampGA " + cmd;
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
