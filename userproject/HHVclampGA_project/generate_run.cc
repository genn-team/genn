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
  if (argc < 6)
  {
    cerr << "usage: generate_run <CPU=0, AUTO GPU=1, GPU n= \"n+2\"> <protocol> <nPop> <totalT> <outdir> <OPTIONS> \n\
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
  int protocol = atoi(argv[2]);
  int nPop = atoi(argv[3]);
  double totalT = atof(argv[4]);
  string outDir = toString(argv[5]) + "_output";  

   int argStart= 6;
#include "parse_options.h"  // parse options

  // write model parameters
  string fname = "model/HHVClampParameters.h";
  ofstream os(fname.c_str());
  os << "#define NPOP " << nPop << endl;
  os << "#define TOTALT " << totalT << endl;
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
  if (which > 1) {
      os << "#define fixGPU " << which-2 << endl;
  }
  os.close();
  
  // build it
#ifdef _WIN32
  cmd= "cd model && buildmodel.bat HHVClamp DEBUG=" + toString(dbgMode);
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }
  cmd += " && nmake /nologo /f WINmakefile clean && nmake /nologo /f WINmakefile";
  if (dbgMode == 1) {
    cmd += " DEBUG=1";
  }
  if (cpu_only) {
      cmd += " CPU_ONLY=1";
  }

#else // UNIX
  cmd = "cd model && buildmodel.sh HHVClamp DEBUG=" + toString(dbgMode);
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
    cmd = "model\\VClampGA.exe " + cmd;
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
