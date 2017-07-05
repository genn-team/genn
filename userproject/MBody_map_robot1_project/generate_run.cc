/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/MBody1_project/generate_run.cc

\brief This file is part of a tool chain for running the classol/MBody1 example model.

This file compiles to a tool that wraps all the other tools into one chain of tasks, including running all the gen_* tools for generating connectivity, providing the population size information through ./model/sizes.h to the MBody1 model definition, running the GeNN code generation and compilation steps, executing the model and collecting some timing information. This tool is the recommended way to quickstart using GeNN as it only requires a single command line to execute all necessary tasks.
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

#include "stringUtils.h"
#include "command_line_processing.h"

//--------------------------------------------------------------------------
/*! \brief Main entry point for generate_run.
 */
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    if ((argc < 9) || (argc > 13))
  {
      cerr << "usage: generate_run <CPU=0, AUTO GPU=1, GPU n= \"n+2\"> <nAL> <nMB> <nLHI> <nLb> <gscale> <outdir> <model name> <OPTIONS> \n\
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
  int nAL = atoi(argv[2]);
  int nMB = atoi(argv[3]);
  int nLHI = atoi(argv[4]);
  int nLB = atoi(argv[5]);
  double gscale = atof(argv[6]);
  string outdir = toString(argv[7]) + "_output";  
  string modelName = argv[8];
 
  int argStart= 9;
#include "parse_options.h"  // parse options

  double pnkc_gsyn = 625.0 / nAL * gscale;
  double pnkc_gsyn_sigma = 300.0 / nAL * gscale / 15.0; 
  double kcdn_gsyn = 2500.0 / nMB * 0.05 * gscale; 
  double kcdn_gsyn_sigma = 2500.0 / (sqrt((double) 1000.0) * sqrt((double) nMB)) * 0.005 * gscale; 
  double pnlhi_theta = 100.0 / nAL * 4.0 * gscale;

  // write neuron population sizes
  string fname = "./model/sizes.h";
  ofstream os(fname.c_str());
  if (which > 1) {
      os << "#define nGPU " << which-2 << endl;
      which= 1;
  }
  os << "#define _NAL " << nAL << endl;
  os << "#define _NMB " << nMB << endl;
  os << "#define _NLHI " << nLHI << endl;
  os << "#define _NLB " << nLB << endl;
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
  cmd = "cd model && genn-buildmodel.bat ";
#else // UNIX
  cmd = "cd model && genn-buildmodel.sh ";
#endif
  cmd += modelName + ".cc";
  if (dbgMode) cmd += " -d";
  if (cpu_only) cmd += " -c";
#ifdef _WIN32
  cmd += " && nmake /nologo /f WINmakefile clean all ";
#else // UNIX
  cmd += " && make clean all ";
#endif
  cmd += "SIM_CODE=" + modelName + "_CODE";
  if (dbgMode) cmd += " DEBUG=1";
  if (cpu_only) cmd += " CPU_ONLY=1";
  cout << cmd << endl;
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

  if (fixsynapse == 0) {
      // generate pnkc synapses
      cmd = gennPath + "/userproject/tools/gen_pnkc_syns ";
      cmd += toString(nAL) + " ";
      cmd += toString(nMB) + " ";
      cmd += "0.5 ";
      cmd += toString(pnkc_gsyn) + " ";
      cmd += toString(pnkc_gsyn_sigma) + " ";
      cmd += outdir + "/" + toString(argv[7]) + ".pnkc 2>&1 ";
#ifndef _WIN32
      cmd += "|tee "+ outdir + "/" + toString(argv[7]) + ".pnkc.msg";
#endif // _WIN32
      retval=system(cmd.c_str());
      if (retval != 0){
	  cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
	  cerr << "Exiting..." << endl;
	  exit(1);
      }

      // generate kcdn synapses
      cmd = gennPath + "/userproject/tools/gen_kcdn_syns ";
      cmd += toString(nMB) + " ";
      cmd += toString(nLB) + " ";
      cmd += toString(kcdn_gsyn) + " ";
      cmd += toString(kcdn_gsyn_sigma) + " ";
      cmd += "1e-20 ";
      cmd += outdir + "/" + toString(argv[7]) + ".kcdn 2>&1 ";
#ifndef _WIN32
      cmd += "|tee "+ outdir + "/" + toString(argv[7]) + ".kcdn.msg";
#endif // _WIN32
      retval=system(cmd.c_str());
      if (retval != 0){
	  cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
	  cerr << "Exiting..." << endl;
	  exit(1);
      }
  // generate pnlhi synapses
      cmd = gennPath + "/userproject/tools/gen_pnlhi_syns ";
      cmd += toString(nAL) + " ";
      cmd += toString(nLHI) + " ";
      cmd += toString(pnlhi_theta) + " 1 ";
      cmd += outdir + "/" + toString(argv[7]) + ".pnlhi 2>&1 ";
#ifndef _WIN32
      cmd += "|tee "+ outdir + "/" + toString(argv[7]) + ".pnlhi.msg";
#endif // _WIN32
      retval = system(cmd.c_str());
      if (retval != 0){
	  cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
	  cerr << "Exiting..." << endl;
	  exit(1);
      }
  // generate input patterns
      cmd = gennPath + "/userproject/tools/gen_input_structured ";
      cmd += toString(nAL) + " ";
      cmd += "10 10 0.1 0.05 1.0 2e-04 ";
      cmd += outdir + "/" + toString(argv[7]) + ".inpat 2>&1 ";
#ifndef _WIN32
      cmd += "|tee "+ outdir + "/" + toString(argv[7]) + ".inpat.msg";
#endif // _WIN32
      retval = system(cmd.c_str());
      if (retval != 0){
	  cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
	  cerr << "Exiting..." << endl;
	  exit(1);
      }
  }
  // run it!
  cout << "running test..." << endl;
#ifdef _WIN32
  if (dbgMode == 1) {
      cmd = "devenv /debugexe model\\classol_sim.exe " + toString(argv[7]) + " " + toString(which);
  }
  else {
    cmd = "model\\classol_sim.exe " + toString(argv[7]) + " " + toString(which);
  }
#else // UNIX
  if (dbgMode == 1) {
    cmd = "cuda-gdb -tui --args model/classol_sim "+ toString(argv[7]) + " " + toString(which);
  }
  else {
    cmd = "model/classol_sim "+ toString(argv[7]) + " " + toString(which);
  }
#endif
  retval = system(cmd.c_str());
  if (retval != 0){
    cerr << "ERROR: Following call failed with status " << retval << ":" << endl << cmd << endl;
    cerr << "Exiting..." << endl;
    exit(1);
  }
  return 0;
}
