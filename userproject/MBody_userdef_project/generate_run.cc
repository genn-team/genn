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

\brief This file is part of a tool chain for running the classol/MBody_userdef example model.

This file compiles to a tool that wraps all the other tools into one chain of tasks, including running all the gen_* tools for generating connectivity, providing the population size information through ../userproject/include/sizes.h to the MBody_userdef model definition, running the GeNN code generation and compilation steps, executing the model and collecting some timing information. This tool is the recommended way to quickstart using GeNN as it only requires a single command line to execute all necessary tasks.
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
  if (!(argc == 10 || argc == 11))
  {
    cerr << "usage: generate_run <CPU=0, AUTO GPU=1, GPU n= \"n+2\"> <nAL> <nMB> <nLHI> <nLb> <gscale> <outdir> <model name> <debug mode? (0/1)> <use previous connectivity?(optional atm) (0/1)>" << endl;
    exit(1);
  }

  string cmd;
  string gennPath = getenv("GENN_PATH");
  string outdir = toString(argv[7]) + "_output";  
  string modelName = argv[8];
  int dbgMode = atoi(argv[9]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  int fixsynapse;
  if (argc == 10) fixsynapse = 0;
  if (argc == 11) fixsynapse = atoi(argv[10]); // if this is not 0 network generation is skipped
  int which = atoi(argv[1]);
  int nAL = atoi(argv[2]);
  int nMB = atoi(argv[3]);
  int nLHI = atoi(argv[4]);
  int nLB = atoi(argv[5]);
  float gscale = atof(argv[6]);
  
  float pnkc_gsyn = 100.0f / nAL * gscale;
  float pnkc_gsyn_sigma = 100.0f / nAL * gscale / 15.0f; 
  float kcdn_gsyn = 2500.0f / nMB * 0.1f * gscale; 
  float kcdn_gsyn_sigma = 2500.0f / nMB * 0.01f * gscale; 
  float pnlhi_theta = 100.0f / nAL * 14.0f * gscale;

  // write neuron population sizes
  string fname = gennPath + "/userproject/include/sizes.h";
  ofstream os(fname.c_str());
  os << "#define _NAL " << nAL << endl;
  os << "#define _NMB " << nMB << endl;
  os << "#define _NLHI " << nLHI << endl;
  os << "#define _NLB " << nLB << endl;
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
  cmd += " && make clean && make";
  if (dbgMode == 1) {
    cmd += " debug";
  }
#endif
  system(cmd.c_str());

  // create output directory
#ifdef _WIN32
  _mkdir(outdir.c_str());
#else // UNIX
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  }
#endif
  
  if (fixsynapse == 0){
  // generate pnkc synapses
  cmd = gennPath + "/userproject/tools/gen_pnkc_syns ";
  cmd += toString(nAL) + " ";
  cmd += toString(nMB) + " ";
  cmd += "0.5 ";
  cmd += toString(pnkc_gsyn) + " ";
  cmd += toString(pnkc_gsyn_sigma) + " ";
  cmd += outdir + "/" + toString(argv[7]) + ".pnkc";
  cmd += " 1> " + outdir + "/" + toString(argv[7]) + ".pnkc.msg 2>&1";
  system(cmd.c_str()); 

  // generate kcdn synapses
  cmd = gennPath + "/userproject/tools/gen_kcdn_syns ";
  cmd += toString(nMB) + " ";
  cmd += toString(nLB) + " ";
  cmd += toString(kcdn_gsyn) + " ";
  cmd += toString(kcdn_gsyn_sigma) + " ";
  cmd += toString(kcdn_gsyn_sigma) + " ";
  cmd += outdir + "/" + toString(argv[7]) + ".kcdn";
  cmd += " 1> " + outdir + "/" + toString(argv[7]) + ".kcdn.msg 2>&1";
  system(cmd.c_str());

  // generate pnlhi synapses
  cmd = gennPath + "/userproject/tools/gen_pnlhi_syns ";
  cmd += toString(nAL) + " ";
  cmd += toString(nLHI) + " ";
  cmd += toString(pnlhi_theta) + " 15 ";
  cmd += outdir + "/" + toString(argv[7]) + ".pnlhi";
  cmd += " 1> " + outdir + "/" + toString(argv[7]) + ".pnlhi.msg 2>&1";
  system(cmd.c_str());

  // generate input patterns
  cmd = gennPath + "/userproject/tools/gen_input_structured ";
  cmd += toString(nAL) + " ";
  cmd += "10 10 0.1 0.05 1.0 2e-04 ";
  cmd += outdir + "/" + toString(argv[7]) + ".inpat";
  cmd += " 1> " + outdir + "/" + toString(argv[7]) + ".inpat.msg 2>&1";
  system(cmd.c_str());

  }
  else{
    cout << "Skipping network generation...." << endl;
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
    cmd = "cuda-gdb -tui --args model/classol_sim " + toString(argv[7]) + " " + toString(which);
  }
  else {
    cmd = "model/classol_sim " + toString(argv[7]) + " " + toString(which);
  }
#endif
  system(cmd.c_str());

  return 0;
}
