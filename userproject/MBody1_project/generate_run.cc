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
  if (argc != 11)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <nAL> <nMB> <nLHI> <nLb> <gscale> <outdir> <executable name> <model name> <debug mode? (0/1)>";
    exit(1);
  }

  string cmd;
  string execName = argv[8];
  string modelName = argv[9];
  int DBGMODE = atoi(argv[10]); // set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  
  int which = atoi(argv[1]);
  int nAL = atoi(argv[2]);
  int nMB = atoi(argv[3]);
  int nLHI = atoi(argv[4]);
  int nLB = atoi(argv[5]);
  float gscale = atof(argv[6]);
  string outdir = toString(argv[7]) + "_output";  
  
  float pnkc_gsyn = 100.0f / nAL * gscale;
  float pnkc_gsyn_sigma = 100.0f / nAL * gscale / 15.0f; 
  float kcdn_gsyn = 2500.0f / nMB * 0.024f * gscale / 0.9; 
  float kcdn_gsyn_sigma = 2500.0f / nMB * 0.06f * gscale / 0.9; 
  float pnlhi_theta = 200.0f / nAL * 7.0f * gscale / 0.9;

  #ifdef _WIN32
  _mkdir(outdir.c_str());
  #else 
  if (mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH) == -1) {
    cerr << "Directory cannot be created. It may exist already." << endl;
  } 
  #endif
  
  // generate pnkc synapses
  cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/gen_pnkc_syns ";
  cmd += toString(nAL) + " ";
  cmd += toString(nMB) + " ";
  cmd += "0.5 ";
  cmd += toString(pnkc_gsyn) + " ";
  cmd += toString(pnkc_gsyn_sigma) + " ";
  cmd += outdir + "/" + toString(argv[7]) + ".pnkc";
  cmd += " &> " + outdir + "/" + toString(argv[7]) + ".pnkc.msg";
  system(cmd.c_str()); 

  // generate kcdn synapses
  cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/gen_kcdn_syns ";
  cmd += toString(nMB) + " ";
  cmd += toString(nLB) + " ";
  cmd += toString(kcdn_gsyn) + " ";
  cmd += toString(kcdn_gsyn_sigma) + " ";
  cmd += toString(kcdn_gsyn_sigma) + " ";
  cmd += outdir + "/" + toString(argv[7]) + ".kcdn";
  cmd += " &> " + outdir + "/" + toString(argv[7]) + ".kcdn.msg";
  system(cmd.c_str());

  // generate pnlhi synapses
  cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/gen_pnlhi_syns ";
  cmd += toString(nAL) + " ";
  cmd += toString(nLHI) + " ";
  cmd += toString(pnlhi_theta) + " 15 ";
  cmd += outdir + "/" + toString(argv[7]) + ".pnlhi";
  cmd += " &> " + outdir + "/" + toString(argv[7]) + ".pnlhi.msg";
  system(cmd.c_str());

  // generate input patterns
  cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/gen_input_structured ";
  cmd += toString(nAL) + " ";
  cmd += "10 10 0.1 0.1 32768 17 ";
  cmd += outdir + "/" + toString(argv[7]) + ".inpat";
  cmd += " &> " + outdir + "/" + toString(argv[7]) + ".inpat.msg";
  system(cmd.c_str());

  string GeNNPath = getenv("GeNNPATH");
  cerr << GeNNPath << endl;
  string fname = GeNNPath + string("/userproject/include/sizes.h");
  ofstream os(fname.c_str());
  os << "#define _NAL " << nAL << endl;
  os << "#define _NMB " << nMB << endl;
  os << "#define _NLHI " << nLHI << endl;
  os << "#define _NLB " << nLB << endl;
  os.close();
  
  cmd = "cd $GeNNPATH/userproject/" + toString(modelName) + "_project/model && " + BUILDMODEL + " " + toString(modelName) + " " + toString(DBGMODE);
  cout << "Debug mode " << DBGMODE << endl;
  cout << "script call was:" << cmd.c_str() << endl;
  system(cmd.c_str());
  cmd = "cd $GeNNPATH/userproject/" + toString(modelName) + "_project/model && ";
  if(DBGMODE == 1) {
    cmd += "make clean && make debug";
  }
  else {
    cmd += "make clean && make";
  }
  system(cmd.c_str());

  // run it!
  cout << "running test..." <<endl;

#ifdef _WIN32
  if(DBGMODE == 1) { // debug
    cerr << "Debugging with gdb is not possible on cl platform." << endl;
  }
  else { // release
    cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/model/bin/" + toString(execName) + " " + toString(argv[7]) + " " + toString(which);
  }

#else
  if(DBGMODE == 1) { // debug
    cmd = "cuda-gdb -tui --args $GeNNPATH/userproject/" + toString(modelName) + "_project/model/bin/" + toString(execName) + " " + toString(argv[7]) + " " + toString(which);
  }
  else { // release
    cmd = "$GeNNPATH/userproject/" + toString(modelName) + "_project/model/bin/" + toString(execName) + " " + toString(argv[7]) + " " + toString(which);
  }

#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
}
