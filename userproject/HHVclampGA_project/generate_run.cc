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
using namespace std;

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else
#include <sys/stat.h> //needed for mkdir
#endif

//--------------------------------------------------------------------------
/*! \brief Template function for string conversion 
 */
//--------------------------------------------------------------------------

template<typename T>
std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 


int main(int argc, char *argv[])
{
  if (argc != 6)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <nPop> <totalT> <outdir> <debug mode? (0/1)>" << endl;
    exit(1);
  }

  int DBGMODE = atoi(argv[5]); // set this to 1 if you want to enable gdb and 
                               // cuda-gdb debugging to 0 for release
  int which= atoi(argv[1]);
  int nPop= atoi(argv[2]);
  double totalT= atof(argv[3]);
  string OutDir = toString(argv[4]) +"_output";  
  string cmd;
 
  #ifdef _WIN32
  _mkdir(OutDir.c_str());
  #else 
  if (mkdir(OutDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH)==-1){
  	cerr << "Directory cannot be created. It may exist already." << endl;
  	}; 
  #endif
  
  string GeNNPath= getenv("GeNNPATH");
  cerr << GeNNPath << endl;
  string fname= GeNNPath+string("/userproject/include/HHVClampParameters.h");
  ofstream os(fname.c_str());
  os << "#define NPOP " << nPop << endl;
  os << "#define TOTALT " << totalT << endl;
  os.close();
  
  cmd= toString("cd model && buildmodel ")+ toString("HHVClamp ")+ toString(DBGMODE);
  cerr << "Debug mode: " << DBGMODE << endl;
  cerr << "Script call was:" << cmd.c_str() << endl;
  system(cmd.c_str());
  if(DBGMODE==1) {
    cmd= toString("cd model && make clean debug && make debug");
  }
  else{
    cmd= toString("cd model && make clean && make");  
  }	
  system(cmd.c_str());

  cmd= toString("echo $GeNNOSTYPE");
  system(cmd.c_str());

  // run it!
  cout << "running test..." <<endl;
#if defined _WIN32 || defined __CYGWIN__
  //cout << "win32" <<endl;
  if(DBGMODE==1) {
    cerr << "Debugging with gdb is not possible on cl platform." << endl;
  }
  else {
    cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); model/bin/$GeNNOSTYPE/release/VClampGA "+  toString(argv[4]) + toString(" ") + toString(which);
  }

#else
   if(DBGMODE == 1) {
  //debug 
     cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); cuda-gdb -tui --args model/bin/$GeNNOSTYPE/debug/VClampGA ") + toString(argv[4]) + toString(" ") + toString(which);
  }
  else
  {
//release  
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); model/bin/$GeNNOSTYPE/release/VClampGA ") + toString(argv[4]) + toString(" ") + toString(which);
  	}
#endif
  cerr << cmd << endl;
  system(cmd.c_str());
  return 0;
}

