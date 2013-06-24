/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/


using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#else
#include <sys/stat.h> //needed for mkdir?
#endif
////////////////////////////////////////////////////////////////////////////////
// Template function for string conversion 
////////////////////////////////////////////////////////////////////////////////

template<typename T>
std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 


int main(int argc, char *argv[])
{
  if (argc != 11)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <nAL> <nMB> <nLHI> <nLb> <executable name> <model name> <debug mode? (0/1)>";
    cerr << "<gPNKC_base> <basename>" << endl;
    exit(1);
  }

  int DBGMODE = atoi(argv[10]); //set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  string cmd;
  string execName = argv[8];
  string modelName = argv[9];
  
  int which= atoi(argv[1]);
  int nAL= atoi(argv[2]);
  int nMB= atoi(argv[3]);
  int nLHI= atoi(argv[4]);
  int nLB= atoi(argv[5]);
  float g0= atof(argv[6]);
  string OutDir = toString(argv[7]) +"_output";  
  
  float pnkc_gsyn= 100.0f/nAL*g0;
  float pnkc_gsyn_sigma= 100.0f/nAL*g0/15.0f; 
  float kcdn_gsyn= 2500.0f/nMB*0.12f*g0/0.9; 
  float kcdn_gsyn_sigma= 2500.0f/nMB*0.025f*g0/0.9; 
  float pnlhi_theta= 200.0f/nAL*7.0f*g0/0.9;

  #ifdef _WIN32
  _mkdir(OutDir.c_str());
  #else 
  if (mkdir(OutDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH)==-1){
  	cerr << "Directory cannot be created. It may exist already." << endl;
  	}; 
  #endif
  
  // generate pnkc synapses
  cmd= toString("$GeNNPATH/tools/gen_pnkc_syns ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString(nMB) + toString(" ") ;
  cmd+= toString("0.5 ") ;
  cmd+= toString(pnkc_gsyn) + toString(" ") ;
  cmd+= toString(pnkc_gsyn_sigma) + toString(" ") ;
  cmd+= toString(argv[7]) + toString(".pnkc");
  cmd+= toString(" &> ") + OutDir+ "/"+ toString(argv[7]) + toString(".pnkc.msg");
  system(cmd.c_str()); 

  // generate kcdn synapses
  cmd= toString("$GeNNPATH/tools/gen_kcdn_syns ");
  cmd+= toString(nMB) + toString(" ") ;
  cmd+= toString(nLB) + toString(" ") ;
  cmd+= toString(kcdn_gsyn) + toString(" ") ;
  cmd+= toString(kcdn_gsyn_sigma) + toString(" ");
  cmd+= toString(kcdn_gsyn_sigma) + toString(" ");
  cmd+= toString(argv[7]) + toString(".kcdn");
  cmd+= toString(" &> ") + OutDir+ "/"+ toString(argv[7]) + toString(".kcdn.msg");
  system(cmd.c_str());

  // generate pnlhi synapses
  cmd= toString("$GeNNPATH/tools/gen_pnlhi_syns ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString(nLHI) + toString(" ") ;
  cmd+= toString(pnlhi_theta) + toString(" 15 ") ;
  cmd+= toString(argv[7]) + toString(".pnlhi");
  cmd+= toString(" &> ") + OutDir+ "/"+ toString(argv[7]) + toString(".pnlhi.msg");
  system(cmd.c_str());

  // generate input patterns
  cmd= toString("$GeNNPATH/tools/gen_input_fixfixfixno_struct ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString("10 10 0.1 0.1 32768 17 ") ;
  cmd+= toString(argv[7]) + toString(".inpat");
  cmd+= toString(" &> ") + OutDir+ "/"+ toString(argv[7]) + toString(".inpat.msg");
  system(cmd.c_str());

  ofstream os("../userproject/include/sizes.h");
  os << "#define _NAL " << nAL << endl;
  os << "#define _NMB " << nMB << endl;
  os << "#define _NLHI " << nLHI << endl;
  os << "#define _NLB " << nLB << endl;
  os.close();
  
  cmd= toString("cd ../userproject/ && buildmodel ")+ toString(modelName)+ toString(" ") + toString(DBGMODE);

  
  cout << "Debug mode " << DBGMODE << endl;

  cout << "script call was:" << cmd.c_str() << endl;
  system(cmd.c_str());
  cmd= toString("cd ../userproject && ");
  if(DBGMODE==1) {
		cmd+= toString("make clean dbg=1 && make dbg=1");
  }
  else{
		cmd+= toString("make clean && make");  
  	}	
  system(cmd.c_str());

  cmd= toString("echo $GeNNOSTYPE");
  system(cmd.c_str());

  // run it!
  cout << "running test..." <<endl;
#if defined _WIN32 || defined __CYGWIN__
  //cout << "win32" <<endl;
  if(DBGMODE==1) {
		cmd= toString("GeNNOSTYPE=Win32; cuda-gdb -tui ../userproject/bin/$GeNNOSTYPE/debug/")+ execName + toString(" ")+  toString(argv[7]) + toString(" ") + toString(which);
	}
	else {
  		cmd= toString("GeNNOSTYPE=Win32; ../userproject/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[7]) + toString(" ") + toString(which);
	}

#else
  //cout << "not win" <<endl;
  //cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); ../userproject/$GeNNOSTYPE/release/classol_sim ")+  toString(argv[7]) + toString(" ") + toString(which);
   if(DBGMODE==1) {
  //debug 
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); cuda-gdb -tui --args ../userproject/bin/$GeNNOSTYPE/debug/")+execName + toString(" ")+  toString(argv[7]) + toString(" ") + toString(which);
  }
  else
  {
//release  
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); ../userproject/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[7]) + toString(" ") + toString(which);
  	}
#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
  
}

