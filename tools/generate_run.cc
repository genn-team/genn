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
  if (argc != 8)
  {
    cerr << "usage: generate_run <CPU=0, GPU=1> <nAL> <nMB> <nLHI> <nLb> ";
    cerr << "<gPNKC_base> <basename>" << endl;
    exit(1);
  }

  int which= atoi(argv[1]);
  int nAL= atoi(argv[2]);
  int nMB= atoi(argv[3]);
  int nLHI= atoi(argv[4]);
  int nLB= atoi(argv[5]);
  float g0= atof(argv[6]);

  float pnkc_gsyn= 100.0f/nAL*g0;
  float pnkc_gsyn_sigma= 100.0f/nAL*g0/15.0f; 
  float kcdn_gsyn= 2500.0f/nMB*0.12f*g0/0.9; 
  float kcdn_gsyn_sigma= 2500.0f/nMB*0.025f*g0/0.9; 
  float pnlhi_theta= 200.0f/nAL*7.0f*g0/0.9;

  string cmd;

  // generate pnkc synapses
  cmd= toString("$GeNNPATH/tools/gen_pnkc_syns ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString(nMB) + toString(" ") ;
  cmd+= toString("0.5 ") ;
  cmd+= toString(pnkc_gsyn) + toString(" ") ;
  cmd+= toString(pnkc_gsyn_sigma) + toString(" ") ;
  cmd+= toString(argv[7]) + toString(".pnkc");
  cmd+= toString(" &> ") + toString(argv[7]) + toString(".pnkc.msg");
  system(cmd.c_str()); 

  // generate kcdn synapses
  cmd= toString("$GeNNPATH/tools/gen_kcdn_syns ");
  cmd+= toString(nMB) + toString(" ") ;
  cmd+= toString(nLB) + toString(" ") ;
  cmd+= toString(kcdn_gsyn) + toString(" ") ;
  cmd+= toString(kcdn_gsyn_sigma) + toString(" ");
  cmd+= toString(kcdn_gsyn_sigma) + toString(" ");
  cmd+= toString(argv[7]) + toString(".kcdn");
  cmd+= toString(" &> ") + toString(argv[7]) + toString(".kcdn.msg");
  system(cmd.c_str());

  // generate pnlhi synapses
  cmd= toString("$GeNNPATH/tools/gen_pnlhi_syns ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString(nLHI) + toString(" ") ;
  cmd+= toString(pnlhi_theta) + toString(" 15 ") ;
  cmd+= toString(argv[7]) + toString(".pnlhi");
  cmd+= toString(" &> ") + toString(argv[7]) + toString(".pnlhi.msg");
  system(cmd.c_str());

  // generate input patterns
  cmd= toString("$GeNNPATH/tools/gen_input_fixfixfixno_struct ");
  cmd+= toString(nAL) + toString(" ") ;
  cmd+= toString("10 10 0.1 0.1 32768 17 ") ;
  cmd+= toString(argv[7]) + toString(".inpat");
  cmd+= toString(" &> ") + toString(argv[7]) + toString(".inpat.msg");
  system(cmd.c_str());

  ofstream os("../userproject/sizes.h");
  os << "#define _NAL " << nAL << endl;
  os << "#define _NMB " << nMB << endl;
  os << "#define _NLHI " << nLHI << endl;
  os << "#define _NLB " << nLB << endl;
  os.close();
  cmd= toString("cd ../userproject/ && . buildmodel MBody1");
  system(cmd.c_str());
  cmd= toString("cd ../userproject && ");
  cmd+= toString("make clean && make");
  system(cmd.c_str());

  cmd= toString("echo $GeNNOSTYPE");
  system(cmd.c_str());

  // run it!
  cmd= toString("../userproject/$GeNNOSTYPE/release/classol_sim ")+  toString(argv[7]) + toString(" ") + toString(which);
  system(cmd.c_str());
   
  return 0;
  
}

