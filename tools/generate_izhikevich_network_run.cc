/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generate_izhikevich_network_run.cc

\brief This file is part of a tool chain for running the classIzh/Izh_sparse example model.

This file compiles to a tool that wraps all the other tools into one chain of tasks, including running all the gen_* tools for generating connectivity, providing the population size information through ../userproject/include/sizes.h to the model definition, running the GeNN code generation and compilation steps, executing the model and collecting some timing information. This tool is the recommended way to quickstart using GeNN as it only requires a single command line to execute all necessary tasks.
*/ 
//--------------------------------------------------------------------------

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
/////////////////////////
unsigned int openFileGetMax(unsigned int * array, unsigned int size, string name){
	FILE * f;
	unsigned int maxConn =0;
  f= fopen(name.c_str(),"r");
  fread(array, (size+1)*sizeof(unsigned int),1,f);

	for (unsigned int i=0;i<size;i++){
		unsigned int connNo = array[i+1]-array[i];
		if (connNo>maxConn) maxConn=connNo;
	}
	fprintf(stderr, " \n maximum postsynaptic connection per neuron in the 1st group is %u \n", maxConn);
	return maxConn;
}
/////////////////////////
int main(int argc, char *argv[])
{
  if (argc != 10)
  {
    cerr << "usage: generate_izhikevich_network_run <CPU=0, GPU=1> <nNeurons> <nConn> <gscale> <outdir> <executable name> <model name> <debug mode? (0/1)> <use previous connectivity? (0/1)>";
    exit(1);
  }

  string cmd;
  
  int which= atoi(argv[1]);
  
  int nTotal = atoi(argv[2]);
  int nExc= ceil(4*nTotal/5);
  int nInh= nTotal-nExc;
  
  int nConn= atoi(argv[3]);
  float gscale= atof(argv[4]);
  string OutName = toString(argv[5]);
  string OutDir = OutName +"_output";  
  string OutDir_g = "inputfiles";  
  string execName = argv[6];
  string modelName = argv[7];
  int DBGMODE = atoi(argv[8]); //set this to 1 if you want to enable gdb and cuda-gdb debugging to 0 for release
  int fixsynapse=atoi(argv[9]); //same synapse patterns should be used to compare CPU to GPU
  
  float meangExc= 0.5*gscale;
  float meangInh= -1.0*gscale;
  string debugmethod;
  
  if (which==1){
  	debugmethod=toString("cuda-gdb");
  }
  else {
  	debugmethod=toString("gdb");
  }
  
  #ifdef _WIN32
  _mkdir(OutDir.c_str());
  _mkdir(OutDir_g.c_str());
  #else 
  if (mkdir(OutDir.c_str(), S_IRWXU | S_IRWXG | S_IXOTH)==-1){
  	cerr << "Directory cannot be created. It may exist already." << endl;
  	}; 
  if (mkdir(OutDir_g.c_str(), S_IRWXU | S_IRWXG | S_IXOTH)==-1){
  	cerr << "Directory cannot be created. It may exist already." << endl;
  	}; 
  #endif
  
  if (fixsynapse==0){
	  // generate network connectivity patterns
	  cmd= toString("$GeNNPATH/tools/gen_syns_sparse_izhModel ");
	  cmd+= toString(nTotal) + toString(" ") ;
	  cmd+= toString(nConn) + toString(" ") ;
	  cmd+= toString(meangExc) + toString(" ") ;
	  cmd+= toString(meangInh) + toString(" ") ;
	  cmd+= "inputfiles/g"+modelName;
	  system(cmd.c_str()); 
	  cout << "connectivity generation script call was:" << cmd.c_str() << endl;
  }
	
	//read connectivity patterns to get maximum connection per neuron for each synapse population
	//population neuron numbers are for sources in the order in currentmodel.cc
  unsigned int *gIndInG= new unsigned int[nTotal]; //allocate the biggest possible, the we will only use what we need

	string name= toString("inputfiles/g"+modelName+"_postIndInG_ee");
	unsigned int maxN0 = openFileGetMax(gIndInG, nExc, name);

	name= toString("inputfiles/g"+modelName+"_postIndInG_ei");
	unsigned int maxN1 = openFileGetMax(gIndInG, nExc, name);

	name= toString("inputfiles/g"+modelName+"_postIndInG_ie");
	unsigned int maxN2 = openFileGetMax(gIndInG, nInh, name);

	name= toString("inputfiles/g"+modelName+"_postIndInG_ii");
	unsigned int maxN3 = openFileGetMax(gIndInG, nInh, name);

	delete [] gIndInG;
////////////////////////////////
  
  string GeNNPath= getenv("GeNNPATH");
  cerr << GeNNPath << endl;
  string fname= GeNNPath+string("/userproject/include/sizes.h");
  ofstream os(fname.c_str());
  os << "#define _NExc " << nExc << endl;
  os << "#define _NInh " << nInh << endl;
	os << "#define _NMaxConnP0 " << maxN0 << endl;
	os << "#define _NMaxConnP1 " << maxN1 << endl;
	os << "#define _NMaxConnP2 " << maxN2 << endl;
	os << "#define _NMaxConnP3 " << maxN3 << endl;
  os.close();
  
  cmd= toString("cd $GeNNPATH/userproject/")+toString(modelName)+("_project/ && buildmodel ")+ toString(modelName)+ toString(" ") + toString(DBGMODE);

  
  cout << "Debug mode " << DBGMODE << endl;

  cout << "script call was:" << cmd.c_str() << endl;
  system(cmd.c_str());
  cmd= toString("cd $GeNNPATH/userproject/")+modelName+("_project && ");
  if(DBGMODE==1) {
		cmd+= toString("make clean debug && make debug");
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
 if(DBGMODE==1) {
	cerr << "Debugging with gdb is not possible on cl platform." << endl;
	}
	else {
  		cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[5]) + toString(" ") + toString(which);
	}
cout << "1..." << endl;
#else
  
   if(DBGMODE==1) {
  //debug 
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); "+debugmethod+" -tui --args $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/debug/")+execName + toString(" ")+  toString(argv[5]) + toString(" ") + toString(which);
  }
  else
  {
//release  
  cmd= toString("GeNNOSTYPE=$(echo $(uname) | tr A-Z a-z); $GeNNPATH/userproject/")+modelName+("_project/bin/$GeNNOSTYPE/release/")+execName + toString(" ")+  toString(argv[5]) + toString(" ") + toString(which);
  	}
  	
cout << "2..." << endl;
#endif
  cout << cmd << endl;
  system(cmd.c_str());
  return 0;
  
}

