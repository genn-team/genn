
#ifndef TESTPREVARSINSIMCODE_CU
#define TESTPREVARSINSIMCODE_CU

#include <cstdlib>
#include <cfloat>
#include <iostream>
#include <fstream>

using namespace std;

#include "hr_time.cpp"
#include "utils.h"
#include "testHelper.h"

#include "testPreVarsInPostLearn.h"
#include "preVarsInPostLearn_CODE/definitions.h"
#include "preVarsInPostLearn_CODE/runner.cc"



preVarsInPostLearn::preVarsInPostLearn()
{
  allocateMem();
  initialize();
  init_synapses();
  init_neurons();
}

preVarsInPostLearn::~preVarsInPostLearn()
{
  freeMem();
  delete[] theW;
}

void preVarsInPostLearn::init_synapses() {
    theW= new float*[10];
    theW[0]= wsyn0;
    theW[1]= wsyn1;
    theW[2]= wsyn2;
    theW[3]= wsyn3;
    theW[4]= wsyn4;
    theW[5]= wsyn5;
    theW[6]= wsyn6;
    theW[7]= wsyn7;
    theW[8]= wsyn8;
    theW[9]= wsyn9;
}
    
void preVarsInPostLearn::init_neurons() {
    for (int i= 0; i < 10; i++) {
	shiftpre[i]= i*10.0f;
    }
    copyStateToDevice();
}

void preVarsInPostLearn::run(int which)
{
  if (which == GPU)
  {
    stepTimeGPU();
    copyStateFromDevice();
  }
  else
  {
    stepTimeCPU();
  }
}


/*====================================================================
--------------------------- MAIN FUNCTION ----------------------------
====================================================================*/

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    cerr << "usage: preVarsInPostLearnSim <GPU = 1, CPU = 0> <output label> <write output files? 0/1>" << endl;
    return EXIT_FAILURE;
  }

  preVarsInPostLearn *sim = new preVarsInPostLearn();
  int which= atoi(argv[1]);
  int write= atoi(argv[3]);
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream timeOs;
  ofstream neurOs;
  ofstream synOs;
  ofstream expSynOs;
  if (write) {
      timeOs.open((outLabel + "_time.dat").c_str(), ios::app);
      neurOs.open((outLabel + "_neur.dat").c_str());
      synOs.open((outLabel + "_syn.dat").c_str());
      expSynOs.open((outLabel + "_expSyn.dat").c_str());
  }
  float x[10][100];
  if (write) {
      cout << "# DT " << DT << endl;
      cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
      cout << "# REPORT_TIME " << REPORT_TIME << endl;
      cout << "# begin simulating on " << ((which) ? "GPU" : "CPU") << endl;
  }
  timer->startTimer();
  float err= 0.0f;
  for (int d= 0; d < 10; d++) {
      for (int j= 0; j < 10; j++) {
	  for (int k= 0; k < 10; k++) {
	      x[d][j*10+k]= 0.0f;
	  }
      }
  }
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {      
      t = i*DT;
      if (write) {
	  neurOs << t << " ";
	  synOs << t << " ";
	  expSynOs << t << " ";
      }
      for (int d= 0; d < 10; d++) { // for each delay
	  for (int j= 0; j < 10; j++) { // for all pre-synaptic neurons 
	      for (int k= 0; k < 10; k++) { // for all post-syn neurons
              // generate expected values
		  if ((t > 2.0001) && (fmod(t-2*DT+5e-5,2.0f) < 1e-4)) {
		      x[d][j*10+k]= t-DT-(d+1)*DT+10*j;
		  }
		  if (write) {
		      synOs << sim->theW[d][j*10+k] << " ";
		      expSynOs << x[d][j*10+k] << " ";
		  }
	      }
	  }		  
	  err+= absDiff(x[d], sim->theW[d], 100);
	  if (write) {
	      synOs << "    ";
	      expSynOs << "    ";
	  }
      }
      if (write) {
	  for (int j= 0; j < 10; j++) {
	      neurOs << xpre[spkQuePtrpre*10+j] << " ";
	  }
	  neurOs << "    ";
      }
      neurOs << endl;
      synOs << endl;
      expSynOs << endl;
      sim->run(which);
      if (fmod(t+5e-5, REPORT_TIME) < 1e-4)
      {
	  cout << "\r" << t;
      }
  }
  cout << "\r";
  timer->stopTimer();
  cout << "# done in " << timer->getElapsedTime() << " seconds" << endl;
  if (write) {
      timeOs << timer->getElapsedTime() << endl;
      timeOs.close();
      neurOs.close();
      synOs.close();
      expSynOs.close();
  }

  delete sim;
  delete timer;
  
  float tolerance= 3e-2;
  int success;
  string result;
  if (abs(err) < tolerance) {
      success= 1;
      result= tS("\033[1;32m PASS \033[0m");
  } else {
      success= 0;
      result= tS("\033[1;31m FAIL \033[0m");
  }
  cout << "# test preVarsInPostLearn: Result " << result << endl;
  cout << "# the error was: " << err << " against tolerance " << tolerance << endl;
  cout << "#-----------------------------------------------------------" << endl;
  if (success)
      return EXIT_SUCCESS;
  else 
      return EXIT_FAILURE;
}

#endif // TESTPREVARSINSIMCODE_CU
