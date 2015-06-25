
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

#include "testEGPInSimCode.h"
#include "EGPInSimCode_CODE/definitions.h"
#include "EGPInSimCode_CODE/runner.cc"



EGPInSimCode::EGPInSimCode()
{
  allocateMem();
  initialize();
  init_neurons();
}

EGPInSimCode::~EGPInSimCode()
{
  freeMem();
}

void EGPInSimCode::init_neurons() {
    for (int i= 0; i < 10; i++) {
	shiftpre[i]= i*10.0f;
    }
    copyStateToDevice();
}

void EGPInSimCode::run(int which, unsigned int copy)
{
  if (which == GPU)
  {
    stepTimeGPU(copy);
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
    cerr << "usage: EGPInSimCodeSim <GPU = 1, CPU = 0> <output label> <write output files? 0/1>" << endl;
    return EXIT_FAILURE;
  }

  EGPInSimCode *sim = new EGPInSimCode();
  int which= atoi(argv[1]);
  int write= atoi(argv[3]);
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream timeOs;
  ofstream neurOs;
  ofstream expNeurOs;
  unsigned int copy;
  if (write) {
      timeOs.open((outLabel + "_time.dat").c_str(), ios::app);
      neurOs.open((outLabel + "_neur.dat").c_str());
      expNeurOs.open((outLabel + "_expNeur.dat").c_str());
  }
  float x[10];
  if (write) {
      cout << "# DT " << DT << endl;
      cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
      cout << "# REPORT_TIME " << REPORT_TIME << endl;
      cout << "# begin simulating on " << ((which) ? "GPU" : "CPU") << endl;
  }
  timer->startTimer();
  float err= 0.0f;
  for (int j= 0; j < 10; j++) {
      x[j]= 0.0f;
  }
  inputpre= 0.0f;
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {      
      if (write) {
	  neurOs << t << " ";
	  expNeurOs << t << " ";
      }
      for (int j= 0; j < 10; j++) { // for all pre-synaptic neurons 
	  // generate expected values
	  if (which == GPU) {
	      if (i%2 == 1) {
		  x[j]= (t-DT)+pow(t-DT,2.0)+j*10;
	      }
	      else {
		  if (i > 0) x[j]= (t-DT)+pow(t-2*DT,2.0)+j*10;
	      }
	  }
	  else {
	      if (i > 0) x[j]= (t-DT)+pow(t-DT,2.0)+j*10;
	  }
	  if (write) {
	      neurOs << xpre[glbSpkShiftpre+j] << " ";
	      expNeurOs << x[j] << " ";
	  }
      }		  
      err+= absDiff(x, xpre+glbSpkShiftpre, 10);
      if (write) {
	  neurOs << endl;
	  expNeurOs << endl;
      }
      inputpre= pow(t, 2.0);
      if (i%2 == 0) copy=GENN_FLAGS::COPY;
      else copy= GENN_FLAGS::NOCOPY;
      sim->run(which, copy);
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
      expNeurOs.close();
  }

  delete sim;
  delete timer;
  
  float tolerance= 2e-2;
  int success;
  string result;
  if (abs(err) < tolerance) {
      success= 1;
      result= tS("\033[1;32m PASS \033[0m");
  } else {
      success= 0;
      result= tS("\033[1;31m FAIL \033[0m");
  }
  cout << "# test EGPInSimCode: Result " << result << endl;
  cout << "# the error was: " << err << " against tolerance " << tolerance << endl;
  cout << "#-----------------------------------------------------------" << endl;
  if (success)
      return EXIT_SUCCESS;
  else 
      return EXIT_FAILURE;
}

#endif // TESTPREVARSINSIMCODE_CU
