
#ifndef SYNDELAYSIM_CU
#define SYNDELAYSIM_CU

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

#include "utils.h"
#include "hr_time.cpp"

#include "SynDelaySim.hpp"
#include "SynDelay_CODE/runner.cc"


SynDelay::SynDelay(bool usingGPU_arg)
{
  usingGPU = usingGPU_arg;
  t = 0.0f;
  sumInput = 0;
  sumInterneuron = 0;
  sumOutput = 0;
  allocateMem();
  initialize();
  if (usingGPU)
  {
    copyGToDevice();
    copyStateToDevice();
  }
}

SynDelay::~SynDelay()
{
  freeMem();
  if (usingGPU)
  {
    freeDeviceMem();
  }
}

void SynDelay::run()
{
  void *devPtr;
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {
    if (usingGPU)
    {
      stepTimeGPU(t);
      copyStateFromDevice();
    }
    else
    {
      stepTimeCPU(t);
    }
    
    
    
    t += DT;
    if (t % REPORT_TIME == 0) cout << "time " << t << endl;
  }
}


/*====================================================================
--------------------------- MAIN FUNCTION ----------------------------
====================================================================*/

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cerr << "usage: SynDelaySim <GPU = true, CPU = false> <basename>" << endl;
    return EXIT_FAILURE;
  }

  SynDelay *sim = new SynDelay(atoi(argv[1]));
  CStopWatch *timer = new CStopWatch();

  // open files for storing spikes and spike counts
  string basename = argv[2];
  string outDir = basename + "_output/";
  ofstream fileSpk;
  ofstream fileSpkCnt;
  fileSpk.open((outDir + "spikes").c_str(), ios::app);
  fileSpkCnt.open((outDir + "spike_counts").c_str(), ios::app);

  // begin simulating
  cout << "# DT " << DT << endl;
  cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
  cout << "# REPORT_TIME " << REPORT_TIME << endl;
  timer -> startTimer();
  sim -> run();
  timer -> stopTimer();
  cout << "done in " << timer -> getElapsedTime() << " seconds" << endl;
  fileSpk.close();
  fileSpkCnt.close();
  cout << "spikes and spike counts are saved in " << outDir << endl;

  delete sim;
  return EXIT_SUCCESS;
}

#endif // SYNDELAYSIM_CU
