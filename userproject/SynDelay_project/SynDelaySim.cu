
#ifndef SYNDELAYSIM_CU
#define SYNDELAYSIM_CU

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

#include "hr_time.cpp"
#include "utils.h"

#include "SynDelaySim.h"
#include "SynDelay_CODE/runner.cc"


SynDelay::SynDelay(bool usingGPU)
{
  this->usingGPU = usingGPU;
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

void SynDelay::run(float t)
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
}


/*====================================================================
--------------------------- MAIN FUNCTION ----------------------------
====================================================================*/

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    cerr << "usage: SynDelaySim <GPU = 1, CPU = 0> <output label>" << endl;
    return EXIT_FAILURE;
  }

  float t = 0.0f;
  SynDelay *sim = new SynDelay(atoi(argv[1]));
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream fileTime;
  ofstream fileV;
  fileTime.open((outLabel + "_time").c_str(), ios::out | ios::app);
  fileV.open((outLabel + "_Vm").c_str(), ios::out | ios::trunc);

  cout << "# DT " << DT << endl;
  cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
  cout << "# REPORT_TIME " << REPORT_TIME << endl;
  cout << "# begin simulating on " << (atoi(argv[1]) ? "GPU" : "CPU") << endl;
  timer->startTimer();
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {
    sim->run(t);
    t += DT;

    fileV << "Time: " << t
	  << "\t\tInput: " << VInput[spkQuePtrInput * 500]
	  << "\t\tInter: " << VInter[0]
	  << "\t\tOutput: " << VOutput[0]
	  << endl;

    if ((int) t % (int) REPORT_TIME == 0)
    {
      cout << "time " << t << endl;
    }
  }
  timer->stopTimer();
  cout << "# done in " << timer->getElapsedTime() << " seconds" << endl;
  fileTime << timer->getElapsedTime() << endl;
  fileTime.close();
  fileV.close();

  delete sim;
  delete timer;
  return EXIT_SUCCESS;
}

#endif // SYNDELAYSIM_CU
