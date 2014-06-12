
#ifndef SYNDELAYSIM_CPP
#define SYNDELAYSIM_CPP

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "utils.h"
#include "hr_time.cpp"
#include "SynDelaySim.h"
#include "SynDelay_CODE_HOST/host.h"
#include "SynDelay_CODE_CUDA0/cuda0.h"
#include "SynDelay_CODE_CUDA1/cuda1.h"

using namespace std;


SynDelay::SynDelay(int usingGPU)
{
  this->usingGPU = usingGPU;
  allocateMemHost();
  initializeHost();
  if (usingGPU)
  {
    allocateMemCuda0();
    copyGToCuda0();
    copyStateToCuda0();
    allocateMemCuda1();
    copyGToCuda1();
    copyStateToCuda1();
  }
}

SynDelay::~SynDelay()
{
  freeMemHost();
  if (usingGPU)
  {
    freeMemCuda0();
    freeMemCuda1();
  }
}

void SynDelay::run(float t)
{
  if (usingGPU)
  {
    stepTimeCuda0(t);
    copyStateFromCuda0();
    stepTimeCuda1(t);
    copyStateFromCuda1();

    // temporary solution - use peer-to-peer memory copies soon
    //copyStateToCuda1();
  }
  else
  {
    stepTimeHost(t);
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
	  << "\t\tInput: " << VInput[spkEvntQuePtrInput * 500]
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

#endif // SYNDELAYSIM_CPP
