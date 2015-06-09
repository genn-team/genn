
#ifndef SYNDELAYSIM_CU
#define SYNDELAYSIM_CU

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

#include "hr_time.cpp"
#include "utils.h"

#include "SynDelaySim.h"
#include "SynDelay_CODE/definitions.h"
#include "SynDelay_CODE/runner.cc"


SynDelay::SynDelay(bool usingGPU)
{
  this->usingGPU = usingGPU;
  allocateMem();
  initialize();
}

SynDelay::~SynDelay()
{
  freeMem();
}

void SynDelay::run(float t)
{
  if (usingGPU)
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
  if (argc != 3)
  {
    cerr << "usage: SynDelaySim <GPU = 1, CPU = 0> <output label>" << endl;
    return EXIT_FAILURE;
  }

  SynDelay *sim = new SynDelay(atoi(argv[1]));
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream fileTime;
  ofstream fileV;
  ofstream fileStInput;
  ofstream fileStInter;
  ofstream fileStOutput;
  fileTime.open((outLabel + "_time").c_str(), ios::out | ios::app);
  fileV.open((outLabel + "_Vm").c_str(), ios::out | ios::trunc);
  fileStInput.open((outLabel + "_input_st").c_str(), ios::out | ios::trunc);
  fileStInter.open((outLabel + "_inter_st").c_str(), ios::out | ios::trunc);
  fileStOutput.open((outLabel + "_output_st").c_str(), ios::out | ios::trunc);
  cout << "# DT " << DT << endl;
  cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
  cout << "# REPORT_TIME " << REPORT_TIME << endl;
  cout << "# begin simulating on " << (atoi(argv[1]) ? "GPU" : "CPU") << endl;
  timer->startTimer();
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {
    sim->run(t);
    t += DT;

    fileV << t
	  << " " << VInput[0]
	  << " " << VInter[0]
	  << " " << VOutput[0]
	  << endl;

    for (int i= 0; i < glbSpkCntInput[spkQuePtrInput]; i++) {
	fileStInput << t << " " << glbSpkInput[glbSpkShiftInput+i] << endl;
    }
    for (int i= 0; i < glbSpkCntInter[0]; i++) {
	fileStInter << t << " " << glbSpkInter[i] << endl;
    }
    for (int i= 0; i < glbSpkCntOutput[0]; i++) {
	fileStOutput << t << " " << glbSpkOutput[i] << endl;
    }

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
