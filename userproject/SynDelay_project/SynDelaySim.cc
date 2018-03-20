//--------------------------------------------------------------------------
//   Author:    James Turner
//  
//   Institute: Center for Computational Neuroscience and Robotics
//              University of Sussex
//              Falmer, Brighton BN1 9QJ, UK 
//  
//   email to:  J.P.Turner@sussex.ac.uk
//  
//--------------------------------------------------------------------------

#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

#include "hr_time.h"

#include "SynDelay_CODE/definitions.h"


#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

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

    const bool usingGPU = (atoi(argv[1]) == 1);
#ifdef CPU_ONLY
    if (usingGPU)
    {
        cerr << "Cannot use GPU in a CPU_ONLY binary." << endl;
        cerr << "Recompile without CPU_ONLY to run a GPU simulation." << endl;
        return EXIT_FAILURE;
    }
#endif // CPU_ONLY

    allocateMem();
    initialize();
    initSynDelay();

    CStopWatch timer;
    string outLabel = argv[2];
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
    timer.startTimer();
    for (int i = 0; i < (TOTAL_TIME / DT); i++)
    {
        if (usingGPU)
        {
#ifndef CPU_ONLY
            stepTimeGPU();
            copyStateFromDevice();
            pullInputCurrentSpikesFromDevice();
#endif // CPU_ONLY
        }
        else
        {
            stepTimeCPU();
        }

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
    timer.stopTimer();
    cout << "# done in " << timer.getElapsedTime() << " seconds" << endl;
    fileTime << timer.getElapsedTime() << endl;
    fileTime.close();
    fileV.close();
    fileStInput.close();
    fileStInter.close();
    fileStOutput.close();

    freeMem();

    return EXIT_SUCCESS;
}

