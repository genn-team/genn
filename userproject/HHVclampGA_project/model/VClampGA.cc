/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cc

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU. 
*/
//--------------------------------------------------------------------------
#include <iostream>
#include <random>

#include "HHVClamp_CODE/definitions.h"

#include "helper.h"

#include "GA.h"

namespace
{
void truevar_init()
{
    for (int n= 0; n < NPOP; n++) {
        VHH[n]= initialHHValues[0];
        mHH[n]= initialHHValues[1];
        hHH[n]= initialHHValues[2];
        nHH[n]= initialHHValues[3];
        errHH[n]= 0.0;
    }

    pushVHHToDevice();
    pushmHHToDevice();
    pushhHHToDevice();
    pushnHHToDevice();
    pusherrHHToDevice();

}

void truevar_initexpHH()
{
    Vexp= initialHHValues[0];
    mexp= initialHHValues[1];
    hexp= initialHHValues[2];
    nexp= initialHHValues[3];
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the project
*/
//--------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: VClampGA <basename> <protocol> \n");
        return 1;
    }
    const int protocol= atoi(argv[2]);
    const std::string outLabel = argv[1];
    const std::string outDir = "../" + outLabel + "_output";

    write_para();
    FILE *osf= fopen((outDir + "/"+ outLabel + ".out.I").c_str(),"w");
    FILE *osb= fopen((outDir + "/"+ outLabel + ".out.best").c_str(),"w");

    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    allocateMem();
    initialize();
    initializeSparse();
    fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");

    // Initialise model of HH cell simulated using CPU
    Vexp = initialHHValues[0];
    mexp = initialHHValues[1];
    hexp = initialHHValues[2];
    nexp = initialHHValues[3];
    gNaexp = initialHHValues[4];
    ENaexp = initialHHValues[5];
    gKexp = initialHHValues[6];
    EKexp = initialHHValues[7];
    glexp = initialHHValues[8];
    Elexp = initialHHValues[9];
    Cexp = initialHHValues[10];

    // Build array of pointers to HH cell parameters so they can be modified algorithmically
    double *theExp_p[7];
    theExp_p[0]= &gNaexp;
    theExp_p[1]= &ENaexp;
    theExp_p[2]= &gKexp;
    theExp_p[3]= &EKexp;
    theExp_p[4]= &glexp;
    theExp_p[5]= &Elexp;
    theExp_p[6]= &Cexp;

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    fprintf(stderr, "# We are running with fixed time step %f \n", DT);

    double oldt;
    inputSpec I;
    initI(I);
    stepVGHH= I.baseV;
    const int iTN= (int) (I.t/DT);
    //CStopWatch timer;
    //timer.startTimer();
    while (t < TOTALT) {
        truevar_init();
        truevar_initexpHH();
        int sn= 0;
        for (int i= 0; i < iTN; i++) {
            oldt= t;
            runexpHH();
            stepTime();
            fprintf(osf,"%f %f \n", t, stepVGHH);
            if ((sn < I.N) && (oldt < I.st[sn]) && (t >= I.st[sn])) {
                stepVGHH= I.V[sn];
                sn++;
            }
        }
        pullHHStateFromDevice();
        fprintf(osb, "%f %f %f %f %f %f %f %f ", t, gNaexp, ENaexp, gKexp, EKexp, glexp, Elexp, Cexp);
        procreatePop(osb, rng);
        if (protocol >= 0) {
            if (protocol < 7) {
                if (protocol%2 == 0) {
                    *(theExp_p[protocol])=  initialHHValues[protocol+4]*(1+0.5*sin(3.1415927*t/40000));
                } else {
                    *(theExp_p[protocol])=  initialHHValues[protocol+4]+40.0*(sin(3.1415927*t/40000));
                }
            }
            else {
                for (int pn= 0; pn < 7; pn++) {
                    double fac;
                    if (pn%2 == 0) {
                        fac= 1+0.005*uniform(rng);
                        *(theExp_p[pn])*= fac;
                    }
                    else {
                        fac= 0.04*uniform(rng);
                        *(theExp_p[pn])+= fac;
                    }
                }
            }
        }
        std::cerr << "% " << t << std::endl;
    }
    //timer.stopTimer();
    //fprintf(timef,"%f \n",timer.getElapsedTime());
    // close files
    fclose(osf);

    fclose(osb);
    return 0;
}
