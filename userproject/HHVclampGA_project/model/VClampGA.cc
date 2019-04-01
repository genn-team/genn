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
const double limit[7][2]= {{1.0, 200.0}, // gNa
                           {0.0, 100.0}, // ENa
                           {1.0, 100.0}, // gKd
                           {-100.0, -20.0}, // EKd
                           {1.0, 50.0}, // gleak
                           {-100.0, -20.0}, // Eleak
                           {1e-1, 10.0}}; // C

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
    int protocol= atoi(argv[2]);
    std::string OutDir = std::string(argv[1]) +"_output";
    std::string name;
    FILE *timef= fopen((OutDir+ "/"+ argv[1] + ".time").c_str(),"a");
    write_para();
    FILE *osf= fopen((OutDir+ "/"+ argv[1] + ".out.I").c_str(),"w");
    FILE *osb= fopen((OutDir+ "/"+ argv[1] + ".out.best").c_str(),"w");

    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    allocateMem();
    initialize();


    for (int n= 0; n < NPOP; n++) {
        gNaHH[n]= limit[0][0]+uniform(rng)*(limit[0][1]-limit[0][0]); // uniform in allowed interval
        ENaHH[n]= limit[1][0]+uniform(rng)*(limit[1][1]-limit[1][0]); // uniform in allowed interval
        gKHH[n]= limit[2][0]+uniform(rng)*(limit[2][1]-limit[2][0]); // uniform in allowed interval
        EKHH[n]= limit[3][0]+uniform(rng)*(limit[3][1]-limit[3][0]); // uniform in allowed interval
        glHH[n]= limit[4][0]+uniform(rng)*(limit[4][1]-limit[4][0]); // uniform in allowed interval
        ElHH[n]= limit[5][0]+uniform(rng)*(limit[5][1]-limit[5][0]); // uniform in allowed interval
        CHH[n]= limit[6][0]+uniform(rng)*(limit[6][1]-limit[6][0]); // uniform in allowed interval
    }


    initializeSparse();
    fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");

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
        pullerrHHFromDevice();
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
    fclose(timef);
    fclose(osb);
    return 0;
}
