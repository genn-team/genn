/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "PoissonIzh-model.h"

// GeNN includes
#include "hr_time.h"

#include "PoissonIzh_CODE/definitions.h"

#include "sizes.h"

// other stuff:
#define T_REPORT_TME 1000.0
#define SYN_OUT_TME 2000.0

#define TOTAL_TME 5000

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: PoissonIzh_sim <basename> <CPU=0, GPU=1> \n");
        return 1;
    }

    int which= atoi(argv[2]);
    string OutDir = string(argv[1]) +"_output";
    string name;
    name= OutDir+ "/"+ argv[1] + ".time";
    FILE *timef= fopen(name.c_str(),"w");

    CStopWatch timer;
    timer.startTimer();
    fprintf(stderr, "# DT %f \n", DT);
    fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(stderr, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
    fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);

    name= OutDir+ "/" + argv[1] + ".out.Vm";
    FILE *osf= fopen(name.c_str(),"w");
    //-----------------------------------------------------------------
    // build the neuronal circuitry
    classol PNIzhNN;

    /*/SPARSE CONNECTIVITY

    name= OutDir+ "/gPoissonIzh";
    fprintf(stderr, "# reading PN-Izh1 synapses from file %s", name.c_str());
    FILE *f= fopen(name.c_str(),"rb");
    name= OutDir+ toString("/gPoissonIzh_info");
    FILE *f_info= fopen(name.c_str(),"rb");
    name= OutDir+ toString("/gPoissonIzh_postIndInG");
    FILE *f_postIndInG= fopen(name.c_str(),"rb");
    name= OutDir+ toString("/gPoissonIzh_postind");
    FILE *f_postind= fopen(name.c_str(),"rb");

    fread(&gPNIzh1.connN,sizeof(unsigned int),1,f_info);
    fprintf(stderr, "read %u times %d bytes \n", gPNIzh1.connN,sizeof(float));
    allocateAllSparseArrays();

    PNIzhNN.read_sparsesyns_par("PN", gPNIzh1, f_postind,f_postIndInG,f);
    fclose(f);
    fclose(f_info);
    fclose(f_postIndInG);
    fclose(f_postind);
    initializeAllSparseArrays();
    //SPARSE CONNECTIVITY END */

    //DENSE CONNECTIVITY

    name= OutDir+ "/gPoissonIzh_nonopt";
    cout << "# reading PN-Izh1 synapses from file "<< name << endl;
    FILE *f= fopen(name.c_str(),"rb");
    PNIzhNN.read_PNIzh1syns(gPNIzh1 , f);
    fclose(f);
    //DENSE CONNECTIVITY END

    PNIzhNN.init(which);         // this includes copying g's for the GPU version

    fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    fprintf(stderr, "# We are running with fixed time step %f \n", DT);
    fprintf(stderr, "# initial wait time execution ... \n");

    t= 0.0;
    int done= 0;
    double last_t_report=  t;
    while (!done)
    {
//    if (which == GPU) PNIzhNN.getSpikeNumbersFromGPU();

        PNIzhNN.run(DT, which); // run next batch

        if (which == 1) {
#ifndef CPU_ONLY
            //PNIzhNN.getSpikeNumbersFromGPU();
            PNIzhNN.getSpikesFromGPU();
            pullIzh1StateFromDevice();
            pullPNStateFromDevice();
#endif
        }

        PNIzhNN.sum_spikes();

        //    PNIzhNN.output_spikes(os, which);
        //   PNIzhNN.output_state(os, which);  // while outputting the current one ...
        fprintf(osf, "%f ", t);
        for(int i=0;i<10;i++) {
            fprintf(osf, "%f ", float(VIzh1[i]));

        }
        fprintf(osf, "\n");
        //      cudaThreadSynchronize();

        // report progress
        if (t - last_t_report >= T_REPORT_TME)
        {
            fprintf(stderr, "time %f \n", t);
            last_t_report= t;
            //PNIzhNN.output_state(os);
        }
        // output synapses occasionally
        // if (synwrite) {
        //   lastsynwrite= synwriteT;
        //   name= toString(argv[1]) + toString(".") + toString((int) synwriteT);
        //   name+= toString(".syn");
        //   f= fopen(name.c_str(),"w");
        //   PNIzhNN.write_kcdnsyns(f);
        //   fclose(f);
        //   synwrite= 0;
        // }
        // if (t - lastsynwrite >= SYN_OUT_TME) {
        //   PNIzhNN.get_kcdnsyns();
        //   synwrite= 1;
        //   synwriteT= t;
        // }
        done= (t >= TOTAL_TME);
    }
    //  PNIzhNN.output_state(os);
    //    if (which == GPU) PNIzhNN.getSpikeNumbersFromGPU();
    //    if (which == GPU) PNIzhNN.getSpikesFromGPU();
    //    PNIzhNN.output_spikes(os, which);
    // if (synwrite) {
    //   lastsynwrite= t;
    //   name= toString(argv[1]) + toString(".") + toString((int) t);
    //   name+= toString(".syn");
    //   f= fopen(name.c_str());
    //   PNIzhNN.write_kcdnsyns(f);
    // fclose(f);
    //   synwrite= 0;
    // }

    timer.stopTimer();
    cerr << "Output files are created under the current directory." << endl;
    float elapsedTime= timer.getElapsedTime();
    fprintf(timef, "%d %d %f \n", PNIzhNN.sumPN, PNIzhNN.sumIzh1, elapsedTime);
    fprintf(stdout, "%d Poisson spikes evoked spikes on %d Izhikevich neurons in %f seconds.\n", PNIzhNN.sumPN, PNIzhNN.sumIzh1, elapsedTime);

    return 0;
}
