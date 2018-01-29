/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu

   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>

// GeNN includes
#include "hr_time.h"
#include "modelSpec.h"

// Model includes
#include "Izh_sparse_model.h"

// Auto generated includes
#include "sizes.h"
#include "Izh_sparse_CODE/definitions.h"

#define DBG_SIZE 5000

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 5000.0
#define TOTAL_TME 1000.0

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: Izh_sparse_sim <basename> <CPU=0, GPU=1> \n");
        return 1;
    }
    int which= atoi(argv[2]);
    string OutDir = string(argv[1]) + "_output";
    string name;
    unsigned int retval = 0; //make the compiler happy

    name= OutDir+ "/" + argv[1]+ ".time";
    FILE *timef= fopen(name.c_str(),"a");


    name= OutDir+ "/"+ argv[1] + ".out.Vm";
    cout << name << endl;
    FILE *osf= fopen(name.c_str(),"w");
    if (which == CPU) {
        name= OutDir+ "/" + argv[1] + ".out.St.CPU";
    }
    else {
        name= OutDir+ "/" + argv[1] + ".out.St.GPU";
    }
    FILE *osf2= fopen(name.c_str(),"w");

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    fprintf(stdout, "#creating classIzh\n");
    classIzh PCNN;
    PCNN.initializeAllVars(which);
    fprintf(stdout, "#classIzh created\n");

    //open log file
    string logname=OutDir+ "/logfile";
    //gettimeofday(&timeforlog, NULL);
    FILE *flog= fopen(logname.c_str(),"a");

    time_t timeforlog=time(0);
    tm * timeinfo = localtime(&timeforlog);
    fprintf(flog,"%d/%d/%d, %d:%d\n",timeinfo->tm_mday,timeinfo->tm_mon+1,timeinfo->tm_year+1900,timeinfo->tm_hour,timeinfo->tm_min);

    fprintf(flog,"Izh_sparse_sim, ");
    if (which == GPU ) {
        fprintf(flog,"GPU simulation\n");
    }
    else {
        fprintf(flog,"CPU simulation\n");
    }
    fprintf(flog, "# DT %f \n", DT);
    fprintf(flog, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(flog, "# TOTAL_TME %f \n", TOTAL_TME);

    unsigned int sumSynapses=0;

    fprintf(stdout, "#reading synapses ... \n");
    FILE *f_info, *f, *f_indInG,*f_ind;
    unsigned int connN;
    //ee
    f_info= fopen("inputfiles/gIzh_sparse_info_ee","rb");
    retval = fread((void *) &connN,sizeof(unsigned int),1,f_info);
    allocateExc_Exc(connN);
    fprintf(stdout, "%u connN, read %u times %lu bytes, fread returned %d values \n", connN, CExc_Exc.connN,sizeof(unsigned int), retval);
    fprintf(flog, "%u connections in gExc_Exc\n",CExc_Exc.connN);
    sumSynapses+=connN;
    fclose(f_info);

    //ei
    f_info= fopen("inputfiles/gIzh_sparse_info_ei","rb");
    retval = fread(&connN,1,sizeof(unsigned int),f_info);
    allocateExc_Inh(connN);
    fprintf(stdout, "read %u times %lu bytes \n",CExc_Inh.connN,sizeof(unsigned int));
    fprintf(flog, "%u connections in gExc_Inh\n",CExc_Inh.connN);
    sumSynapses+=connN;
    fclose(f_info);

    //ie
    f_info= fopen("inputfiles/gIzh_sparse_info_ie","rb");
    retval = fread(&connN,1,sizeof(unsigned int),f_info);
    allocateInh_Exc(connN);
    fprintf(stdout, "read %u times %lu bytes \n",CInh_Exc.connN,sizeof(unsigned int));
    fprintf(flog, "%u connections in gInh_Exc\n",CInh_Exc.connN);
    sumSynapses+=connN;
    fclose(f_info);

    //ii
    f_info= fopen("inputfiles/gIzh_sparse_info_ii","rb");
    retval = fread(&connN,1, sizeof(unsigned int),f_info);
    allocateInh_Inh(connN);
    fprintf(stdout, "read %u times %lu bytes \n",CInh_Inh.connN,sizeof(unsigned int));
    fprintf(flog, "%u connections in gInh_Inh\n",CInh_Inh.connN);
    sumSynapses+=connN;
    fclose(f_info);

    //open and read conductance arrays from files
    //ee
    f= fopen("inputfiles/gIzh_sparse_ee","rb");
    f_indInG= fopen("inputfiles/gIzh_sparse_indInG_ee","rb");
    f_ind= fopen("inputfiles/gIzh_sparse_ind_ee","rb");
    PCNN.read_sparsesyns_par(_NExc, CExc_Exc, f_ind, f_indInG,f,gExc_Exc);
    fclose(f);
    fclose(f_indInG);
    fclose(f_ind);

    //ei
    f= fopen("inputfiles/gIzh_sparse_ei","rb");
    f_indInG= fopen("inputfiles/gIzh_sparse_indInG_ei","rb");
    f_ind= fopen("inputfiles/gIzh_sparse_ind_ei","rb");
    PCNN.read_sparsesyns_par(_NExc, CExc_Inh, f_ind,f_indInG,f,gExc_Inh);
    fclose(f);
    fclose(f_indInG);
    fclose(f_ind);

    //ie
    f= fopen("inputfiles/gIzh_sparse_ie","rb");
    f_indInG= fopen("inputfiles/gIzh_sparse_indInG_ie","rb");
    f_ind= fopen("inputfiles/gIzh_sparse_ind_ie","rb");
    PCNN.read_sparsesyns_par(_NInh, CInh_Exc, f_ind,f_indInG,f,gInh_Exc);
    fclose(f);
    fclose(f_indInG);
    fclose(f_ind);

    //ii
    f= fopen("inputfiles/gIzh_sparse_ii","rb");
    f_indInG= fopen("inputfiles/gIzh_sparse_indInG_ii","rb");
    f_ind= fopen("inputfiles/gIzh_sparse_ind_ii","rb");
    PCNN.read_sparsesyns_par(_NInh, CInh_Inh, f_ind,f_indInG,f,gInh_Inh);
    fclose(f);
    fclose(f_indInG);
    fclose(f_ind);

    // Initialise sparse data structures and copy everything to device
    initIzh_sparse();

    /*//use this if network size is <= 1000
    PCNN.gen_alltoall_syns(gpExc_Exc, 0, 0, 0.5); //exc to exc
    PCNN.gen_alltoall_syns(gpExc_Inh, 0, 1, 0.5); //exc to  inh
    PCNN.gen_alltoall_syns(gpInh_Exc, 1, 0, -1.0); //inh to exc
    PCNN.gen_alltoall_syns(gpInh_Inh, 1, 1, -1.0); //inh to inh
    PCNN.init(which);         // this includes copying g's for the GPU version
*/
    fprintf(stdout, "\nThere are %u synapses in the model.\n", sumSynapses);
    fprintf(stdout, "# neuronal circuitry built, start computation ... \n\n");

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation


    fprintf(stdout, "# DT %f \n", DT);
    fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);

    fprintf(stdout, "# We are running with fixed time step %f \n", DT);
    fprintf(stdout, "# initial wait time execution ... \n");

    CStopWatch timer;
    timer.startTimer();

    t= 0.0;
    int done= 0;
    float last_t_report=  t;
    while (!done) {
        if (which == GPU){
#ifndef CPU_ONLY
            stepTimeGPU();

            copySpikeNFromDevice();
            copySpikesFromDevice();
#endif
        }
        else {
            stepTimeCPU();
        }

        PCNN.sum_spikes();
        for (int i= 0; i < glbSpkCntPExc[0]; i++) {
            fprintf(osf2,"%f %d\n", t, glbSpkPExc[i]);
        }

        for (int i= 0; i < glbSpkCntPInh[0]; i++) {
            fprintf(osf2, "%f %d\n", t, _NExc+glbSpkPInh[i]);
        }
        // report progress
        if (t - last_t_report >= T_REPORT_TME)
        {
            fprintf(stderr, "time %f \n", t);
            last_t_report= t;
        }

        done= (t >= TOTAL_TME);
    }

    timer.stopTimer();

    cout << "Output files are created under the current directory. Output and parameters are logged in: " << logname << endl;
    fprintf(timef, "%d %d %u %u %.4f %.2f %.1f %.2f %u %s %d\n",which, _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, logname.c_str(), _FTYPE);
    fprintf(flog, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal time %f at DT+%f \nTotal number of synapses in the model is %u, %d precision\n\n#################\n", _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, _FTYPE);
    fprintf(stdout, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal simulation of %f ms at DT=%f \nTotal number of synapses in the model is %u, %d precision\n\n#################\n", _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, _FTYPE);
    fclose(osf);
    fclose(timef);
    fclose(osf2);
    fclose(flog);

    return 0;
}


