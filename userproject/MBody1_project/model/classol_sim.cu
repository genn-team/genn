/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file classol_sim.cu

\brief Main entry point for the classol (CLASSification in OLfaction) model simulation. Provided as a part of the complete example of simulating the MBody1 mushroom body model. 
*/
//--------------------------------------------------------------------------


#include "classol_sim.h"

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------


int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: classol_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int GPUarg= atoi(argv[2]);
  string OutDir = toString(argv[1]) +"_output";
  string name;
  name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  

  int which;
  if (GPUarg > 1) {
     which= 1;
     nGPU= GPUarg-2;
  }    
  else {
     which= GPUarg;	
     nGPU= AUTODEVICE;
  }
  patSetTime= (int) (PAT_TIME/DT);
  patFireTime= (int) (PATFTIME/DT);
  fprintf(stdout, "# DT %f \n", DT);
  fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stdout, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
  fprintf(stdout, "# PATFTIME %f \n", PATFTIME); 
  fprintf(stdout, "# patFireTime %d \n", patFireTime);
  fprintf(stdout, "# PAT_TIME %f \n", PAT_TIME);
  fprintf(stdout, "# patSetTime %d \n", patSetTime);
  fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);
  
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.st"); 
  FILE *osf2= fopen(name.c_str(),"w");

#ifdef TIMING
  name= OutDir+ "/"+ toString(argv[1]) + toString(".timingprofile"); 
  FILE *timeros= fopen(name.c_str(),"w");
  sdkCreateTimer(&timer_gen);
  double tme;
#endif

  //-----------------------------------------------------------------
  // build the neuronal circuitery
  classol locust;

#ifdef TIMING
  sdkStartTimer(&timer_gen);
#endif

  fprintf(stdout, "# reading PN-KC synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnkc");
  FILE *f= fopen(name.c_str(),"r");
  locust.read_pnkcsyns(f);
  fclose(f);

#ifdef TIMING
  sdkStopTimer(&timer_gen);
  tme= sdkGetTimerValue(&timer_gen);
  fprintf(timeros, "%% Reading PN-KC synapses: %f \n", tme);
  sdkResetTimer(&timer_gen);
  sdkStartTimer(&timer_gen);
#endif

  fprintf(stdout, "# reading PN-LHI synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnlhi");
  f= fopen(name.c_str(), "r");
  locust.read_pnlhisyns(f);
  fclose(f);   

#ifdef TIMING
  sdkStopTimer(&timer_gen);
  tme= sdkGetTimerValue(&timer_gen);
  fprintf(timeros, "%% Reading PN-LHI synapses: %f \n", tme);
  sdkResetTimer(&timer_gen);
  sdkStartTimer(&timer_gen);
#endif
  
  fprintf(stdout, "# reading KC-DN synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".kcdn");
  f= fopen(name.c_str(), "r");
  locust.read_kcdnsyns(f);

#ifdef TIMING
  sdkStopTimer(&timer_gen);
  tme= sdkGetTimerValue(&timer_gen);
  fprintf(timeros, "%% Reading KC-DN synapses: %f \n", tme);
  sdkResetTimer(&timer_gen);
  sdkStartTimer(&timer_gen);
#endif

  fprintf(stdout, "# reading input patterns ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".inpat");
  f= fopen(name.c_str(), "r");
  locust.read_input_patterns(f);
  fclose(f);

#ifdef TIMING
  sdkStopTimer(&timer_gen);
  tme= sdkGetTimerValue(&timer_gen);
  fprintf(timeros, "%% Reading input patterns: %f \n", tme);
  sdkResetTimer(&timer_gen);
  sdkStartTimer(&timer_gen);
#endif

  locust.generate_baserates();
  if (which == GPU) {
    locust.allocate_device_mem_patterns();
  }
  locust.init(which);         // this includes copying g's for the GPU version

#ifdef TIMING
  sdkStopTimer(&timer_gen);
  tme= sdkGetTimerValue(&timer_gen);
  fprintf(timeros, "%% Initialisation: %f \n", tme);
  sdkResetTimer(&timer_gen);
#endif

  fprintf(stdout, "# neuronal circuitery built, start computation ... \n\n");

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stdout, "# We are running with fixed time step %f \n", DT);
  fprintf(stdout, "# initial wait time execution ... \n");

  t= 0.0;
  int done= 0;
  float last_t_report=  t;
  locust.run(DT, which);
  float synwriteT= 0.0f;
  float lastsynwrite= 0.0f;
  int synwrite= 0;
  timer.startTimer();
  while (!done) 
  {
    if (which == GPU) {
      locust.getSpikeNumbersFromGPU();
      locust.getSpikesFromGPU();
    }
    locust.run(DT, which); // run next batch
    // if (which == GPU) {  
//	pullDNfromDevice();
    //   }
    
#ifdef TIMING
    if (which == CPU) {
	fprintf(timeros, "%f %f %f \n", sdkGetTimerValue(&neuron_timer), sdkGetTimerValue(&synapse_timer), sdkGetTimerValue(&learning_timer));
    }
    else {
	fprintf(timeros, "%f %f %f \n", neuron_tme, synapse_tme, learning_tme);
    }
#endif

    locust.sum_spikes();
    locust.output_spikes(osf2, which);

 /*   fprintf(osf, "%f ", t);
    //  for (int i= 0; i < 100; i++) {
    //     fprintf(osf, "%f ", VDN[i]);
    //   }
    // fprintf(osf,"\n");
*/
    // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      fprintf(stdout, "time %f \n", t);
      last_t_report= t;
    }
    // output synapses occasionally
    if (synwrite) {
       lastsynwrite= synwriteT;
       name= OutDir+ "/"+ tS(argv[1]) + tS(".") + tS((int) synwriteT) + tS(".syn"); 
       f= fopen(name.c_str(),"w");
       locust.write_kcdnsyns(f);
       fclose(f);
       synwrite= 0;
    }
    if (t - lastsynwrite >= SYN_OUT_TME) {
       locust.get_kcdnsyns();
       synwrite= 1;
       synwriteT= t;
    }
    done= (t >= TOTAL_TME);
  }
  timer.stopTimer();
  cerr << "output files are created under the current directory." << endl;
  fprintf(timef, "%d %u %u %u %u %u %.4f %.2f %.1f %.2f\n",which, locust.model.sumNeuronN[locust.model.neuronGrpN-1], locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);
  fclose(osf);
  fclose(osf2);
  fclose(timef);
  freeDeviceMem();
  cudaDeviceReset();

#ifdef TIMING
  fclose(timeros);
#endif

  return 0;
}
