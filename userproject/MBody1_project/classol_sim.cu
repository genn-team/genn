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
  int which= atoi(argv[2]);
  string OutDir = toString(argv[1]) +"_output";
  string name;
  name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  

  timer.startTimer();
  patSetTime= (int) (PAT_TIME/DT);
  patFireTime= (int) (PATFTIME/DT);
  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stderr, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
  fprintf(stderr, "# PATFTIME %f \n", PATFTIME); 
  fprintf(stderr, "# patFireTime %d \n", patFireTime);
  fprintf(stderr, "# PAT_TIME %f \n", PAT_TIME);
  fprintf(stderr, "# patSetTime %d \n", patSetTime);
  fprintf(stderr, "# TOTAL_TME %f \n", TOTAL_TME);
  
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".out.St");
  FILE *osf2= fopen(name.c_str(),"w");

  //-----------------------------------------------------------------
  // build the neuronal circuitery
  classol locust;

  fprintf(stderr, "# reading PN-KC synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnkc");
  FILE *f= fopen(name.c_str(),"r");
  locust.read_pnkcsyns(f);
  fclose(f);
 
  fprintf(stderr, "# reading PN-LHI synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnlhi");
  f= fopen(name.c_str(), "r");
  locust.read_pnlhisyns(f);
  fclose(f);   
  
  fprintf(stderr, "# reading KC-DN synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".kcdn");
  f= fopen(name.c_str(), "r");
  locust.read_kcdnsyns(f);

  fprintf(stderr, "# reading input patterns ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".inpat");
  f= fopen(name.c_str(), "r");
  locust.read_input_patterns(f);
  fclose(f);
  locust.generate_baserates();

  if (which == GPU) {
    locust.allocate_device_mem_patterns();
  }
  locust.init(which);         // this includes copying g's for the GPU version

  fprintf(stderr, "# neuronal circuitery built, start computation ... \n\n");

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);
  fprintf(stderr, "# initial wait time execution ... \n");

  t= 0.0;
//  float lastsynwrite= t;
  int done= 0;
  float last_t_report=  t;
//  locust.output_state(os, which);  
//  locust.output_spikes(os, which);  
  locust.run(DT, which);
//  locust.output_state(os, which);  
//  float synwriteT= 0.0f;
//  int synwrite= 0;
//  unsigned int sum= 0;
  while (!done) 
  {
    if (which == GPU) {
      locust.getSpikeNumbersFromGPU();
      locust.getSpikesFromGPU();
    }
//    if (which == GPU) locust.getSpikeNumbersFromGPU();
    locust.run(DT, which); // run next batch
    if (which == GPU) {  
     CHECK_CUDA_ERRORS(cudaMemcpy(VDN, d_VDN, 10*sizeof(float), cudaMemcpyDeviceToHost));
    }
    locust.sum_spikes();
//    locust.output_spikes(osf, which);
//    locust.output_state(os, which);  // while outputting the current one ...
    locust.output_spikes(osf2, which);
    fprintf(osf, "%f ", t);
    for (int i= 0; i < 10; i++) {
    fprintf(osf, "%f ", VDN[i]);
   }
    fprintf(osf,"\n");
//    cudaThreadSynchronize();

    // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      fprintf(stderr, "time %f \n", t);
      last_t_report= t;
//      locust.output_state(os);
    }
    // output synapses occasionally
    // if (synwrite) {
    //   lastsynwrite= synwriteT;
    //   name= toString(argv[1]) + toString(".") + toString((int) synwriteT);
    //   name+= toString(".syn");
    //   f= fopen(name.c_str(),"w");
    //   locust.write_kcdnsyns(f);
    //   fclose(f);
    //   synwrite= 0;
    // }
    // if (t - lastsynwrite >= SYN_OUT_TME) {
    //   locust.get_kcdnsyns();
    //   synwrite= 1;
    // ¯  synwriteT= t;
    // }
    done= (t >= TOTAL_TME);
  }
//  locust.output_state(os);
//    if (which == GPU) locust.getSpikesFromGPU();
//    locust.output_spikes(os, which);
  // if (synwrite) {
  //   lastsynwrite= t;
  //   name= toString(argv[1]) + toString(".") + toString((int) t);
  //   name+= toString(".syn");
  //   f= fopen(name.c_str());
  //   locust.write_kcdnsyns(f);
  // fclose(f);
  //   synwrite= 0;
  // }

  timer.stopTimer();
  cerr << "output files are created under the current directory." << endl;
  fprintf(timef, "%d %d %d %d %f \n", locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime());
  fclose(osf);
  fclose(osf2);
  fclose(timef);
  freeDeviceMem();
  cudaDeviceReset();
  return 0;
}
