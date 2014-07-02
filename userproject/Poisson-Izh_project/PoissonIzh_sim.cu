/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "PoissonIzh_sim.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: PoissonIzh_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int which= atoi(argv[2]);
  string OutDir = toString(argv[1]) +"_output";
  string name;
  name= OutDir+ "/"+toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"w");  

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
  fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);
  
  name= OutDir+ "/" + toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ "/" + toString(argv[1]) + toString(".outPN.Vm"); 
  FILE *osfpn= fopen(name.c_str(),"w");
  //-----------------------------------------------------------------
  // build the neuronal circuitery
  classol locust;
 
    
  fprintf(stderr, "# reading PN-Izh1 synapses ... \n");
  name= OutDir+ "/"+ toString(argv[1]) + toString(".pnkc");
  FILE *f= fopen(name.c_str(),"r");
  locust.read_PNIzh1syns(f);
  fclose(f);   
 
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
  void *devPtr;
  int done= 0;
  float last_t_report=  t;
  locust.run(DT, which);
  unsigned int sum= 0;
  while (!done) 
  {
//   if (which == GPU) locust.getSpikesFromGPU();
//    if (which == GPU) locust.getSpikeNumbersFromGPU();
    locust.run(DT, which); // run next batch
    if (which == GPU) {  
      cudaGetSymbolAddress(&devPtr, d_VIzh1);
      CHECK_CUDA_ERRORS(cudaMemcpy(VIzh1, devPtr, 10*sizeof(float), cudaMemcpyDeviceToHost));
	} 
//    locust.sum_spikes();
//    locust.output_spikes(os, which);
//   locust.output_state(os, which);  // while outputting the current one ...
   fprintf(osf, "%f ", t);
   fprintf(osfpn,"%f ",t);
   for(int i=0;i<10;i++) {
   fprintf(osf, "%f ", VIzh1[i]);
   fprintf(osfpn, "%f ", VPN[i]);
   	
   	}
   fprintf(osf, "\n");
   fprintf(osfpn, "\n");
//      cudaThreadSynchronize();

   // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      fprintf(stderr, "time %f \n", t);
      last_t_report= t;
      //locust.output_state(os);
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
    //   synwriteT= t;
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
  cerr << "Output files are created under the current directory." << endl;
  fprintf(timef, "%d %d %d %d %f \n", locust.sumPN, locust.sumIzh1, timer.getElapsedTime());

  return 0;
}
