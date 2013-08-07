/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "OneComp_sim.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: OneComp_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int which= atoi(argv[2]);
  string OutDir = toString(argv[1]) +"_output";
  string name, name2; 

  timer.startTimer();
  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);
  
  name= OutDir+ "/" + toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name2= OutDir+ "/" + toString(argv[1]) + toString(".explinp"); 
  FILE *osf2= fopen(name2.c_str(),"w");
  //-----------------------------------------------------------------
  // build the neuronal circuitry
  classol locust;
    
  locust.init(which);         // this includes copying g's for the GPU version

  fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");
  unsigned int outno;
  if (locust.model.neuronN[0]>10) 
  outno=10;
  else outno=locust.model.neuronN[0];

  if (which == GPU) locust.allocate_device_mem_input(); 
  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);
  fprintf(stderr, "# initial wait time execution ... \n");

  t= 0.0;
  void *devPtr;
  int done= 0;
  float last_t_report=  t;
  locust.run(DT, which);
  while (!done) 
  {
    for (int k=0;k<locust.model.neuronGrpN;k++)
    {
      if (locust.model.receivesInputCurrent[k]==2) 
      {
        FILE * ff;
        ff=fopen("../../tools/expoutf","r");
        locust.read_input_values(ff);
        fclose(ff);
      }
      if (locust.model.receivesInputCurrent[k]==3)
      {
        FILE * ff;
	locust.create_input_values(ff, t);
      }
    } 
    if (which == GPU) locust.copy_device_mem_input();
    locust.run(DT, which); // run next batch
    if (which == GPU) {  
      cudaGetSymbolAddress(&devPtr, d_VIzh1);
      CHECK_CUDA_ERRORS(cudaMemcpy(VIzh1, devPtr, outno*sizeof(float), cudaMemcpyDeviceToHost));
	} 
   fprintf(osf, "%f ", t);
   locust.write_input_to_file(osf2);
   
   for(int i=0;i<outno;i++) {
     fprintf(osf, "%f ", VIzh1[i]);
    }
  
   fprintf(osf, "\n");
   cudaThreadSynchronize();
   

   // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      fprintf(stderr, "time %f \n", t);
      last_t_report= t;
    }

    done= (t >= TOTAL_TME);
  }

  timer.stopTimer();
  cerr << "Output files are created under the current directory." << endl;
  fclose(osf);
  fclose(osf2);
  return 0;
}
