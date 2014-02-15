/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "Izh_sparse_sim.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: Izh_sparse_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int which= atoi(argv[2]);
  string OutDir = toString(argv[1]) +toString("_output");
  string name; 

  name= OutDir+ toString("/") + toString(argv[1])+ toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  
  
  timer.startTimer();
  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);
 
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.Vm"); 
  cerr << name << endl;
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.St"); 
  FILE *osf2= fopen(name.c_str(),"w");
  
  //-----------------------------------------------------------------
  // build the neuronal circuitry
  classol PCNN;
  
  name= OutDir+ toString("/") + toString(argv[1])+ toString(".params");
  FILE *fparams = fopen(name.c_str(),"w");
  name= OutDir+ toString("/") + toString(argv[1])+ toString(".fg");
  FILE *fg = fopen(name.c_str(),"w");
  PCNN.initializeAllVars(which); 
 
  if (PCNN.model.synapseConnType[0]==SPARSE)
  {   
  	fprintf(stderr, "# qqqqqqqqqqqqqqqqqqqq reading synapses ... \n");
 
 		//ee
    name= toString("inputfiles/gIzh_sparse_info_ee");
 	  FILE *f_info= fopen(name.c_str(),"r");
  	fread(&gExc_Exc.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %d bytes \n",gExc_Exc.connN,sizeof(unsigned int));
 		fclose(f_info);
  	
  	//ei
    name= toString("inputfiles/gIzh_sparse_info_ei");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gExc_Inh.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Inh.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %d bytes \n",gExc_Inh.connN,sizeof(unsigned int));
 		fclose(f_info);
  	
  	//ie
    name= toString("inputfiles/gIzh_sparse_info_ie");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gInh_Exc.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %d bytes \n",gInh_Exc.connN,sizeof(unsigned int));
 		fclose(f_info);
  	
  	//ii
    name= toString("inputfiles/gIzh_sparse_info_ii");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gInh_Inh.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %d bytes \n",gInh_Inh.connN,sizeof(unsigned int));
 		fclose(f_info);
  	
  	allocateAllSparseArrays();


	  //ee
  	name= toString("inputfiles/gIzh_sparse_ee");
 		FILE *f= fopen(name.c_str(),"r"); 
  	name= toString("inputfiles/gIzh_sparse_postIndInG_ee");
  	FILE *f_postIndInG= fopen(name.c_str(),"r");
  	name= toString("inputfiles/gIzh_sparse_postind_ee");
  	FILE *f_postind= fopen(name.c_str(),"r");  
		PCNN.read_sparsesyns_par(0, gExc_Exc, f_postind,f_postIndInG,f);
  	fclose(f); 
  	fclose(f_postIndInG); 
  	fclose(f_postind);   
  	
	  //ei
  	name= toString("inputfiles/gIzh_sparse_ei");
 		f= fopen(name.c_str(),"r"); 
  	name= toString("inputfiles/gIzh_sparse_postIndInG_ei");
  	f_postIndInG= fopen(name.c_str(),"r");
  	name= toString("inputfiles/gIzh_sparse_postind_ei");
  	f_postind= fopen(name.c_str(),"r");  
		PCNN.read_sparsesyns_par(1, gExc_Inh, f_postind,f_postIndInG,f);
  	fclose(f); 
  	fclose(f_postIndInG); 
  	fclose(f_postind);   
  	
	  //ie
  	name= toString("inputfiles/gIzh_sparse_ie");
 		f= fopen(name.c_str(),"r"); 
  	name= toString("inputfiles/gIzh_sparse_postIndInG_ie");
  	f_postIndInG= fopen(name.c_str(),"r");
  	name= toString("inputfiles/gIzh_sparse_postind_ie");
  	f_postind= fopen(name.c_str(),"r");  
		PCNN.read_sparsesyns_par(2, gInh_Exc, f_postind,f_postIndInG,f);
  	fclose(f); 
  	fclose(f_postIndInG); 
  	fclose(f_postind);   
  	
	  //ii
  	name= toString("inputfiles/gIzh_sparse_ii");
 		f= fopen(name.c_str(),"r"); 
  	name= toString("inputfiles/gIzh_sparse_postIndInG_ii");
  	f_postIndInG= fopen(name.c_str(),"r");
  	name= toString("inputfiles/gIzh_sparse_postind_ii");
  	f_postind= fopen(name.c_str(),"r");  
		PCNN.read_sparsesyns_par(3, gInh_Inh, f_postind,f_postIndInG,f);
  	fclose(f); 
  	fclose(f_postIndInG); 
  	fclose(f_postind);   
  	
  	initializeAllSparseArrays();
  }
  /*else
  {
  	//use this if network size is <= 1000
  	PCNN.gen_alltoall_syns(gpExc_Exc, 0, 0, 0.5); //exc to exc
  	PCNN.gen_alltoall_syns(gpExc_Inh, 0, 1, 0.5); //exc to  inh
  	PCNN.gen_alltoall_syns(gpInh_Exc, 1, 0, -1.0); //inh to exc
  	PCNN.gen_alltoall_syns(gpInh_Inh, 1, 1, -1.0); //inh to inh
  	PCNN.init(which);         // this includes copying g's for the GPU version
  }*/
  
  fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");
  unsigned int outno;
  if (PCNN.model.neuronN[0]>10) 
  outno=10;
  else outno=PCNN.model.neuronN[0];

  if (which == GPU) PCNN.allocate_device_mem_input(); 
	fprintf(stderr, "set input...\n");
  PCNN.setInput(which);
  
  PCNN.output_params(fparams, fg);
  fclose(fparams); 
  fclose(fg);
  
  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);
  fprintf(stderr, "# initial wait time execution ... \n");

  t= 0.0;
  void *devPtr;
  size_t sizept=0; 
  int done= 0;
  float last_t_report=  t;
  PCNN.run(DT, which);
  while (!done) 
  {
  	 if (which == GPU) PCNN.getSpikesFromGPU();
    
    PCNN.setInput(which);
    PCNN.run(DT, which); // run next batch
    if (which == GPU) { 
      CHECK_CUDA_ERRORS(cudaMemcpy(VPExc,d_VPExc, outno*sizeof(PCNN.model.ftype), cudaMemcpyDeviceToHost));
	} 
   PCNN.sum_spikes();
   PCNN.output_spikes(osf2, which);
   
   //PCNN.output_state(os, which);
   fprintf(osf, "%f ", t);
   //PCNN.write_input_to_file(osf2);
   
   for(int i=0;i<outno;i++) {
     fprintf(osf, "%f ", VPExc[i]);
    }
  
   fprintf(osf, "\n");
   

   // report progress
    if (t - last_t_report >= T_REPORT_TME)
    {
      fprintf(stderr, "time %f \n", t);
      last_t_report= t;
    }

    done= (t >= TOTAL_TME);
   if (which == GPU) {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPExc,d_VPExc, sizeof(PCNN.model.ftype), cudaMemcpyDeviceToHost));
}
  }

  timer.stopTimer();
  cerr << "Output files are created under the current directory." << endl;
  fprintf(timef, "%d %d %u %f %f \n",which, PCNN.model.neuronN[0], PCNN.sumPExc, timer.getElapsedTime(),VPExc[0]);
  fclose(osf);
  fclose(timef);
  fclose(osf2);
  cudaDeviceReset();
  return 0;
}
