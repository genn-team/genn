/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "Izh_sparse_sim.h"
#include "../GeNNHelperKrnls.cu"

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
  fprintf(stderr, "# TOTAL_TME %f \n", TOTAL_TME);
 
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.Vm"); 
  cerr << name << endl;
  FILE *osf= fopen(name.c_str(),"w");
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.St"); 
  FILE *osf2= fopen(name.c_str(),"w");
  
  //-----------------------------------------------------------------
  // build the neuronal circuitry
  fprintf(stderr, "#creating classIzh\n");
	classIzh PCNN;
  fprintf(stderr, "#classIzh created\n");
	
	//open log file
  string logname=OutDir+ toString("/logfile");	
	//gettimeofday(&timeforlog, NULL);
	FILE *flog= fopen(logname.c_str(),"a");

	struct tm * timeinfo;
	time_t timeforlog=time(0);
	timeinfo = localtime(&timeforlog);
	fprintf(flog,"%d/%d/%d, %d:%d\n",timeinfo->tm_mday,timeinfo->tm_mon+1,timeinfo->tm_year+1900,timeinfo->tm_hour,timeinfo->tm_min);

	fprintf(flog,"Izh_sparse_sim, ");	
	if (which == GPU ) fprintf(flog,"GPU simulation\n"); else fprintf(flog,"CPU simulation\n");
	fprintf(flog, "# DT %f \n", DT);
  fprintf(flog, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(flog, "# TOTAL_TME %f \n", TOTAL_TME);

  name= OutDir+ toString("/") + toString(argv[1])+ toString(".params");
  FILE *fparams = fopen(name.c_str(),"w");
  name= OutDir+ toString("/") + toString(argv[1])+ toString(".fg");
  FILE *fg = fopen(name.c_str(),"w");
  PCNN.initializeAllVars(which); 
 	unsigned int sumSynapses=0;
  
  fprintf(stderr, "#reading synapses ... \n");
 	FILE *f_info, *f, *f_postIndInG,*f_postind;
 		//ee
    name= toString("inputfiles/gIzh_sparse_info_ee");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gExc_Exc.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %lu bytes \n",gExc_Exc.connN,sizeof(unsigned int));
  	fprintf(flog, "%u connections in gExc_Exc\n",gExc_Exc.connN);
		sumSynapses+=gExc_Exc.connN;
 		fclose(f_info);
  	
  	//ei
    name= toString("inputfiles/gIzh_sparse_info_ei");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gExc_Inh.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Inh.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %lu bytes \n",gExc_Inh.connN,sizeof(unsigned int));
  	fprintf(flog, "%u connections in gExc_Inh\n",gExc_Inh.connN);
		sumSynapses+=gExc_Inh.connN;
 		fclose(f_info);
  	
  	//ie
    name= toString("inputfiles/gIzh_sparse_info_ie");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gInh_Exc.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %lu bytes \n",gInh_Exc.connN,sizeof(unsigned int));
  	fprintf(flog, "%u connections in gInh_Exc\n",gInh_Exc.connN);
		sumSynapses+=gInh_Exc.connN;
 		fclose(f_info);
  	
  	//ii
    name= toString("inputfiles/gIzh_sparse_info_ii");
 	  f_info= fopen(name.c_str(),"r");
  	fread(&gInh_Inh.connN,sizeof(unsigned int),1,f_info);
  	//gExc_Exc.connN=sizeof(unsigned int)*(PCNN.model.neuronN[0]+PCNN.model.neuronN[1]);
  	fprintf(stderr, "read %u times %lu bytes \n",gInh_Inh.connN,sizeof(unsigned int));
  	fprintf(flog, "%u connections in gInh_Inh\n",gInh_Inh.connN);
		sumSynapses+=gInh_Inh.connN;
 		fclose(f_info);
  	
  	allocateAllSparseArrays();

	  //ee
  	name= toString("inputfiles/gIzh_sparse_ee");
 		f= fopen(name.c_str(),"r"); 
  	name= toString("inputfiles/gIzh_sparse_postIndInG_ee");
  	f_postIndInG= fopen(name.c_str(),"r");
  	name= toString("inputfiles/gIzh_sparse_postind_ee");
  	f_postind= fopen(name.c_str(),"r");  
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

  	/*//use this if network size is <= 1000
  	PCNN.gen_alltoall_syns(gpExc_Exc, 0, 0, 0.5); //exc to exc
  	PCNN.gen_alltoall_syns(gpExc_Inh, 0, 1, 0.5); //exc to  inh
  	PCNN.gen_alltoall_syns(gpInh_Exc, 1, 0, -1.0); //inh to exc
  	PCNN.gen_alltoall_syns(gpInh_Inh, 1, 1, -1.0); //inh to inh
  	PCNN.init(which);         // this includes copying g's for the GPU version
  */
  fprintf(stderr, "\nThere are %u synapses in the model.\n", sumSynapses);
  fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");
  unsigned int outno;
  curandState * devStates;
  if (PCNN.model.neuronN[0]>10) 
  outno=10;
  else outno=PCNN.model.neuronN[0];

  if (which == GPU) PCNN.allocate_device_mem_input(); 

	fprintf(stderr, "set input...\n");
  int sampleBlkNo0 = ceil(float((PCNN.model.neuronN[0])/float(BlkSz)));
  int sampleBlkNo1 = ceil(float((PCNN.model.neuronN[1])/float(BlkSz)));
  dim3 sThreads(BlkSz,1);
  dim3 sGrid0(sampleBlkNo0,1);
  dim3 sGrid1(sampleBlkNo1,1);
  if (which==CPU) 
    {
    	PCNN.setInput(which);
    }
  else{

  		 CHECK_CUDA_ERRORS(cudaMalloc((void **)&devStates, PCNN.model.neuronN[0]*sizeof(curandState)));
  	   xorwow_setup(devStates, PCNN.model.neuronN[0]); //setup the prng for the bigger network only

  		 generate_random_gpuInput_xorwow<<<sGrid0,sThreads>>>(devStates, PCNN.d_input1, PCNN.model.neuronN[0], 5.0, 0.0);
  		 generate_random_gpuInput_xorwow<<<sGrid1,sThreads>>>(devStates, PCNN.d_input2, PCNN.model.neuronN[1], 2.0, 0.0); 

  	}  
  
  PCNN.output_params(fparams, fg);
  fclose(fparams); 
  fclose(fg);

  
  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);
  fprintf(stderr, "# initial wait time execution ... \n");

  t= 0.0;
  int done= 0;
  float last_t_report=  t;
  PCNN.run(DT, which);
  while (!done) 
  {
  	 if (which == GPU){ 
  	 PCNN.getSpikeNumbersFromGPU();
     PCNN.getSpikesFromGPU();
    
  	 generate_random_gpuInput_xorwow<<<sGrid0,sThreads>>>(devStates, PCNN.d_input1, PCNN.model.neuronN[0], 5.0, 0.0);
  	 generate_random_gpuInput_xorwow<<<sGrid1,sThreads>>>(devStates, PCNN.d_input2, PCNN.model.neuronN[1], 2.0, 0.0); 

		}
		if (which == CPU){
		PCNN.setInput(which);
		}
     PCNN.run(DT, which); // run next batch
     /*if (which == GPU) { 
     CHECK_CUDA_ERRORS(cudaMemcpy(VPExc,d_VPExc, outno*sizeof(PCNN.model.ftype), cudaMemcpyDeviceToHost));
	} */
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
    CHECK_CUDA_ERRORS(cudaMemcpy(VPExc,d_VPExc, 10*sizeof(PCNN.model.ftype), cudaMemcpyDeviceToHost));
  }
  }
	
  timer.stopTimer();
	
  cerr << "Output files are created under the current directory. Output and parameters are logged in: " << logname << endl;
  fprintf(timef, "%d %d %u %u %.4f %.2f %.1f %.2f %u %s\n",which, PCNN.model.sumNeuronN[PCNN.model.neuronGrpN-1], PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, logname.c_str());
  fprintf(flog, "%d neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal number of synapses in the model is %u\n\n#################\n", PCNN.model.sumNeuronN[PCNN.model.neuronGrpN-1], PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses);
  fclose(osf);
  fclose(timef);
  fclose(osf2);
	fclose(flog);
	freeDeviceMem();
  cudaDeviceReset();
  return 0;
}


