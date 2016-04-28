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
#include "Izh_sparse_sim.h"
#ifndef CPU_ONLY
#include "../GeNNHelperKrnls.cu"
#endif

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
  unsigned int retval = 0; //make the compiler happy

  name= OutDir+ toString("/") + toString(argv[1])+ toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  
  
 
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.Vm"); 
  cout << name << endl;
  FILE *osf= fopen(name.c_str(),"w");
  if (which == 0) name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.St.CPU");
  if (which == 1) name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.St.GPU"); 
  FILE *osf2= fopen(name.c_str(),"w");
  
  //-----------------------------------------------------------------
  // build the neuronal circuitry
  fprintf(stdout, "#creating classIzh\n");
	classIzh PCNN;
  PCNN.initializeAllVars(which);
  fprintf(stdout, "#classIzh created\n");
	
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

  unsigned int sumSynapses=0;
  
  fprintf(stdout, "#reading synapses ... \n");
 	FILE *f_info, *f, *f_indInG,*f_ind;
 	unsigned int connN;
 	//ee
  name= toString("inputfiles/gIzh_sparse_info_ee");
  f_info= fopen(name.c_str(),"rb");
 	retval = fread((void *) &connN,sizeof(unsigned int),1,f_info);
	allocateExc_Exc(connN);
  fprintf(stdout, "%u connN, read %u times %lu bytes, fread returned %d values \n", connN, CExc_Exc.connN,sizeof(unsigned int), retval);
  fprintf(flog, "%u connections in gExc_Exc\n",CExc_Exc.connN);
	sumSynapses+=connN;
	fclose(f_info);
  	
  	//ei
  name= toString("inputfiles/gIzh_sparse_info_ei");
  f_info= fopen(name.c_str(),"rb");
  retval = fread(&connN,1,sizeof(unsigned int),f_info);
 	allocateExc_Inh(connN);
 	fprintf(stdout, "read %u times %lu bytes \n",CExc_Inh.connN,sizeof(unsigned int));
  fprintf(flog, "%u connections in gExc_Inh\n",CExc_Inh.connN);
	sumSynapses+=connN;
 	fclose(f_info);
  	
  //ie
  name= toString("inputfiles/gIzh_sparse_info_ie");
 	f_info= fopen(name.c_str(),"rb");
  retval = fread(&connN,1,sizeof(unsigned int),f_info);
  allocateInh_Exc(connN);
  fprintf(stdout, "read %u times %lu bytes \n",CInh_Exc.connN,sizeof(unsigned int));
  fprintf(flog, "%u connections in gInh_Exc\n",CInh_Exc.connN);
	sumSynapses+=connN;
 	fclose(f_info);
  	
  	//ii
  name= toString("inputfiles/gIzh_sparse_info_ii");
 	f_info= fopen(name.c_str(),"rb");
  retval = fread(&connN,1, sizeof(unsigned int),f_info);
  allocateInh_Inh(connN);
  fprintf(stdout, "read %u times %lu bytes \n",CInh_Inh.connN,sizeof(unsigned int));
  fprintf(flog, "%u connections in gInh_Inh\n",CInh_Inh.connN);
	sumSynapses+=connN;
 	fclose(f_info);
  	
  //open and read conductance arrays from files
	//ee
  name= toString("inputfiles/gIzh_sparse_ee");
 	f= fopen(name.c_str(),"rb"); 
  name= toString("inputfiles/gIzh_sparse_indInG_ee");
  f_indInG= fopen(name.c_str(),"rb");
  name= toString("inputfiles/gIzh_sparse_ind_ee");
  	f_ind= fopen(name.c_str(),"rb");  
	PCNN.read_sparsesyns_par(0, CExc_Exc, f_ind, f_indInG,f,gExc_Exc);
  	fclose(f); 
  	fclose(f_indInG); 
  	fclose(f_ind);   
  	
	//ei
  	name= toString("inputfiles/gIzh_sparse_ei");
 	f= fopen(name.c_str(),"rb"); 
  	name= toString("inputfiles/gIzh_sparse_indInG_ei");
  	f_indInG= fopen(name.c_str(),"rb");
  	name= toString("inputfiles/gIzh_sparse_ind_ei");
  	f_ind= fopen(name.c_str(),"rb");  
	PCNN.read_sparsesyns_par(1, CExc_Inh, f_ind,f_indInG,f,gExc_Inh);
  	fclose(f); 
  	fclose(f_indInG); 
  	fclose(f_ind);   
  	
	//ie
  	name= toString("inputfiles/gIzh_sparse_ie");
 	f= fopen(name.c_str(),"rb"); 
  	name= toString("inputfiles/gIzh_sparse_indInG_ie");
  	f_indInG= fopen(name.c_str(),"rb");
  	name= toString("inputfiles/gIzh_sparse_ind_ie");
  	f_ind= fopen(name.c_str(),"rb");  
        PCNN.read_sparsesyns_par(2, CInh_Exc, f_ind,f_indInG,f,gInh_Exc);
  	fclose(f); 
  	fclose(f_indInG); 
  	fclose(f_ind);   
  	
	//ii
  	name= toString("inputfiles/gIzh_sparse_ii");
 	f= fopen(name.c_str(),"rb"); 
  	name= toString("inputfiles/gIzh_sparse_indInG_ii");
  	f_indInG= fopen(name.c_str(),"rb");
  	name= toString("inputfiles/gIzh_sparse_ind_ii");
  	f_ind= fopen(name.c_str(),"rb");  
	PCNN.read_sparsesyns_par(3, CInh_Inh, f_ind,f_indInG,f,gInh_Inh);
  	fclose(f); 
  	fclose(f_indInG); 
  	fclose(f_ind);   
#ifndef CPU_ONLY
  	initializeAllSparseArrays();
#endif

  	/*//use this if network size is <= 1000
  	PCNN.gen_alltoall_syns(gpExc_Exc, 0, 0, 0.5); //exc to exc
  	PCNN.gen_alltoall_syns(gpExc_Inh, 0, 1, 0.5); //exc to  inh
  	PCNN.gen_alltoall_syns(gpInh_Exc, 1, 0, -1.0); //inh to exc
  	PCNN.gen_alltoall_syns(gpInh_Inh, 1, 1, -1.0); //inh to inh
  	PCNN.init(which);         // this includes copying g's for the GPU version
  */
  fprintf(stdout, "\nThere are %u synapses in the model.\n", sumSynapses);
  fprintf(stdout, "# neuronal circuitry built, start computation ... \n\n");
#ifndef CPU_ONLY
  curandState * devStates;
#endif
  /*unsigned int outno;
  if (PCNN.model.neuronN[0]>10) 
  outno=10;
  else outno=PCNN.model.neuronN[0];
*/

  fprintf(stdout, "set input...\n");
#ifndef CPU_ONLY
  int sampleBlkNo0 = ceil(float((PCNN.model.neuronN[0])/float(BlkSz)));
  int sampleBlkNo1 = ceil(float((PCNN.model.neuronN[1])/float(BlkSz)));
  dim3 sThreads(BlkSz,1);
  dim3 sGrid0(sampleBlkNo0,1);
  dim3 sGrid1(sampleBlkNo1,1);
#endif
  if (which==CPU) 
    {
    	PCNN.setInput(which);
    }
  else{
#ifndef CPU_ONLY
  		 CHECK_CUDA_ERRORS(cudaMalloc((void **)&devStates, PCNN.model.neuronN[0]*sizeof(curandState)));
  	   xorwow_setup(devStates, PCNN.model.neuronN[0]); //setup the prng for the bigger network only

	   generate_random_gpuInput_xorwow<<<sGrid0,sThreads>>>(devStates, d_I0PExc, PCNN.model.neuronN[0], meanInpExc, (scalar) 0.0);
	   generate_random_gpuInput_xorwow<<<sGrid1,sThreads>>>(devStates, d_I0PInh, PCNN.model.neuronN[1], meanInpInh, (scalar) 0.0); 
#endif
  	}  
  
  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation
  

  fprintf(stdout, "# DT %f \n", DT);
  fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);

  fprintf(stdout, "# We are running with fixed time step %f \n", DT);
  fprintf(stdout, "# initial wait time execution ... \n");
  timer.startTimer();

  t= 0.0;
  int done= 0;
  float last_t_report=  t;
  PCNN.run(DT, which);
#ifndef CPU_ONLY
  copySpikeNFromDevice();
  copySpikesFromDevice();
#endif
  PCNN.sum_spikes();
 
  if (which == GPU){ 
#ifndef CPU_ONLY
      while (!done) {
	  generate_random_gpuInput_xorwow<<<sGrid0,sThreads>>>(devStates, d_I0PExc, PCNN.model.neuronN[0], meanInpExc, (scalar) 0.0);
	  generate_random_gpuInput_xorwow<<<sGrid1,sThreads>>>(devStates, d_I0PInh, PCNN.model.neuronN[1], meanInpInh, (scalar) 0.0); 
	  
	  stepTimeGPU();
	  t+= DT;
	  copySpikeNFromDevice();
	  copySpikesFromDevice();
	  PCNN.sum_spikes();
	  
	  for (int i= 0; i < glbSpkCntPExc[0]; i++) {
	      fprintf(osf2,"%f %d\n", t, glbSpkPExc[i]);
	  }
	  
	  for (int i= 0; i < glbSpkCntPInh[0]; i++) {
	      fprintf(osf2, "%f %d\n", t, PCNN.model.sumNeuronN[0]+glbSpkPInh[i]);
	  }
	  // report progress
	  if (t - last_t_report >= T_REPORT_TME)
	  {
	      fprintf(stderr, "time %f \n", t);
	      last_t_report= t;
	  }
	  
	  done= (t >= TOTAL_TME);
      }
#endif
  }
  if (which == CPU){
      while (!done) {
	  PCNN.setInput(which);
	  stepTimeCPU();
	  t+= DT;
	  
	  PCNN.sum_spikes();
	  for (int i= 0; i < glbSpkCntPExc[0]; i++) {
	      fprintf(osf2,"%f %d\n", t, glbSpkPExc[i]);
	  }
	  
	  for (int i= 0; i < glbSpkCntPInh[0]; i++) {
	      fprintf(osf2, "%f %d\n", t, PCNN.model.sumNeuronN[0]+glbSpkPInh[i]);
	  }
	  // report progress
	  if (t - last_t_report >= T_REPORT_TME)
	  {
	      fprintf(stderr, "time %f \n", t);
	      last_t_report= t;
	  }
	  
	  done= (t >= TOTAL_TME);
    }
  }
  
  timer.stopTimer();
  
  cout << "Output files are created under the current directory. Output and parameters are logged in: " << logname << endl;
  fprintf(timef, "%d %d %u %u %.4f %.2f %.1f %.2f %u %s %s\n",which, PCNN.model.sumNeuronN[PCNN.model.neuronGrpN-1], PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, logname.c_str(), PCNN.model.ftype.c_str());
  fprintf(flog, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal time %f at DT+%f \nTotal number of synapses in the model is %u, %s precision\n\n#################\n", PCNN.model.sumNeuronN[PCNN.model.neuronGrpN-1], PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, PCNN.model.ftype.c_str());
  fprintf(stdout, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal simulation of %f ms at DT=%f \nTotal number of synapses in the model is %u, %s precision\n\n#################\n", PCNN.model.sumNeuronN[PCNN.model.neuronGrpN-1], PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, PCNN.model.ftype.c_str());
  fclose(osf);
  fclose(timef);
  fclose(osf2);
  fclose(flog);
  
  return 0;
}


