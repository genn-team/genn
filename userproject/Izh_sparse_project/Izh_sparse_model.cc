/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _IZH_SPARSE_MODEL_CC_
#define _IZH_SPARSE_MODEL_CC_

#include "Izh_sparse_CODE/runner.cc"
#include "../../lib/include/numlib/randomGen.h"
#include "../../lib/include/numlib/gauss.h"

randomGauss RG;
randomGen R;

classol::classol()
{
  modelDefinition(model);
  input1=new float[model.neuronN[0]];
  input2=new float[model.neuronN[1]];
  allocateMem();
  initialize();
  sumPExc = 0;
  sumPInh = 0;


}

void classol::randomizeVar(float * Var, float strength, unsigned int neuronGrp)
{
	//kernel if gpu?
  
  for (int j=0; j< model.neuronN[neuronGrp]; j++){
      	Var[j]=Var[j]+strength*R.n();
  	}
}

void classol::randomizeVarSq(float * Var, float strength, unsigned int neuronGrp)
{
	//kernel if gpu?
  //randomGen R;
  float randNbr;
  for (int j=0; j< model.neuronN[neuronGrp]; j++){
  	      randNbr=R.n();
      	Var[j]=Var[j]+strength*randNbr*randNbr;
  	}
}


void classol::initializeAllVars(unsigned int which)
{
	randomizeVar(aPInh,0.08,1);	
	randomizeVar(bPInh,-0.05,1);
	randomizeVarSq(cPExc,15.0,0);
	randomizeVarSq(dPExc,-6.0,0);
	
	
	for (int j=0; j< model.neuronN[1]; j++){
		UPExc[j]=bPExc[j]*VPExc[j];
		UPInh[j]=bPInh[j]*VPInh[j];	
	}
	for (int j=model.neuronN[1]; j< model.neuronN[0]; j++){
		UPExc[j]=bPExc[j]*VPExc[j];
	}
	
  	if (which == GPU) {
		CHECK_CUDA_ERRORS(cudaMemcpy(d_aPInh, aPInh, sizeof(float)*model.neuronN[1], cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(d_bPInh, bPInh, sizeof(float)*model.neuronN[1], cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(d_cPExc, cPExc, sizeof(float)*model.neuronN[0], cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(d_dPExc, dPExc, sizeof(float)*model.neuronN[0], cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(d_UPExc, UPExc, sizeof(float)*model.neuronN[0], cudaMemcpyHostToDevice));
		CHECK_CUDA_ERRORS(cudaMemcpy(d_UPInh, UPInh, sizeof(float)*model.neuronN[1], cudaMemcpyHostToDevice));
  	}
}
	
void classol::init(unsigned int which)
{
  if (which == CPU) {
  }
  if (which == GPU) {
    copyGToDevice(); 
    copyStateToDevice();
  }
}


void classol::allocate_device_mem_input()
{
  unsigned int size;

  size= model.neuronN[0]*sizeof(float);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_input1, size));
    
  size= model.neuronN[1]*sizeof(float);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_input2, size));
}
void classol::copy_device_mem_input()
{
  cudaMemcpy(d_input1,input1, model.neuronN[0]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input2,input2, model.neuronN[1]*sizeof(float), cudaMemcpyHostToDevice);
}
void classol::free_device_mem()
{
  // clean up memory                          
  printf("input is const, no need to copy");                                     
  CHECK_CUDA_ERRORS(cudaFree(d_input1));                                   
  CHECK_CUDA_ERRORS(cudaFree(d_input2));
}



classol::~classol()
{
  free(input1);
  free(input2);
}


void classol::write_input_to_file(FILE *f)
{
  printf("input is const, no need to write");
  unsigned int outno;
  if (model.neuronN[0]>10) 
  outno=10;
  else outno=model.neuronN[0];

  fprintf(f, "%f ", t);
  for(int i=0;i<outno;i++) {
    fprintf(f, "%f ", input1[i]);
  }
  fprintf(f,"\n");
}

void classol::read_input_values(FILE *f)
{
  fread(input1, model.neuronN[0]*sizeof(float),1,f);
}


void classol::create_input_values(float t) //define your explicit input rule here
{
  for (int x= 0; x < model.neuronN[0]; x++) {
    input1[x]= 5*RG.n();
  }
  
  for (int x= 0; x < model.neuronN[1]; x++) {
    input2[x]= 2*RG.n();
  }
}

void classol::read_sparsesyns_par(int synInd, Conductance C, FILE *f_ind,FILE *f_indInG,FILE *f_g //!< File handle for a file containing sparse conductivity values
			    )
{
  //allocateSparseArray(synInd,C.connN);

  fread(C.gp, C.connN*sizeof(model.ftype),1,f_g);
  fprintf(stderr,"%d active synapses. \n",C.connN);
  fread(C.gIndInG, (model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int),1,f_indInG);
  fread(C.gInd, C.connN*sizeof(int),1,f_ind);


  // general:
  fprintf(stderr,"Read conductance ... \n");
  fprintf(stderr, "Size is %d for synapse group %d. Values start with: \n",C.connN, synInd);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%f ", C.gp[i]);
  }
  fprintf(stderr,"\n\n");
  
  
  fprintf(stderr, "%d indices read. Index values start with: \n",C.connN);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%d ", C.gInd[i]);
  }  
  fprintf(stderr,"\n\n");
  
  
  fprintf(stderr, "%d g indices read. Index in g array values start with: \n", model.neuronN[model.synapseSource[synInd]]+1);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%d ", C.gIndInG[i]);
  }  
}

void classol::gen_alltoall_syns( float * g, unsigned int nPre, unsigned int nPost, float gscale//!< Generate random conductivity values for an all to all network
			    )
{
  //randomGen R;
  for(int i= 0; i < model.neuronN[nPre]; i++) {
  	 for(int j= 0; j < model.neuronN[nPost]; j++){
      g[i*model.neuronN[nPost]+j]=gscale*R.n();
    }
  }
  fprintf(stderr,"\n\n");
}

void classol::setInput(unsigned int which)
{
	create_input_values(t);
	if (which == GPU) copy_device_mem_input();
}

void classol::run(float runtime, unsigned int which)
{
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (which == GPU){
       stepTimeGPU(d_input1,d_input2, t);
    }
    if (which == CPU)
       stepTimeCPU(input1, input2,t);
    t+= DT;
    iT++;
  }
}

void classol::sum_spikes()
{
  sumPExc+= glbscntPExc;
  sumPInh+= glbscntPInh;
}

//--------------------------------------------------------------------------
// output functions

void classol::output_state(FILE *f, unsigned int which)
{
  if (which == GPU) 
    copyStateFromDevice();

  fprintf(f, "%f ", t);

   for (int i= 0; i < model.neuronN[0]-1; i++) {
     fprintf(f, "%f ", VPExc[i]);
   }
   
   for (int i= 0; i < model.neuronN[1]-1; i++) {
     fprintf(f, "%f ", VPInh[i]);
   }
   
  fprintf(f,"\n");
}

void classol::output_params(FILE *f, FILE *f2)
{
	
	for (int i= 0; i < model.neuronN[0]-1; i++) {
		fprintf(f, "%f ", aPExc[i]);
		fprintf(f, "%f ", bPExc[i]);
		fprintf(f, "%f ", cPExc[i]);
		fprintf(f, "%f ", dPExc[i]);
		fprintf(f, "%f ", input1[i]);
		fprintf(f,"\n");
	}
	for (int i= 0; i < model.neuronN[1]-1; i++) {
		fprintf(f, "%f ", aPInh[i]);
		fprintf(f, "%f ", bPInh[i]);
		fprintf(f, "%f ", cPInh[i]);
		fprintf(f, "%f ", dPInh[i]);
		fprintf(f, "%f ", input2[i]);
		fprintf(f,"\n");
		
	}
	
	for (int i= 0; i < gExc_Exc.connN; i++) {
	  fprintf(f2,"%f ", gExc_Exc.gp[i]);
	  fprintf(f2,"\n");
	}
	for (int i= 0; i < gExc_Inh.connN; i++) {
	  fprintf(f2,"%f ", gExc_Inh.gp[i]);
	  fprintf(f2,"\n");
	}
	for (int i= 0; i < gInh_Exc.connN; i++) {
	  fprintf(f2,"%f ", gInh_Exc.gp[i]);
	  fprintf(f2,"\n");
	}
	for (int i= 0; i < gInh_Inh.connN; i++) {
	  fprintf(f2,"%f ", gInh_Inh.gp[i]);
 		fprintf(f2,"\n");
	}
	fprintf(stderr, "4...\n");
}
//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void classol::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step
 
This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void classol::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}


void classol::output_spikes(FILE *f, unsigned int which)
{
	if (which == GPU) {
		getSpikesFromGPU();
	 	getSpikeNumbersFromGPU();
	}
 
  for (int i= 0; i < glbscntPExc; i++) {
		fprintf(f,"%f %d\n", t, glbSpkPExc[i]);
  }

  for (int i= 0; i < glbscntPInh; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[0]+glbSpkPInh[i]);
  }
}
#endif	

 
