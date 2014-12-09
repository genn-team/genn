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
#include "Izh_sparse_model.h"

randomGauss RG;
randomGen R;

classIzh::classIzh()
{
  modelDefinition(model);
  input1=new scalar[model.neuronN[0]];
  input2=new scalar[model.neuronN[1]];
  allocateMem();
  initialize();
  sumPExc = 0;
  sumPInh = 0;
}

void classIzh::importArray(scalar *dest, double *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classIzh::exportArray(double *dest, scalar *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classIzh::randomizeVar(scalar * Var, scalar strength, unsigned int neuronGrp)
{
  //kernel if gpu?
  for (int j=0; j< model.neuronN[neuronGrp]; j++){
    Var[j]=Var[j]+strength*R.n();
  }
}

void classIzh::randomizeVarSq(scalar * Var, scalar strength, unsigned int neuronGrp)
{
  //kernel if gpu?
  //randomGen R;
  scalar randNbr;
  for (int j=0; j< model.neuronN[neuronGrp]; j++){
    randNbr=R.n();
    Var[j]=Var[j]+strength*randNbr*randNbr;
  }
}


void classIzh::initializeAllVars(unsigned int which)
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
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aPInh, aPInh, sizeof(scalar)*model.neuronN[1], cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bPInh, bPInh, sizeof(scalar)*model.neuronN[1], cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_cPExc, cPExc, sizeof(scalar)*model.neuronN[0], cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_dPExc, dPExc, sizeof(scalar)*model.neuronN[0], cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UPExc, UPExc, sizeof(scalar)*model.neuronN[0], cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UPInh, UPInh, sizeof(scalar)*model.neuronN[1], cudaMemcpyHostToDevice));
  }
}
	
void classIzh::init(unsigned int which)
{
  if (which == CPU) {
  }
  if (which == GPU) {
    copyStateToDevice();
  }
}


void classIzh::allocate_device_mem_input()
{
  unsigned int size;

  size= model.neuronN[0]*sizeof(scalar);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_input1, size));
    
  size= model.neuronN[1]*sizeof(scalar);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_input2, size));
}

void classIzh::copy_device_mem_input()
{
  CHECK_CUDA_ERRORS(cudaMemcpy(d_input1,input1, model.neuronN[0]*sizeof(scalar), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_input2,input2, model.neuronN[1]*sizeof(scalar), cudaMemcpyHostToDevice));
}

void classIzh::free_device_mem()
{
  // clean up memory                          
  printf("input is const, no need to copy");                                     
  CHECK_CUDA_ERRORS(cudaFree(d_input1));                                   
  CHECK_CUDA_ERRORS(cudaFree(d_input2));
}

classIzh::~classIzh()
{
  free(input1);
  free(input2);
  freeMem();
}

// Functions related to explicit input
void classIzh::write_input_to_file(FILE *f)
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

/*void classIzh::read_input_values(FILE *f)
{
  fread(input1, model.neuronN[0]*sizeof(scalar),1,f);
}*/


void classIzh::create_input_values() //define your explicit input rule here
{

  		for (int x= 0; x < model.neuronN[0]; x++) {
   		input1[x]= 5.0*RG.n();
 		 }
  
 		for (int x= 0; x < model.neuronN[1]; x++) {
   		input2[x]= 2.0*RG.n();
  		}


}

void classIzh::read_sparsesyns_par(int synInd, Conductance C, FILE *f_ind, FILE *f_indInG, FILE *f_g, scalar *g //!< File handle for a file containing sparse conductivity values
			    )
{
  unsigned int retval=0; //to make the compiler happy
  double * gtemp = new double[C.connN]; //we need this now as files will always be generated as double but we may run the model with single precision
  //retval=fread(g, 1, C.connN*sizeof(double),f_g);
  fprintf(stdout,"Reading conductance values ... \n");
  retval=fread(gtemp, 1, C.connN*sizeof(double),f_g);

  importArray(g, gtemp, C.connN);
  
  fprintf(stdout,"Conductances are converted. \n");

  if (retval!=C.connN*sizeof(double)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
  fprintf(stdout,"%d active synapses in group %d. \n",C.connN,synInd);
  retval=fread(C.indInG, 1, (model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int),f_indInG);
  if (retval!=(model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");
  retval=fread(C.ind, 1, C.connN*sizeof(int),f_ind);
  if (retval!=C.connN*sizeof(int)) fprintf(stderr, "ERROR: Number of elements read is different than it should be.");

  // general:
  fprintf(stdout,"Read conductance ... \n");
  fprintf(stdout, "Size is %d for synapse group %d. Values start with: \n",C.connN, synInd);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%f ", float(g[i]));
  }
  fprintf(stdout,"\n\n");
  fprintf(stdout, "%d indices read. Index values start with: \n",C.connN);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%d ", C.ind[i]);
  }  
  fprintf(stdout,"\n\n");  
  fprintf(stdout, "%d g indices read. Index in g array values start with: \n", model.neuronN[model.synapseSource[synInd]]+1);
  for(int i= 0; i < 20; i++) {
    fprintf(stdout, "%d ", C.indInG[i]);
  }  
	fprintf(stdout,"\n\n");
  delete [] gtemp;
}

void classIzh::gen_alltoall_syns( scalar * g, unsigned int nPre, unsigned int nPost, scalar gscale//!< Generate random conductivity values for an all to all network
			    )
{
  //randomGen R;
  for(int i= 0; i < model.neuronN[nPre]; i++) {
  	 for(int j= 0; j < model.neuronN[nPost]; j++){
      g[i*model.neuronN[nPost]+j]=gscale*R.n();
    }
  }
  fprintf(stdout,"\n\n");
}

void classIzh::setInput(unsigned int which)
{
	create_input_values();
	//if (which == GPU) copy_device_mem_input();
}



void classIzh::run(double runtime, unsigned int which)
{
  int riT= (int) (runtime/DT);
  if (which == GPU){
    for (int i= 0; i < riT; i++) {
      stepTimeGPU(d_input1,d_input2, t);
      t+= DT;
      iT++;
    }
  }
  
  if (which == CPU){
    for (int i= 0; i < riT; i++) {
      stepTimeCPU(input1, input2,t);
      t+= DT;
      iT++;
		}
  }

}

void classIzh::sum_spikes()
{
  sumPExc+= glbSpkCntPExc;
  sumPInh+= glbSpkCntPInh;
}

//--------------------------------------------------------------------------
// output functions

void classIzh::output_state(FILE *f, unsigned int which)
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

void classIzh::output_params(FILE *f, FILE *f2)
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
		fprintf(f2, "%f ", aPInh[i]);
		fprintf(f2, "%f ", bPInh[i]);
		fprintf(f2, "%f ", cPInh[i]);
		fprintf(f2, "%f ", dPInh[i]);
		fprintf(f2, "%f ", input2[i]);
		fprintf(f2,"\n");
		
	}
}
//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void classIzh::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step
 
This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void classIzh::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}


void classIzh::output_spikes(FILE *f, unsigned int which)
{
	if (which == GPU) {
	 	getSpikeNumbersFromGPU();
		getSpikesFromGPU();
	}
 
  for (int i= 0; i < glbSpkCntPExc; i++) {
		fprintf(f,"%f %d\n", t, glbSpkPExc[i]);
  }

  for (int i= 0; i < glbSpkCntPInh; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[0]+glbSpkPInh[i]);
  }
}
#endif	

 
