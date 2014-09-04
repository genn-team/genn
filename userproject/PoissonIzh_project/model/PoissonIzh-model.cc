/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _POISSONIZHMODEL_CC_
#define _POISSONIZHMODEL_CC_

#include "PoissonIzh_CODE/runner.cc"

classol::classol()
{
  modelDefinition(model);
  pattern= new unsigned int[model.neuronN[0]*PATTERNNO];
  baserates= new unsigned int[model.neuronN[0]];
  allocateMem();
  initialize();
  sumPN= 0;
  sumIzh1= 0;
}

void classol::init(unsigned int which)
{
  if (which == CPU) {
    theRates= baserates;
  }
  if (which == GPU) {
    copyGToDevice(); 
    copyStateToDevice();
    theRates= d_baserates;
  }
}

void classol::allocate_device_mem_patterns()
{
  unsigned int size;

  // allocate device memory for input patterns
  size= model.neuronN[0]*PATTERNNO*sizeof(unsigned int);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_pattern, size));
  fprintf(stderr, "allocated %u elements for pattern.\n", size/sizeof(unsigned int));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(unsigned int);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_baserates, size));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
}

void classol::allocate_device_mem_input()
{
  unsigned int size;

  // allocate device memory for explicit input
  size= model.neuronN[0]*PATTERNNO*sizeof(unsigned int);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_pattern, size));
  fprintf(stderr, "allocated %u elements for pattern.\n", size/sizeof(unsigned int));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(unsigned int);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_baserates, size));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
}

void classol::free_device_mem()
{
  // clean up memory
  CHECK_CUDA_ERRORS(cudaFree(d_pattern));
  CHECK_CUDA_ERRORS(cudaFree(d_baserates));
}

classol::~classol()
{
  delete [] pattern;
  delete [] baserates;
  freeMem();
}

void classol::read_PNIzh1syns(float *gp, FILE *f)
{
  fprintf(stderr, "%u\n", model.neuronN[0]*model.neuronN[1]*sizeof(float));
  fread(gp, model.neuronN[0]*model.neuronN[1]*sizeof(float),1,f);
  fprintf(stderr,"read PNIzh1 ... \n");
  fprintf(stderr, "values start with: \n");
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%f ", gp[i]);
  }
  fprintf(stderr,"\n\n");
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


void classol::read_input_patterns(FILE *f)
{
  // we use a predefined pattern number
  fread(pattern, model.neuronN[0]*PATTERNNO*sizeof(unsigned int),1,f);
  fprintf(stderr, "read patterns ... \n");
  fprintf(stderr, "input pattern values start with: \n");
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%d ", pattern[i]);
  }
  fprintf(stderr, "\n\n");
}

void classol::generate_baserates()
{
  // we use a predefined pattern number
  for (int i= 0; i < model.neuronN[0]; i++) {
    baserates[i]= INPUTBASERATE;
  }
  fprintf(stderr, "generated basereates ... \n");
  fprintf(stderr, "baserate value: %d ", INPUTBASERATE);
  fprintf(stderr, "\n\n");  
}

void classol::run(float runtime, unsigned int which)
{
  unsigned int pno;
  unsigned int offset= 0;
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (iT%patSetTime == 0) {
      pno= (iT/patSetTime)%PATTERNNO;
      if (which == CPU)
				theRates= pattern;
      if (which == GPU)
				theRates= d_pattern;
      offset= pno*model.neuronN[0];
    }
    if (iT%patSetTime == patFireTime) {
      if (which == CPU)
	theRates= baserates;
      if (which == GPU)
	theRates= d_baserates;
      offset= 0;
    }
    if (which == GPU)
       stepTimeGPU(theRates, offset, t);
    if (which == CPU)
       stepTimeCPU(theRates, offset, t);
    t+= DT;
    iT++;
  }
}

//--------------------------------------------------------------------------
// output functions

void classol::output_state(FILE *f, unsigned int which)
{
  if (which == GPU) 
    copyStateFromDevice();

  fprintf(f, "%f ", t);
  for (int i= 0; i < model.neuronN[0]; i++) {
    fprintf(f, "%f ", VPN[i]);
   }

   for (int i= 0; i < model.neuronN[1]; i++) {
     fprintf(f, "%f ", VIzh1[i]);
   }

  fprintf(f,"\n");
}

void classol::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

void classol::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}

void classol::output_spikes(FILE *f, unsigned int which)
{
  for (int i= 0; i < glbscntPN; i++) {
    fprintf(f, "%f %d\n", t, glbSpkPN[i]);
  }
  for (int i= 0; i < glbscntIzh1; i++) {
    fprintf(f,  "%f %d\n", t, model.sumNeuronN[0]+glbSpkIzh1[i]);
  }
}

void classol::sum_spikes()
{
  sumPN+= glbscntPN;
  sumIzh1+= glbscntIzh1;
}


#endif	

