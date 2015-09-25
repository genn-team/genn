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
#include "PoissonIzh-model.h"
#include "PoissonIzh_CODE/runner.cc"
#include "modelSpec.h"
#include "modelSpec.cc"

classol::classol()
{
  modelDefinition(model);
  baserates= new uint64_t[model.neuronN[0]];
  allocateMem();
  initialize();
  sumPN= 0;
  sumIzh1= 0;
}

void classol::init(unsigned int which)
{
  if (which == CPU) {
    ratesPN= baserates;
  }
  if (which == GPU) {
#ifndef CPU_ONLY
    copyStateToDevice();
    ratesPN= d_baserates;
#endif
  }
}

#ifndef CPU_ONLY
void classol::free_device_mem()
{
  // clean up memory
  CHECK_CUDA_ERRORS(cudaFree(d_baserates));
}
#endif

classol::~classol()
{
  free(baserates);
  freeMem();
}
void classol::importArray(scalar *dest, double *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classol::exportArray(double *dest, scalar *src, int sz) 
{
    for (int i= 0; i < sz; i++) {
	dest[i]= (scalar) src[i];
    }
}

void classol::read_PNIzh1syns(scalar *gp, FILE *f)
{

  int sz= model.neuronN[0]*model.neuronN[1];
  double *tmpg= new double[sz];
  unsigned int retval = fread(tmpg, 1, sz * sizeof(double),  f);
  importArray(gp, tmpg, sz);
  fprintf(stderr,"read PNIzh1 ... \n");
  fprintf(stderr, "%u bytes, values start with: \n", retval);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%f ", float(gp[i]));
  }
  fprintf(stderr,"\n\n");
}

void classol::read_sparsesyns_par(int synInd, SparseProjection C, FILE *f_ind,FILE *f_indInG,FILE *f_g, double * g //!< File handle for a file containing sparse conductivity values
				  )
{
  //allocateSparseArray(synInd,C.connN);

  unsigned int retval = fread(g, C.connN*sizeof(model.ftype),1,f_g);
  fprintf(stderr,"%d active synapses. \n",C.connN);
  retval = fread(C.indInG, (model.neuronN[model.synapseSource[synInd]]+1)*sizeof(unsigned int),1,f_indInG);
  retval = fread(C.ind, C.connN*sizeof(int),1,f_ind);


  // general:
  fprintf(stderr,"Read conductance ... \n");
  fprintf(stderr, "Size is %d for synapse group %d. Values start with: \n",C.connN, synInd);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%f ", float(g[i]));
  }
  fprintf(stderr,"\n\n");
  
  
  fprintf(stderr, "%d indices read. Index values start with: \n",C.connN);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%d ", C.ind[i]);
  }  
  fprintf(stderr,"\n\n");
  
  
  fprintf(stderr, "%u bytes of %d g indices read. Index in g array values start with: \n", retval, model.neuronN[model.synapseSource[synInd]]+1);
  for(int i= 0; i < 100; i++) {
    fprintf(stderr, "%d ", C.indInG[i]);
  }  
}


void classol::generate_baserates()
{
  // we use a predefined pattern number
    uint64_t inputBase;
    convertRateToRandomNumberThreshold(&InputBaseRate, &inputBase, 1);
  for (int i= 0; i < model.neuronN[0]; i++) {
    baserates[i]= inputBase;
  }
  fprintf(stderr, "generated baserates ... \n");
  fprintf(stderr, "baserate value %f, converted random number: %llu ", InputBaseRate, inputBase);
  fprintf(stderr, "\n\n"); 
#ifndef CPU_ONLY
  int size= model.neuronN[0]*sizeof(uint64_t);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_baserates, size));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
#endif
}

void classol::run(float runtime, unsigned int which)
{
  unsigned int offsetPN= 0;
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (which == GPU) {
#ifndef CPU_ONLY
       stepTimeGPU();
#endif
    }
    if (which == CPU) {
	stepTimeCPU();
    }
    t+= DT;
    iT++;
  }
}

//--------------------------------------------------------------------------
// output functions

void classol::output_state(FILE *f, unsigned int which)
{
  if (which == GPU) 
#ifndef CPU_ONLY
    copyStateFromDevice();
#endif

  fprintf(f, "%f ", t);
  for (int i= 0; i < model.neuronN[0]; i++) {
    fprintf(f, "%f ", VPN[i]);
   }

   for (int i= 0; i < model.neuronN[1]; i++) {
     fprintf(f, "%f ", VIzh1[i]);
   }

  fprintf(f,"\n");
}

#ifndef CPU_ONLY
void classol::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

void classol::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}
#endif

void classol::output_spikes(FILE *f, unsigned int which)
{
  for (int i= 0; i < glbSpkCntPN[0]; i++) {
    fprintf(f, "%f %d\n", t, glbSpkPN[i]);
  }
  for (int i= 0; i < glbSpkCntIzh1[0]; i++) {
    fprintf(f,  "%f %d\n", t, model.sumNeuronN[0]+glbSpkIzh1[i]);
  }
}

void classol::sum_spikes()
{
  sumPN+= glbSpkCntPN[0];
  sumIzh1+= glbSpkCntIzh1[0];
}


#endif	

