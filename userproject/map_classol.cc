/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _MAP_CLASSOL_CC_
#define _MAP_CLASSOL_CC_

#include "MBody1_CODE/runner.cc"

classol::classol()
{
  modelDefinition(model);
  pattern= new unsigned int[model.neuronN[0]*PATTERNNO];
  baserates= new unsigned int[model.neuronN[0]];
  allocateMem();
  initialize();
  sumPN= 0;
  sumKC= 0;
  sumLHI= 0;
  sumDN= 0;
}

void classol::init(unsigned int which)
{
  initGRaw();
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
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_pattern, size));
  checkCudaErrors(cudaMalloc((void**) &d_pattern, size));
  fprintf(stderr, "allocated %u elements for pattern.\n", size/sizeof(unsigned int));
  //CUDA_SAFE_CALL(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(unsigned int);
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_baserates, size));
  //CUDA_SAFE_CALL(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMalloc((void**) &d_baserates, size));
  checkCudaErrors(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
}


void classol::free_device_mem()
{
  // clean up memory                                                            
  //CUDA_SAFE_CALL(cudaFree(d_pattern));
  //CUDA_SAFE_CALL(cudaFree(d_baserates));
                                       
  checkCudaErrors(cudaFree(d_pattern));
  checkCudaErrors(cudaFree(d_baserates));
}



classol::~classol()
{
  free(pattern);
  free(baserates);
}


void classol::read_pnkcsyns(FILE *f)
{
  // version 1
  fprintf(stderr, "%u\n", model.neuronN[0]*model.neuronN[1]*sizeof(float));
  fread(gpPNKC, model.neuronN[0]*model.neuronN[1]*sizeof(float),1,f);
  // version 2
  /*  unsigned int UIntSz= sizeof(unsigned int)*8;   // in bit!
  unsigned int logUIntSz= (int) (logf((float) UIntSz)/logf(2.0f)+1e-5f);
  unsigned int tmp= model.neuronN[0]*model.neuronN[1];
  unsigned size= (tmp >> logUIntSz);
  if (tmp > (size << logUIntSz)) size++;
  size= size*sizeof(unsigned int);
  is.read((char *)gpPNKC, size);*/

  // general:
  //assert(is.good());
  fprintf(stderr,"read pnkc ... \n");
  fprintf(stderr, "values start with: \n");
  for(int i= 0; i < 20; i++) {
    fprintf(stderr, "%f ", gpPNKC[i]);
  }
  fprintf(stderr,"\n\n");
}


void classol::write_pnkcsyns(FILE *f)
{
  fwrite(gpPNKC, model.neuronN[0]*model.neuronN[1]*sizeof(float),1,f);
  fprintf(stderr, "wrote pnkc ... \n");
}


void classol::read_pnlhisyns(FILE *f)
{
  fread(gpPNLHI, model.neuronN[0]*model.neuronN[2]*sizeof(float),1,f);
  fprintf(stderr,"read pnlhi ... \n");
  fprintf(stderr, "values start with: \n");
  for(int i= 0; i < 20; i++) {
    fprintf(stderr, "%f ", gpPNLHI[i]);
  }
  fprintf(stderr, "\n\n");
}

void classol::write_pnlhisyns(FILE *f)
{
  fwrite(gpPNLHI, model.neuronN[0]*model.neuronN[2]*sizeof(float),1,f);
  fprintf(stderr, "wrote pnlhi ... \n");
}


void classol::read_kcdnsyns(FILE *f)
{
  fread(gpKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(float),1,f);
  fprintf(stderr, "read kcdn ... \n");
  fprintf(stderr, "values start with: \n");
  for(int i= 0; i < 20; i++) {
    fprintf(stderr, "%f ", gpKCDN[i]);
  }
  fprintf(stderr, "\n\n");
}


void classol::write_kcdnsyns(FILE *f)
{
  fwrite(gpKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(float),1,f);
  fprintf(stderr, "wrote kcdn ... \n");
}


void classol::read_input_patterns(FILE *f)
{
  // we use a predefined pattern number
  fread(pattern, model.neuronN[0]*PATTERNNO*sizeof(unsigned int),1,f);
  fprintf(stderr, "read patterns ... \n");
  fprintf(stderr, "values start with: \n");
  for(int i= 0; i < 20; i++) {
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
    if (iT%PAT_SETTIME == 0) {
      pno= (iT/PAT_SETTIME)%PATTERNNO;
      if (which == CPU)
	theRates= pattern;
      if (which == GPU)
	theRates= d_pattern;
      offset= pno*model.neuronN[0];
      fprintf(stderr, "setting pattern, pattern offset: %d\n", offset);
    }
    if (iT%PAT_SETTIME == PAT_FIRETIME) {
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
    //    fprintf(stderr, "%f\n", t);
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
  // for (int i= 0; i < model.neuronN[0]; i++) {
  //   os << seedsPN[i] << " ";
  // }
  // for (int i= 0; i < model.neuronN[0]; i++) {
  //   os << spikeTimesPN[i] << " ";
  // }
  //  os << glbscnt0 << "  ";
  // for (int i= 0; i < glbscntPN; i++) {
  //   os << glbSpkPN[i] << " ";
  // }
  // os << " * ";
  // os << glbscntKC << "  ";
  // for (int i= 0; i < glbscntKC; i++) {
  //   os << glbSpkKC[i] << " ";
  // }
  // os << " * ";
  // os << glbscntLHI << "  ";
  // for (int i= 0; i < glbscntLHI; i++) {
  //   os << glbSpkLHI[i] << " ";
  // }
  // os << " * ";
  // os << glbscntDN << "  ";
  // for (int i= 0; i < glbscntDN; i++) {
  //   os << glbSpkDN[i] << " ";
  // }
 //   os << " * ";
 // for (int i= 0; i < 20; i++) {
 //   os << VKC[i] << " ";
 // }
   for (int i= 0; i < model.neuronN[1]; i++) {
     fprintf(f, "%f ", VKC[i]);
   }
  //os << " * ";
  //for (int i= 0; i < model.neuronN[1]; i++) {
  //  os << inSynKC0[i] << " ";
  //}
  //  os << " * ";
  //  for (int i= 0; i < 20; i++) {
  //    os << inSynKC1[i] << " ";
  //  }
  //  os << " * ";
  //  for (int i= 0; i < 20; i++) {
  //    os << inSynLHI0[i] << " ";
  //  }
  //  os << " * ";
  //  for (int i= 0; i < model.neuronN[3]; i++) {
  //    os << inSynDN0[i] << " ";
  //  }
  //  os << endl;
  //  os << " * ";
  //  for (int i= 0; i < model.neuronN[3]; i++) {
  //    os << inSynDN1[i] << " ";
  //  }
  for (int i= 0; i < model.neuronN[2]; i++) {
    fprintf(f, "%f ", VLHI[i]);
  }
  for (int i= 0; i < model.neuronN[3]; i++) {
    fprintf(f, "%f ", VDN[i]);
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

  //    fprintf(stderr, "%f %f %f %f %f\n", t, glbscntPN, glbscntKC, glbscntLHI,glbscntDN);
  for (int i= 0; i < glbscntPN; i++) {
    fprintf(f, "%f %d\n", t, glbSpkPN[i]);
  }
  for (int i= 0; i < glbscntKC; i++) {
    fprintf(f,  "%f %d\n", t, model.sumNeuronN[0]+glbSpkKC[i]);
  }
  for (int i= 0; i < glbscntLHI; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[1]+glbSpkLHI[i]);
  }
  for (int i= 0; i < glbscntDN; i++) {
    fprintf(f, "%f %d\n", t, model.sumNeuronN[2]+glbSpkDN[i]);
  }
}

void classol::sum_spikes()
{
  sumPN+= glbscntPN;
  sumKC+= glbscntKC;
  sumLHI+= glbscntLHI;
  sumDN+= glbscntDN;
}

void classol::get_kcdnsyns()
{
  void *devPtr;
  cudaGetSymbolAddress(&devPtr, "d_gpKCDN");
  //CUDA_SAFE_CALL(cudaMemcpy(gpKCDN, devPtr, 
  //model.neuronN[1]*model.neuronN[3]*sizeof(float), 
  //  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(gpKCDN, devPtr,
    model.neuronN[1]*model.neuronN[3]*sizeof(float), 
    cudaMemcpyDeviceToHost));
}



#endif	

