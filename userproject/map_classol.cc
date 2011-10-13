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
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_pattern, size));
  cerr << "allocated " << size/sizeof(unsigned int) << " elements for pattern" << endl;
  CUDA_SAFE_CALL(cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice));
  size= model.neuronN[0]*sizeof(unsigned int);
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_baserates, size));
  CUDA_SAFE_CALL(cudaMemcpy(d_baserates, baserates, size, cudaMemcpyHostToDevice)); 
}


void classol::free_device_mem()
{
  // clean up memory                                                            
  CUDA_SAFE_CALL(cudaFree(d_pattern));
  CUDA_SAFE_CALL(cudaFree(d_baserates));
}



classol::~classol()
{
  delete[] pattern;
  delete[] baserates;
}


void classol::read_pnkcsyns(istream &is)
{
  // version 1
  cerr << model.neuronN[0]*model.neuronN[1]*sizeof(float) << endl;
  is.read((char *)gpPNKC, model.neuronN[0]*model.neuronN[1]*sizeof(float));
  // version 2
  /*  unsigned int UIntSz= sizeof(unsigned int)*8;   // in bit!
  unsigned int logUIntSz= (int) (logf((float) UIntSz)/logf(2.0f)+1e-5f);
  unsigned int tmp= model.neuronN[0]*model.neuronN[1];
  unsigned size= (tmp >> logUIntSz);
  if (tmp > (size << logUIntSz)) size++;
  size= size*sizeof(unsigned int);
  is.read((char *)gpPNKC, size);*/

  // general:
  assert(is.good());
  cerr << "read pnkc ... " << endl;
  cerr << "values start with: ";
  for(int i= 0; i < 20; i++) {
    cerr << gpPNKC[i] << " ";
  }
  cerr << endl << endl;
}


void classol::write_pnkcsyns(ostream &os)
{
  os.write((char *)gpPNKC, model.neuronN[0]*model.neuronN[1]*sizeof(float));
  cerr << "wrote pnkc ... " << endl;
}


void classol::read_pnlhisyns(istream &is)
{
  is.read((char *)gpPNLHI, model.neuronN[0]*model.neuronN[2]*sizeof(float));
  assert(is.good());
  cerr << "read pnlhi ... " << endl;
  cerr << "values start with: ";
  for(int i= 0; i < 20; i++) {
    cerr << gpPNLHI[i] << " ";
  }
  cerr << endl << endl;
}

void classol::write_pnlhisyns(ostream &os)
{
  os.write((char *)gpPNLHI, model.neuronN[0]*model.neuronN[2]*sizeof(float));
  cerr << "wrote pnlhi ... " << endl;
}


void classol::read_kcdnsyns(istream &is)
{
  is.read((char *)gpKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(float));
  assert(is.good());
  cerr << "read kcdn ... " << endl;
  cerr << "values start with: ";
  for(int i= 0; i < 20; i++) {
    cerr << gpKCDN[i] << " ";
  }
  cerr << endl << endl;
}


void classol::write_kcdnsyns(ostream &os)
{
  os.write((char *)gpKCDN, model.neuronN[1]*model.neuronN[3]*sizeof(float));
  cerr << "wrote kcdn ... " << endl;
}


void classol::read_input_patterns(istream &is)
{
  // we use a predefined pattern number
  is.read((char *)pattern, model.neuronN[0]*PATTERNNO*sizeof(unsigned int));
  assert(is.good());
  cerr << "read patterns ... " << endl;
  cerr << "values start with: ";
  for(int i= 0; i < 20; i++) {
    cerr << pattern[i] << " ";
  }
  cerr << endl << endl;
}

void classol::generate_baserates()
{
  // we use a predefined pattern number
  for (int i= 0; i < model.neuronN[0]; i++) {
    baserates[i]= INPUTBASERATE;
  }
  cerr << "generated basereates ... " << endl;
  cerr << "baserate value: " << INPUTBASERATE;
  cerr << endl << endl;  
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
      cerr << "setting pattern, pattern offset:" << offset << endl;
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
    //    cerr << t << endl;
  }
}

//--------------------------------------------------------------------------
// output functions

void classol::output_state(ostream &os, unsigned int which)
{
  if (which == GPU) 
    copyStateFromDevice();

  os << t << " ";
  for (int i= 0; i < model.neuronN[0]; i++) {
     os << VPN[i] << " ";
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
     os << VKC[i] << " ";
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
    os << VLHI[i] << " ";
  }
  for (int i= 0; i < model.neuronN[3]; i++) {
    os << VDN[i] << " ";
  }
  os << endl;
}

void classol::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

void classol::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}

void classol::output_spikes(ostream &os, unsigned int which)
{

  //    cerr << t << " " << glbscntPN << " " << glbscntKC << " ";
  //    cerr << glbscntLHI << " " << glbscntDN << endl;
  for (int i= 0; i < glbscntPN; i++) {
    os << t << " " << glbSpkPN[i] << endl;
  }
  for (int i= 0; i < glbscntKC; i++) {
    os << t << " " << model.sumNeuronN[0]+glbSpkKC[i] << endl;
  }
  for (int i= 0; i < glbscntLHI; i++) {
    os << t << " " << model.sumNeuronN[1]+glbSpkLHI[i] << endl;
  }
  for (int i= 0; i < glbscntDN; i++) {
    os << t << " " << model.sumNeuronN[2]+glbSpkDN[i] << endl;
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
  CUDA_SAFE_CALL(cudaMemcpy(gpKCDN, devPtr, 
    model.neuronN[1]*model.neuronN[3]*sizeof(float), 
    cudaMemcpyDeviceToHost));
}



#endif	

