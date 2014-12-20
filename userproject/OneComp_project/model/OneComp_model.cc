/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#ifndef _ONECOMP_MODEL_CC_
#define _ONECOMP_MODEL_CC_

#include "OneComp_CODE/runner.cc"

neuronpop::neuronpop()
{
  modelDefinition(model);
  input1=new float[model.neuronN[0]];
  allocateMem();
  initialize();
  sumIzh1 = 0;
}

void neuronpop::init(unsigned int which)
{
  if (which == CPU) {
  }
  if (which == GPU) {
    copyStateToDevice();
  }
}


void neuronpop::allocate_device_mem_input()
{
  unsigned int size;

  size= model.neuronN[0]*sizeof(float);
  CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_input1, size));
}

void neuronpop::copy_device_mem_input()
{
  CHECK_CUDA_ERRORS(cudaMemcpy(d_input1, input1, model.neuronN[0]*sizeof(float), cudaMemcpyHostToDevice));
}

neuronpop::~neuronpop()
{
  free(input1);
  freeMem();
}

void neuronpop::write_input_to_file(FILE *f)
{
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

void neuronpop::read_input_values(FILE *f)
{
  fread(input1, model.neuronN[0]*sizeof(float),1,f);
}


void neuronpop::create_input_values(float t) //define your explicit input rule here. The model uses constant input instead at the moment.
{
  const double pi = 4*atan(1.);
  float frequency =5;
  for (int x= 0; x < model.neuronN[0]; x++) {
    input1[x]= 5*float(sin(frequency*(t+10*(x+1))*0.001*2*pi));
  }
}


void neuronpop::run(float runtime, unsigned int which)
{
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (which == GPU){
       /****if receivesInputCurrent==1***/
       stepTimeGPU(t);
       /****if receivesInputCurrent>1***/
       //stepTimeGPU(d_input1, t);
    }
    if (which == CPU)
       /****if receivesInputCurrent==1***/
       stepTimeCPU(t);
       /****if receivesInputCurrent>1***/
       //stepTimeCPU(input1, t);
    t+= DT;
    iT++;
  }
}

void neuronpop::sum_spikes()
{
  sumIzh1+= glbSpkCntIzh1[0];
}

//--------------------------------------------------------------------------
// output functions

void neuronpop::output_state(FILE *f, unsigned int which)
{
  if (which == GPU) 
    copyStateFromDevice();

  fprintf(f, "%f ", t);

   for (int i= 0; i < model.neuronN[0]-1; i++) {
     fprintf(f, "%f ", VIzh1[i]);
   }

  fprintf(f,"\n");
}


//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void neuronpop::getSpikesFromGPU()
{
  copySpikesFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step
 
This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void neuronpop::getSpikeNumbersFromGPU() 
{
  copySpikeNFromDevice();
}


void neuronpop::output_spikes(FILE *f, unsigned int which)
{

   for (int i= 0; i < glbSpkCntIzh1[0]; i++) {
     fprintf(f, "%f %d\n", t, glbSpkIzh1[i]);
   }

}
#endif	

