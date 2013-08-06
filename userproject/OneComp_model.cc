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

#include "IzhEx_CODE/runner.cc"

classol::classol()
{
  modelDefinition(model);
 input1=new float[model.neuronN[0]];
  allocateMem();
  initialize();

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
}
void classol::copy_device_mem_input()
{
  CHECK_CUDA_ERRORS(cudaMemcpy(d_input1, input1, model.neuronN[0]*sizeof(float), cudaMemcpyHostToDevice));
}
void classol::free_device_mem()
{
  // clean up memory                          
                                       
  CHECK_CUDA_ERRORS(cudaFree(d_input1));
}



classol::~classol()
{
  free(input1);
}


void classol::write_input_to_file(FILE *f)
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

void classol::read_input_values(FILE *f)
{
  fread(input1, model.neuronN[0]*sizeof(float),1,f);
}


void classol::create_input_values(FILE * f, float t) //define your explicit input rule here
{
  const double pi = 4*atan(1.);
  float frequency =5;
  for (int x= 0; x < model.neuronN[0]; x++) {
    input1[x]= 5*float(sin(frequency*(t+10*(x+1))*0.001*2*pi));
  }
}


void classol::run(float runtime, unsigned int which)
{
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (which == GPU){
       stepTimeGPU(d_input1,t);
    }
    if (which == CPU)
       stepTimeCPU(input1,t);
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

   for (int i= 0; i < model.neuronN[0]-1; i++) {
     fprintf(f, "%f ", VIzh1[i]);
   }

  fprintf(f,"\n");
}

#endif	

