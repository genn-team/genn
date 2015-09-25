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
  allocateMem();
  initialize();
  sumIzh1 = 0;
}

void neuronpop::init(unsigned int which)
{
  if (which == CPU) {
  }
  if (which == GPU) {
#ifndef CPU_ONLY
    copyStateToDevice();
#endif
  }
}


neuronpop::~neuronpop()
{
  freeMem();
}

void neuronpop::run(float runtime, unsigned int which)
{
  int riT= (int) (runtime/DT);

  for (int i= 0; i < riT; i++) {
    if (which == GPU){
#ifndef CPU_ONLY
       stepTimeGPU();
#endif
    }
    if (which == CPU)
       stepTimeCPU();
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
#ifndef CPU_ONLY
    copyStateFromDevice();
#endif

  fprintf(f, "%f ", t);

   for (int i= 0; i < model.neuronN[0]-1; i++) {
     fprintf(f, "%f ", VIzh1[i]);
   }

  fprintf(f,"\n");
}

#ifndef CPU_ONLY
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
#endif

void neuronpop::output_spikes(FILE *f, unsigned int which)
{

   for (int i= 0; i < glbSpkCntIzh1[0]; i++) {
     fprintf(f, "%f %d\n", t, glbSpkIzh1[i]);
   }

}
#endif	

