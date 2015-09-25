/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#ifndef ONECOMP_H 
#define ONECOMP_H

#include "OneComp.cc"

class neuronpop
{
 public:
  NNmodel model;
  unsigned int sumIzh1;
  neuronpop();
  ~neuronpop();
  void init(unsigned int);
  void run(float, unsigned int);
#ifndef CPU_ONLY
  void getSpikesFromGPU(); 
  void getSpikeNumbersFromGPU();
#endif 
  void output_state(FILE *, unsigned int);
  void output_spikes(FILE *, unsigned int);
  void sum_spikes();
};

#endif
