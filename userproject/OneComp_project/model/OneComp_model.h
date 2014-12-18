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
  float *input1;
  //------------------------------------------------------------------------
  // on the device:
  float *d_input1;
  //------------------------------------------------------------------------
  unsigned int sumIzh1;
  neuronpop();
  ~neuronpop();
  void init(unsigned int);
  void allocate_device_mem_patterns();
  void allocate_device_mem_input();
  void copy_device_mem_input();
  void write_input_to_file(FILE *);
  void read_input_values(FILE *);
  void create_input_values(float t);
  void run(float, unsigned int);
  void getSpikesFromGPU(); 
  void getSpikeNumbersFromGPU(); 
  void output_state(FILE *, unsigned int);
  void output_spikes(FILE *, unsigned int);
  void sum_spikes();
};

#endif
