/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#ifndef POISSONIZH_H 
#define POISSONIZH_H

#include "Poisson-Izh.cc"

class classol
{
 public:
  NNmodel model;
  unsigned int *theRates;
  unsigned int *pattern;
  unsigned int *baserates;
  //------------------------------------------------------------------------
  // on the device:
  unsigned int *d_pattern;
  unsigned int *d_baserates;
  //------------------------------------------------------------------------
  unsigned int sumPN, sumIzh1;
  // end of data fields 

  classol();
  ~classol();
  void init(unsigned int);
  void allocate_device_mem_patterns();
  void allocate_device_mem_input();
  void free_device_mem();
  void read_PNIzh1syns(FILE *);
  void write_PNIzh1syns(FILE *);
  void read_input_patterns(FILE *);
  void generate_baserates();
  void run(float, unsigned int);
  void output_state(FILE *, unsigned int);
  void getSpikesFromGPU();
  void getSpikeNumbersFromGPU();
  void output_spikes(FILE *, unsigned int);
  void sum_spikes();
};

#endif
