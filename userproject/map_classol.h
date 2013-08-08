/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#ifndef CLASSOL_H
#define CLASSOL_H

#include "MBody1.cc"

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
  unsigned int sumPN, sumKC, sumLHI, sumDN;
  // end of data fields 

  classol();
  ~classol();
  void init(unsigned int);
  void allocate_device_mem_patterns();
  void free_device_mem();
  void read_pnkcsyns(FILE *);
  void write_pnkcsyns(FILE *);
  void read_pnlhisyns(FILE *);
  void write_pnlhisyns(FILE *);
  void read_kcdnsyns(FILE *);
  void write_kcdnsyns(FILE *);
  void read_input_patterns(FILE *);
  void generate_baserates();
  void run(float, unsigned int);
  void output_state(FILE *, unsigned int);
  void getSpikesFromGPU();
  void getSpikeNumbersFromGPU();
  void output_spikes(FILE *, unsigned int);
  void sum_spikes();
  void get_kcdnsyns();
};

#endif
