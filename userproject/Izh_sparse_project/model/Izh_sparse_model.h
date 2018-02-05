/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#pragma once

#include "Izh_sparse_CODE/definitions.h"

class classIzh
{
 private:
  void importArray(scalar *, double *, int);
  void exportArray(double *, scalar *, int);
 public:
  //------------------------------------------------------------------------
  unsigned int sumPExc, sumPInh;
  classIzh();
  ~classIzh();
  void allocate_device_mem_patterns();
  void allocate_device_mem_input();
  void read_sparsesyns_par(unsigned int, SparseProjection&, FILE *,FILE *,FILE *, scalar *);
  void free_device_mem();
  void write_input_to_file(FILE *);
  void read_input_values(FILE *);
  void create_input_values();
  void run(double, unsigned int);
  void getSpikesFromGPU(); 
  void getSpikeNumbersFromGPU(); 
  void output_state(FILE *, unsigned int);
  void output_spikes(FILE *, unsigned int);
  void output_params(FILE *, FILE *);
  void sum_spikes();
  void initializeAllVars(unsigned int);
};
