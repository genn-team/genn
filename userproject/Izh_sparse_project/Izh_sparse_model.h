/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#ifndef IZH_SPARSE_MODEL_H 
#define IZH_SPARSE_MODEL_H

#include "Izh_sparse.cc"

class classol
{
 public:
  NNmodel model;
  float *input1, *input2;
  //------------------------------------------------------------------------
  // on the device:
  float *d_input1, *d_input2;
  //------------------------------------------------------------------------
  unsigned int sumPExc, sumPInh;
  classol();
  ~classol();
  void init(unsigned int);
  void allocate_device_mem_patterns();
  void allocate_device_mem_input();
  void copy_device_mem_input();
  void read_sparsesyns_par(int, struct Conductance, FILE *,FILE *,FILE *); 
  void gen_alltoall_syns(float *, unsigned int, unsigned int, float);
  void free_device_mem();
  void write_input_to_file(FILE *);
  void read_input_values(FILE *);
  void create_input_values(float);
  void run(float, unsigned int);
  void getSpikesFromGPU(); 
  void getSpikeNumbersFromGPU(); 
  void output_state(FILE *, unsigned int);
  void output_spikes(FILE *, unsigned int);
  void output_params(FILE *, FILE *);
  void sum_spikes();
  void setInput(unsigned int);
  void randomizeVar(float *, float, unsigned int);
  void randomizeVarSq(float *, float, unsigned int);
  void initializeAllVars(unsigned int);
};

#endif
