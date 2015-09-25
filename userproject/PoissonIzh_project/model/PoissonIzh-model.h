/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#ifndef POISSONIZHMODEL_H 
#define POISSONIZHMODEL_H


class classol
{
private:
    void importArray(scalar *, double *, int);
    void exportArray(double *, scalar *, int);
 public:
  NNmodel model;
  uint64_t *baserates;
  //------------------------------------------------------------------------
  // on the device:
  uint64_t *d_baserates;
  //------------------------------------------------------------------------
  unsigned int sumPN, sumIzh1;
  // end of data fields 

  classol();
  ~classol();
  void init(unsigned int);
#ifndef CPU_ONLY
  void allocate_device_mem_input();
  void free_device_mem();
#endif
  void read_PNIzh1syns(scalar *, FILE *);
  void read_sparsesyns_par(int, struct SparseProjection, FILE *,FILE *,FILE *, double *);
  void generate_baserates();
  void run(float, unsigned int);
  void output_state(FILE *, unsigned int);
#ifndef CPU_ONLY
  void getSpikesFromGPU();
  void getSpikeNumbersFromGPU();
#endif
  void output_spikes(FILE *, unsigned int);
  void sum_spikes();
};

#endif
