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
  unsigned int offset;
  uint64_t *theRates;
  scalar *p_pattern; 
  uint64_t *pattern;
  uint64_t *baserates;
  //------------------------------------------------------------------------
  // on the device:
  uint64_t *d_pattern;
  uint64_t *d_baserates;
  //------------------------------------------------------------------------
  unsigned int sumPN, sumIzh1;
  // end of data fields 

  classol();
  ~classol();
  void init(unsigned int);
  void allocate_device_mem_patterns();
  void allocate_device_mem_input();
  void free_device_mem();
  void read_PNIzh1syns(scalar *, FILE *);
  void read_sparsesyns_par(int, struct Conductance, FILE *,FILE *,FILE *, double *);
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
