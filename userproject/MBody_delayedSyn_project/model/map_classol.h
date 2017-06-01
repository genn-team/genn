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

#include <stdint.h>

//--------------------------------------------------------------------------
/*! \file userproject/MBody_delayedSyn_project/model/map_classol.h

\brief Header file containing the class definition for classol (CLASSification OLfaction model), which contains the methods for setting up, initialising, simulating and saving results of a model of the insect mushroom body.

The "classol" class is provided as part of a complete example of using GeNN in a user application. The model is a reimplementation of the model in

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005), doi:10.1007/s00422-005-0019-7 
*/
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
/*! \brief This class cpontains the methods for running the MBody_delayedSyn example model.
 */
//--------------------------------------------------------------------------

class classol
{
private:
    void importArray(scalar *, double *, int);
    void exportArray(double *, scalar *, int);
 public:
  NNmodel model;
  scalar *p_pattern;  
  uint64_t *pattern;
  uint64_t *baserates;
  //------------------------------------------------------------------------
  // on the device:
  uint64_t *d_pattern;
  uint64_t *d_baserates;
  //------------------------------------------------------------------------
  unsigned int sumPN, sumKC, sumLHI, sumDN;
  unsigned int size_g; //number of connections
  // end of data fields 

  classol();
  ~classol();
  void init(unsigned int); 
#ifndef CPU_ONLY
  void allocate_device_mem_patterns();
  void free_device_mem(); 
#endif
  void read_pnkcsyns(FILE *);
  void read_sparsesyns_par(const char*, struct SparseProjection, scalar *, FILE *,FILE *,FILE *);
  void write_pnkcsyns(FILE *); 
  void read_pnlhisyns(FILE *); 
  void write_pnlhisyns(FILE *); 
  void read_kcdnsyns(FILE *); 
  void write_kcdnsyns(FILE *); 
  void read_input_patterns(FILE *); 
  void generate_baserates(); 
#ifndef CPU_ONLY
  void runGPU(scalar); 
#endif
  void runCPU(scalar); 
  void output_state(FILE *, unsigned int); 
  void getSpikesFromGPU(); 
  void getSpikeNumbersFromGPU(); 
  void output_spikes(FILE *, unsigned int); 
  void sum_spikes(); 
  void get_kcdnsyns(); 
};

#endif
