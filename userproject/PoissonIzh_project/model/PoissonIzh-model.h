/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#pragma once

#include <cstdint>

#include "PoissonIzh_CODE/definitions.h"

class classol
{
private:
    void importArray(scalar *, double *, int);
    void exportArray(double *, scalar *, int);

public:
    //------------------------------------------------------------------------
    unsigned int sumPN, sumIzh1;
    // end of data fields

    classol();
    ~classol();

    void init(unsigned int);
    void read_PNIzh1syns(scalar *, FILE *);
    void read_sparsesyns_par(unsigned int numPre, SparseProjection &, FILE *,FILE *,FILE *, double *);
    void run(float, unsigned int);
    void output_state(FILE *, unsigned int);
#ifndef CPU_ONLY
    void getSpikesFromGPU();
    void getSpikeNumbersFromGPU();
#endif
    void output_spikes(FILE *, unsigned int);
    void sum_spikes();
};
