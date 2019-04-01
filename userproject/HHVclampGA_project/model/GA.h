/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-25
  
--------------------------------------------------------------------------*/
#pragma once

#include <algorithm>


typedef struct {
    unsigned int id;
    double err;
} errTupel;

int compareErrTupel(const void *x, const void *y) 
{
    if (((errTupel *)x)->err < ((errTupel *)y)->err) return -1;
    if (((errTupel *)x)->err > ((errTupel *)y)->err) return 1;
    return 0;
}

void single_var_reinit(int n, double fac, std::mt19937 &rng, std::normal_distribution<double> &normal)
{
    gNaHH[n]*= (1.0+fac*sigGNa*normal(rng)); // multiplicative Gaussian noise
    ENaHH[n]+= fac*sigENa*normal(rng); // additive Gaussian noise
    gKHH[n]*= (1.0+fac*sigGK*normal(rng)); // multiplicative Gaussian noise
    EKHH[n]+= fac*sigEK*normal(rng); // additive Gaussian noise
    glHH[n]*= (1.0+fac*sigGl*normal(rng)); // multiplicative Gaussian noise
    ElHH[n]+= fac*sigEl*normal(rng); // additive Gaussian noise
    CHH[n]*= (1.0+fac*sigC*normal(rng)); // multiplicative Gaussian noise
}

void copy_var(int src, int trg)
{
    gNaHH[trg]= gNaHH[src];
    ENaHH[trg]= ENaHH[src];
    gKHH[trg]= gKHH[src];
    EKHH[trg]=EKHH[src];
    glHH[trg]= glHH[src];
    ElHH[trg]= ElHH[src];
    CHH[trg]= CHH[src];
}


void procreatePop(FILE *osb, std::mt19937 &rng)
{
    static errTupel errs[NPOP];
    for (int i= 0; i < NPOP; i++) {
        errs[i].id= i;
        errs[i].err= errHH[i];
    }
    qsort((void *) errs, NPOP, sizeof(errTupel), compareErrTupel);

#ifdef DEBUG_PROCREATE
    cerr << "% sorted fitness: ";
    for (int i= 0; i < NPOP; i++) {
        cerr << errs[i].err << " ";
    }
    cerr << endl;
#endif
    fprintf(osb, "%f %f %f %f %f %f %f %f \n", gNaHH[errs[0].id], ENaHH[errs[0].id], gKHH[errs[0].id], EKHH[errs[0].id], glHH[errs[0].id], ElHH[errs[0].id], CHH[errs[0].id], errHH[errs[0].id]);

    // mutate the second half of the instances
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i= NPOP/2, j= 0; i < NPOP; i++) {
        copy_var(errs[j].id, errs[i].id); // copy good ones over bad ones
        single_var_reinit(errs[i].id, 0.1, rng, normal); // jiggle the new copies a bit
        j++;
    }

    copyStateToDevice();
}


