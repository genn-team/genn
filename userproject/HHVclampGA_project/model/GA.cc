/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Informatics
              University of Sussex 
              Brighton BN1 9QJ, UK
  
   email to:  t.nowotny@sussex.ac.uk
  
   initial version: 2014-06-25
  
--------------------------------------------------------------------------*/

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

void procreatePop(FILE *osb)
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
  for (int i= NPOP/2, j= 0; i < NPOP; i++) {
    copy_var(errs[j].id, errs[i].id); // copy good ones over bad ones
    single_var_reinit(errs[i].id, 0.1); // jiggle the new copies a bit
    j++;
  }
  copyStateToDevice();	
}


