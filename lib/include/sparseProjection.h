/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2015-08-19
   
--------------------------------------------------------------------------*/

#ifndef SPARSE_PROJECTION
#define SPARSE_PROJECTION

//! \brief class (struct) for defining a spars connectivity projection
struct SparseProjection{
    unsigned int *indInG;
    unsigned int *ind;
    unsigned int *preInd;
    unsigned int *revIndInG;
    unsigned int *revInd;
    unsigned int *remap;
    unsigned int connN; 
};

#endif
