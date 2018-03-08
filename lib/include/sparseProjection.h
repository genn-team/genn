/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2015-08-19
   
--------------------------------------------------------------------------*/
#pragma once

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

//! Row-major ordered sparse matrix structure in 'ragged' format
template<typename PostIndexType>
struct RaggedProjection {
    //! Length of each row of matrix
    unsigned int *rowLength;

    //! Indices of target neurons
    PostIndexType *trgInd;

    //unsigned int *preInd;
    //unsigned int *revIndInG;
    //unsigned int *revInd;
    //unsigned int *remap;
};

#endif  // CPU_ONLY
