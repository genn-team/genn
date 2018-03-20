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
    RaggedProjection(unsigned int maxRow, unsigned int maxCol) 
    : maxRowLength(maxRow), maxColLength(maxCol)
    {}

    //! Maximum dimensions of matrices (used for sizing of ind and remap)
    const unsigned int maxRowLength;
    const unsigned int maxColLength;

    //! Length of each row of matrix
    unsigned int *rowLength;

    //! Ragged row-major matrix, padded to maxRowLength containing indices of target neurons
    PostIndexType *ind;

    //! Length of each column of matrix
    unsigned int *colLength;
    
    //! Ragged column-major matrix, padded to maxColLength containing indices back into ind
    unsigned int *remap;
};
