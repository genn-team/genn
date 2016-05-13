
#ifndef SPARSE_UTILS_H
#define SPARSE_UTILS_H

#include <string>

using namespace std;


/*---------------------------------------------------------------------
 Utility to count how many entries above a specified value exist in a float array
 ---------------------------------------------------------------------*/

template <class DATATYPE> unsigned int countEntriesAbove(DATATYPE *Array, int sz, double includeAbove);


/*---------------------------------------------------------------------
 Utility to get a synapse weight from a SPARSE structure by x,y coordinates
 NB: as the SparseProjection struct doesnt hold the preN size (it should!) it is not possible
 to check the parameter validity. This fn may therefore crash unless user knows max poss X
 ---------------------------------------------------------------------*/

template <class DATATYPE> DATATYPE getG(DATATYPE *wuvar, SparseProjection *sparseStruct, int x, int y);


/*---------------------------------------------------------------------
Setting the values of SPARSE connectivity matrix
----------------------------------------------------------------------*/

template <class DATATYPE> void setSparseConnectivityFromDense(DATATYPE *wuvar, int preN, int postN, DATATYPE *tmp_gRNPN, SparseProjection *sparseStruct);


/*---------------------------------------------------------------------
 Utility to generate the SPARSE connectivity structure from a simple all-to-all array
 ---------------------------------------------------------------------*/

template <class DATATYPE> void createSparseConnectivityFromDense(DATATYPE *wuvar, int preN, int postN, DATATYPE *tmp_gRNPN, SparseProjection *sparseStruct, bool runTest);


//---------------------------------------------------------------------
/*! \brief  Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 */
//---------------------------------------------------------------------

void createPosttoPreArray(unsigned int preN, unsigned int postN, SparseProjection *C);


//--------------------------------------------------------------------------
/*! \brief function to create the mapping from the normal index array "ind" to the "reverse" array revInd, i.e. the inverse mapping of remap. This is needed if SynapseDynamics accesses pre-synaptic variables.
 */
//--------------------------------------------------------------------------

void createPreIndices(unsigned int preN, unsigned int postN, SparseProjection *C);


#ifndef CPU_ONLY
// ------------------------------------------------------------------------
// initializing conductance arrays for sparse matrices

void initializeSparseArray(SparseProjection C,  unsigned int *dInd, unsigned int *dIndInG, unsigned int preN);

void initializeSparseArrayRev(SparseProjection C,  unsigned int *dRevInd, unsigned int *dRevIndInG, unsigned int *dRemap, unsigned int postN);

void initializeSparseArrayPreInd(SparseProjection C,  unsigned int *dPreInd);
#endif

// is this used anywhere? Suggest to remove (TN)
//!!!!!find var to check if a string is used in a code (atm it is used to create post-to-pre arrays)
void strsearch(string &s, const string trg);

#endif
