
#ifndef SPARSEUTILS_CC
#define SPARSEUTILS_CC

#include "sparseUtils.h"
#include "utils.h"

#include <vector>


//---------------------------------------------------------------------
/*! \brief  Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 */
//---------------------------------------------------------------------

void createPosttoPreArray(unsigned int preN, unsigned int postN, SparseProjection * C) {
    vector<vector<unsigned int> > tempvectInd(postN); //temporary vector to keep indices
    vector<vector<unsigned int> > tempvectV(postN); //temporary vector to keep connectivity values
    unsigned int glbcounter = 0;
    
    for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
	for (int j = 0; j < (C->indInG[i+1]-C->indInG[i]); j++){ //for every postsynaptic neuron j
	    tempvectInd[C->ind[C->indInG[i]+j]].push_back(i); //C->ind[C->indInG[i]+j]: index of postsynaptic neuron
	    tempvectV[C->ind[C->indInG[i]+j]].push_back(C->indInG[i]+j); //this should give where we can find the value in the array
	    glbcounter++;
	}
    }
    unsigned int lcounter =0;

    C->revIndInG[0]=0;
    for (int k = 0; k < postN; k++){
	C->revIndInG[k+1]=C->revIndInG[k]+tempvectInd[k].size();
	for (int p = 0; p< tempvectInd[k].size(); p++){ //if k=0?
	    C->revInd[lcounter]=tempvectInd[k][p];
	    C->remap[lcounter]=tempvectV[k][p];
	    lcounter++;
	}
    }
}


//--------------------------------------------------------------------------
/*! \brief Function to create the mapping from the normal index array "ind" to the "reverse" array revInd, i.e. the inverse mapping of remap. 
This is needed if SynapseDynamics accesses pre-synaptic variables.
 */
//--------------------------------------------------------------------------

void createPreIndices(unsigned int preN, unsigned int postN, SparseProjection * C) 
{
    // let's not assume anything and create from the minimum available data, i.e. indInG and ind
    vector<vector<unsigned int> > tempvect(postN); //temporary vector to keep indices
    for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
	for (int j = 0; j < (C->indInG[i+1]-C->indInG[i]); j++){ //for every postsynaptic neuron j
	    C->preInd[C->indInG[i]+j]= i; // simmple array of the presynaptic neuron index of each synapse
	}
    }
}


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Function for initializing conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------

void initializeSparseArray(SparseProjection C,  unsigned int * dInd, unsigned int * dIndInG, unsigned int preN)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
} 


//--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------

void initializeSparseArrayRev(SparseProjection C,  unsigned int * dRevInd, unsigned int * dRevIndInG, unsigned int * dRemap, unsigned int postN)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dRevInd, C.revInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dRevIndInG, C.revIndInG, (postN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dRemap, C.remap, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}


//--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance arrays presynaptic indices for sparse matrices on  the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------

void initializeSparseArrayPreInd(SparseProjection C,  unsigned int * dPreInd)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dPreInd, C.preInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}
#endif

#endif // SPARSEUTILS_CC
