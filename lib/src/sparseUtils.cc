#include "sparseUtils.h"

// Standard C++ includes
#include <numeric>
#include <vector>

// Standard C includes
#include <cassert>

// GeNN includes
#include "utils.h"


//---------------------------------------------------------------------
/*! \brief  Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 */
//---------------------------------------------------------------------
void createPosttoPreArray(unsigned int preN, unsigned int postN, SparseProjection * C)
{
    // Zero reverse lookup indices
    std::fill_n(C->revIndInG, postN + 1, 0);

    // First calculate column lengths in revIndInG
    for (unsigned int i = 0; i < preN; i++){ //i : index of presynaptic neuron
        for(unsigned int j = C->indInG[i]; j < C->indInG[i + 1]; j++) { //for every postsynaptic neuron j
            C->revIndInG[C->ind[j] + 1]++;
        }
    }

    // Compute the partial sum so revIndInG is now correctly initialised
    std::partial_sum(&C->revIndInG[1], &C->revIndInG[postN + 1], &C->revIndInG[1]);
    assert(C->revIndInG[postN] == C->connN);

    // Create vector to count connections made to each postsynaptic neuron
    std::vector<unsigned int> postCount(postN);

    // Loop through presynaptic neurons
    for (unsigned int i = 0; i < preN; i++) {
        // Loop through synapses in corresponding matrix row
        for(unsigned int s = C->indInG[i]; s < C->indInG[i + 1]; s++) {
            // Get index of postsynaptic target for synapse
            const unsigned int postIndex = C->ind[s];

            // Get synapse index of connection
            const unsigned int synIndex = C->revIndInG[postIndex] + postCount[postIndex];
            assert(synIndex < C->revIndInG[postIndex + 1]);

            // Store index of presynaptic reverse target and synapse in appropriate arrays
            C->revInd[synIndex] = i;
            C->remap[synIndex] = s;

            // Update connection count
            postCount[postIndex]++;
        }
    }
}

//--------------------------------------------------------------------------
/*! \brief Function to create the mapping from the normal index array "ind" to the "reverse" array revInd, i.e. the inverse mapping of remap. 
This is needed if SynapseDynamics accesses pre-synaptic variables.
 */
//--------------------------------------------------------------------------

void createPreIndices(unsigned int preN, unsigned int, SparseProjection * C)
{
    // let's not assume anything and create from the minimum available data, i.e. indInG and ind
    for (unsigned int i = 0; i< preN; i++){ //i : index of presynaptic neuron
        for (unsigned int j = 0; j < (C->indInG[i+1]-C->indInG[i]); j++){ //for every postsynaptic neuron j
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

void initializeSparseArray(const SparseProjection &C,  unsigned int * dInd, unsigned int * dIndInG, unsigned int preN)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
} 


//--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------

void initializeSparseArrayRev(const SparseProjection &C,  unsigned int * dRevInd, unsigned int * dRevIndInG, unsigned int * dRemap, unsigned int postN)
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

void initializeSparseArrayPreInd(const SparseProjection &C,  unsigned int * dPreInd)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dPreInd, C.preInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}
#endif  // CPU_ONLY
