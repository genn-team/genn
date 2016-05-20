//-----------------------------------------------------------------------
/*!  \file sparseUtils.cc 
  
  \brief Contains functions related to setting up sparse connectivity.
*/
//--------------------------------------------------------------------------


#ifndef sparse_utils_cc
#define sparse_utils_cc

#include <cstdio>
#include <cmath>

//--------------------------------------------------------------------------
/*!
  \brief Utility to count how many entries above a specified value exist in a float array
*/
//--------------------------------------------------------------------------


template <class DATATYPE>
unsigned int countEntriesAbove(DATATYPE * Array, //!< Pointer to the all-to-all array
                               int sz, //!< Length of the all-to-all array
                               double includeAbove //!< Threshold for considering an array element as existing (in an all-to-all array non-existing connections are set to 0 or similarly low value)
)
{
	int count = 0;
	for (int i = 0; i < sz; i++) {
		if (abs(Array[i]) > includeAbove) count++;
	}
	fprintf(stdout, "\nCounted %u nonzero entries\n\n", count);
	return count;

}

//--------------------------------------------------------------------------
/*!
  \brief DEPRECATED Utility to get a synapse weight from a SPARSE structure by x,y coordinates
 NB: as the SparseProjection struct doesnt hold the preN size (it should!) it is not possible
 to check the parameter validity. This fn may therefore crash unless user knows max poss X
*/
//--------------------------------------------------------------------------


template <class DATATYPE>
DATATYPE getG(DATATYPE * wuvar, SparseProjection  * sparseStruct, int x, int y)
{
  fprintf(stderr,"WARNING: This function is deprecated, and if you are still using it \n\
  you are probably trying to use the old sparse structures containing the g array.  \n\
  Conductance structures have changed: conductance values should be defined as synapse variables now; \n\
  the structure is renamed as \"SparseProjection\" and contains only indexing arrays. \n\n\
  The replacement function for getG is \n\
  getSparseVar(DATATYPE * wuvar, SparseProjection  * sparseStruct, int x, int y).\n\n\
  calling getSparseVar...");
  getSparseVar(wuvar, &sparseStruct, x, y);
}
template <class DATATYPE>
float getSparseVar(DATATYPE * wuvar, SparseProjection  * sparseStruct, int x, int y)
{
  DATATYPE g = 0.0; //default return value implies zero weighted for x,y

	int startSynapse = sparseStruct->indInG[x];
	int endSynapse = sparseStruct->indInG[x+1];

	for (int syn = startSynapse; syn < endSynapse; syn++) {
		if (sparseStruct->ind[syn]==y) {//look for index y
			g = wuvar[syn]; //set the real g
			break; //stop looking
		}
	}
	return g;

}

//--------------------------------------------------------------------------
/*!
  \brief Function for setting the values of SPARSE connectivity matrix
*/
//--------------------------------------------------------------------------

template <class DATATYPE>
void setSparseConnectivityFromDense(DATATYPE * wuvar, //!< Pointer to the weightUpdateModel var array
                                    int preN, //!< Number of presynaptic neurons
                                    int postN, //!< Number of postsynaptic neurons
                                    DATATYPE * tmp_var, //!< Pointer to the all-to-all equivalent of theweightUpdateModel var array to be converted
                                    SparseProjection * sparseStruct //!< Structure that contains sparse connectivity
)
{
  int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.
  
	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_var[pre * postN + post];
			if (g > GENN_PREFERENCES::asGoodAsZero) {
				sparseStruct->ind[synapse] = post;
				wuvar[synapse] = g;
				synapse ++;
			}
		}
		sparseStruct->indInG[pre + 1] = synapse; //write start of next group
	}
}

//--------------------------------------------------------------------------
/*!
  \brief Utility to generate the SPARSE connectivity structure from a simple all-to-all array
*/
//--------------------------------------------------------------------------
template <class DATATYPE>
void createSparseConnectivityFromDense(DATATYPE * wuvar, //!< Pointer to the weightUpdateModel var array
                                       int preN, //!< Number of presynaptic neurons
                                       int postN, //!< Number of postsynaptic neurons
                                       DATATYPE * tmp_var, //!< Pointer to the all-to-all equivalent of theweightUpdateModel var array to be converted
                                       SparseProjection * sparseStruct, //!< Structure that contains sparse connectivity
                                       bool runTest //!< Flag to enable testing
) {

	
	sparseStruct->connN = countEntriesAbove(tmp_var, preN * postN, GENN_PREFERENCES::asGoodAsZero);

	int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.

	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_var[pre * postN + post];
			if (g > GENN_PREFERENCES::asGoodAsZero) {
				sparseStruct->ind[synapse] = post;
				wuvar[synapse] = g;
				synapse ++;
			}
		}
		sparseStruct->indInG[pre + 1] = synapse; //write start of next group
	}
	if (!runTest) return;

	//test correct
	int fails = 0;
	for (int test = 0; test < 10; ++test) {
		int randX = rand() % preN;
		int randY = rand() % postN;
		float denseResult = tmp_var[randX * postN + randY];
		float sparseResult = getG(wuvar, sparseStruct,randX,randY);
		if (abs(denseResult-sparseResult) > GENN_PREFERENCES::asGoodAsZero) fails++;
	}
	if (fails > 0 ) {
		fprintf(stderr, "ERROR: Sparse connectivity generator failed for %u out of 10 random checks.\n", fails);
		exit(1);
	}
	fprintf(stdout, "Sparse connectivity generator passed %u out of 10 random checks.\n", fails);

}



//---------------------------------------------------------------------
/*! \brief  Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 */
//---------------------------------------------------------------------

void createPosttoPreArray(unsigned int preN, //!< Number of presynaptic neurons
                      unsigned int postN, //!< Number of postsynaptic neurons
                      SparseProjection * C //!< Structure that contains sparse connectivity
)
{
  vector<vector<unsigned int> > tempvectInd(postN); // temporary vector to keep indices
  vector<vector<unsigned int> > tempvectV(postN); // temporary vector to keep connectivity values
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

void createPreIndices(unsigned int preN, //!< Number of presynaptic neurons
                      unsigned int postN, //!< Number of postsynaptic neurons
                      SparseProjection * C //!< Structure that contains sparse connectivity
) 
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

void initializeSparseArray(SparseProjection C, //!< Structure that contains sparse connectivity
                           unsigned int * dInd, //!< Pointer to the sparse Ind variable array on the GPU 
                           unsigned int * dIndInG, //!< Pointer to the sparse IndInG variable array on the GPU
                           unsigned int preN //!< Number of presynaptic neurons
)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
} 

 //--------------------------------------------------------------------------
/*! \brief Function for initializing reversed conductance array indices for sparse matrices on the GPU
(by copying the values from the host)
 */
//--------------------------------------------------------------------------
void initializeSparseArrayRev(SparseProjection C, //!< Structure that contains sparse connectivity
                              unsigned int * dRevInd, //!< Pointer to the sparse RevInd variable array on the GPU 
                              unsigned int * dRevIndInG, //!< Pointer to the sparse RevndInG variable array on the GPU
                              unsigned int * dRemap,  //!< Pointer to the sparse remap variable array on the GPU
                              unsigned int postN //!< Number of postsynaptic neurons
)
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
void initializeSparseArrayPreInd(SparseProjection C, //!< Structure that contains sparse connectivity
                                 unsigned int * dPreInd //!< Pointer to the sparse PreInd variable array on the GPU
)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dPreInd, C.preInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

#endif

#endif
