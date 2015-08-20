#ifndef sparse_utils_cc
#define sparse_utils_cc

#include <cstdio>
#include <cmath>

/*---------------------------------------------------------------------
 Utility to count how many entries above a specified value exist in a float array
 ---------------------------------------------------------------------*/
template <class DATATYPE>
unsigned int countEntriesAbove(DATATYPE * Array, int sz, double includeAbove)
{
	int count = 0;
	for (int i = 0; i < sz; i++) {
		if (abs(Array[i]) > includeAbove) count++;
	}
	fprintf(stdout, "\nCounted %u nonzero entries\n\n", count);
	return count;

}

/*---------------------------------------------------------------------
 Utility to get a synapse weight from a SPARSE structure by x,y coordinates
 NB: as the SparseProjection struct doesnt hold the preN size (it should!) it is not possible
 to check the parameter validity. This fn may therefore crash unless user knows max poss X
 ---------------------------------------------------------------------*/
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

/*---------------------------------------------------------------------
Setting the values of SPARSE connectivity matrix
----------------------------------------------------------------------*/
template <class DATATYPE>
void setSparseConnectivityFromDense(DATATYPE * wuvar, int preN,int postN,DATATYPE * tmp_gRNPN, SparseProjection * sparseStruct){
  int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.
  
	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_gRNPN[pre * postN + post];
			if (g > GENN_PREFERENCES::asGoodAsZero) {
				sparseStruct->ind[synapse] = post;
				wuvar[synapse] = g;
				synapse ++;
			}
		}
		sparseStruct->indInG[pre + 1] = synapse; //write start of next group
	}
}

/*---------------------------------------------------------------------
 Utility to generate the SPARSE connectivity structure from a simple all-to-all array
 ---------------------------------------------------------------------*/
template <class DATATYPE>
void createSparseConnectivityFromDense(DATATYPE * wuvar, int preN,int postN,DATATYPE * tmp_gRNPN, SparseProjection * sparseStruct, bool runTest) {

	
	sparseStruct->connN = countEntriesAbove(tmp_gRNPN, preN * postN, GENN_PREFERENCES::asGoodAsZero);
	//sorry -- this is not functional anymore 
	//allocateSparseArray(sparseStruct, sparseStruct.connN, preN, false);

	int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.

	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_gRNPN[pre * postN + post];
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
		float denseResult = tmp_gRNPN[randX * postN + randY];
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
/*! \brief function to create the mapping from the normal index array "ind" to the "reverse" array revInd, i.e. the inverse mapping of remap. This is needed if SynapseDynamics accesses pre-synaptic variables.
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
    // ------------------------------------------------------------------------
    // initializing conductance arrays for sparse matrices

void initializeSparseArray(SparseProjection C,  unsigned int * dInd, unsigned int * dIndInG, unsigned int preN)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dInd, C.ind, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dIndInG, C.indInG, (preN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
} 

void initializeSparseArrayRev(SparseProjection C,  unsigned int * dRevInd, unsigned int * dRevIndInG, unsigned int * dRemap, unsigned int postN)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dRevInd, C.revInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dRevIndInG, C.revIndInG, (postN+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(dRemap, C.remap, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void initializeSparseArrayPreInd(SparseProjection C,  unsigned int * dPreInd)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dPreInd, C.preInd, C.connN*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

#endif

// is this used anywhere? Suggest to remove (TN)
//!!!!!find var to check if a string is used in a code (atm it is used to create post-to-pre arrays)
void strsearch(string &s, const string trg)
{
  size_t found= s.find(trg);
  if (found != string::npos) {
    //createPosttoPreArray(var)...
  }
}
#endif
