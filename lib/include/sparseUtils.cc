#ifndef sparse_utils_cc
#define sparse_utils_cc

#include <cstdio>
#include <cmath>

/*---------------------------------------------------------------------
 Utility to count how many entries above a specified value exist in a float array
 ---------------------------------------------------------------------*/
template <class DATATYPE>
unsigned int countEntriesAbove(DATATYPE * Array, int sz, DATATYPE includeAbove)
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
 NB: as the Conductance struct doesnt hold the preN size (it should!) it is not possible
 to check the parameter validity. This fn may therefore crash unless user knows max poss X
 ---------------------------------------------------------------------*/
template <class DATATYPE>
DATATYPE getG(DATATYPE * wuvar, Conductance  * sparseStruct, int x, int y)
{
  fprintf(stderr,"WARNING: This function is deprecated, and if you are still using it \n\
  you are probably trying to use the old sparse structures containing g array.  \n\
  Conductance structures have changed: conductance values should be defined as synapse variable now; \n\
  the structure contains only indexing arrays. \n\n\
  Replacement function for getG is \n\
  getSparseVar(DATATYPE * wuvar, Conductance  * sparseStruct, int x, int y).\n\n\
  calling getSparseVar...");
  getSparseVar(wuvar, &sparseStruct, x, y);
}
template <class DATATYPE>
float getSparseVar(DATATYPE * wuvar, Conductance  * sparseStruct, int x, int y)
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
void setSparseConnectivityFromDense(DATATYPE * wuvar, int preN,int postN,DATATYPE * tmp_gRNPN, Conductance * sparseStruct){
  int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.
  float asGoodAsZero = 0.0001f;//as far as we are concerned. Remember floating point errors.
	
	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_gRNPN[pre * postN + post];
			if (g > asGoodAsZero) {
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
void createSparseConnectivityFromDense(DATATYPE * wuvar, int preN,int postN,DATATYPE * tmp_gRNPN, Conductance * sparseStruct, bool runTest) {

	
	float asGoodAsZero = 0.0001f;//as far as we are concerned. Remember floating point errors.
	sparseStruct->connN = countEntriesAbove(tmp_gRNPN, preN * postN, asGoodAsZero);
	//sorry -- this is not functional anymore 
	//allocateSparseArray(sparseStruct, sparseStruct.connN, preN, false);

	int synapse = 0;
	sparseStruct->indInG[0] = 0; //first neuron always gets first synapse listed.

	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			DATATYPE g = tmp_gRNPN[pre * postN + post];
			if (g > asGoodAsZero) {
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
		if (abs(denseResult-sparseResult) > asGoodAsZero) fails++;
	}
	if (fails > 0 ) {
		fprintf(stderr, "ERROR: Sparse connectivity generator failed for %u out of 10 random checks.\n", fails);
		exit(1);
	}
	fprintf(stdout, "Sparse connectivity generator passed %u out of 10 random checks.\n", fails);

}



/*---------------------------------------------------------------------
 Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 ---------------------------------------------------------------------*/
void createPosttoPreArray(unsigned int preN, unsigned int postN, Conductance * sparseStruct) {
  vector<vector<unsigned int> > tempvectInd(postN); //temporary vector to keep indices
  vector<vector<unsigned int> > tempvectV(postN); //temporary vector to keep conductance values
	unsigned int glbcounter = 0;

	for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
		for (int j = 0; j < (sparseStruct->indInG[i+1]-sparseStruct->indInG[i]); j++){ //for every postsynaptic neuron c
			tempvectInd[sparseStruct->ind[sparseStruct->indInG[i]+j]].push_back(i); //sparseStruct->ind[sparseStruct->indInG[i]+j]: index of postsynaptic neuron
			//old way : tempvectG[sparseStruct->ind[sparseStruct->indInG[i]+j]].push_back(sparseStruct->gp[sparseStruct->indInG[i]+j]);
			tempvectV[sparseStruct->ind[sparseStruct->indInG[i]+j]].push_back(sparseStruct->indInG[i]+j); //this should give where we can find the value in the array
			glbcounter++;
      //fprintf(stdout,"i:%d j:%d val pushed to G is:%f , sparseStruct->gIndInG[i]=%d\n", i, j, sparseStruct->gp[sparseStruct->gIndInG[i]+j],sparseStruct->gIndInG[i]);
		}
	}
	//which one makes more s?ense - probably the one on top
  //float posttoprearray = new float[glbcounter];
	unsigned int lcounter =0;

	sparseStruct->revIndInG[0]=0;
	for (int k = 0; k < postN; k++){
		sparseStruct->revIndInG[k+1]=sparseStruct->revIndInG[k]+tempvectInd[k].size();
		for (int p = 0; p< tempvectInd[k].size(); p++){ //if k=0?
			sparseStruct->revInd[lcounter]=tempvectInd[k][p];
			sparseStruct->remap[lcounter]=tempvectV[k][p];
			lcounter++;
		}
	}
}



//!!!!!find var to check if a string is used in a code (atm it is used to create post-to-pre arrays)
void strsearch(string &s, const string trg)
{
  size_t found= s.find(trg);
  if (found != string::npos) {
    //createPosttoPreArray(var)...
  }
}
#endif
