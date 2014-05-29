#ifndef sparse_utils_cc
#define sparse_utils_cc

#include <stdio.h>
#include <math.h>

/*---------------------------------------------------------------------
 Utility to count how many entries above a specified value exist in a float array
 ---------------------------------------------------------------------*/
int  countEntriesAbove(float * floatArray, int sz, float includeAbove)
{
	int count = 0;
	for (int i = 0; i < sz; i++) {
		if (floatArray[i] > includeAbove) count++;
	}
	return count;

}

/*---------------------------------------------------------------------
 Utility to get a synapse weight from a SPARSE structure by x,y coordinates
 NB: as the Conductance struct doesnt hold the preN size (it should!) it is not possible
 to check the parameter validity. This fn may therefore crash unless user knows max poss X
 ---------------------------------------------------------------------*/
float getG(Conductance  * sparseStruct, int x, int y)
{
	float g = 0.0f; //default return value implies zero weighted for x,y

	int startSynapse = sparseStruct->gIndInG[x];
	int endSynapse = sparseStruct->gIndInG[x+1];

	for (int syn = startSynapse; syn < endSynapse; syn++) {
		if (sparseStruct->gInd[syn]==y) {//look for index y
			g = sparseStruct->gp[syn]; //set the real g
			break; //stop looking
		}
	}
	return g;

}

/*---------------------------------------------------------------------
 Utility to generate the SPARSE connectivity structure from a simple all-to-all array
 ---------------------------------------------------------------------*/
void createSparseConnectivityFromDense(int preN,int postN,float * tmp_gRNPN, Conductance  * sparseStruct, bool runTest) {

	float asGoodAsZero = 0.0001f;//as far as we are concerned. Remember floating point errors.
	sparseStruct->connN = countEntriesAbove(tmp_gRNPN, preN * postN, asGoodAsZero);
	allocateSparseArray(sparseStruct, preN, false);

	int synapse = 0;
	sparseStruct->gIndInG[0] = 0; //first neuron always gets first synapse listed.

	for (int pre = 0; pre < preN; ++pre) {
		for (int post = 0; post < postN; ++post) {
			float g = tmp_gRNPN[pre * postN + post];
			if (g > asGoodAsZero) {
				sparseStruct->gInd[synapse] = post;
				sparseStruct->gp[synapse] = g;
				synapse ++;
			}
		}
		sparseStruct->gIndInG[pre + 1] = synapse; //write start of next group
	}
	if (!runTest) return;

	//test correct
	int fails = 0;
	for (int test = 0; test < 10; ++test) {
		int randX = rand() % preN;
		int randY = rand() % postN;
		float denseResult = tmp_gRNPN[randX * postN + randY];
		float sparseResult = getG(sparseStruct,randX,randY);
		if (abs(denseResult-sparseResult) > asGoodAsZero) fails++;
	}
	if (fails > 0 ) {
		fprintf(stderr, "ERROR: Sparse connectivity generator failed for %u out of 10 random checks.\n", fails);
		exit(1);
	}
	fprintf(stdout, "Sparse connectivity generator passed %u out of 10 random checks.\n", fails);

}


#endif
