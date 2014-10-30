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
Setting the values of SPARSE connectivity matrix
----------------------------------------------------------------------*/
void setSparseConnectivityFromDense(int preN,int postN,float * tmp_gRNPN, Conductance * sparseStruct){
  int synapse = 0;
	sparseStruct->gIndInG[0] = 0; //first neuron always gets first synapse listed.
  float asGoodAsZero = 0.0001f;//as far as we are concerned. Remember floating point errors.
	
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
}

/*---------------------------------------------------------------------
 Utility to generate the SPARSE connectivity structure from a simple all-to-all array
 ---------------------------------------------------------------------*/
void createSparseConnectivityFromDense(int preN,int postN,float * tmp_gRNPN, Conductance * sparseStruct, bool runTest) {

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



/*---------------------------------------------------------------------
 Utility to generate the SPARSE array structure with post-to-pre arrangement from the original pre-to-post arrangement where postsynaptic feedback is necessary (learning etc)
 ---------------------------------------------------------------------*/
void createPosttoPreArray(int preN,int postN, Conductance * sparseStruct, Conductance * sparseStructPost) {
  float * posttoprearray = new float[sparseStruct->connN];
  vector<vector<unsigned int> > tempvectInd(postN); //temporary vector to keep indices
  vector<vector<float> > tempvectG(postN); //temporary vector to keep conductance values
	unsigned int glbcounter = 0;

	sparseStructPost->connN=sparseStruct->connN;
	for (int i = 0; i< preN; i++){ //i : index of presynaptic neuron
		for (int j = 0; j < (sparseStruct->gIndInG[i+1]-sparseStruct->gIndInG[i]); j++){ //for every postsynaptic neuron c
			tempvectInd[sparseStruct->gInd[sparseStruct->gIndInG[i]+j]].push_back(i); //sparseStruct->gInd[sparseStruct->gIndInG[i]+j]: index of postsynaptic neuron
			tempvectG[sparseStruct->gInd[sparseStruct->gIndInG[i]+j]].push_back(sparseStruct->gp[sparseStruct->gIndInG[i]+j]);
			glbcounter++;
      //fprintf(stdout,"i:%d j:%d val pushed to G is:%f , sparseStruct->gIndInG[i]=%d\n", i, j, sparseStruct->gp[sparseStruct->gIndInG[i]+j],sparseStruct->gIndInG[i]);
		}
	}
	fprintf(stdout,"that's it");
  //which one makes more s?ense - probably the one on top
  //float posttoprearray = new float[glbcounter];
	unsigned int lcounter =0;

	sparseStructPost->gIndInG[0]=0;
	for (int k = 0; k < postN; k++){
		sparseStructPost->gIndInG[k+1]=sparseStructPost->gIndInG[k]+tempvectInd[k].size();
		for (int p = 0; p< tempvectInd[k].size(); p++){ //if k=0?
			sparseStructPost->gInd[lcounter]=tempvectInd[k][p];
			sparseStructPost->gp[lcounter]=tempvectG[k][p];
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
