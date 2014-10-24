#include <curand_kernel.h>

#define BlkSz 256 //NOTE: This was the best value for our machine even though it supports up to 1204 threads per block. You may try to change it to another value if it works better.

/*********************************/       
//kernels for random number generation
/*********************************/
__global__ void setup_kernel(curandState *state, unsigned long seed, int sizeofResult){
	int id = threadIdx.x+blockIdx.x*BlkSz;	
	if (id < sizeofResult) curand_init(seed, id, 0, &state[id]);
}

/*********************************/
__global__ void generate_random_gpuInput_xorwow(curandState * state, float * result, int sizeofResult, float Rstrength, float Rshift)
{
	int id = threadIdx.x+blockIdx.x* BlkSz; //TODO: use neuron kernel params
	float x;
	
	if (id < sizeofResult){
		curandState localstate = state[id];
		x =curand_normal(&localstate); 
		result[id] = x*Rstrength+Rshift;
		state[id]=localstate;
	}
}

/*********************************/
//function to setup the random number generator using the xorwow algorithm
/*********************************/
void xorwow_setup(curandState * devStates, long int sampleSize){
    int sampleBlkNo = ceil(float(sampleSize/float(BlkSz)));
    dim3 sThreads(BlkSz,1);
    dim3 sGrid(sampleBlkNo,1); 

		long long int seed = 117; 
		setup_kernel<<<sGrid,sThreads>>>(devStates, seed, sampleSize);
}
