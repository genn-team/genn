
#include <curand_kernel.h>

const int BlkSz = 256; //NOTE: This was the best value for our machine even though it supports up to 1204 threads per block. You may try to change it to another value if it works better.

/*********************************/       
//kernels for random number generation
/*********************************/

__global__ void setup_kernel (curandState *state, unsigned long seed, int sizeofResult);


/*********************************/
//function to setup the random number generator using the xorwow algorithm
/*********************************/

void xorwow_setup (curandState *devStates, long int sampleSize, long long int seed);


/*********************************/

template <typename T>
__global__ void generate_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T Rstrength, T Rshift);


/*********************************/

template <typename T>
void generate_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T Rstrength, T Rshift, dim3 sGrid, dim3 sThreads);
