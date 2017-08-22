
#include "GeNNHelperKrnls.h"

/*********************************/       
//kernels for random number generation
/*********************************/

__global__ void setup_kernel (curandState *state, unsigned long seed, int sizeofResult)
{
    int id = threadIdx.x+blockIdx.x*BlkSz;	
    if (id < sizeofResult) curand_init(seed, id, 0, &state[id]);
}


/*********************************/
//function to setup the random number generator using the xorwow algorithm
/*********************************/

void xorwow_setup (curandState * devStates, long int sampleSize, long long int seed)
{
    int sampleBlkNo = ceilf(float(sampleSize/float(BlkSz)));
    dim3 sThreads(BlkSz,1);
    dim3 sGrid(sampleBlkNo,1); 

    setup_kernel <<<sGrid, sThreads>>> (devStates, seed, sampleSize);
}


/*********************************/

template <typename T>
__global__ void generate_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T Rstrength, T Rshift)
{
    int id = threadIdx.x+blockIdx.x* BlkSz; //TODO: use neuron kernel params
    T x;

    if (id < sizeofResult){
        curandState localstate = state[id];
        x = curand_normal(&localstate);
        result[id] = x*Rstrength+Rshift;
        state[id]=localstate;
    }
}

/*********************************/

template <typename T>
__global__ void generate_uniform_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T min, T range)
{
    int id = threadIdx.x+blockIdx.x* BlkSz; //TODO: use neuron kernel params
    T x;

    if (id < sizeofResult){
        curandState localstate = state[id];
        x = curand_uniform(&localstate);
        result[id] = min + (x * range);
        state[id]=localstate;
    }
}

template <typename T>
void generate_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T Rstrength, T Rshift, dim3 sGrid, dim3 sThreads)
{
    generate_random_gpuInput_xorwow <<<sGrid, sThreads>>> (state, result, sizeofResult, Rstrength, Rshift);
}

template
void generate_random_gpuInput_xorwow <float> (curandState *state, float *result, int sizeofResult, float Rstrength, float Rshift, dim3 sGrid, dim3 sThreads);

template
void generate_random_gpuInput_xorwow <double> (curandState *state, double *result, int sizeofResult, double Rstrength, double Rshift, dim3 sGrid, dim3 sThreads);


template <typename T>
void generate_uniform_random_gpuInput_xorwow (curandState *state, T *result, int sizeofResult, T min, T max, dim3 sGrid, dim3 sThreads)
{
    generate_uniform_random_gpuInput_xorwow <<<sGrid, sThreads>>> (state, result, sizeofResult, min, max - min);
}

template
void generate_uniform_random_gpuInput_xorwow <float> (curandState *state, float *result, int sizeofResult, float min, float max, dim3 sGrid, dim3 sThreads);

template
void generate_uniform_random_gpuInput_xorwow <double> (curandState *state, double *result, int sizeofResult, double min, double max, dim3 sGrid, dim3 sThreads);
