%module(package="genn_wrapper") FixedNumberTotalPreCalc
%{
#define SWIG_FILE_WITH_INIT // for numpy
#include "fixedNumberTotalPreCalc.h"
%}
%rename("%(undercase)s", %$isfunction, notregexmatch$name="add[a-zA-Z]*Population", notregexmatch$name="addCurrentSource", notregexmatch$name="assignExternalPointer[a-zA-Z]*") "";
%include "numpy.i" 

%init %{
import_array();
%}

// Expose C++ functions with unsigned int **subRowLengths, int *numSubRowLengths parameters in their signatures 
// i.e. preCalcRowLengths below into python functions which return a numpy array (of size specified by the function itself)
%apply (unsigned int** ARGOUTVIEW_ARRAY1, int *DIM1) {(unsigned int **subRowLengths, int *numSubRowLengths)}


%inline %{

void preCalcRowLengths(unsigned int numPre, unsigned int numPost, size_t numConnections,
                       unsigned int **subRowLengths, int *numSubRowLengths, 
                       std::mt19937 &rng, unsigned int numThreadsPerSpike = 1)
{
    *subRowLengths = new unsigned int[numPre * numThreadsPerSpike];
    preCalcRowLengths(numPre, numPost, numConnections, *subRowLengths, rng, numThreadsPerSpike);
    *numSubRowLengths = numPre * numThreadsPerSpike;
}

std::mt19937 createMT19937(unsigned int seed)
{
    // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
    std::seed_seq seeds{seed};
    
    std::mt19937 rng;
    rng.seed(seeds);
    return rng;
}

std::mt19937 createMT19937()
{
    uint32_t seedData[std::mt19937::state_size];
    std::random_device seedSource;
    for(int i = 0; i < std::mt19937::state_size; i++) {
        seedData[i] = seedSource();
    }
    std::seed_seq seeds(std::begin(seedData), std::end(seedData));
    
    std::mt19937 rng;
    rng.seed(seeds);
    return rng;
}
%}
