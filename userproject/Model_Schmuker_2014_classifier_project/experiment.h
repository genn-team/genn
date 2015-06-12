/*--------------------------------------------------------------------------
   Author: Alan Diamond
  
   initial version: 2014-Mar-05
  
--------------------------------------------------------------------------

Header file containing global variables, constants and macros used in running the Schmuker_2014_classifier.

-------------------------------------------------------------------------- */

#ifdef _WIN32
#define divi "\\"
#else
#define divi "/"
#endif

typedef unsigned int UINT;

//pseudo random generator used in synapseFnct.cc and synapseKrnl.cc.
#define D_MAX_RANDOM_NUM 32767
//#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & D_MAX_RANDOM_NUM

/*--------------------------------------------------------------------------
//Experiment constants
 --------------------------------------------------------------------------*/
#define RECORDINGS_DIR "recordings_iris_data"
#define CACHE_DIR "cached_iris_data"
#define OUTPUT_DIR "output_iris"
#define VR_DATA_FILENAME "VR-recordings-iris.data"
#define DATASET_NAME "Iris"

#define TOTAL_RECORDINGS 150 //How many sets of sensor readings will be presented
#define N_FOLDING 5 //e.g. five fold folding of training vs test data  -> 80:20 split
#define RECORDING_TIME_MS  1000 //How long is each recording in ms
#define REPEAT_LEARNING_SET 2 //Repeat the learning set N times, for more exposure to early observations

//learning parameters
#define SPIKING_ACTIVITY_THRESHOLD_HZ 5 // if rate greater than threshold (set to 35 sp/s in the paper) then we can expect this neuron response to its VR contributed to winning class cluster

//switchable flags
//#define FLAG_GENERATE_RASTER_PLOT 1
//#define DEVICE_MEM_ALLOCATED_ON_DEVICE 1 //set this for GeNN 1.0 compatibility
#define FLAG_RUN_ON_CPU 1


/*--------------------------------------------------------------------------
Data and extra network dimensions (see also Model file)
 --------------------------------------------------------------------------*/

//this range is used to scale the input data into firing rates (Hz)
#define MAX_FIRING_RATE_HZ 70 //Hz
#define MIN_FIRING_RATE_HZ 20 //Hz


// weight and connectivity prob between an RN cluster and corresponding PN cluster
#define GLOBAL_WEIGHT_SCALING  1.0 //0.375 //Use this to tune absolute weight levels to neuron models. Relative weights are taken from Schmuker paper
#define WEIGHT_RN_PN  0.5
#define CONNECTIVITY_RN_PN 0.5

//#define WEIGHT_WTA_PN_PN   0.7 * 0.133 // paper uses an intermediate LN population PN-LN 0.7, LN-PN 0.133
//#define WEIGHT_WTA_AN_AN   0.5 * 1.0 // paper uses an intermediate adjoint inhibitory population PN-AI 0.5, AI-PN 1.0
#define WEIGHT_WTA_PN_PN 0.01
#define WEIGHT_WTA_AN_AN 0.01


#define CONNECTIVITY_PN_PN 0.5 //  connectivity between a PN cluster and neurons in other clusters
#define CONNECTIVITY_AN_AN 0.5 //  connectivity between an AN cluster and neurons in other clusters

// weight and connectivity prob between PN  and AN populations
#define CONNECTIVITY_PN_AN 0.5 //50% connectivity between PN and AN
#define MIN_WEIGHT_PN_AN 0.1 //0.2
#define MAX_WEIGHT_PN_AN 0.4 //0.66
#define WEIGHT_DELTA_PN_AN 0.04 //increment/decrement of plastic synapse weights
#define PLASTICITY_INTERVAL_MS 330 //how often to update the plastic synapse weights

using namespace std;
#include <cassert>
#include "hr_time.cpp"
#include "utils.h" // for CHECK_CUDA_ERRORS
//#include <cuda_runtime.h>

/*--------------------------------------------------------------------------
global variables
--------------------------------------------------------------------------*/

CStopWatch timer;


//--------------------------------------------------------------------------
#include "Schmuker2014_classifier.cu"
