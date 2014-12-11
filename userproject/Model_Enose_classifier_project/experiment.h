/*--------------------------------------------------------------------------
   Author: Alan Diamond
  
   initial version: 2014-Mar-05
  
--------------------------------------------------------------------------

Header file containing global variables, constants and macros used in running the Schmuker_2014_classifier.

-------------------------------------------------------------------------- */

//pseudo random generator used in synapseFnct.cc and synapseKrnl.cc.
#define D_MAX_RANDOM_NUM 32767
//#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & D_MAX_RANDOM_NUM

/*--------------------------------------------------------------------------
//Experiment constants
 --------------------------------------------------------------------------*/
#define PYTHON_DIR "~/genn/userproject/Model_Enose_classifier_project/python" //where to find python scripts used by experiment
#define PYTHON_RUNTIME "python" // the runtime engine to invoke .py files.

#define DATASET_NAME "Enose"
#define RECORDINGS_DIR "recordings_enose_data/2010-high/fullData_clean/SubSample%ss%"

#define FILENAME_DATA_DESCRIPTION  "DataDescription.txt"

#define RECORDING_FILENAME_TEMPLATE "Enose SensorRecording%i_SubSample%ss%_x100.csv"
#define VR_DATA_FILENAME_TEMPLATE "tmpTrainingDataVRSet.csv"
#define CREATE_VR_FROM_EACH_LEARNING_SET 1
#define NUM_EPOCHS 100 //indicates how many re-exposures were used when invoking the clustering algorithm (neural gas)
#define VR_DIR "VRData" //subdir below recordings dir where VR files now stored


//std name for file holding all the discrete samples or subsamples in the dataset
//This file is primarly used to locate max/imn/avg distances betweeen samples
#define FILENAME_ALL_DISTINCT_SAMPLES "AllDistinctSamples.csv"

#define FILENAME_CLASS_LABELLING "ClassLabelling.csv"

#define CACHE_DIR "enose_cached_data" //location for cached input data after transformation to set of VR responses and then rate code
#define OUTPUT_DIR "output"

//Control of statisitics gathering / scoring
#define N_FOLDING 10  //e.g. 10-fold split of training vs test data  -> 90:10 split
#define USE_FOLDINGS N_FOLDING //set this to less than N_FOLDING to move on quicker. e.g. get quick results with just 2 foldings

#define N_REPEAT_TRIAL_OF_PARAMETER 1 //Repeat each crossvalidation of a parameter how many times . e.g. N=50. performance variation (Std deviation) can be high for low no of repeats

#define TOTAL_RECORDINGS 200 //How many sets of sensor readings will be presented
#define RECORDING_TIME_MS 299.5f //How long is each recording in ms

//learning parameters
#define SPIKING_ACTIVITY_THRESHOLD_HZ 10 // if rate greater than threshold (set to 35 sp/s in the paper) then we assume this PN neuron's response to the input contributed to winning class cluster

//Generating and displaying data during a run

//#define WRITE_VR_ACTIVATION_RESULTS 1 //write files totting up VR activation
//#define FLAG_GENERATE_VR_ACTIVATION_DATA_PER_RECORDING 1 //write a file per recording showing VR response in RN and PN layers

//#define DEVICE_MEM_ALLOCATED_ON_DEVICE 1 //set this for GeNN 1.0 compatibility

//#define FLAG_OUTPUT_INDIVIDUAL_RESULTS 1 //write a file giving the result details of every classification decision
//#define OUTPUT_MISTAKES_ONLY 1 //for the above file , include all results or only mistakes

//#define FLAG_OVERRIDE_RECORDING_SELECTED 1 // query user to select a specific recording

//#define FLAG_GENERATE_RASTER_PLOT_DURING_TRAINING_REPEAT 0 //specifiy rpt in which to create rasters. not defined = none. Use 0 for immedate rater plots. Use REPEAT_LEARNING_SET-1 for plots in last stage of training
#define FLAG_GENERATE_RASTER_PLOT_DURING_TESTING 1

#define RASTER_FREQ 1 //generate a raster plot every N recordings. For Enose data, N = 10 produces one plot per class
#define DISPLAY_RASTER_IMMEDIATELY 1

#define RASTER_PLOT_EXTRA "ACT" //As well as raster plot also show either ACTivation bars or HEATmap
//#define RASTER_PLOT_EXTRA "HEAT"

//switch between normal test stage (same data+duration for training and test stage)
// and "continuous realtime input" test stage (switches to delayed full length data for test stage only)
//#define USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE 1
/*--------------------------------------------------------------------------
Settings related to continous data mode (set by USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE)
 --------------------------------------------------------------------------*/
#define LIMITED_DURATION_FOR_FULL_LENGTH_RECORDINGS 40 //don't use whole recording , main interest is up until all samples points are passed
//#define LIMITED_DURATION_FOR_FULL_LENGTH_RECORDINGS RECORDING_TIME_MS
#define DELAYED_RECORDING_FILENAME_TEMPLATE "Enose SensorRecording%i_Delayed%ss%_x100.csv" //what are the delayed recording named
#define DELAYED_RECORDING_SUBDIR "../Recordings Delayed%ss%" // and where to find them (relative to current recordinbgs dir)
//#define INFER_CLASS_EVALUATION_WINDOW_FROM_AN 1 //if this not set then use PN layer activity to locate classification activity
#define CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN 20//Hz. In PN, what avg spike rate threshold in a class cluster switches the class evaluation mechansim on/off
#define CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN 30 //As above, for AN
#define MIN_CLASS_EVALUATION_TIMESTEPS 10 //the minimum time window used to evalute the class. Each timestep is 0.5sec
//#define FLAG_GENERATE_SPIKE_FREQ_DATA_FILE 1//create a file per recording showing how the max spiking rate per cluster changes over time. This file is useful to set threhold values for class evaluation window

/*--------------------------------------------------------------------------
Data and extra network dimensions (see also Model file)
 --------------------------------------------------------------------------*/

//this range is used to scale the input data into firing rates (Hz)
#define MAX_FIRING_RATE_HZ 70 //Hz
#define MIN_FIRING_RATE_HZ 00 //Hz

// weight and connectivity prob between an RN cluster and corresponding PN cluster
#define GLOBAL_WEIGHT_SCALING  1.0 //Use this to tune absolute weight levels to neuron models. Relative weights are taken from Schmuker paper
#define WEIGHT_RN_PN  0.5 //NB THis is not parameterised as tricky to alter on the fly if using SPARSE conectivity encoding. Optimum value has been ascertained using DENSE connectivity
#define CONNECTIVITY_RN_PN 0.5

//#define WEIGHT_WTA_PN_PN   0.7 * 0.133 // paper uses an intermediate LN population PN-LN 0.7, LN-PN 0.133
//#define WEIGHT_WTA_AN_AN   0.5 * 1.0 // paper uses an intermediate adjoint inhibitory population PN-AI 0.5, AI-PN 1.0
#define WEIGHT_WTA_PN_PN 0.0075
//#define WEIGHT_WTA_PN_PN 0.003
#define WEIGHT_WTA_AN_AN 0.01

#define CONNECTIVITY_PN_PN 0.5 //  connectivity between a PN cluster and neurons in other clusters
#define CONNECTIVITY_AN_AN 0.5 //  connectivity between an AN cluster and neurons in other clusters

//Plasticity and Learning settings
#define REPEAT_LEARNING_SET 4 //Repeat the learning set N times, for more exposure to early observations
#define PLASTICITY_INTERVAL_MS (RECORDING_TIME_MS/8.0) //how often to update the plastic synapse weights
// weight and connectivity prob between PN  and AN populations
#define CONNECTIVITY_PN_AN 0.5 //50% connectivity between PN and AN
#define MIN_WEIGHT_PN_AN 0.1 //0.2
#define MAX_WEIGHT_PN_AN 0.4 //0.66
#define WEIGHT_INC_DELTA_PN_AN 0.06 //increment of plastic synapse weights
#define WEIGHT_DEC_MULTIPLIER_PN_AN 0.5 //decrement plastic synapse weights by a proportion of INC param
#define START_WEIGHT_MAX_PN_AN MAX_WEIGHT_PN_AN //start weights are random, this is the upper limit
#define START_WEIGHT_MIN_PN_AN MIN_WEIGHT_PN_AN //start weights are random, this is the lower limit

//define VR response function
#define USE_NON_LINEAR_VR_RESPONSE 1
#define VR_RESPONSE_SIGMA_SCALING 1.0f
#define VR_RESPONSE_POWER 0.7f //only used if non linear rsponse specified
#define VR_RESPONSE_DISTANCE_SCALING 5

using namespace std;
#include <cassert>
#include "hr_time.cpp"
#include "utils.h" // for CHECK_CUDA_ERRORS
//#include "utilities.h"

//#include <cuda_runtime.h>

/*--------------------------------------------------------------------------
global variables
--------------------------------------------------------------------------*/

CStopWatch timer;
bool global_RunOnCPU = false;

//--------------------------------------------------------------------------
#include "Enose_classifier.cu"
