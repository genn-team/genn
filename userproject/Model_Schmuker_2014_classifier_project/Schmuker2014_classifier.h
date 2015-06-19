/*--------------------------------------------------------------------------
   Author: Alan Diamond
--------------------------------------------------------------------------*/


#ifndef SCHMUKER2014_CLASSIFIER_H
#define SCHMUKER2014_CLASSIFIER_H

//--------------------------------------------------------------------------
/*! \file Schmuker2014_classifier.h

\brief Header file containing the class definition for the Schmuker2014 classifier, which contains the methods for 
setting up, initialising, simulating and saving results of a multivariate classifier imspired by the insect olfactory system.
See "A neuromorphic network for generic multivariate data classification, Michael Schmuker, Thomas Pfeilc, and Martin Paul Nawrota, 2014"

*/
//--------------------------------------------------------------------------


#include "Model_Schmuker_2014_classifier.cc"

//--------------------------------------------------------------------------
/*! \brief This class cpontains the methods for running the Schmuker_2014_classifier example model.
 */
//--------------------------------------------------------------------------

class Schmuker2014_classifier
{
 public:
	double d_maxRandomNumber; //number used to scale correcly scale poission neuron firing probabilities

	NNmodel model;
	uint64_t *inputRates; //dataset (2D array) of required poisson neuron firing rates that represent the input data to the network
	unsigned int inputRatesSize; //cache the size of the input data, this is used at multiple times
	float * vrData; //2D array of vectors in feature space that will act as the virtual receptor (VR) points
	unsigned int *classLabel; //array holding set of classes labelling the recordings, indexed by recordingIdx
	unsigned int *individualSpikeCountPN;  //stores total of  spikes on every PN neuron during the run time. Needed for the plasticity rule.
	unsigned int *overallWinnerSpikeCountAN; ////stores total of the spikes witihn each AN cluster during the overall run. Needed to determine the overall winning output cluster class
	unsigned int *clusterSpikeCountAN; //stores total of the spikes witihn each AN cluster during each plasticity window. Needed to determine the winning output cluster class during that window,
	float * plasticWeights; //2D array copy of PN AN weights which is updated by plasticity during a presentation


	//------------------------------------------------------------------------
	// on the device:
	uint64_t *d_inputRates; //copy of inputRates that will be passed en block to device, which will use a specified offset to select the particular vector of rates

	//------------------------------------------------------------------------
	//convenience holders for getting population sizes
	unsigned int countRN, countPN, countAN, countPNAN;
	static const unsigned int  timestepsPerRecording = RECORDING_TIME_MS / DT ; // = num timesteps contained in each data recording;

	//cached max and min values generated from data set used for scaling VR responses
	float * sampleDistance;  //array of 2 floats 0 = max, 1 = min

	//holders for the directories used to locate recordings, cached data and output files
	string recordingsDir, cacheDir, outputDir, datasetName, uniqueRunId;

	UINT correctClass; //holds the id for the class labelled to the current recording
	int winningClass; //stores the id of the class that "won" the classification during the last input presentation (invocation of run() method). Set by calculateOverallWinner method

	enum data_type {data_type_int, data_type_uint, data_type_float,data_type_double};

	FILE * log; //opened general logging file

	//Adjustable parameters (defaulted from #define constants)
	UINT param_SPIKING_ACTIVITY_THRESHOLD_HZ;
	UINT param_MAX_FIRING_RATE_HZ;
	UINT param_MIN_FIRING_RATE_HZ;
	float param_GLOBAL_WEIGHT_SCALING;
	float param_WEIGHT_RN_PN;
	float param_CONNECTIVITY_RN_PN;
	float param_WEIGHT_WTA_PN_PN;
	float param_WEIGHT_WTA_AN_AN;
	float param_CONNECTIVITY_PN_PN;
	float param_CONNECTIVITY_AN_AN;
	float param_CONNECTIVITY_PN_AN;
	float param_MIN_WEIGHT_PN_AN;
	float param_MAX_WEIGHT_PN_AN;
	float param_WEIGHT_DELTA_PN_AN;
	float param_PLASTICITY_INTERVAL_MS;

	bool clearedDownDevice;

	// end of data fields

	Schmuker2014_classifier();
	~Schmuker2014_classifier();
	void allocateHostAndDeviceMemory();
	void populateDeviceMemory();
	void update_input_data_on_device();
	void clearDownDevice();
	void run(float runtime, string filename_rasterPlot,bool usePlasticity);
	void getSpikesFromGPU();
	void getSpikeNumbersFromGPU();
	void outputSpikes(FILE *, string delim);

	void initialiseWeights_SPARSE_RN_PN();
	void initialiseWeights_WTA_PN_PN();
	void initialiseWeights_DENSE_PN_AN();
	void initialiseWeights_WTA_AN_AN();

	void createWTAConnectivity(float * synapse, UINT populationSize, UINT clusterSize, float synapseWeight, float probability);

	bool randomEventOccurred(float probability);

	void updateWeights_PN_AN_on_device();

	void generate_or_load_inputrates_dataset(unsigned int recordingIdx);
	void generate_inputrates_dataset(unsigned int recordingIdx);
	FILE * openRecordingFile(UINT recordingIndex);

	void applyLearningRuleSynapses(float * synapsesPNAN);


	void initialiseInputData();
	void load_VR_data();
	void setCorrectClass(UINT recordingIdx);
	UINT getClassCluster(UINT anIdx);
	void loadClassLabels();

	void addInputRate(float * samplePoint,UINT timeStep);
	uint64_t convertToRateCode(float inputRateHz) ;
	float calculateVrResponse(float * samplePoint, float * vrPoint);

	void setMaxMinSampleDistances();
	void findMaxMinSampleDistances(float * samples, UINT startAt,UINT totalSamples);
	float getSampleDistance(UINT max0_min1);
	float getManhattanDistance(float * pointA,float * pointB, UINT numElements);

	float getRand0to1();
	UINT calculateOverallWinner();
	UINT calculateWinner(unsigned int * clusterSpikeCount);
	UINT calculateCurrentWindowWinner();

	void updateIndividualSpikeCountPN() ;
	void resetIndividualSpikeCountPN() ;
	void updateClusterSpikeCountAN() ;
	void resetClusterSpikeCountAN() ;
	void resetOverallWinner();
	void updateWeights_PN_AN();

	UINT getClusterIndex(UINT neuronIndex, UINT clusterSize);

	void generateSimulatedTimeSeriesData();

	string getRecordingFilename(UINT recordingIdx) ;

	//Common utility fns. Move to a utilities file/class
	bool loadArrayFromTextFile(string path, void * array, string delim, UINT arrayLen, data_type dataType);
	void checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints);
	void checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints, string delim);
	void printSeparator();

	void resetDevice();

	void startLog() ;

};

#endif
