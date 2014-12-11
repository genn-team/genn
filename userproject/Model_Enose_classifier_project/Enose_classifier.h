/*--------------------------------------------------------------------------
   Author: Alan Diamond
--------------------------------------------------------------------------*/


#ifndef ENOSE_CLASSIFIER_H
#define ENOSE_CLASSIFIER_H

//--------------------------------------------------------------------------
/*! \file Enose_classifier.h

\brief Header file containing the class definition for the Enose classifier, which contains the methods for
setting up, initialising, simulating and saving results of a multivariate classifier imspired by the insect olfactory system.
See "A neuromorphic network for generic multivariate data classification, Michael Schmuker, Thomas Pfeilc, and Martin Paul Nawrota, 2014"

*/
//--------------------------------------------------------------------------

//#include "utilities.h"
#include "utilities.cc"

#include "Model_Enose_classifier.cc"

//--------------------------------------------------------------------------
/*! \brief This class cpontains the methods for running the Enose_classifier  model.
 */
//--------------------------------------------------------------------------

class Enose_classifier
{
 public:
	float t; //absolute time elapsed for network

	NNmodel model;
	unsigned int *inputRates; //dataset (2D array) of required poisson neuron firing rates that represent the input data to the network
	unsigned int inputRatesSize; //cache the size of the input data, this is used at multiple times
	float * vrData; //2D array of vectors in feature space that will act as the virtual receptor (VR) points
	unsigned int *classLabel; //array holding set of classes labelling the recordings, indexed by recordingIdx
	unsigned int *individualSpikeCountPN;  //stores total of  spikes on every PN neuron during the run time. Needed for the plasticity rule.
	unsigned int *overallWinnerSpikeCountAN; ////stores total of the spikes witihn each AN cluster during the overall run. Needed to determine the overall winning output cluster class
	unsigned int *clusterSpikeCountAN; //stores total of the spikes witihn each AN cluster during each plasticity window. Needed to determine the winning output cluster class during that window,
	unsigned int *clusterSpikeCountPN; //stores total of the spikes witihn each PN cluster during input presentation.
	unsigned int *clusterSpikeCountRN; //stores total of the spikes witihn each RN cluster during input presentation.
	unsigned int *clusterSpikeCountPerTimestepAN; //stores total spikes witihn each AN cluster during each separate timestep
	float * plasticWeights; //2D array copy of PN AN weights which is updated by plasticity during a presentation
	enum class_evaluation_stage {stage_pre_evaluation, stage_evaluation_active, stage_post_evaluation};
	class_evaluation_stage currClassEvaluationStage;
	UINT classEvaluationStartTimestep, classEvaluationEndTimestep;

	//------------------------------------------------------------------------
	// on the device:
	unsigned int *d_inputRates; //copy of inputRates that will be passed en block to device, which will use a specified offset to select the particular vector of rates

	//------------------------------------------------------------------------
	//convenience holders for getting population sizes
	unsigned int countRN, countPN, countAN, countPNAN;
	static const unsigned int  timestepsPerRecording = (RECORDING_TIME_MS / DT) ; // = num timesteps contained in each data recording;

	//cached max and min values generated from data set used for scaling VR responses
	float * allDistinctSamples; //cached set of all distinct samples, one per recording
	float * sampleDistance;  //array of 3 floats 0 = max, 1 = min, 2 = average
	double totalledDistance; //all sample distances added up. used to calcualte average
	UINT samplesCompared; //how many samples were compared, Needed with totalledDistance to get the average

	//holders for the directories used to locate recordings, cached data and output files
	string recordingsDir, recordingFilenameTemplate, cacheDir, outputDir, datasetName, uniqueRunId;

	UINT correctClass; //holds the id for the class labelled to the current recording
	int winningClass; //stores the id of the class that "won" the classification during the last input presentation (invocation of run() method). Set by calculateOverallWinner method

	FILE * log; //opened general logging file

	//Adjustable parameters (defaulted from #define constants)
	UINT param_SPIKING_ACTIVITY_THRESHOLD_HZ;
	UINT param_MAX_FIRING_RATE_HZ;
	UINT param_MIN_FIRING_RATE_HZ;
	UINT param_REPEAT_LEARNING_SET;
	float param_GLOBAL_WEIGHT_SCALING;
	float param_CONNECTIVITY_RN_PN;
	float param_WEIGHT_WTA_PN_PN;
	float param_WEIGHT_WTA_AN_AN;
	float param_CONNECTIVITY_PN_PN;
	float param_CONNECTIVITY_AN_AN;
	float param_CONNECTIVITY_PN_AN;
	float param_MIN_WEIGHT_PN_AN;
	float param_MAX_WEIGHT_PN_AN;
	//float param_WEIGHT_DELTA_PN_AN;
	float param_WEIGHT_INC_DELTA_PN_AN;
	float param_WEIGHT_DEC_MULTIPLIER_PN_AN;
	float param_START_WEIGHT_MAX_PN_AN;
	float param_START_WEIGHT_MIN_PN_AN;
	float param_PLASTICITY_INTERVAL_MS;
	float param_VR_RESPONSE_POWER;
	float param_VR_RESPONSE_SIGMA_SCALING;
	float param_VR_RESPONSE_DISTANCE_SCALING;
	float param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN;
	float param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN;
	float param_MIN_CLASS_EVALUATION_TIMESTEPS;

	bool clearedDownDevice;

	//temp - used to locate an accurate spiking rate threshold for making a positive clas identification
	float highestWinningSpikeFreqHzAN;

	// end of data fields

	Enose_classifier();
	~Enose_classifier();
	void allocateHostAndDeviceMemory();
	void populateDeviceMemory();
	void update_input_data_on_device();
	void clearDownDevice();
	void run(float runtime, bool createRasterPlot, string filename_rasterPlot,bool usePlasticity,UINT stretchTime);
	void getSpikesFromGPU();
	void getSpikeNumbersFromGPU();
	void outputSpikes(FILE *, string delim, float startTime);

	void initialiseWeights_SPARSE_RN_PN();
	void initialiseWeights_WTA_PN_PN();
	void initialiseWeights_DENSE_PN_AN();
	void initialiseWeights_WTA_AN_AN();

	void createWTAConnectivity(float * synapse, UINT populationSize, UINT clusterSize, float synapseWeight, float probability);

	void updateWeights_PN_AN_on_device();

	void generate_or_load_inputrates_dataset(unsigned int recordingIdx);
	void generate_inputrates_dataset(unsigned int recordingIdx);
	FILE * openRecordingFile(UINT recordingIndex);

	void applyLearningRuleSynapses(float * synapsesPNAN, UINT winner);


	void initialiseInputData();
	void initialise_VR_data();
	bool initialiseDistinctSamples();
	void load_VR_data();
	void setCorrectClass(UINT recordingIdx);
	UINT getClassCluster(UINT anIdx);
	void loadClassLabels();

	void addInputRate(UINT timeStep, float * vrResponses);
	UINT convertToRateCode(float inputRateHz) ;
	float calculateVrResponse(float * samplePoint, float * vrPoint);

	void setMaxMinSampleDistances();
	void findMaxMinSampleDistances(float * samples, UINT startAt,UINT totalSamples);
	float getSampleDistance(UINT max0_min1);
	float getManhattanDistance(float * pointA,float * pointB, UINT numElements);

	UINT calculateOverallWinner(bool restrictToEvaluationPeriod);
	UINT calculateWinner(unsigned int * clusterSpikeCount);
	UINT calculateCurrentWindowWinner();
	int calculateEvaluationPeriodWinner();

	void checkSpikeRateThresholdAN(float sampleTimeMs, float periodMs, UINT timestep, UINT lastTimestep, ofstream & spikeFreqFile);
	void checkSpikeRateThresholdPN(float sampleTimeMs, float periodMs, UINT timestep, UINT lastTimestep, ofstream & spikeFreqFile);
	float calculateAvgSpikeFreqHz(int spikeCount,float periodMs,int numNeurons);

	void updateIndividualSpikeCountPN() ;
	void resetIndividualSpikeCountPN() ;
	void updateClusterSpikeCountAN(UINT timestep) ;
	void updateClusterSpikeCountPN();
	void updateClusterSpikeCountRN();
	void resetClusterSpikeCountAN() ;
	void resetClusterSpikeCountPerTimestepAN();
	void resetClusterSpikeCountRN() ;
	void resetClusterSpikeCountPN() ;
	void resetOverallWinner();
	void updateWeights_PN_AN();

	UINT getClusterIndex(UINT neuronIndex, UINT clusterSize);

	void generateSimulatedTimeSeriesData();

	string getRecordingFilePath(UINT recordingIdx) ;

	void resetDevice();

	void startLog() ;

	void findMaxMinSamplesDistancesOnDevice(float * allSamples);

	string getVrResponseFilename(UINT recordingIdx);
	void completeVrResponseSet(float * vrResponses,UINT recordingIdx);
	void setMaxSampleDistance(float d);
	float getMaxSampleDistance();
	void setMinSampleDistance(float d);
	float getMinSampleDistance();
	void setAverageSampleDistance(float d);
	float getAverageSampleDistance();

	bool displayRasterPlot(string srcdir , string srcFilename, float endTime, bool displayActivation, int stretchTime);
};

#endif
