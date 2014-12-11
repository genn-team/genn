#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767

/*--------------------------------------------------------------------------
   Author: Alan Diamond

--------------------------------------------------------------------------
 Main entry point for the experiment using the classifier design based on Schmuker 2014 hardware classifier

 --------------------------------------------------------------------------*/
#include "experiment.h"
#include <time.h>
#include <algorithm> //for random_shuffle
//#include <array>
#include <vector>
//#include "utilities.h"
#include "utilities.cc"


//this class does most the work, controlled by this experiment
Enose_classifier classifier;


//generic parameter struct to allow generic testing and reporting of different parameter settings
typedef struct Parameter {
	string name;
	string value;
} Parameter;


bool clearInputRateCache() {
	return wildcardFileDelete(classifier.cacheDir,"*.cache",false);
}

bool clearResponseVRCache() {
	bool ok =  wildcardFileDelete(classifier.cacheDir,"VR*.csv",false);
	if (!ok) {
		cerr << "clearResponseVRCache: wildcard delete reported failure";
		return false;
	} else  {
		return clearInputRateCache();//if VR responses change then cached input rates also become invalid
	}
}

//Uses pipe/shell to invoke Python MDP package to run neural gas clustering to generate set of VR points from data set
bool generateVRs(string srcDir, string srcFilename, string destFilename , int numVR , int epochs)
{
	stringstream commandLine;
	commandLine <<  PYTHON_RUNTIME << SPACE <<  PYTHON_DIR << "/GenerateVRs.py ";
	commandLine << srcDir << SPACE << srcFilename << SPACE << destFilename;
	commandLine << SPACE << numVR << SPACE << epochs;
	string result;
	cout << "Invoking shell command:\n" << commandLine.str() << endl;
	bool callWorked = invokeLinuxShellCommand(commandLine.str().c_str(),result);
	if (!callWorked) {
		cerr << "Unable to invoke VR python script" << endl;
		exit(1);
	}

	cout << "Result:\n" << result;
	return true;
}

bool createVRFilesFromRecordings(string recordingDir) {

	stringstream vrFilename;
	UINT numVR, epochs;
	cout << "Enter number of VR and number of epochs:" << endl;
	cin >> numVR;
	cin >> epochs;
	vrFilename << "VRData_x" << numVR << "_epochs" << epochs << ".csv";
	bool ok = generateVRs(recordingDir,FILENAME_ALL_DISTINCT_SAMPLES,vrFilename.str(),numVR,epochs);
	if (!ok) {
		cerr << "A problem was reported generating VR file " << vrFilename << endl;
		exit(1);
	}
	return true;

}



/*-----------------------------------------------------------------
Uses a timestamp plus network parameters used to create an id string unique to this run
-----------------------------------------------------------------*/

string getUniqueRunId()
{
	string timestamp = toString(time (NULL));
	string id = timestamp +
			"-" + classifier.datasetName;
	return id;
}


/*-----------------------------------------------------------------
Write to matching file the parameters used to create this run
-----------------------------------------------------------------*/
void outputRunParameters()
{
	string paramFilename = classifier.outputDir;
	paramFilename.append(SLASH).append(classifier.uniqueRunId).append(" Run Parameters.txt");
	FILE * file = fopen(paramFilename.c_str(),"w");
	fprintf(file,"DATASET_NAME\t\t%s\n",toString(DATASET_NAME).c_str());
	fprintf(file,"DT\t\t%f\n",DT);
	fprintf(file,"global_NumVR\t\t%d\n",global_NumVR);
	fprintf(file,"VR_DATA_FILENAME_TEMPLATE\t\t%s\n",toString(VR_DATA_FILENAME_TEMPLATE).c_str());
	fprintf(file,"DISTINCT_SAMPLES_PER_RECORDING\t\t%d\n",DISTINCT_SAMPLES_PER_RECORDING);
	fprintf(file,"NUM_SENSORS_RECORDED\t\t%d\n",NUM_SENSORS_RECORDED);
	fprintf(file,"NUM_SENSORS_CHOSEN\t\t%d\n",NUM_SENSORS_CHOSEN);
	fprintf(file,"NUM_FEATURES\t\t%d\n",NUM_FEATURES);
	fprintf(file,"NUM_CLASSES\t\t%d\n",NUM_CLASSES);
	fprintf(file,"NETWORK_SCALE\t\t%d\n",NETWORK_SCALE);
	fprintf(file,"CLUST_SIZE_RN\t\t%d\n",CLUST_SIZE_RN);
	fprintf(file,"CLUST_SIZE_PN\t\t%d\n",CLUST_SIZE_PN);
	fprintf(file,"CLUST_SIZE_AN\t\t%d\n",CLUST_SIZE_AN);
	fprintf(file,"SYNAPSE_TAU_RNPN\t\t%f\n",SYNAPSE_TAU_RNPN);
	fprintf(file,"SYNAPSE_TAU_PNPN\t\t%f\n",SYNAPSE_TAU_PNPN);
	fprintf(file,"SYNAPSE_TAU_PNAN\t\t%f\n",SYNAPSE_TAU_PNAN);
	fprintf(file,"SYNAPSE_TAU_ANAN\t\t%f\n",SYNAPSE_TAU_ANAN);

	fprintf(file,"MAX_FIRING_RATE_HZ\t\t%d\n",MAX_FIRING_RATE_HZ);
	fprintf(file,"MIN_FIRING_RATE_HZ\t\t%d\n",MIN_FIRING_RATE_HZ);
	fprintf(file,"GLOBAL_WEIGHT_SCALING\t\t%f\n",GLOBAL_WEIGHT_SCALING);
	fprintf(file,"WEIGHT_RN_PN\t\t%f\n",WEIGHT_RN_PN);
	fprintf(file,"WEIGHT_WTA_PN_PN\t\t%f\n",WEIGHT_WTA_PN_PN);
	fprintf(file,"WEIGHT_WTA_AN_AN\t\t%f\n",WEIGHT_WTA_AN_AN);
	fprintf(file,"CONNECTIVITY_RN_PN\t\t%f\n",CONNECTIVITY_RN_PN);
	fprintf(file,"CONNECTIVITY_PN_PN\t\t%f\n",CONNECTIVITY_PN_PN);
	fprintf(file,"CONNECTIVITY_AN_AN\t\t%f\n",CONNECTIVITY_AN_AN);
	fprintf(file,"CONNECTIVITY_PN_AN\t\t%f\n",CONNECTIVITY_PN_AN);
	fprintf(file,"MIN_WEIGHT_PN_AN\t\t%f\n",MIN_WEIGHT_PN_AN);
	fprintf(file,"MAX_WEIGHT_PN_AN\t\t%f\n",MAX_WEIGHT_PN_AN);
	fprintf(file,"WEIGHT_INC_DELTA_PN_AN\t\t%f\n",WEIGHT_INC_DELTA_PN_AN);
	fprintf(file,"WEIGHT_DEC_MULTIPLIER_PN_AN\t\t%f\n",WEIGHT_DEC_MULTIPLIER_PN_AN);
	fprintf(file,"START_WEIGHT_MAX_PN_AN\t\t%f\n",START_WEIGHT_MAX_PN_AN);
	fprintf(file,"START_WEIGHT_MIN_PN_AN\t\t%f\n",START_WEIGHT_MIN_PN_AN);
	fprintf(file,"PLASTICITY_INTERVAL_MS\t\t%f\n",PLASTICITY_INTERVAL_MS);
	fprintf(file,"REPEAT_LEARNING_SET\t\t%d\n",REPEAT_LEARNING_SET);
	fprintf(file,"SPIKING_ACTIVITY_THRESHOLD_HZ\t\t%d\n",SPIKING_ACTIVITY_THRESHOLD_HZ);
	fprintf(file,"TOTAL_RECORDINGS\t\t%d\n",TOTAL_RECORDINGS);
	fprintf(file,"N_FOLDING\t\t%d\n",N_FOLDING);
	fprintf(file,"N_REPEAT_TRIAL_OF_PARAMETER\t\t%d\n",N_REPEAT_TRIAL_OF_PARAMETER);
	fprintf(file,"RECORDING_TIME_MS\t\t%f\n",RECORDING_TIME_MS);
	fprintf(file,"VR_RESPONSE_SIGMA_SCALING\t\t%f\n",VR_RESPONSE_SIGMA_SCALING);
	fprintf(file,"VR_RESPONSE_POWER\t\t%f\n",VR_RESPONSE_POWER);
	fprintf(file,"VR_RESPONSE_DISTANCE_SCALING\t\t%f\n",VR_RESPONSE_DISTANCE_SCALING);
#ifdef USE_NON_LINEAR_VR_RESPONSE
	fprintf(file,"USE_NON_LINEAR_VR_RESPONSE\t\t%d\n",1);
#else
	fprintf(file,"USE_NON_LINEAR_VR_RESPONSE\t\t%d\n",0);
#endif
	fprintf(file,"VR_RESPONSE_DISTANCE_SCALING\t\t%f\n",VR_RESPONSE_DISTANCE_SCALING);
#ifdef INFER_CLASS_EVALUATION_WINDOW_FROM_AN
	fprintf(file,"CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN\t\t%f\n",CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN);
#else
	fprintf(file,"CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN\t\t%f\n",CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN);
#endif
	fclose(file);
}

/*-----------------------------------------------------------------
Load the specified recording and apply it to the classifier as a set of input rates
-----------------------------------------------------------------*/
bool applyInputToClassifier(UINT recordingIdx, float durationMs, bool usePlasticity, bool createRasterPlot,  string filename_rasterPlot, UINT stretchTime, bool useEvaluationWindowForClassification )
{

	//printf("Presenting recording %u to classifier.. \n",recordingIdx);

	classifier.resetOverallWinner();

	//printf( "Loading data recording %d ..\n", recordingIdx);

	//get the set of input rate data for the recording (this will be first generated from the sensor data and the VR set, then cached in a uniquely named file)
	classifier.generate_or_load_inputrates_dataset(recordingIdx);

	//move the rates data across to the device
	classifier.update_input_data_on_device();

	//get the correct class label for this recording and store
	classifier.setCorrectClass(recordingIdx);

	//run the model for the duration of the recording, collecting the relevant spike sets on each timestep (if raster plot specified in FLAGS )
	classifier.run(durationMs,createRasterPlot, filename_rasterPlot,usePlasticity,stretchTime);

	int winner  = classifier.calculateOverallWinner(useEvaluationWindowForClassification);

	bool classifiedCorrectly =  winner == classifier.correctClass;

	//string yesNo  = toString(classifiedCorrectly ? "YES" : "NO");
	//cout << "Classified Correctly? " << yesNo << endl;

	return classifiedCorrectly;

}

/*--------------------------------------------------------------------------
 assign default values to the main classifier parameters
 -------------------------------------------------------------------------- */
void setDefaultParamValues()
{
	classifier.param_SPIKING_ACTIVITY_THRESHOLD_HZ = SPIKING_ACTIVITY_THRESHOLD_HZ;
	classifier.param_MAX_FIRING_RATE_HZ  = MAX_FIRING_RATE_HZ;
	classifier.param_MIN_FIRING_RATE_HZ = MIN_FIRING_RATE_HZ;
	classifier.param_GLOBAL_WEIGHT_SCALING = GLOBAL_WEIGHT_SCALING;
	classifier.param_CONNECTIVITY_RN_PN = CONNECTIVITY_RN_PN;
	classifier.param_WEIGHT_WTA_PN_PN = WEIGHT_WTA_PN_PN;
	classifier.param_WEIGHT_WTA_AN_AN = WEIGHT_WTA_AN_AN;
	classifier.param_CONNECTIVITY_PN_PN = CONNECTIVITY_PN_PN;
	classifier.param_CONNECTIVITY_AN_AN = CONNECTIVITY_AN_AN;
	classifier.param_CONNECTIVITY_PN_AN = CONNECTIVITY_PN_AN;
	classifier.param_MIN_WEIGHT_PN_AN = MIN_WEIGHT_PN_AN;
	classifier.param_MAX_WEIGHT_PN_AN = MAX_WEIGHT_PN_AN;
	classifier.param_WEIGHT_INC_DELTA_PN_AN  = WEIGHT_INC_DELTA_PN_AN;
	classifier.param_WEIGHT_DEC_MULTIPLIER_PN_AN  = WEIGHT_DEC_MULTIPLIER_PN_AN;
	classifier.param_START_WEIGHT_MAX_PN_AN = START_WEIGHT_MAX_PN_AN;
	classifier.param_START_WEIGHT_MIN_PN_AN = START_WEIGHT_MIN_PN_AN;
	classifier.param_PLASTICITY_INTERVAL_MS = PLASTICITY_INTERVAL_MS;
	classifier.param_VR_RESPONSE_POWER = VR_RESPONSE_POWER;
	classifier.param_VR_RESPONSE_SIGMA_SCALING = VR_RESPONSE_SIGMA_SCALING;
	classifier.param_VR_RESPONSE_DISTANCE_SCALING = VR_RESPONSE_DISTANCE_SCALING;
	classifier.param_REPEAT_LEARNING_SET = REPEAT_LEARNING_SET;
	classifier.param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN = CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN;
	classifier.param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN = CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN;
	classifier.param_MIN_CLASS_EVALUATION_TIMESTEPS = MIN_CLASS_EVALUATION_TIMESTEPS;

}

string generateRasterPlotFilename(UINT recordingIdx,UINT folding, UINT rpt, string stage,Parameter param, UINT index)
{
	stringstream rasterPlotFilename;
	rasterPlotFilename << classifier.uniqueRunId;
	rasterPlotFilename << "-RasterData-Rec" << toString(recordingIdx) << "-Class" << toString(classifier.correctClass);
	rasterPlotFilename << "-" << param.name << "=" << param.value;
	rasterPlotFilename << "-" << "rpt" << toString(rpt);
	rasterPlotFilename << "-" << "folding" << toString(folding);
	rasterPlotFilename << "-" << stage;
	if (stage=="training") rasterPlotFilename << "[" << toString(index) << "]";
	rasterPlotFilename << ".csv";
	return rasterPlotFilename.str();
}
/*--------------------------------------------------------------------------
check that the data in this directory match the current global settings e.g. NUM_SENSORS_CHOSEN
-------------------------------------------------------------------------- */
void checkDataCompatibility()
{
	string path  = classifier.recordingsDir + SLASH + FILENAME_DATA_DESCRIPTION;
	ifstream in (path.c_str());
	if(!in.is_open()) {
		cerr << "Failed to open data description file " << path << endl;
		exit(1);
	} else {
		cout << "Loaded data description file " << path << endl;
	}

	string line;
	getline(in,line);//skip label header row
	UINT numRec, samplesPerRec, numSensorsChosen;
	in >> numRec;
	in >> samplesPerRec;
	in >> numSensorsChosen;
	in.close();
	if (numRec != TOTAL_RECORDINGS) {
		cerr << "COMPATIBILITY ERROR: Expected " << TOTAL_RECORDINGS << " recordings but detected " << numRec << endl;
		exit(1);
	}
	if (samplesPerRec != DISTINCT_SAMPLES_PER_RECORDING) {
		cerr << "COMPATIBILITY ERROR: Expected " << DISTINCT_SAMPLES_PER_RECORDING << " samples per recording but detected " << samplesPerRec << endl;
		exit(1);
	}
	if (numSensorsChosen != NUM_SENSORS_CHOSEN) {
		cerr << "COMPATIBILITY ERROR: Expected " << NUM_SENSORS_CHOSEN << " sensors to be used but detected " << numSensorsChosen << endl;
		exit(1);
	}
}

//Try to detect and report weird GPU error wher spikes disappear
void checkDeviceError(float testPercent)
{
	if (testPercent  < 10.0) {
		cerr << "No Spiking issue, experimemt terminated." << endl;
		cudaThreadSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			cerr << "CUDA error at " << __FILE__ << ", " << __LINE__ << ": " << cudaGetErrorString(error) << endl;
		}
		exit(1);
	}
}




bool createVRsFromLearningSet(vector<int> & shuffledRecordings,int firstTestingSet,int lastTestingSet,int numVR) {

	cout << "Creating VR set from training data set, excluding test set.." << endl;
	vector<string> recordingsFiles;

	string trainingDataRecordingsFilename = "tmpTrainingDataRecordings.csv";
	string trainingDataRecordingsFilePath = classifier.outputDir + SLASH + trainingDataRecordingsFilename;
	if(fileExists(trainingDataRecordingsFilePath)) {
		wildcardFileDelete(classifier.outputDir , trainingDataRecordingsFilename,false);
	}

	string testDataRecordingsFilename = "tmpTestDataRecordings.csv";
	string testDataRecordingsFilePath = classifier.outputDir + SLASH + testDataRecordingsFilename;
	if(fileExists(testDataRecordingsFilePath)) {
		wildcardFileDelete(classifier.outputDir , testDataRecordingsFilename,false);
	}

	//create a new file of separate training and test data.
	//The training file is for the VR generator . The other is for use in activation graphs etc
	FILE * trainingDataFile = fopen(trainingDataRecordingsFilePath.c_str(),"a");
	FILE * testDataFile = fopen(testDataRecordingsFilePath.c_str(),"a");

	FILE * dataFile; //ptr to swtich between test and training files
	for (int i=0; i<TOTAL_RECORDINGS; i++ ) {

		//separate test data set from training data set
		if (i<firstTestingSet || i>lastTestingSet)
		{//its training data
			dataFile = trainingDataFile;
		} else {//its test data
			dataFile = testDataFile;
		}

		//get the id of the recording to be inlcuded in the training data
		UINT recordingIdx = shuffledRecordings[i];

		//get pointer to start of cached array entry for that recording
		float * recordArray = & classifier.allDistinctSamples[recordingIdx*NUM_FEATURES];
		//add that sample to the training set
		writeArrayToTextFile(dataFile,recordArray,1,NUM_FEATURES,data_type_float,5,false,COMMA,false);
		fprintf(dataFile, "\n");

	}

	fclose(trainingDataFile);
	fclose(testDataFile);

	//invoke the SOM generator , passing the new training data file created
	string vrFilename = "../";
	vrFilename.append(VR_DIR).append(SLASH).append("tmpTrainingDataVRSet.csv");//place result in the VR subdir
	generateVRs(classifier.outputDir,trainingDataRecordingsFilename,vrFilename,numVR,NUM_EPOCHS);
	cout << "Created VR set " << trainingDataRecordingsFilename << endl;
}


/*--------------------------------------------------------------------------
 Write a file per recording that compares the activation (spike totals) of VRs in the RN and PN layers
 Graphing these files informs on how the WTA in the PN layer is performing
 Return path of data file created for this recording
 -------------------------------------------------------------------------- */
string writeVrActivationResults(UINT recordingIdx, string filename, ofstream & overallWtaResults) {

	string path  = classifier.outputDir;
	path.append("/").append(filename);

#ifdef FLAG_GENERATE_VR_ACTIVATION_DATA_PER_RECORDING
	ofstream output(path.c_str());
#endif

	//write col headings
	//output << "VR,RN,RN_ORDER,PN,PN_ORDER" << endl;

	//calc totals (for %) and ordering
	int totalRN = 0;
	int firstRN_VR = -1;
	int firstRN_activation = -1;
	int secondRN_VR = -1;
	int secondRN_activation = -1;
	int totalPN = 0;
	int firstPN_VR = -1;
	int firstPN_activation = -1;
	int secondPN_VR = -1;
	int secondPN_activation = -1;

	for (int vr = 0; vr < global_NumVR; ++vr) {
		int activationRN = classifier.clusterSpikeCountRN[vr];
		int activationPN = classifier.clusterSpikeCountPN[vr];

		totalRN += activationRN;
		totalPN += activationPN;

		if (activationRN > firstRN_activation) {
			//push leader down to second
			secondRN_VR = firstRN_VR;
			secondRN_activation = firstRN_activation;
			//update leader with  current VR
			firstRN_VR = vr;
			firstRN_activation = activationRN;

		} else if(activationRN > secondRN_activation) {
			secondRN_VR = vr;
			secondRN_activation = activationRN;
		}

		if (activationPN > firstPN_activation) {
			//push leader down to second
			secondPN_VR = firstPN_VR;
			secondPN_activation = firstPN_activation;
			//update leader with  current VR
			firstPN_VR = vr;
			firstPN_activation = activationPN;

		} else if(activationPN > secondPN_activation) {
			secondPN_VR = vr;
			secondPN_activation = activationPN;
		}
	}

#ifdef FLAG_GENERATE_VR_ACTIVATION_DATA
	//write data file for this recording
	for (int vr = 0; vr < global_NumVR; ++vr) {
		int activationRN = classifier.clusterSpikeCountRN[vr];
		int activationPN = classifier.clusterSpikeCountPN[vr];
		output << vr << COMMA;
		//output << (100  * ((float)activationRN) / totalRN) << COMMA;
		output << (100  * ((float)activationRN) / firstRN_activation) << COMMA;
		//output << (100  * ((float)activationPN) / totalPN) << COMMA;
		output << (100  * ((float)activationPN) / firstPN_activation) << COMMA;
		int orderRN  = vr==firstPN_VR ? 1 : vr==secondPN_VR ? 2 : 0 ;
		output << orderRN << COMMA;
		int orderPN  = vr==firstPN_VR ? 1 : vr==secondPN_VR ? 2 : 0 ;
		output << orderPN ;
		output << endl;
	}
	output.close();
#endif

	//extend data file for all recordings
	float firstPercentage = 100 * ((float)firstPN_activation)/totalPN ;
	float secondPercentage = 100 * ((float)secondPN_activation)/totalPN;
	string matchFirst  = firstPN_VR==firstRN_VR ? "true" : "false";
	string matchSecond = secondPN_VR==secondRN_VR  ? "true" : "false";
	float winnerConfidence  = ((float)firstPN_activation)/(secondPN_activation==0? 1 : secondPN_activation);

	string matchTopTwo  = (firstPN_VR==firstRN_VR && secondPN_VR==secondRN_VR) || (firstPN_VR==secondRN_VR && firstRN_VR==secondPN_VR) ? "true" : "false";

	float firstPlus2ndActivation = firstPN_activation + secondPN_activation;
	float remainingActivation = totalPN - firstPlus2ndActivation;
	float topTwoConfidence  = firstPlus2ndActivation/remainingActivation;

	overallWtaResults  << recordingIdx << COMMA << firstPercentage << COMMA << secondPercentage  << COMMA << matchFirst << COMMA << matchSecond << COMMA << winnerConfidence << COMMA << matchTopTwo << COMMA << topTwoConfidence << endl;
	overallWtaResults.flush();

	return path;

}

/*--------------------------------------------------------------------------
 This function is the entry point for running the experiment
 -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{

	if (argc != 3)
	{
		fprintf(stderr, "usage: experiment <runOnGPU> <baseDir> \n");
		return 1;
	}

	int runONCPU = atoi(argv[1]);
	global_RunOnCPU = runONCPU==0;
	cout << "Run on: " << (global_RunOnCPU ? "CPU" : "GPU")  << endl;

	string baseDir = toString(argv[2]);
	cout << "Base directory set: " << baseDir << endl;

	bool queryClearDown = true;
	string modelParamExplored = toString("na");
	string modelParamValue = toString("na");

	string recordingsDir =  baseDir;
	recordingsDir.append(SLASH).append(RECORDINGS_DIR);
	string recordingFilenameTemplate = RECORDING_FILENAME_TEMPLATE;

	//if a subsample dataset was selected (presence of %ss% in constants) then replace with correct code
	string subsampleCode = toString(DISTINCT_SAMPLES_PER_RECORDING);
	subsampleCode.append(DOT).append(toString(NUM_SENSORS_CHOSEN));

	recordingsDir = replace(recordingsDir,"%ss%", subsampleCode);
	recordingFilenameTemplate = replace(recordingFilenameTemplate,"%ss%", subsampleCode);

	classifier.recordingsDir = recordingsDir;
	classifier.recordingFilenameTemplate = recordingFilenameTemplate;


	//-----------------------------------------------------------------
	//NETWORK INITIALISATION

	printf( "Network initialisation commenced..\n");

	cout << "Simulation will be run on: " << (global_RunOnCPU ? "CPU" : "GPU")  << endl;
	classifier.resetDevice(); //clear out any possible memory leaks etc from previous runs

	//check that the data in the recordings directory match the current global settings e.g. NUM_SENSORS_CHOSEN
	checkDataCompatibility();

	//set up file locations
	classifier.datasetName = DATASET_NAME;
	classifier.cacheDir = baseDir + SLASH + CACHE_DIR;
	createDirIfNotExists(classifier.cacheDir);

	classifier.outputDir = classifier.recordingsDir + SLASH + OUTPUT_DIR;
	createDirIfNotExists(classifier.outputDir);

	classifier.uniqueRunId = getUniqueRunId();
	classifier.startLog();

	printf( "Recordings input directory set to %s\n", classifier.recordingsDir.c_str());
	printf( "Cache directory set to %s\n", classifier.cacheDir.c_str());
	printf( "Output directory set to %s\n", classifier.outputDir.c_str());



	if (queryClearDown && promptYN("Clear down output directory?")) {
			wildcardFileDelete(classifier.outputDir,"*Enose*",true);
		}


	//Uncomment and edit fn to generate simulated timeseries data
	//classifier.generateSimulatedTimeSeriesData();
	//exit(1);

	//assign default values to the main classifier parameters
	setDefaultParamValues();

	//allocate the memory arrays used in the network on the host and device
	classifier.allocateHostAndDeviceMemory();


	//seed the random number generator for creating random connections
	srand(time(NULL));
	//srand(222); //TODO reset

	//cache an array of all the individual distinct samples, one per recording
	classifier.initialiseDistinctSamples();

	//initialise the set of weights for the SPARSE 1:1 subcluster-subcluster synapses RN-PN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_SPARSE_RN_PN();

	//initialise the set of weights for the DENSE subcluster-subcluster WTA synapses PN-PN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_WTA_PN_PN();

	//NB: This is now called at the start of each folding trial (see main method) to reset the plastic weights
	//initialise the set of weights for the DENSE plastic synapses PN-AN (GeNN has no automatic function for what we need)
	//classifier.initialiseWeights_DENSE_PN_AN();

	//initialise the set of weights for the DENSE subcluster-subcluster WTA synapses AN-AN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_WTA_AN_AN();

	//init sorage for set of virtual receptor points VR to be used to generate input levels
	classifier.initialise_VR_data();

	//allocate storage on CPU and GPU for the dataset of input rates to the poisson neurons
	classifier.initialiseInputData();


	//load classes labelling the recording data sets
	classifier.loadClassLabels();

#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
	//now that all SPARSE arrays have been set up on device , we can user helper fn to allocate device storage for them
	allocateAllDeviceSparseArrays();
#endif

	// Finally, move all the data arrays initialised across to the device
	classifier.populateDeviceMemory();


	//re-seed the random number generator after device setup (which used srand in its own way)
	srand(time(NULL));
	//srand(222); //TODO reset


	outputRunParameters();
	printf( "Network initialisation completed.\n");


	//-----------------------------------------------------------------
	//SET UP TRAINING AND TESTING DATASETS

	//define the indexes of the test set
	unsigned int sizeTestingSet = TOTAL_RECORDINGS / N_FOLDING ; //e.g. 100 recordings into 5 = 20 recordings per bucket

	//set up a list of all the recording id's shuffled into a random order,
	//this allows a simple linear split of the training and test data to achieve rigorous cross validation
	//this algorithm creates a semi-random list but where entries are taken (randomly) from each class in turn
	//the classes themselves are also placed in a random order.
	//The idea is create an even spread across classes to avoid biasing the learning towards a certain class
	//Using a purely random order has been found to create large standard deviations in results across a n-fold cross validation

	vector<int> recordings;
	for (int i=0; i<TOTAL_RECORDINGS; i++ ) {//enter all recordings in order
		recordings.push_back(i);
	}
	vector<int> shuffledClasses;
	for (int i=0; i<NUM_CLASSES; i++ ) {//enter all classes in order
		shuffledClasses.push_back(i);
	}
	random_shuffle(shuffledClasses.begin(),shuffledClasses.end());
	random_shuffle(recordings.begin(),recordings.end());

	vector<int> shuffledRecordings;
	UINT recordingsPerClass = TOTAL_RECORDINGS/NUM_CLASSES;
	for (int k=0; k<recordingsPerClass; k++) {
		for (int i=0; i<NUM_CLASSES; i++ ) {
			int currClass = shuffledClasses[i];
			int j=0;
			//get index of first remaining recording of the specified class
			while(currClass != recordings[j]/recordingsPerClass && j<recordings.size()) {
				j++;
			}
			if (j<recordings.size()) {//found a recording of right class
				shuffledRecordings.push_back(recordings[j]); //add it to shuffled list
				recordings.erase(recordings.begin()+j); //take it out of the list of possibles
			}
		}
	}


	//re-seed the random number generator after shuffle fn (which may have used srand in its own way)
	srand(time(NULL));
	//srand(222);//TODO reset


	//-----------------------------------------------------------------
	//set up parameter exploration , if any
	//-----------------------------------------------------------------
	bool clearInputRates = false;
	bool clearResponseVRs = false;
	bool requiresSynapseRebuild = false;

	//default
	string paramName = modelParamExplored;
	int paramValues[] {atoi(modelParamValue.c_str())};

	//string paramName = "REPEAT_LEARNING_SET";
	//int paramValues[] {4};

	//string paramName = "global_NumVR";
	//int paramValues[] {global_NumVR};

	//string paramName = "VR_RESPONSE_POWER";
	//float paramValues[] {0.4,0.5};
	//clearResponseVRs = true;

	//string paramName = "VR_RESPONSE_SIGMA_SCALING";
	//float paramValues[] {1.0};
	//float paramValues[] {0.15};
	//float paramValues[] {1.0,0.5,0.3,0.15};
	//clearResponseVRs = true;

	//string paramName = "SpkActivityThresholdHz";
	//int paramValues[] {5,10,20,30,40};

	//string paramName = "WEIGHT_INC_DELTA_PN_AN";
	//float paramValues[] {0.3};

	//string paramName = "START_WEIGHT_MAX_PN_AN";
	//float paramValues[] {0.4,0.3,0.25};

	//string paramName = "WEIGHT_DEC_MULTIPLIER_PN_AN";
	//float paramValues[] {0.5,1.0};

	//string paramName = "WeightDeltaPNAN";
	//float paramValues[] {0.04,0.06,0.08};

	/*
	string paramName = "PLASTICITY_INTERVAL_MS";
	float paramValues[] {RECORDING_TIME_MS/8.0,RECORDING_TIME_MS/8.0,
						 RECORDING_TIME_MS/4.0,RECORDING_TIME_MS/4.0,
						 RECORDING_TIME_MS/2.0,RECORDING_TIME_MS/2.0,
						 RECORDING_TIME_MS,RECORDING_TIME_MS};
	 */

	//string paramName = "GlobalWeightScaling";
	//float paramValues[] {0.1,0.25,0.5,4,6,10};
	//requiresSynapseRebuild = true

	//string paramName = "WeightWTA-ANAN";
	//float paramValues[] {0.01};
	//requiresSynapseRebuild = true;

	//string paramName = "CONNECTIVITY_RN_PN";
	//float paramValues[] {0.01,0.1,0.25,0.4,0.5,0.6,0.75,0.99};
	//requiresSynapseRebuild = true

	//string paramName = "WEIGHT_WTA_PN_PN";
	//float paramValues[] {0.0075,0.007};
	//requiresSynapseRebuild = true;

	//string paramName = "MAX_FIRING_RATE_HZ";
	//int paramValues[] {50,60,70,80};
	//clearInputRates = true;

	//string paramName = "CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN";
	//float paramValues[] {32.5};

	//string paramName = "MIN_CLASS_EVALUATION_TIMESTEPS";
	//float paramValues[] {10};

	//-----------------------------------------------------------------

	//track the overall performance of the classifier
	stringstream overallResultsFilename;
	overallResultsFilename << classifier.outputDir << SLASH << classifier.uniqueRunId << " Overall Results.txt";
	//<< " for varying " << paramName  << ".txt";
	cout << "Overall results file: " << overallResultsFilename.str() << endl;
	FILE * overallResultsFile = fopen(overallResultsFilename.str().c_str(),"w");
	fprintf(overallResultsFile, "%s,NumVR,AvgPercentScore,StdDev\n", paramName.c_str());

	/*
	stringstream testingMistakesFilename;
	testingMistakesFilename << classifier.outputDir << SLASH << classifier.uniqueRunId << " Testing Mistakes varying " << paramName  << ".txt";
	ofstream testingMistakesFile(testingMistakesFilename.str().c_str());
	testingMistakesFile << paramName << ",Class,MistakenForClass" << endl;
	*/

	//track performance per recording
	UINT recordingClassifiedWrongInTest[TOTAL_RECORDINGS];
	UINT recordingClassifiedWrongInTraining[TOTAL_RECORDINGS];
	UINT classClassifiedWrongInTraining[NUM_CLASSES];
	zeroArray(classClassifiedWrongInTraining,NUM_CLASSES);
	zeroArray(recordingClassifiedWrongInTest,TOTAL_RECORDINGS);
	zeroArray(recordingClassifiedWrongInTraining,TOTAL_RECORDINGS);


	//Run full cross validation, stepping through parameter values supplied
	for (int paramIndex = 0; paramIndex < sizeof(paramValues)/sizeof(paramValues[0]); paramIndex++) {

		//Apply next param value
		Parameter param = {paramName,toString(paramValues[paramIndex])};
		//classifier.param_SPIKING_ACTIVITY_THRESHOLD_HZ = paramValues[paramIndex];
		//classifier.param_WEIGHT_DELTA_PN_AN = paramValues[paramIndex];
		//classifier.param_PLASTICITY_INTERVAL_MS = paramValues[paramIndex];
		//classifier.param_GLOBAL_WEIGHT_SCALING = paramValues[paramIndex];
		//classifier.param_WEIGHT_WTA_PN_PN = paramValues[paramIndex];
		//classifier.param_WEIGHT_WTA_AN_AN = paramValues[paramIndex];
		//classifier.param_CONNECTIVITY_RN_PN = paramValues[paramIndex];
		//classifier.param_MAX_FIRING_RATE_HZ  = paramValues[paramIndex];
		//classifier.param_REPEAT_LEARNING_SET  = paramValues[paramIndex];
		//classifier.param_VR_RESPONSE_SIGMA_SCALING = paramValues[paramIndex];
		//classifier.param_VR_RESPONSE_POWER = paramValues[paramIndex];
		//classifier.param_START_WEIGHT_MAX_PN_AN = paramValues[paramIndex];
		//classifier.param_WEIGHT_DEC_MULTIPLIER_PN_AN = paramValues[paramIndex];
		//classifier.param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN = paramValues[paramIndex];
		//classifier.param_MIN_CLASS_EVALUATION_TIMESTEPS = paramValues[paramIndex];

		//clear cache data if required between param values
		if (clearInputRates) clearInputRateCache();
		if (clearResponseVRs) clearResponseVRCache();

		if (requiresSynapseRebuild) {
			//regenerate non plastic synapse weight arrays
			//(these values depend on tuneable parameters - weight, connectivty etc)
			//classifier.initialiseWeights_SPARSE_RN_PN(); exclude paramterising as tricky to vary if using SPARSE encoding
			classifier.initialiseWeights_WTA_PN_PN();
			classifier.initialiseWeights_WTA_AN_AN();
			if (!global_RunOnCPU) {
				copyGToDevice();//update the device copy if using GPU
			}
		}



		//track the performance across each param setting, per folding
		stringstream perParamResultsFilename;
		perParamResultsFilename << classifier.outputDir << SLASH << classifier.uniqueRunId << " Totalled Results for " << param.name << "-" << param.value  << ".txt";
		FILE * perParamResultsFile = fopen(perParamResultsFilename.str().c_str(),"w");
		fprintf(perParamResultsFile,"%s,Folding,Stage,Correct,OutOf,Percent\n",param.name.c_str());

		//track the WTA performance of the PN layer
		string perParamWtaPerformanceFilenamePN = replace(perParamResultsFilename.str(),"Totalled Results","PN Layer WTA Performance");
		ofstream perParamWtaPerformanceFilePN(perParamWtaPerformanceFilenamePN.c_str());
		perParamWtaPerformanceFilePN  << "RecordingId,WinnerPercentage,2ndPlacePercentage,PN-RN-WinnersMatch,PN-RN-2ndPlaceMatch,ConfidenceCorrectWinner,TopTwoMatch,ConfidenceCorrectTopTwo" << endl;



#ifdef FLAG_OUTPUT_INDIVIDUAL_RESULTS
		//track the detailed performance of the classifier
		stringstream individualResultsFilename;
		individualResultsFilename << classifier.outputDir << SLASH << classifier.uniqueRunId << " Individual Results for " << param.name << "-" << param.value  << ".txt";
		FILE * individualResultsFile = fopen(individualResultsFilename.str().c_str(),"w");
		fprintf(individualResultsFile,"%s,folding,recordingIdx,classifierSelectedClass,correctClass\n",param.name.c_str());
		printf("Individual training results will be saved to the file: %s\n", individualResultsFilename.str().c_str());
#endif

		int totalTestScore = 0;
		int totalTestCount = 0;

		vector<float> vecFoldingResults;//holder for the result of each folding, will be averaged/stdDev at the end of cross validation

		for (int rptTrial = 0; rptTrial < N_REPEAT_TRIAL_OF_PARAMETER; rptTrial++) {

			cout << "Beginning iteration " << (rptTrial+1) << " of " << N_REPEAT_TRIAL_OF_PARAMETER << " trials of parameter " << param.name << "=" << param.value << endl;

			for (int folding = 0; folding < USE_FOLDINGS; folding++) { //USE_FOLDINGS usually = N_FOLDING (use less for quick test of param value range)

				unsigned int firstTestingSet = folding * sizeTestingSet;
				unsigned int lastTestingSet = firstTestingSet +  sizeTestingSet -1;

				//reset the weights for the plastic synapses to starting values
				classifier.initialiseWeights_DENSE_PN_AN();
				//update on the device
				classifier.updateWeights_PN_AN_on_device();

				//-----------------------------------------------------------------
				//RUN TRAINING
				string stage = "training";
				//printf( "%s %s, trial %u, folding %u, %s stage commenced..\n",param.name.c_str(),param.value.c_str(),(rptTrial+1),folding,stage.c_str());
				cout << param.name << "=" << param.value << COMMA << "trial " << (rptTrial+1) << SLASH << N_REPEAT_TRIAL_OF_PARAMETER << ", folding " << folding << ", " << stage << " stage commenced.." << endl;

				//reset to original (in case testing stage used diffferent (e.g. delayed) versions of recordings)
				classifier.recordingsDir = recordingsDir;
				classifier.recordingFilenameTemplate = recordingFilenameTemplate;
				cout << "Reset recording dir to " << classifier.recordingsDir << endl;
				cout << "Reset recording filename to " << classifier.recordingFilenameTemplate << endl;
				clearResponseVRCache();//clear cache as using different input data from testing stage

#ifdef CREATE_VR_FROM_EACH_LEARNING_SET
				//create and save a VR set from the current learning set only, using SOM/neural gas
				//This will therefore change on every folding
				clearResponseVRCache(); //removed cached responses to VR set from previous folding
				//create new VR set
				createVRsFromLearningSet(shuffledRecordings,firstTestingSet,lastTestingSet,global_NumVR);
#endif
				//load VRs into classifier
				classifier.load_VR_data();


				UINT trainingCount = 0;
				UINT trainingScore = 0;

				//Repeat the training set X times, for more exposure to early observations
				for (int rpt = 0; rpt < classifier.param_REPEAT_LEARNING_SET ; rpt++) {

					timer.startTimer();

					//for each recording in the training set
					for (int i=0; i<TOTAL_RECORDINGS; i++ ) {

						//leave out nominated test data set, only use training data set
						if (i<firstTestingSet || i>lastTestingSet)
						{
							UINT recordingIdx = shuffledRecordings[i];
							//cout << i << SPACE << recordingIdx << endl;


							#ifdef FLAG_OVERRIDE_RECORDING_SELECTED
							if (promptYN("Do want to override recording id selected?")){
								cout << "Enter id:" << endl;
								cin >> recordingIdx;
							}
							#endif

							classifier.setCorrectClass(recordingIdx); //do this now so that raster plots get the correct filename
							bool usePlasticity  = true;
							bool createRasterPlot = false;

							#ifdef FLAG_GENERATE_RASTER_PLOT_DURING_TRAINING_REPEAT
							if (rpt==FLAG_GENERATE_RASTER_PLOT_DURING_TRAINING_REPEAT) {
								createRasterPlot = recordingIdx % RASTER_FREQ == 0 ? true : false;
							}
							#endif

							string rasterFilename = generateRasterPlotFilename(recordingIdx, folding,  rptTrial,stage,param,i);
							bool classifiedCorrectly  = applyInputToClassifier(recordingIdx, RECORDING_TIME_MS, usePlasticity,createRasterPlot,rasterFilename,1, false);
							//cout << classifiedCorrectly << endl;


							#ifdef FLAG_OUTPUT_INDIVIDUAL_RESULTS
							//write classifier decision to file alongside correct class
							if (!(OUTPUT_MISTAKES_ONLY) || classifier.winningClass!=classifier.correctClass) {
								fprintf(individualResultsFile,"%s,%u,%s,%u,%u,%u\n", param.value.c_str(), folding, stage.c_str(), recordingIdx , classifier.winningClass, classifier.correctClass);
							}
							#endif

							#ifdef WRITE_VR_ACTIVATION_RESULTS
							string vrActivationFilePath;
							if (folding==0 && rpt==0) {
								string filename = replace(rasterFilename,"RasterData","VRActivation");
								vrActivationFilePath = writeVrActivationResults(recordingIdx,filename,perParamWtaPerformanceFilePN);
							}
							#endif


							#ifdef DISPLAY_RASTER_IMMEDIATELY
							if (createRasterPlot)  {
								const char * rasterPlotExtra = RASTER_PLOT_EXTRA;
								bool displayActivation = (rasterPlotExtra=="ACT");
								classifier.displayRasterPlot(classifier.outputDir,rasterFilename,RECORDING_TIME_MS,displayActivation, 1);
							}
							#endif

							trainingCount++;
							if (classifiedCorrectly) {
								trainingScore++;
							} else if (rpt==(classifier.param_REPEAT_LEARNING_SET-1)){ //last re-exposure - i.e. late learnning
								//track classify performance per recording
								recordingClassifiedWrongInTraining[recordingIdx] = recordingClassifiedWrongInTraining[recordingIdx] + 1;
								UINT recordingsPerClass = TOTAL_RECORDINGS / NUM_CLASSES;
								UINT classId = recordingIdx / recordingsPerClass;
								classClassifiedWrongInTraining[classId] = classClassifiedWrongInTraining[classId] + 1;

								//cout << "Wrong classification (right/answered): "  << classId << SPACE << classifier.winningClass << endl;
							}
						}

					}//end of recordings
					timer.stopTimer();
					printf( "Presented %u recordings for training in time:%f\n",TOTAL_RECORDINGS-sizeTestingSet, timer.getElapsedTime());
				} //end repeats


				float trainingPercent  =100*((float)trainingScore)/((float)trainingCount);
				printf( "%s=%s, Folding %u: Classifier training completed. Score:%u/%u (%f percent) \n",param.name.c_str(),param.value.c_str(),folding,trainingScore,trainingCount,trainingPercent);
				fprintf(perParamResultsFile,"%s,%u,%s,%u,%u,%f\n",param.value.c_str(),folding,stage.c_str(),trainingScore,trainingCount,trainingPercent);

				//-----------------------------------------------------------------
				//TESTING
				stage = "testing";

#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
				//switch to delayed complete recordings for testing
				classifier.recordingsDir.append("/").append(replace(DELAYED_RECORDING_SUBDIR,"%ss%", subsampleCode));
				classifier.recordingFilenameTemplate = replace(recordingFilenameTemplate,"SubSample","Delayed");
				cout << "Switched recording dir to " << classifier.recordingsDir << endl;
				cout << "Switched recording filename to " << classifier.recordingFilenameTemplate  << endl;
				clearResponseVRCache();//clear cache as using different input data from training stage
#endif

				cout << param.name << "=" << param.value << COMMA << "trial " << (rptTrial+1) << SLASH << N_REPEAT_TRIAL_OF_PARAMETER << ", folding " << folding << ", " << stage << " stage commenced.." << endl;

				UINT testCount = 0;
				UINT testScore = 0;

				//for each recording in the test set
				for (int i=firstTestingSet; i<=lastTestingSet; i++ )
					//for (int i=0; i<=testingSet.size(); i++ )
				{
					int recordingIdx = shuffledRecordings[i];
					classifier.setCorrectClass(recordingIdx);

#ifdef FLAG_GENERATE_RASTER_PLOT_DURING_TESTING
							bool createRasterPlot = true;
#else
							bool createRasterPlot = false;
#endif

					string filename_rasterPlot = generateRasterPlotFilename(recordingIdx, folding, rptTrial,stage,param,i);

#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
					//settings to use when testing against delayed real-time enose recordings
					UINT stretchTime = 1000; //stretch from ms into secs (ie. real time for enose data)
					float durationMs = LIMITED_DURATION_FOR_FULL_LENGTH_RECORDINGS; //use only 1st  N secs. using the whole 5 mins will take ages to create for no gain
					bool useEvaluationWindowForClassification = true;
#else
					//settings to use when testing against single input data samples (as used for training)
					UINT stretchTime = 1; //dont stretch
					float durationMs = RECORDING_TIME_MS; //use the whole recording
					bool useEvaluationWindowForClassification = false;
#endif

					bool classifiedCorrectly  = applyInputToClassifier(recordingIdx,durationMs, false,createRasterPlot,filename_rasterPlot,stretchTime,useEvaluationWindowForClassification);//no plasticity


					string vrActivationFilePath;
					if (folding==0) {
						vrActivationFilePath = writeVrActivationResults(recordingIdx,replace(filename_rasterPlot,"RasterData","VRActivation"),perParamWtaPerformanceFilePN);
					}
					if (createRasterPlot)  {
						#ifdef DISPLAY_RASTER_IMMEDIATELY
							const char * rasterPlotExtra = RASTER_PLOT_EXTRA;
							bool displayActivation = (rasterPlotExtra=="ACT");
							classifier.displayRasterPlot(classifier.outputDir,filename_rasterPlot,durationMs*stretchTime,displayActivation, stretchTime);
						#endif
					}


					//write classifier decision to file alongside correct class
#ifdef FLAG_OUTPUT_INDIVIDUAL_RESULTS
					if (!(OUTPUT_MISTAKES_ONLY) || classifier.winningClass!=classifier.correctClass) {
						fprintf(individualResultsFile,"%s,%u,%s,%u,%u,%u\n", param.value.c_str(), folding, stage.c_str(), recordingIdx , classifier.winningClass, classifier.correctClass);
					}
#endif

					testCount++;
					if (classifiedCorrectly) {
						testScore++;
					} else {
						//track classifiy performance per recording
						recordingClassifiedWrongInTest[recordingIdx] = recordingClassifiedWrongInTest[recordingIdx] + 1;
					}

#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
					//if any discrepancy between correct answer and the 2 types of evaluation then flag it
					int winnerOverEvaluationWindow = classifier.calculateOverallWinner(true);
					int winnerOverWholeRecording = classifier.calculateOverallWinner(false);
					if (!(classifier.correctClass==winnerOverEvaluationWindow && classifier.correctClass==winnerOverWholeRecording)) {
						cout << "Recording:" << recordingIdx << "  Correct class:" << classifier.correctClass <<  "  Eval.Winner:" << winnerOverEvaluationWindow << "(" << classifier.classEvaluationStartTimestep * stretchTime * DT  << " to " << classifier.classEvaluationEndTimestep * stretchTime * DT <<  ")  Overall Winner:" << winnerOverWholeRecording << endl;
					}
#endif

				}


				float testPercent = 100*((float)testScore)/((float)testCount);
				printf( "Folding %u: Classifier Testing completed. Score:%u/%u (%f percent) \n",folding, testScore,testCount,testPercent);
				fprintf(perParamResultsFile,"%s,%u,%s,%u,%u,%f\n",param.value.c_str(), folding,stage.c_str(),testScore,testCount,testPercent);

				//save the score for this folding for later processing (we will want the std dev)
				vecFoldingResults.push_back(testPercent);

				totalTestScore += testScore;
				totalTestCount += testCount;

				//try to detect error where GPU seems to pack up
				//checkDeviceError(testPercent);


				//if (promptYN("End of folding " + toString(folding) + ". Enter Y to continue.")) exit(0);

			} //goto next folding
			//end of foldings
		}//next repeat trial of same param value

		fclose(perParamResultsFile);
		perParamWtaPerformanceFilePN.close();

#ifdef FLAG_OUTPUT_INDIVIDUAL_RESULTS
		fclose(individualResultsFile);
#endif

		//analyze per param results
		float avg = getAverage(vecFoldingResults);
		float stdDev = getStdDev(vecFoldingResults,avg);

		float totalTestPercent = 100*((float)totalTestScore)/((float)totalTestCount);
		printf( "Classifier Training/Testing completed for %s=%s. Total Score:%u/%u (%f percent) \n", param.name.c_str(),param.value.c_str(),totalTestScore,totalTestCount,totalTestPercent);
		fprintf(overallResultsFile,"%s,%u,%f,%f\n",param.value.c_str(),global_NumVR,avg,stdDev);


	} //goto next param value
	//end of param exploration

	fclose(overallResultsFile);

	//output individual performance per recording and per class
	string path  = classifier.outputDir;
	path.append(SLASH).append(classifier.uniqueRunId).append(" PerformancePerRecording.txt");
	ofstream perRecPerformanceFile(path.c_str());
	perRecPerformanceFile << "RecIdx,WrongInLateTraining,WrongInTesting" << endl;
	for (int recIdx = 0; recIdx < TOTAL_RECORDINGS; ++recIdx) {
		perRecPerformanceFile << recIdx << COMMA << recordingClassifiedWrongInTraining[recIdx] << COMMA << recordingClassifiedWrongInTest[recIdx] << endl;
	}
	perRecPerformanceFile.close();

	path  = classifier.outputDir;
	path.append(SLASH).append(classifier.uniqueRunId).append(" PerformancePerClass.txt");
	ofstream perClassPerformanceFile(path.c_str());
	perClassPerformanceFile << "Class,WrongInLateTraining" << endl;
	for (int classIdx = 0; classIdx < NUM_CLASSES; ++classIdx) {
		perClassPerformanceFile << classIdx << COMMA << classClassifiedWrongInTraining[classIdx] << endl;
	}
	perClassPerformanceFile.close();


	//shut down device before classifier instance destroyed
	classifier.clearDownDevice();

	printTextFile(overallResultsFilename.str());
	printf( "End of Run %s\n", classifier.uniqueRunId.c_str());

	return 0;
}//END OF MAIN



