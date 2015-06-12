#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767

/*--------------------------------------------------------------------------
   Author: Alan Diamond

--------------------------------------------------------------------------
 Main entry point for the experiment using the classifier design based on Schmuker 2014 hardware classifier

 --------------------------------------------------------------------------*/
#include "experiment.h"
#include <time.h>
#include <algorithm> //for std:find

//for stat command (file info interrgation
#include <sys/types.h>
#include <sys/stat.h>

#ifndef S_ISDIR
#define S_ISDIR(mode)  (((mode) & S_IFMT) == S_IFDIR)
#endif

//this class does most the work, controlled by this experiment
Schmuker2014_classifier classifier;

//generic parameter struct to allow generic testing and reporting of different parameter settings
typedef struct Parameter {
	string name;
	string value;
} Parameter;


/*-----------------------------------------------------------------
Utility to determine if named directory exists
-----------------------------------------------------------------*/
bool directoryExists(string const &path) {
	struct stat st;
	if (stat(path.c_str(), &st) == -1) {
		return false;
	}
	return  S_ISDIR(st.st_mode);
}



/*-----------------------------------------------------------------
Utility to create a directory on Linux if not exisiting
-----------------------------------------------------------------*/
bool createDirectory(string path) {
	if (directoryExists(path)) {
		cout << "INFO: instructed to create directory " <<  path << " that already exists. Ignoring.." << endl;
		return false;
	} else {
		string cmd  = "mkdir \"" + path + "\"";
		return system(cmd.c_str());
	}
}


/*-----------------------------------------------------------------
Utilities to get the average and stdDev from a vector of floats
-----------------------------------------------------------------*/
float getAverage(vector<float> &v)
{
	float total = 0.0f;
	for (vector<float>::iterator it = v.begin(); it != v.end(); ++it)
	    total += *it;
	return total / v.size();
}

float getStdDev(vector<float> &v, float avg)
{
	float totalDiffSquared = 0.0f;
	for (vector<float>::iterator it = v.begin(); it != v.end(); ++it) {
		float diff = (avg - *it);
		totalDiffSquared += diff*diff;
	}
	float variance  = totalDiffSquared / v.size();
	return sqrtf(variance);
}

/*-----------------------------------------------------------------
Utility to write any text file to console
-----------------------------------------------------------------*/
bool printTextFile(string path)
{
 	ifstream file(path.c_str());
    if(!file.is_open()) return false;
    while (!file.eof()) {
    	string line;
    	file >> line;
    	printf("%s\n",line.c_str());
    }
    file.close();
    return true;
}

/*-----------------------------------------------------------------
Uses a timestamp plus network parameters used to create an id string unique to this run
-----------------------------------------------------------------*/

string getUniqueRunId()
{
	string timestamp = toString(time (NULL));
	string id = timestamp +
			"_" + classifier.datasetName;
	return id;
}


/*-----------------------------------------------------------------
Write to matching file the parameters used to create this run
-----------------------------------------------------------------*/
void outputRunParameters()
{
	string paramFilename = classifier.outputDir + divi + classifier.uniqueRunId + "_Run_Parameters.txt";
	FILE * file = fopen(paramFilename.c_str(),"w");
	fprintf(file,"DATASET_NAME\t\t%s\n",toString(DATASET_NAME).c_str());
	fprintf(file,"DT\t\t%f\n",DT);
	fprintf(file,"NUM_VR\t\t%d\n",NUM_VR);
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
	fprintf(file,"WEIGHT_DELTA_PN_AN\t\t%f\n",WEIGHT_DELTA_PN_AN);
	fprintf(file,"PLASTICITY_INTERVAL_MS\t\t%u\n",PLASTICITY_INTERVAL_MS);
	fprintf(file,"SPIKING_ACTIVITY_THRESHOLD_HZ\t\t%d\n",SPIKING_ACTIVITY_THRESHOLD_HZ);
	fprintf(file,"TOTAL_RECORDINGS\t\t%d\n",TOTAL_RECORDINGS);
	fprintf(file,"N_FOLDING\t\t%d\n",N_FOLDING);
	fprintf(file,"RECORDING_TIME_MS\t\t%d\n",RECORDING_TIME_MS);
	fclose(file);
}

/*-----------------------------------------------------------------
Load the specified recording and apply it to the classifier as a set of input rates
-----------------------------------------------------------------*/
bool applyInputToClassifier(UINT recordingIdx,bool usePlasticity)
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
	string filename_rasterPlot = classifier.datasetName + "_" + classifier.uniqueRunId +  "_Recording-" + toString(recordingIdx) + "_Class-" + toString(classifier.correctClass)  + "_Raster_plot_data.txt";

	/*
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	*/
	classifier.run(RECORDING_TIME_MS,filename_rasterPlot,usePlasticity);
	/*
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Run method completed in %f\n", time);
	*/

	int winner  = classifier.calculateOverallWinner();

	bool classifiedCorrectly =  winner == classifier.correctClass;

	string yesNo  = classifiedCorrectly ? "YES" : "NO";
	//cout << "Classified Correctly? " << yesNo << endl;

	return classifiedCorrectly;

}


/*--------------------------------------------------------------------------
 Utility function to determine if a passed vector of ints contains the specified value
 -------------------------------------------------------------------------- */
bool vectorContains(vector<int> &vec ,int lookingFor)
{
	vector<int>::iterator it = find(vec.begin(), vec.end(), lookingFor);
	return (it != vec.end());
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
	classifier.param_WEIGHT_RN_PN = WEIGHT_RN_PN;
	classifier.param_CONNECTIVITY_RN_PN = CONNECTIVITY_RN_PN;
	classifier.param_WEIGHT_WTA_PN_PN = WEIGHT_WTA_PN_PN;
	classifier.param_WEIGHT_WTA_AN_AN = WEIGHT_WTA_AN_AN;
	classifier.param_CONNECTIVITY_PN_PN = CONNECTIVITY_PN_PN;
	classifier.param_CONNECTIVITY_AN_AN = CONNECTIVITY_AN_AN;
	classifier.param_CONNECTIVITY_PN_AN = CONNECTIVITY_PN_AN;
	classifier.param_MIN_WEIGHT_PN_AN = MIN_WEIGHT_PN_AN;
	classifier.param_MAX_WEIGHT_PN_AN = MAX_WEIGHT_PN_AN;
	classifier.param_WEIGHT_DELTA_PN_AN  = WEIGHT_DELTA_PN_AN;
	classifier.param_PLASTICITY_INTERVAL_MS = PLASTICITY_INTERVAL_MS;
}

/*--------------------------------------------------------------------------
 This function is the entry point for running the experiment
 -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		fprintf(stderr, "usage: experiment <output-dir> \n");
		return 1;
	}

	string basename = toString(argv[1]);


	//-----------------------------------------------------------------
	//NETWORK INITIALISATION

	printf( "Network initialisation commenced..\n");

	#ifdef FLAG_RUN_ON_CPU
			printf("Simulation will be run on Host CPU\n");
	#else
			printf("Simulation will be run on Device GPU\n");
	#endif


	classifier.resetDevice(); //clear out any possible memory leaks etc from previous runs


	//set up file locations
	classifier.datasetName = DATASET_NAME;
	classifier.recordingsDir =   basename + divi + RECORDINGS_DIR;
	classifier.cacheDir = basename + divi + CACHE_DIR;
	createDirectory(classifier.cacheDir);
	classifier.outputDir = basename + divi + OUTPUT_DIR;
	createDirectory(classifier.outputDir);
	classifier.uniqueRunId = getUniqueRunId();
	classifier.startLog();

	printf( "Recordings input directory set to %s\n", classifier.recordingsDir.c_str());
	printf( "Cache directory set to %s\n", classifier.cacheDir.c_str());
	printf( "Output directory set to %s\n", classifier.outputDir.c_str());

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

	//initialise the set of weights for the SPARSE 1:1 subcluster-subcluster synapses RN-PN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_SPARSE_RN_PN();

	//initialise the set of weights for the DENSE subcluster-subcluster WTA synapses PN-PN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_WTA_PN_PN();

	//NB: This is now called at the start of each folding trial (see main method) to reset the plastic weights
	//initialise the set of weights for the DENSE plastic synapses PN-AN (GeNN has no automatic function for what we need)
	//classifier.initialiseWeights_DENSE_PN_AN();

	//initialise the set of weights for the DENSE subcluster-subcluster WTA synapses AN-AN (GeNN has no automatic function for what we need)
	classifier.initialiseWeights_WTA_AN_AN();

	//load set of virtual receptor points VR to be used to generate input levels
	classifier.load_VR_data();

	//allocate storage on CPU and GPU for the dataset of input rates to the poisson neurons
	classifier.initialiseInputData();


	//load classes labelling the recording data sets
	classifier.loadClassLabels();

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

	//set up a list of all the recording id's shuffled into a random order, this allows a simple linear split of the training and test data to achieve cross validation
	vector<int> shuffledRecordings; 
	for (int i=0; i<TOTAL_RECORDINGS; i++ ) {//enter all recordings in order
		shuffledRecordings.push_back(i);
	}
	//printf("Re-enable random shuffle!\n");//TODO reset
	random_shuffle(shuffledRecordings.begin(),shuffledRecordings.end());

	//re-seed the random number generator after shuffle fn (which may have used srand in its own way)
	srand(time(NULL));
	//srand(222);//TODO reset


	//-----------------------------------------------------------------
	//set up parameter exploration , if any
	//-----------------------------------------------------------------
	//string paramName = "SpkActivityThresholdHz";
	//int paramValues[] {3,4,6};

	//string paramName = "WeightDeltaPNAN";
	//float paramValues[] {0.005,0.01,0.025,0.05,0.1,0.2,0.4};

	string paramName = "PLASTICITY_INTERVAL_MS";
	//float paramValues[] {50,100,200,330,500,1000};
	float paramValues[]= {330};

	//-----------------------------------------------------------------

	//track the overall performance of the classifier
	string overallResultsFilename = classifier.outputDir + divi + classifier.uniqueRunId + "_Overall_Results_for_varying_" + paramName  + ".txt";
	FILE * overallResultsFile = fopen(overallResultsFilename.c_str(),"w");
	fprintf(overallResultsFile,"%s,AvgPercentScore,StdDev\n",paramName.c_str());

	//Run full cross validation, stepping through parameter values supplied
	for (int paramIndex = 0; paramIndex < sizeof(paramValues)/sizeof(paramValues[0]); paramIndex++) {

		//Apply next param value
		Parameter param = {paramName,toString(paramValues[paramIndex])};
		//classifier.param_SPIKING_ACTIVITY_THRESHOLD_HZ = paramValues[paramIndex];
		//classifier.param_WEIGHT_DELTA_PN_AN = paramValues[paramIndex];
		classifier.param_PLASTICITY_INTERVAL_MS = paramValues[paramIndex];

		//track the performance across each param setting, per folding
		string perParamResultsFilename = classifier.outputDir + divi + classifier.uniqueRunId + "_Totalled_Results_for_" + param.name + "-" + param.value  + ".txt";
		FILE * perParamResultsFile = fopen(perParamResultsFilename.c_str(),"w");
		fprintf(perParamResultsFile,"%s,Folding,Stage,Correct,OutOf,Percent\n",param.name.c_str());

		//track the detailed performance of the classifier
		string individualResultsFilename = classifier.outputDir + divi + classifier.uniqueRunId + "_Individual_Results_for_" + param.name + "-" + param.value  + ".txt";
		FILE * individualResultsFile = fopen(individualResultsFilename.c_str(),"w");
		fprintf(individualResultsFile,"%s,folding,recordingIdx,classifierSelectedClass,correctClass\n",param.name.c_str());
		printf("Individual training results will be saved to the file: %s\n", individualResultsFilename.c_str());

		int totalTestScore = 0;
		int totalTestCount = 0;

		vector<float> vecFoldingResults;//holder for the result of each folding, will be averaged/stdDev at the end of cross validation

		for (int folding = 0; folding < N_FOLDING; folding++) {

			unsigned int firstTestingSet = folding * sizeTestingSet;
			unsigned int lastTestingSet = firstTestingSet +  sizeTestingSet -1;

			//reset the weights for the plastic synapses PN-AN to a random selection
			classifier.initialiseWeights_DENSE_PN_AN();
			//update on the device
			classifier.updateWeights_PN_AN_on_device();


			//-----------------------------------------------------------------
			//RUN TRAINING
			string stage = "training";
			printf( "%s %s, folding %u, %s stage commenced..\n",param.name.c_str(),param.value.c_str(),folding,stage.c_str());


			UINT trainingCount = 0;
			UINT trainingScore = 0;

			//Repeat the training set X times, for more exposure to early observations
			for (int rpt = 0; rpt < REPEAT_LEARNING_SET ; rpt++) {

				timer.startTimer();

				//for each recording in the training set
				for (int i=0; i<TOTAL_RECORDINGS; i++ ) {

					//leave out nominated test data set, only use training data set
					if (i<firstTestingSet || i>lastTestingSet)
					{
						UINT recordingIdx = shuffledRecordings[i];

						bool usePlasticity  = true;
						bool classifiedCorrectly  = applyInputToClassifier(recordingIdx,usePlasticity);

						//write classifier decision to file alongside correct class
						fprintf(individualResultsFile,"%s,%u,%s,%u,%u,%u\n", param.value.c_str(), folding, stage.c_str(), recordingIdx , classifier.winningClass, classifier.correctClass);

						trainingCount++;
						if (classifiedCorrectly) trainingScore++;
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
			printf( "%s %s, folding %u, %s stage commenced..\n",param.name.c_str(),param.value.c_str(),folding,stage.c_str());

			UINT testCount = 0;
			UINT testScore = 0;

			printf( "Classifier testing commenced..\n");

			//for each recording in the test set
			for (int i=firstTestingSet; i<=lastTestingSet; i++ )
			{
				int recordingIdx = shuffledRecordings[i];
				bool classifiedCorrectly  = applyInputToClassifier(recordingIdx,false);//no plasticity


				//write classifier decision to file alongside correct class
				fprintf(individualResultsFile,"%s,%u,%s,%u,%u,%u\n", param.value.c_str(), folding, stage.c_str(), recordingIdx , classifier.winningClass, classifier.correctClass);

				testCount++;
				if (classifiedCorrectly) testScore++;
			}


			float testPercent = 100*((float)testScore)/((float)testCount);
			printf( "Folding %u: Classifier Testing completed. Score:%u/%u (%f percent) \n",folding, testScore,testCount,testPercent);
			fprintf(perParamResultsFile,"%s,%u,%s,%u,%u,%f\n",param.value.c_str(), folding,stage.c_str(),testScore,testCount,testPercent);

			//save the score for this folding for later processing (we will want the std dev)
			vecFoldingResults.push_back(testPercent);

			totalTestScore += testScore;
			totalTestCount += testCount;

		} //goto next folding
		//end of foldings

		fclose(perParamResultsFile);
		fclose(individualResultsFile);

		//analyze per param results
		float avg = getAverage(vecFoldingResults);
		float stdDev = getStdDev(vecFoldingResults,avg);

		float totalTestPercent = 100*((float)totalTestScore)/((float)totalTestCount);
		printf( "Classifier Training/Testing completed for %s=%s. Total Score:%u/%u (%f percent) \n", param.name.c_str(),param.value.c_str(),totalTestScore,totalTestCount,totalTestPercent);
		fprintf(overallResultsFile,"%s,%f,%f\n",param.value.c_str(),avg,stdDev);

	} //goto next param value
	//end of param exploration

	fclose(overallResultsFile);

	//shut down device before classifier instance destroyed
	//classifier.clearDownDevice();
	//This is done in the classifier destructor which also clears CPU memory etc

	printTextFile(overallResultsFilename);
	printf( "End of Run %s\n", classifier.uniqueRunId.c_str());

	return 0;
}//END OF MAIN



