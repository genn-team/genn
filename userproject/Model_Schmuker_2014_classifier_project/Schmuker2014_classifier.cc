/*--------------------------------------------------------------------------
   Author: Alan Diamond

   Institute: University of Sussex


   initial version:  Mar 1 2014

--------------------------------------------------------------------------*/

#ifndef _SCHMUKER2014_CLASSIFIER_
#define _SCHMUKER2014_CLASSIFIER_ //!< macro for avoiding multiple inclusion during compilation

/*--------------------------------------------------------------------------
	 Implementation of the Schmuker2014_classifier class.
  -------------------------------------------------------------------------- */

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h> 
#include <time.h>
#include "Schmuker2014_classifier.h"
#include "Schmuker_2014_classifier_CODE/runner.cc"
#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
	#include "sparseUtils.cc"
#endif

//--------------------------------------------------------------------------

Schmuker2014_classifier::Schmuker2014_classifier():
correctClass(0),winningClass(0),vrData(NULL),inputRatesSize(0),clearedDownDevice(false)
{

	d_maxRandomNumber = pow(2.0, (double) sizeof(uint64_t)*8-16); //work this out only once
	modelDefinition(model);


	//convenience vars
	countRN  = model.neuronN[0]; //size of receptor neurons RN population ( =  size of input )
	countPN  = model.neuronN[1]; //size of projection neurons PN population
	countAN  = model.neuronN[2]; //size of association neurons AN population
	//  timestepsPerRecording = RECORDING_TIME_MS / DT ; // = num timesteps contained in each data recording


	countPNAN  = countPN * countAN; //uses DENSE all-all (although N% will be zero weight)
	plasticWeights = new float [countPNAN];//2D array copy of PN AN weights which is updated by plasticity during a presentation

}

/*--------------------------------------------------------------------------
	Destructor method clears up memeory
 --------------------------------------------------------------------------*/

Schmuker2014_classifier::~Schmuker2014_classifier()
{

	fclose(log);

	// free all user arrays created on the heap
	free(inputRates);
	free(vrData);
	free(sampleDistance);
	free(classLabel);
	free(individualSpikeCountPN);
	free(clusterSpikeCountAN);
	free(overallWinnerSpikeCountAN);
	free(plasticWeights);


	//free mem allocated on the CPU and the GENN-created GPU data
	freeMem();


	if (!clearedDownDevice) clearDownDevice(); //don't try and clear device memory twice
}


void Schmuker2014_classifier::startLog()
{
	string logPath = outputDir + divi + uniqueRunId + "_Log.txt";
	this->log = fopen(logPath.c_str(),"w");

}

/*--------------------------------------------------------------------------
   REset CUDA device to ensure no mem leaks form previous run
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::resetDevice()
{
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());	// Wait for any GPU work to complete
	CHECK_CUDA_ERRORS(cudaDeviceReset());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());	// Wait for the reset
}

/*--------------------------------------------------------------------------
  Allocate the memory arrays used in the network on the host and device
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::allocateHostAndDeviceMemory()
{
	allocateMem();
	//initialise host side data (from GeNN 2 onwards, this also copies host data across to device)
	initialize();


	//cant do this now, don't know the sizes yet, will do individually, as connexctivty matrices are generated
	//allocateAllSparseArrays();
}

/*--------------------------------------------------------------------------
   Method for populating data on the device, e.g. synapse weight arrays
  -------------------------------------------------------------------------- */

void Schmuker2014_classifier::populateDeviceMemory()
{

	printf( "populating data on the device, e.g. synapse weight arrays..\n");

	//the sparse arrays have their own copy fn for some reason
	initializeAllSparseArrays();

	copyStateToDevice();


	printf( "..complete.\n");

}

/*--------------------------------------------------------------------------
 Clear device mem and reset device
 -------------------------------------------------------------------------- */
void Schmuker2014_classifier::clearDownDevice()
{
	// clean up memory allocated outside the model
	CHECK_CUDA_ERRORS(cudaFree((void*)d_inputRates));

	resetDevice();
	clearedDownDevice = true;

}

/*--------------------------------------------------------------------------
   Method for copying the current input dataset across to the device
  -------------------------------------------------------------------------- */

void Schmuker2014_classifier::update_input_data_on_device()
{

	// update device memory with set of input data
	CHECK_CUDA_ERRORS(cudaMemcpy(d_inputRates, inputRates, inputRatesSize, cudaMemcpyHostToDevice));

}

/*--------------------------------------------------------------------------
	convert an input rate (Hz) into a proprietary rateCode (a probability number for the poisson neuron model). 

  NB: This should move to the device code.
--------------------------------------------------------------------------*/

uint64_t  Schmuker2014_classifier::convertToRateCode(float inputRateHz)
{

	/*

  	Pr(spike)  = rateCode / D_MAX_RANDOM_NUM  =   rate(Hz) * DT (seconds)

	so,  rateCode = rate(Hz) * DT (seconds) * D_MAX_RANDOM_NUM

	 */

	double prob = (double)inputRateHz  * DT/1000.0; //timestep DT is held as ms

	if (prob > 1.0)  prob = 1.0;

	uint64_t rateCode = (uint64_t) (prob*d_maxRandomNumber);

	return rateCode;

}


/* --------------------------------------------------------------------------
load set of virtual receptor points VR to be used to generate input levels 
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::load_VR_data()
{
	string filename = recordingsDir + divi + VR_DATA_FILENAME;

	printf("Loading VR data from file %s\n", filename.c_str());
	UINT size = NUM_VR * NUM_FEATURES;
	vrData = new float [size];

	bool ok = loadArrayFromTextFile(filename,vrData,",",size,data_type_float);
	if (!ok) {
		fprintf(stderr,"Failed to load VR file %s\n", filename.c_str());
		exit(1);
	} else {
		printf("VR data loaded from file: %s\n", filename.c_str());
	}
	//checkContents("VR data", vrData,size,4,data_type_float,5);
}


/* --------------------------------------------------------------------------
utility to load an array of the designated data_type from a delimited text file
-------------------------------------------------------------------------- */
bool Schmuker2014_classifier::loadArrayFromTextFile(string path, void * array, string delim, UINT arrayLen, data_type dataType)
{
	using namespace std;

	float * floatArray = (float*)array;
	UINT * 	uintArray = (UINT*)array;
	int * intArray = (int*)array;
	double * doubleArray = (double*)array;

	ifstream file(path.c_str());
	if(!file.is_open()) return false;

	string line;
	UINT index = 0;

	while (!file.eof()) {
		string line;
		file >> line;
		size_t pos1 = 0;
		size_t pos2;
		do {
			pos2 = line.find(delim, pos1);
			string datum = line.substr(pos1, (pos2-pos1));

			switch (dataType) {
			case(data_type_float):
							floatArray[index] = atof(datum.c_str());
			break;
			case(data_type_uint):
							uintArray[index] = atoi(datum.c_str());;
			break;
			case(data_type_double):
							doubleArray[index] = atof(datum.c_str());;
			break;
			case(data_type_int):
							intArray[index] = atoi(datum.c_str());;
			break;
			default:
				fprintf(stderr, "ERROR: Unknown data type:%d\n", dataType);
				exit(1);
			}

			pos1 = pos2+1;
			index ++;
		} while(pos2!=string::npos && index<arrayLen);
		if (index==arrayLen) break;
	}
	file.close();
	return true;
}


/* --------------------------------------------------------------------------
debug utility to output a line of dashes
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::printSeparator()
{
	printf("--------------------------------------------------------------\n");
}


/* --------------------------------------------------------------------------
debug utility to check contents of an array
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints ) {
	checkContents( title, array,howMany, displayPerLine,dataType,decimalPoints,"\t" );//default delim tab
}
void Schmuker2014_classifier::checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints, string delim )
{

	float * floatArray = (float*)array;
	UINT * 	uintArray = (UINT*)array;
	int * intArray = (int*)array;
	double * doubleArray = (double*)array;

	printSeparator();
	printf("\n%s\n", title.c_str());
	for (UINT i = 0; i < howMany; ++i) {

		switch (dataType) {
		case(data_type_float):
					printf("%*.*f%s", 1,decimalPoints, floatArray[i],delim.c_str());
		break;
		case(data_type_uint):
					printf("%d%s", uintArray[i],delim.c_str());
		break;
		case(data_type_double):
					printf("%*.*f%s", 1,decimalPoints, doubleArray[i],delim.c_str());
		break;
		case(data_type_int):
					printf("%d%s", intArray[i],delim.c_str());
		break;
		default:
			fprintf(stderr, "ERROR: Unknown data type:%d\n", dataType);
			exit(1);

		}
		if (i % displayPerLine == displayPerLine -1 ) printf("\n");

	}
	printSeparator();
}

/* --------------------------------------------------------------------------
load set of classes labelling the recordings
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::loadClassLabels()
{
	string filename = recordingsDir + "/ClassLabelling.csv";

	this->classLabel = new UINT [TOTAL_RECORDINGS];

	UINT recordIdLabellings[TOTAL_RECORDINGS *2]; //stores content of file, (recordIdx,ClassID) pairs

	loadArrayFromTextFile(filename,&recordIdLabellings,",",TOTAL_RECORDINGS*2,data_type_uint);
	for (int i = 0; i < TOTAL_RECORDINGS; ++i) {
		UINT recordingIdx = recordIdLabellings[i*2];
		UINT classId = recordIdLabellings[i*2+1];
		classLabel[recordingIdx] = classId;
	}
	printf( "Class Labels loaded from file: %s\n",filename.c_str());
	//checkContents("Class Labels", classLabel,TOTAL_RECORDINGS,TOTAL_RECORDINGS,data_type_uint,0);
}


/* --------------------------------------------------------------------------
get the set of input rate data for the recording 
(this will be first generated from the sensor data and the VR set, then cached in a uniquely named file)
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::generate_or_load_inputrates_dataset(unsigned int recordingIdx)
{
	string cacheFilename = cacheDir + divi + "InputRates_created_from_recording_no._" + toString(recordingIdx) +  "_with_" + toString(NUM_VR) + "_VRs" + ".cache";
	FILE *f= fopen(cacheFilename.c_str(),"r");
	if (f==NULL)  {
		//file doesn't exist
		generate_inputrates_dataset(recordingIdx);
		//write inputRates to cache file
		FILE *f= fopen(cacheFilename.c_str(),"w");
		fwrite(inputRates,inputRatesSize ,1,f);		
		fclose(f);
		printf( "Input rates for recording %d written to cache file.\n", recordingIdx);
		//checkContents("Input Rates written to cache",inputRates,countRN*5,countRN,data_type_uint,0);

	} else { //cached version exists
		//load inputrates from cache file
		size_t dataRead = fread(inputRates,inputRatesSize ,1,f);
		fclose(f);
		//printf( "Input rates for recording %d loaded from cache file.\n", recordingIdx);
		//checkContents("Input Rates " + toString(recordingIdx),inputRates,countRN,countRN,data_type_uint,0);
	}
}

/* --------------------------------------------------------------------------
generate simulated timeseries data from single static data file e.g. Iris or MNIST data
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::generateSimulatedTimeSeriesData()
{
	//load static data file
	string staticDataFilename = recordingsDir + "/IrisFeatureVectors.csv";
	float * featureVectors = new float[TOTAL_RECORDINGS*NUM_FEATURES];
	bool ok = loadArrayFromTextFile(staticDataFilename,featureVectors,",",TOTAL_RECORDINGS*NUM_FEATURES,data_type_float);
	if(!ok) {
		fprintf(stderr,"Failed to load static data file: %s\n", staticDataFilename.c_str());
		exit(1);
	}

	for (int recordingIdx = 0; recordingIdx < TOTAL_RECORDINGS; ++recordingIdx) {
		string recordingFilename = getRecordingFilename(recordingIdx);
		FILE * file = fopen(recordingFilename.c_str(),"w");
		for (int ts = 0; ts < timestepsPerRecording; ++ts) { //for each timestep, output the same feature vector
			for (int feat = 0; feat < NUM_FEATURES; ++feat) {
				int indexToAccess = recordingIdx*NUM_FEATURES + feat;
				fprintf(file,"%f",featureVectors[indexToAccess]);
				if (feat==NUM_FEATURES-1) {//last one in the row
					fprintf(file,"\n");
				} else {
					fprintf(file,",");//delim
				}
			}
		}
		fclose(file);
		printf("Created simulated timeseries data file: %s\n", recordingFilename.c_str());
	}
}

/* --------------------------------------------------------------------------
Knows how to build the individual filenames used for the sensor data
-------------------------------------------------------------------------- */
string Schmuker2014_classifier::getRecordingFilename(UINT recordingIdx)
{
	return this->recordingsDir + divi + this->datasetName + " SensorRecording" + toString(recordingIdx) +  ".csv";
}

/* --------------------------------------------------------------------------
generate the set of input rate data for the recording from the sensor data and the VR set
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::generate_inputrates_dataset(unsigned int recordingIdx)
{


	//open sensor data file
	UINT length = NUM_FEATURES * timestepsPerRecording;
	float * recording = new float[length];
	string recordingFilename = getRecordingFilename(recordingIdx);
	loadArrayFromTextFile(recordingFilename,recording,",",length,data_type_float);

	//for each data point read from file, get a distance metric to each VR point, these will become the input rate levels to the network
	for (UINT ts=0 ; ts < timestepsPerRecording; ts++) {
		addInputRate(&recording[ts*NUM_FEATURES],ts);
	}	
	delete[] recording;
}

/* --------------------------------------------------------------------------
get a handle to the specified sensor recording file
-------------------------------------------------------------------------- */
FILE * Schmuker2014_classifier::openRecordingFile(UINT recordingIdx)
{
	string recordingFilename = recordingsDir + divi + datasetName + " SensorRecording" + toString(recordingIdx) +  ".data";
	FILE *f= fopen(recordingFilename.c_str(),"r");
	if (f==NULL)  {
		//file doesn't exist or cant read
		cout << "ERROR! failed to open recording file " << recordingFilename + "\n" ;
		exit(1);
	}
	return f;
}


/* --------------------------------------------------------------------------
extend the set of input rate data by one, using the response from the set of VRs to a single sample of sensor data (vector in feature space)
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::addInputRate(float * samplePoint,UINT timeStep)
{
	UINT inputRatesDataOffset = timeStep * countRN; //get ptr to start of vector of rates (size = countRN) for this time step

	for (UINT vr=0 ; vr < NUM_VR; vr++) {//step through the set of VRs

		float * vrPoint = &vrData[vr * NUM_FEATURES]; //get a ptr to start of the next VR vector in the set

		//Calculate the response of the VR to the sample point
		float vrResponse  = calculateVrResponse(samplePoint,vrPoint);

		//scale the firing rate (Hz) from the distance 
		float rateHz  = param_MIN_FIRING_RATE_HZ + vrResponse * (float)(param_MAX_FIRING_RATE_HZ - param_MIN_FIRING_RATE_HZ);

		//convert Hz to proprietary rate code used on the device ( this code should move to device code)
		uint64_t rateCode  = convertToRateCode(rateHz);


		//fill in a clusters worth with the same rate (one VR excites one cluster in RN)
		for (UINT i=0; i < CLUST_SIZE_RN; i++) {
			inputRates[inputRatesDataOffset + vr*CLUST_SIZE_RN + i] = rateCode;
		}
	}
}

/* --------------------------------------------------------------------------
Calculate the response of a given VR to a single sample of sensor data (vector in feature space)
-------------------------------------------------------------------------- */
float Schmuker2014_classifier::calculateVrResponse(float * samplePoint, float * vrPoint)
{
	//get the Manhattan distance metric
	float distance = getManhattanDistance(samplePoint, vrPoint, NUM_FEATURES);
	//normalise to a number in range 0..1
	UINT max = 0;
	UINT min = 1;
	float response  =  1 - (distance - getSampleDistance(min)) / (getSampleDistance(max)  - getSampleDistance(min)) ; 
	return response;

}

/* --------------------------------------------------------------------------
Get the max or minimum absolute "distance" existing between 2 points in the full sensor recording data set
If no cached version exists then calculate by interrogating the full data set
-------------------------------------------------------------------------- */
float Schmuker2014_classifier::getSampleDistance(UINT max0_min1)
{
	if (this->sampleDistance == NULL) {//not yet accessed

		//look for saved data to load

		this->sampleDistance = new float[2]; //will store the max and min values
		string path = recordingsDir + "/MaxMinSampleDistances.csv";
		bool ok = loadArrayFromTextFile(path,sampleDistance,",",2,data_type_float);
		if (ok)  {
			printf( "Max (%f) and Min (%f) Sample Distances loaded from file %s.\n",sampleDistance[0],sampleDistance[1],path.c_str());
		} else {
			//file doesn't exist yet, so need to generate values
			setMaxMinSampleDistances();

			//now write to file
			FILE *f= fopen(path.c_str(),"w");
			fprintf(f,"%f,%f",sampleDistance[0],sampleDistance[1]);
			fclose(f);
			printf( "Max and Min Sample Distances written to file: %s.\n",path.c_str());

		}


	} 
	return this->sampleDistance[max0_min1];

}

/* --------------------------------------------------------------------------
Load full data set to find  the max and minimum absolute "distance" existing between 2 points in the sensor recording data set
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::setMaxMinSampleDistances()
{
	//reset values (negative values imply not set)
	this->sampleDistance[0]=-1.0;
	this->sampleDistance[1]=-1.0;


	//load all recordings files into one giant array (because we need to compare all vs all)
	printf( "Interrogating the full data set to find  the max and minimum distances between sample points..\n");

	//UINT recordingLength = NUM_FEATURES * 1; //shortcut for iris data
	UINT recordingLength = NUM_FEATURES * timestepsPerRecording;

	//allocate data to load ALL recordings into
	float * allRecordings = new float [TOTAL_RECORDINGS * recordingLength];

	float * singleRecording = new float[recordingLength];

	for (UINT recordingIdx=0; recordingIdx<TOTAL_RECORDINGS; recordingIdx++) {
		//open sensor data file
		loadArrayFromTextFile(getRecordingFilename(recordingIdx),singleRecording,",",recordingLength,data_type_float);
		for (int i = 0; i < recordingLength; ++i) {
			allRecordings[recordingIdx*recordingLength + i] = singleRecording[i];//copy into big array
		}
	}

	//using big array..
	for (int startAtSample = 0; startAtSample < (TOTAL_RECORDINGS * recordingLength)-1; startAtSample++) {
		findMaxMinSampleDistances(allRecordings,startAtSample,TOTAL_RECORDINGS * timestepsPerRecording);
		//findMaxMinSampleDistances(allRecordings,startAtSample,TOTAL_RECORDINGS * 1);///shortcut for Iris data, only use one sample per recording
	}

	//clear up
	delete [] singleRecording;
	delete [] allRecordings	;

	printf( "..complete");

}

/* --------------------------------------------------------------------------
Go through the full data set to find  the max and minimum absolute "distance" existing between 2 points in the sensor recording data set
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::findMaxMinSampleDistances(float * samples, UINT startAt, UINT totalSamples)
{
	//we need to do an all vs all distance measurement to find the two closest and the two furthest points

	float * sampleA  = &samples[startAt * NUM_FEATURES]; //remember each sample comprises NUM_FEATURES floats

	for (UINT i = startAt+1; i < totalSamples; i++) {
		float * sampleB  = &samples[i * NUM_FEATURES]; 
		printf("Sample B (%f,%f,%f,%f)\n",sampleB[0],sampleB[1],sampleB[2],sampleB[3]);
		float distanceAB = getManhattanDistance(sampleA,sampleB,NUM_FEATURES);
		if (distanceAB > this->sampleDistance[0] || this->sampleDistance[0]<0.0) {
			this->sampleDistance[0] =  distanceAB ; //update max
			printf("Max updated! sample:%d vs sample:%d    max:%f min:%f\n",startAt,i,this->sampleDistance[0],this->sampleDistance[1]);
		} else if (distanceAB < this->sampleDistance[1]|| this->sampleDistance[1]<0.0) {
			this->sampleDistance[1] =  distanceAB ; //update min
			printf("Min updated! sample:%d vs sample:%d    max:%f min:%f\n",startAt,i,this->sampleDistance[0],this->sampleDistance[1]);
		}

	}
	//now call fn recursively with next start point (we have done A:B , A:C, A:D etc, now we need to do B:C, B:D, etc)
	//if (startAt+2 < totalSamples) {
	//	findMaxMinSampleDistances(samples, startAt+1);
	//}
}

/* --------------------------------------------------------------------------
calculate the Manhattan distance metric between two vectors of floats denoting points in for example, feature space
The "Manhattan" distance is simply the sum of all the co-ordinate differences
-------------------------------------------------------------------------- */
float Schmuker2014_classifier::getManhattanDistance(float * pointA,float * pointB, UINT numElements)
{
	float totalDistance = 0.0;
	for (UINT i=0 ; i < numElements; i++) {
		float a  = pointA[i];
		float b = pointB[i];
		float pointDistance = abs(a - b);
		if (pointDistance > 20) {
			fprintf(stderr,"ERROR? Manhattan distance %f v. large.\n",pointDistance);
		}
		totalDistance = totalDistance + pointDistance;
	}
	return totalDistance;
}


/*--------------------------------------------------------------------------
   Method for simulating the model for a specified duration in ms
  -------------------------------------------------------------------------- */

void Schmuker2014_classifier::run(float runtimeMs, string filename_rasterPlot,bool usePlasticity)
{
#ifdef FLAG_GENERATE_RASTER_PLOT
	string path = outputDir + divi + filename_rasterPlot;
	FILE *rasterFile = fopen(path.c_str(),"w");
	if (rasterFile==NULL) {
		fprintf(stderr,"Unable to open raster file for writing %s\n",path.c_str());
		exit(1);
	}
#endif


	int timestepsRequired = (int) (runtimeMs/DT);

	int timestepsBetweenPlasticity = (int) (param_PLASTICITY_INTERVAL_MS/DT);
	int nextPlasticityUpdate = timestepsBetweenPlasticity - 1;

	//t = 0.0f; //reset elapsed time

	//reset spike count totals across the current plasticity window
	resetIndividualSpikeCountPN();
	resetClusterSpikeCountAN();

	for (int timestep= 0; timestep < timestepsRequired; timestep++) {

		offsetRN = timestep * countRN ; //units = num of unsigned ints

#ifdef FLAG_RUN_ON_CPU
		//step simulation by one timestep on CPU
		stepTimeCPU();
#else
		//step simulation by one timestep on GPU
		stepTimeGPU();
		getSpikesFromGPU(); //need these to calculate winning class etc (and for raster plots)
		//cudaDeviceSynchronize();
#endif

		//uncomment this to divert other state data into raster file
		//copyStateFromDevice();
		//fprintf(rasterFile,"%f,%f\n",t,VPN[0]);

		updateIndividualSpikeCountPN(); //total up the spikes on every PN neuron during the run time. Needed for the plasticity rule.
		updateClusterSpikeCountAN(); // total up the spikes witihn each AN cluster during the run. Needed to determine the winning output cluster class


		if (usePlasticity && timestep == nextPlasticityUpdate) {
			//update plastic weights
			applyLearningRuleSynapses(plasticWeights);
			resetIndividualSpikeCountPN();
			resetClusterSpikeCountAN();
			nextPlasticityUpdate += timestepsBetweenPlasticity;
		}

#ifdef FLAG_GENERATE_RASTER_PLOT
		outputSpikes(rasterFile,"\t");
#endif

		//increment global time elapsed
		t+= DT;
	}
	updateWeights_PN_AN(); //at end of presentation, copy plasticity changes over to the actual synapses

	//completed simulation run
#ifdef FLAG_GENERATE_RASTER_PLOT
	fclose(rasterFile);
#endif

}

/* --------------------------------------------------------------------------
   output functions
  -------------------------------------------------------------------------- 
   Method for copying all spikes of the last time step from the GPU

  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
  -------------------------------------------------------------------------- */

void Schmuker2014_classifier::getSpikesFromGPU()
{
#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
	copySpikeNFromDevice(); //GeNN 1.0 does this as part of copySpikesFromDevice();
#endif
	copySpikesFromDevice();
}

/*--------------------------------------------------------------------------
   Method for writing the spikes occurred in the last time step to a file
           This can be used to create a raster plot (by providing coordinates (t,index) of points to plot 
           File format: time|idx of spiked neuron|\n
  -------------------------------------------------------------------------- */

void Schmuker2014_classifier::outputSpikes(FILE *f, string delim )
{

	/*
	printf("RN spikes: %u\n",glbSpkCntRN[0]);
	printf("PN spikes: %u\n",glbSpkCntPN[0]);
	printf("AN spikes: %u\n",glbSpkCntAN[0]);
	 */

	for (int i= 0; i < glbSpkCntRN[0]; i++) {
		fprintf(f, "%f%s%d\n", t, delim.c_str(), glbSpkRN[i]);
	}

	for (int i= 0; i < glbSpkCntPN[0]; i++) {
		fprintf(f,  "%f%s%d\n", t, delim.c_str(), countRN + glbSpkPN[i] );
	}

	for (int i= 0; i < glbSpkCntAN[0]; i++) {
		fprintf(f, "%f%s%d\n", t, delim.c_str(), countRN + countPN + glbSpkAN[i]);
	}
}

/*--------------------------------------------------------------------------
   overwrite actual synapse weights with any plastic changes made
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::updateWeights_PN_AN()
{
	memcpy(gPNAN,plasticWeights,countPNAN * sizeof(float));

	//update to new weights on the device
	#ifndef FLAG_RUN_ON_CPU
		updateWeights_PN_AN_on_device();
	#endif
}

/*--------------------------------------------------------------------------
   Method for updating the conductances of the learning synapses between projection neurons PNs and association ANs on the device memory
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::updateWeights_PN_AN_on_device()
{
#ifdef DEVICE_MEM_ALLOCATED_ON_DEVICE
	void *d_ptrPNAN;
	cudaGetSymbolAddress(&d_ptrPNAN, d_gPNAN);
	CHECK_CUDA_ERRORS(cudaMemcpy(d_ptrPNAN, gPNAN, countPNAN*sizeof(float), cudaMemcpyHostToDevice));
#else
	//New version. device mem allocated using cudaMalloc from the host side
	CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNAN, gPNAN, countPNAN*sizeof(float), cudaMemcpyHostToDevice));
#endif
}


/*--------------------------------------------------------------------------
   Initialise the set of weights for the SPARSE 1:1 subcluster-subcluster synapses RN-PN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::initialiseWeights_SPARSE_RN_PN()
{


	float * tmp_gRNPN = gRNPN; //for DENSE, point at the predefined all-all-array

	//for each synapse from population X to pop Y
	for (UINT x= 0; x < countRN; x++) {
		for (UINT y= 0; y < countPN; y++) {
			UINT synapseIdx = x * countPN + y;
			tmp_gRNPN[synapseIdx] = 0.0f; //default to no connection
			if  (getClusterIndex(x,CLUST_SIZE_RN) == getClusterIndex(y,CLUST_SIZE_PN)) { //same cluster
				if (randomEventOccurred(param_CONNECTIVITY_RN_PN)) {
					tmp_gRNPN[synapseIdx]  = param_WEIGHT_RN_PN * param_GLOBAL_WEIGHT_SCALING;
				}
			}
		}
	}
	//checkContents("RNPN Connections",tmp_gRNPN,countRN * countPN,countPN,data_type_float,2);


	printf("Initialised weights for  SPARSE 1:1 subcluster-subcluster synapses RN-PN.\n");
}



/*--------------------------------------------------------------------------
   initialise the set of weights for the DENSE subcluster-subcluster WTA synapses PN-PN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::initialiseWeights_WTA_PN_PN()
{
	createWTAConnectivity(gPNPN, countPN, CLUST_SIZE_PN, param_WEIGHT_WTA_PN_PN * param_GLOBAL_WEIGHT_SCALING, param_CONNECTIVITY_PN_PN);
	//checkContents("PN PN Connections",gPNPN,countPN*countPN,countPN,data_type_float,1," ");
}

/*  -------------------------------------------------------------------------- 
  initialise the set of weights for the DENSE plastic synapses PN-AN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::initialiseWeights_DENSE_PN_AN()
{
	//uses a DENSE connection matrix (i.e. all-to-all)
	//The paper specifies an N% connectivity (e.g. 50%) , so N% of the weights will be fixed at zero
	for (UINT i= 0; i < countPN; i++) {
		for (UINT j= 0; j < countAN; j++) {
			if (randomEventOccurred(param_CONNECTIVITY_PN_AN)) {
				//set weight randomly between limits
				float weight  = param_MIN_WEIGHT_PN_AN  +  getRand0to1() * (param_MAX_WEIGHT_PN_AN - param_MIN_WEIGHT_PN_AN);
				gPNAN[i*countAN + j] = weight * param_GLOBAL_WEIGHT_SCALING;
			} else {
				gPNAN[i*countAN + j] = 0.0; //zero weighted = no connection
			}
		}
	}
	//checkContents("PN-AN Connections",gPNAN,countPN*countAN,countAN,data_type_float,3);

	//initialise plastic weights as a copy of PN-AN. These weights are updated periodically during a presention but not used in classifier until end of presentation
	memcpy(plasticWeights,gPNAN,countPNAN * sizeof(float));

}
/*  --------------------------------------------------------------------------   
  //initialise the set of weights for the DENSE subcluster-subcluster WTA synapses AN-AN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::initialiseWeights_WTA_AN_AN()
{
	createWTAConnectivity(gANAN, countAN, CLUST_SIZE_AN, param_WEIGHT_WTA_AN_AN * param_GLOBAL_WEIGHT_SCALING, param_CONNECTIVITY_AN_AN);
}

/*  --------------------------------------------------------------------------   
  allocate storage on CPU and GPU for the set of input data (rates) to be processed
  -------------------------------------------------------------------------- */
void Schmuker2014_classifier::initialiseInputData()
{
	//allocate storage for the set of input data (rates) to be processed
	this->inputRatesSize = countRN * sizeof(uint64_t) * timestepsPerRecording;

	//allocate memory on the CPU to hold the current input dataset
	inputRates = new uint64_t[timestepsPerRecording * countRN];

	//allocate corresponding memory on the GPU device to hold the input dataset
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_inputRates, inputRatesSize));
#ifdef FLAG_RUN_ON_CPU
	ratesRN= inputRates;
#else
	ratesRN= d_inputRates;
#endif

	printf("Memory allocated for input rates on CPU and GPU.\n");
}

/* --------------------------------------------------------------------------
set the integer code for the class labelled to the specified recording
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::setCorrectClass(UINT recordingIdx)
{
	this->correctClass = this->classLabel[recordingIdx];
}

/* --------------------------------------------------------------------------
Use the results of the classifier to update the weights of to the outputs
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::applyLearningRuleSynapses(float * synapsesPNAN)
{

	int currentWinner = calculateCurrentWindowWinner();

	//for each synapse from PN to AN	
	for (UINT pnIdx= 0; pnIdx < countPN; pnIdx++) {

		//find out  spiking rate / sec of preSynaptic neuron during the trial, i.e. get spikecount/duration
		float spikerate = ((float)individualSpikeCountPN[pnIdx]) / (param_PLASTICITY_INTERVAL_MS/1000.0);
		UINT pnCluster  =  getClusterIndex(pnIdx,CLUST_SIZE_PN);
		//fprintf(log,"%u,%u,%f\n",pnCluster,pnIdx,spikerate);

		// if rate greater than SPIKING_ACTIVITY_THRESHOLD_HZ (set to 35 sp/s in the paper) then this neuron response to its VR may have contributed to winning class cluster
		if (spikerate >= param_SPIKING_ACTIVITY_THRESHOLD_HZ) {

			for (UINT anIdx= 0; anIdx < countAN; anIdx++) {
				UINT synapseIdx = pnIdx*countAN + anIdx;
				float currWeight  = 0.0;
				currWeight  = synapsesPNAN[synapseIdx] ;

				//if weight != zero (ignore zeroes because this simulates no connection)
				if (currWeight > 0.001) {

					//get the class id assigned to the cluster containing the postSynaptic neuron
					UINT anClass = getClassCluster(anIdx);

					if (anClass == currentWinner) { //if this synapse points into the cluster that "won" the classification in this plasticity window

						//if the current winning class is the correct class then increase the weight, otherwise decrease it
						float polarity  = currentWinner == this->correctClass ? 1.0 : -1.0;
						float weightDelta  = param_WEIGHT_DELTA_PN_AN * param_PLASTICITY_INTERVAL_MS / RECORDING_TIME_MS;
						float weightChange  = polarity * weightDelta * param_GLOBAL_WEIGHT_SCALING;

						//get new weight 
						currWeight += weightChange;

						//constrain to limits w-min, w-max 
						if (currWeight > param_MAX_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING) currWeight = param_MAX_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING;
						if (currWeight < param_MIN_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING) currWeight = param_MIN_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING;

						//update the synapse array with new value
						synapsesPNAN[synapseIdx] = currWeight;

						//string direction = polarity>0?"increased":"decreased";
						//printf("Weight %s from cluster %u to class cluster %u\n", direction.c_str(), pnCluster,anClass);

					}
				}
			} //next synapse
		}
	}
}

/* --------------------------------------------------------------------------
return the id of the class (AN neuron cluster) that the specified neuron belongs to
-------------------------------------------------------------------------- */
UINT Schmuker2014_classifier::getClassCluster(UINT anIdx)
{
	return getClusterIndex(anIdx,CLUST_SIZE_AN) ;
}

/* --------------------------------------------------------------------------
return the index of the neuron cluster that a specified neuron belongs to
-------------------------------------------------------------------------- */
UINT Schmuker2014_classifier::getClusterIndex(UINT neuronIndex, UINT clusterSize)
{
	return neuronIndex / clusterSize ;
}

/* --------------------------------------------------------------------------
fill in the passed connectivity array to create a WTA structure between clusters of a specified size
DENSE Connections are made between neurons (with a specified probability) unless they are in the same cluster
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::createWTAConnectivity(float * synapse, UINT populationSize, UINT clusterSize, float synapseWeight, float probability)
{
	//for each synapse from population X to itself (Y)	
	for (UINT x= 0; x < populationSize; x++) {
		for (UINT y= 0; y < populationSize; y++) {
			UINT synapseIdx = x * populationSize + y;
			synapse[synapseIdx] = 0.0; //default to no connection
			if  (getClusterIndex(x,clusterSize) != getClusterIndex(y,clusterSize)) { //different clusters
				if (randomEventOccurred(probability)) {
					synapse[synapseIdx]  = synapseWeight;
				}
			}
		}
	} 		
}

/* --------------------------------------------------------------------------
create a pseudorandom event with a specified probability
-------------------------------------------------------------------------- */
bool Schmuker2014_classifier::randomEventOccurred(float probability) 
{
	if (probability <0.0 || probability > 1.0)  {
		fprintf(stderr, "randomEventOccurred() fn. ERROR! INVALID PROBABILITY SPECIFIED AS %f.\n", probability);
		exit(1);
	} else {
		return getRand0to1() <= probability;
	}
}



/* --------------------------------------------------------------------------
generate a random number between 0 and 1 inclusive 
-------------------------------------------------------------------------- */
float Schmuker2014_classifier::getRand0to1()
{
	return ((float)rand()) / ((float)RAND_MAX) ;
}

/* --------------------------------------------------------------------------
Use spike counts in AN clusters after a run to decide the winning class , store result in winningClass instance var
-------------------------------------------------------------------------- */
UINT Schmuker2014_classifier::calculateWinner(unsigned int * clusterSpikeCount)
{
	UINT winner = 0;
	UINT max = 0;

	for(UINT clustIdx=0; clustIdx<NUM_CLASSES; clustIdx++) {
		if (clusterSpikeCount[clustIdx] > max) {
			max = clusterSpikeCount[clustIdx];
			winner = clustIdx;
		} else if (clusterSpikeCount[clustIdx] == max) { //draw
			//toss a coin (we don't want to always favour the same cluster)
			if (randomEventOccurred(0.5)) winner = clustIdx;
		}
	}
	return winner;
}


UINT Schmuker2014_classifier::calculateOverallWinner()
{
	UINT winner = calculateWinner(overallWinnerSpikeCountAN);
	this->winningClass = winner;

	return winner;
}


UINT Schmuker2014_classifier::calculateCurrentWindowWinner() {

	UINT winner = calculateWinner(clusterSpikeCountAN);
	return winner;

}

/* --------------------------------------------------------------------------
total up the spikes on every PN neuron during the full run time. Needed for the plasticity rule.
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::updateIndividualSpikeCountPN() 
{
	for (int i= 0; i < glbSpkCntPN[0]; i++) {
		UINT neuronIdx = glbSpkPN[i]; //this array lists by index the PN neurons with "spike events" during last timestep
		individualSpikeCountPN[neuronIdx] = individualSpikeCountPN[neuronIdx]  + 1; //increment count for this neuron
	}
}

/* --------------------------------------------------------------------------
CLear down totals in spike count array - for reset between runs
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::resetIndividualSpikeCountPN() 
{
	if (individualSpikeCountPN==NULL) {
		//create array if not done yet
		individualSpikeCountPN = new UINT[countPN];
	}

	for (int i= 0; i < countPN; i++) {
		individualSpikeCountPN[i] = 0;
	}
}


/* --------------------------------------------------------------------------
totals up the spikes within each AN cluster during the plasticity window. Needed to determine the current(window) and overall winning output cluster class
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::updateClusterSpikeCountAN() 
{
	for (int i= 0; i < glbSpkCntAN[0]; i++) {
		UINT neuronIdx = glbSpkAN[i]; //this array lists by index the AN neurons that "spiked" during last timestep
		UINT classId = getClassCluster(neuronIdx);
		clusterSpikeCountAN[classId] = clusterSpikeCountAN[classId] + 1; //increment count for this cluster/class
		overallWinnerSpikeCountAN[classId] = overallWinnerSpikeCountAN[classId] + 1;
	}
}

/* --------------------------------------------------------------------------
Clear down totals in spike count array - for reset between runs
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::resetClusterSpikeCountAN() 
{
	if (clusterSpikeCountAN==NULL) {
		//create array if not done yet
		clusterSpikeCountAN = new UINT [NUM_CLASSES];
	}

	for (int i= 0; i < NUM_CLASSES; i++) {
		clusterSpikeCountAN[i] = 0;
	}

}

/* --------------------------------------------------------------------------
Clear down overall winner 
-------------------------------------------------------------------------- */
void Schmuker2014_classifier::resetOverallWinner() 
{
	if (overallWinnerSpikeCountAN==NULL) {
		//create array if not done yet
		overallWinnerSpikeCountAN = new UINT [NUM_CLASSES];
	}

	for (int i= 0; i < NUM_CLASSES; i++) {
		overallWinnerSpikeCountAN[i] = 0;
	}

	this->winningClass = -1; //reset winner
}

#endif	

