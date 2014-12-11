/*--------------------------------------------------------------------------
   Author: Alan Diamond

   Institute: University of Sussex


   initial version:  Mar 1 2014

--------------------------------------------------------------------------*/

#ifndef _Enose_classifier_
#define _Enose_classifier_ //!< macro for avoiding multiple inclusion during compilation

/*--------------------------------------------------------------------------
	 Implementation of the Enose_classifier class.
  -------------------------------------------------------------------------- */

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h> 
#include <time.h>
#include "Enose_classifier.h"
#include "Enose_classifier_CODE/runner.cc"
#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
#include "sparseUtils.cc"
#endif

//--------------------------------------------------------------------------

Enose_classifier::Enose_classifier():
correctClass(0),winningClass(0),vrData(NULL),inputRatesSize(0),t(0.0f),clearedDownDevice(false)
{

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

Enose_classifier::~Enose_classifier()
{

	fclose(log);

	// free all user arrays created on the heap
	free(inputRates);
	free(vrData);
	free(sampleDistance);
	free(allDistinctSamples);
	free(classLabel);
	free(individualSpikeCountPN);
	free(clusterSpikeCountAN);
	free(clusterSpikeCountPerTimestepAN);
	free(clusterSpikeCountPN);
	free(clusterSpikeCountRN);
	free(overallWinnerSpikeCountAN);
	free(plasticWeights);


	//free mem allocated on the CPU
	freeMem();


	if (!clearedDownDevice) clearDownDevice(); //don't try and clear device memory twice
}


void Enose_classifier::startLog()
{
	stringstream logPath;
	logPath << outputDir << SLASH << uniqueRunId << " Log.txt";
	this->log = fopen(logPath.str().c_str(),"w");

}

void Enose_classifier::setMaxSampleDistance(float d) {
	this->sampleDistance[0] = d;
}

float Enose_classifier::getMaxSampleDistance() {
	return this->getSampleDistance(0) ;
}

void Enose_classifier::setMinSampleDistance(float d) {
	this->sampleDistance[1] = d;
}

float Enose_classifier::getMinSampleDistance() {
	return this->getSampleDistance(1) ;
}

void Enose_classifier::setAverageSampleDistance(float avg) {
	this->sampleDistance[2] = avg;
}

float Enose_classifier::getAverageSampleDistance() {
	return this->getSampleDistance(2) ;
}

/*--------------------------------------------------------------------------
   REset CUDA device to ensure no mem leaks form previous run
  -------------------------------------------------------------------------- */
void Enose_classifier::resetDevice()
{
	//CHECK_CUDA_ERRORS(cudaDeviceSynchronize());	// Wait for any GPU work to complete
	CHECK_CUDA_ERRORS(cudaDeviceReset());
	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());	// Wait for the reset
}

/*--------------------------------------------------------------------------
  Allocate the memory arrays used in the network on the host and device
  -------------------------------------------------------------------------- */
void Enose_classifier::allocateHostAndDeviceMemory()
{
	allocateMem();

	//cant do this now, don't know the sizes yet, will do individually, as connexctivty matrices are generated
	//allocateAllSparseArrays();
}

/*--------------------------------------------------------------------------
   Method for populating data on the device, e.g. synapse weight arrays
  -------------------------------------------------------------------------- */

void Enose_classifier::populateDeviceMemory()
{

	printf( "populating data on the device, e.g. synapse weight arrays..\n");

	//initialise host side data (from GeNN 2 onwards, this also copies host data across to device)
	initialize();

	//the sparse arrays have their own copy fn for some reason
#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
	initializeAllSparseArrays();
#endif

	//perform explicit copy to maintain compibility with GeNN 1.0 (later versions include this work within initialise() )
	copyGToDevice();
	copyStateToDevice();


	printf( "..complete.\n");

}

/*--------------------------------------------------------------------------
 Clear device mem and reset device
 -------------------------------------------------------------------------- */
void Enose_classifier::clearDownDevice()
{
		//free mem allocated on the CUDA device from the host i.e. cudaMalloc
		#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
			freeDeviceMem();
		#endif

		// clean up memory allocated outside the model
		CHECK_CUDA_ERRORS(cudaFree((void*)d_inputRates));

		resetDevice();
	clearedDownDevice = true;

}

/*--------------------------------------------------------------------------
   Method for copying the current input dataset across to the device
  -------------------------------------------------------------------------- */

void Enose_classifier::update_input_data_on_device()
{

	// update device memory with set of input data
	CHECK_CUDA_ERRORS(cudaMemcpy(d_inputRates, inputRates, inputRatesSize, cudaMemcpyHostToDevice));

}

/*--------------------------------------------------------------------------
	convert an input rate (Hz) into a proprietary rateCode (a probability number for the poisson neuron model). 

  NB: This should move to the device code.
--------------------------------------------------------------------------*/

UINT  Enose_classifier::convertToRateCode(float inputRateHz)
{

	/*

  	Pr(spike)  = rateCode / D_MAX_RANDOM_NUM  =   rate(Hz) * DT (seconds)

	so,  rateCode = rate(Hz) * DT (seconds) * D_MAX_RANDOM_NUM

	 */

	float prob = (float)inputRateHz  * DT/1000.0; //timestep DT is held as ms

	if (prob > 1.0)  prob = 1.0;

	UINT rateCode  = prob * (float)D_MAX_RANDOM_NUM;

	return rateCode;


}


void Enose_classifier::initialise_VR_data() {
	UINT size = global_NumVR * NUM_FEATURES;
	vrData = new float [size];
	cout << "Storage for VR data initialised" << endl;
}
/* --------------------------------------------------------------------------
load set of virtual receptor points VR to be used to generate input levels 
-------------------------------------------------------------------------- */
void Enose_classifier::load_VR_data()
{
	string filename = recordingsDir;
	filename.append(SLASH).append(VR_DIR).append(SLASH).append(VR_DATA_FILENAME_TEMPLATE);
	filename = replace(filename,"%VR%",toString(global_NumVR));
	filename = replace(filename,"%EP%",toString(NUM_EPOCHS));

	printf("Loading VR data from file %s\n", filename.c_str());
	UINT size = global_NumVR * NUM_FEATURES;
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
load set of classes labelling the recordings
-------------------------------------------------------------------------- */
void Enose_classifier::loadClassLabels()
{
	string filename = recordingsDir;
	filename.append(SLASH).append(FILENAME_CLASS_LABELLING);

	this->classLabel = new UINT [TOTAL_RECORDINGS];

	UINT recordIdLabellings[TOTAL_RECORDINGS *2]; //stores content of file, (recordIdx,ClassID) pairs

	bool ok = loadArrayFromTextFile(filename,&recordIdLabellings,",",TOTAL_RECORDINGS*2,data_type_uint);
	if (!ok) {
		cerr << "Failed to load class label file: " << filename << endl;
		exit(1);
	}
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
void Enose_classifier::generate_or_load_inputrates_dataset(unsigned int recordingIdx)
{
	stringstream cacheFilename;
	cacheFilename << cacheDir << SLASH << "InputRates created from recording no. " << recordingIdx <<  " with " << global_NumVR << " VRs" << ".cache";
	if (!fileExists(cacheFilename.str()))  {
		//file doesn't exist
		generate_inputrates_dataset(recordingIdx);
		//write inputRates to cache file
		FILE *f= fopen(cacheFilename.str().c_str(),"w");
		if (f==NULL) {
			cerr << "Unable to open cache file for writing: '" << cacheFilename.str() << "'" << endl;
			exit(1);
		}
		fwrite(inputRates,inputRatesSize ,1,f);		
		fclose(f);
		//printf( "Input rates for recording %d written to cache file.\n", recordingIdx);
		//checkContents("Input Rates written to cache",inputRates,countRN*5,countRN,data_type_uint,0);

	} else { //cached version exists
		FILE *f= fopen(cacheFilename.str().c_str(),"r");
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
void Enose_classifier::generateSimulatedTimeSeriesData()
{
	//load static data file
	string staticDataFilename = recordingsDir;
	staticDataFilename.append("/IrisFeatureVectors.csv");
	float * featureVectors = new float[TOTAL_RECORDINGS*NUM_FEATURES];
	bool ok = loadArrayFromTextFile(staticDataFilename,featureVectors,",",TOTAL_RECORDINGS*NUM_FEATURES,data_type_float);
	if(!ok) {
		fprintf(stderr,"Failed to load static data file: %s\n", staticDataFilename.c_str());
		exit(1);
	}

	for (int recordingIdx = 0; recordingIdx < TOTAL_RECORDINGS; ++recordingIdx) {
		string recordingFilename = getRecordingFilePath(recordingIdx);
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
string Enose_classifier::getRecordingFilePath(UINT recordingIdx)
{
	stringstream path;

	path << this->recordingsDir << SLASH << replace(recordingFilenameTemplate,"%i",toString(recordingIdx));

	return path.str();
}

/* --------------------------------------------------------------------------
generate the set of input rate data for the recording from the sensor data and the VR set
-------------------------------------------------------------------------- */
void Enose_classifier::generate_inputrates_dataset(unsigned int recordingIdx)
{
	//float * vrResponses = new float[global_NumVR*timestepsPerRecording];
	float vrResponses[global_NumVR*timestepsPerRecording];

	string vrPath = getVrResponseFilename(recordingIdx);
	if (fileExists(vrPath)) {
		bool ok = loadArrayFromTextFile(vrPath,vrResponses,",",global_NumVR*timestepsPerRecording,data_type_float);
	} else {
		completeVrResponseSet(vrResponses,recordingIdx);
		FILE * vrResponseFile = fopen(vrPath.c_str(),"w");
		writeArrayToTextFile(vrResponseFile,vrResponses,timestepsPerRecording,global_NumVR,data_type_float,7,false,COMMA,true);
		//cout << "VR Response to recording " << recordingIdx << " written to cache directory." << endl;
	}

	//fill in the input rate data for the poisson neuron clusters using the vr response set
	for (UINT ts=0 ; ts < timestepsPerRecording; ts++) {
		addInputRate(ts,vrResponses);
	}

	//delete[] vrResponses;
}

/* --------------------------------------------------------------------------
complete a set of VR response data for the specfied recording and the VR set
-------------------------------------------------------------------------- */
void Enose_classifier::completeVrResponseSet(float * vrResponses,UINT recordingIdx)
{
	//open sensor data file
	UINT length = NUM_FEATURES * timestepsPerRecording;
	float * recording = new float[length];
	string recordingFilename = getRecordingFilePath(recordingIdx);
	bool ok = loadArrayFromTextFile(recordingFilename,recording,COMMA,length,data_type_float);
	if (!ok) {
		cerr << "generate_inputrates_dataset: unable to read recording file" << endl;
		exit(1);
	}

	//for each data point read from file, get a distance metric to each VR point, these will become the input rate levels to the network
	for (UINT ts=0 ; ts < timestepsPerRecording; ts++) {

		float * samplePoint = &recording[ts * NUM_FEATURES];

		for (UINT vr=0 ; vr < global_NumVR; vr++) {//step through the set of VRs

			float * vrPoint = &vrData[vr * NUM_FEATURES]; //get a ptr to start of the next VR vector in the set

			//Calculate the response of the VR to the sample point
			float vrResponse  = calculateVrResponse(samplePoint,vrPoint);
			vrResponses[ts * global_NumVR + vr] = vrResponse;
		}

	}

	delete[] recording;
}

string Enose_classifier::getVrResponseFilename(UINT recordingIdx)
{
	stringstream vrPath;
	vrPath << CACHE_DIR << "/VRs_x" << global_NumVR <<  "-Response-to-Recording" << recordingIdx <<  ".csv";
	return vrPath.str();

}

/* --------------------------------------------------------------------------
get a handle to the specified sensor recording file
-------------------------------------------------------------------------- */
FILE * Enose_classifier::openRecordingFile(UINT recordingIdx)
{
	string recordingFilename = getRecordingFilePath(recordingIdx);
	FILE *f= fopen(recordingFilename.c_str(),"r");
	if (f==NULL)  {
		//file doesn't exist or cant read
		cout << "ERROR! failed to open recording file " << recordingFilename << endl;
		exit(1);
	}
	return f;
}


/* --------------------------------------------------------------------------
extend the set of input rate data by one, using the response from the set of VRs to a single sample of sensor data (vector in feature space)
-------------------------------------------------------------------------- */
void Enose_classifier::addInputRate(UINT timeStep, float * vrResponses)
{
	UINT inputRatesDataOffset = timeStep * countRN; //get ptr to start of vector of rates (size = countRN) for this time step

	for (UINT vr=0 ; vr < global_NumVR; vr++) {//step through the set of VRs

		//Calculate the response of the VR to the sample point
		float vrResponse  = vrResponses[timeStep * global_NumVR + vr];

		//scale the firing rate (Hz) from the distance 
		float rateHz  = param_MIN_FIRING_RATE_HZ + vrResponse * (float)(param_MAX_FIRING_RATE_HZ - param_MIN_FIRING_RATE_HZ);

		//convert Hz to proprietary rate code used on the device ( this code should move to device code)
		UINT rateCode  = convertToRateCode(rateHz);


		//fill in a clusters worth with the same rate (one VR excites one cluster in RN)
		for (UINT i=0; i < CLUST_SIZE_RN; i++) {
			inputRates[inputRatesDataOffset + vr*CLUST_SIZE_RN + i] = rateCode;
		}
	}
}

/* --------------------------------------------------------------------------
Calculate the response of a given VR to a single sample of sensor data (vector in feature space)
-------------------------------------------------------------------------- */
float Enose_classifier::calculateVrResponse(float * samplePoint, float * vrPoint)
{
	//get the Manhattan distance metric
	float distance = getManhattanDistance(samplePoint, vrPoint, NUM_FEATURES);
	//normalise to a number in range 0..1
	float sigma = param_VR_RESPONSE_SIGMA_SCALING * getAverageSampleDistance();
	#ifdef USE_NON_LINEAR_VR_RESPONSE
		//use parameterised e^(-x) function to get an customised response curve
		float power = param_VR_RESPONSE_POWER;
		float exponent = -powf((distance * param_VR_RESPONSE_DISTANCE_SCALING)/sigma,power);
		float response  = expf(exponent);
		//cout << param_VR_RESPONSE_SIGMA_SCALING << COMMA << distance << COMMA << response << endl;
	
	#else
		//use linear response scaled by the average distance between samples
		float response  =  1 - (distance  / sigma ) ;
	#endif

	return response < 0 ? 0: response; //further than avg cluster points (VR's) contribute no response

}

/* --------------------------------------------------------------------------
Get the max or minimum absolute "distance" existing between 2 points in the full sensor recording data set
If no cached version exists then calculate by interrogating the full data set
-------------------------------------------------------------------------- */
float Enose_classifier::getSampleDistance(UINT max0_min1_avg2)
{
	if (this->sampleDistance == NULL) {//not yet accessed

		//look for saved data to load

		this->sampleDistance = new float[2]; //will store the max and min values
		string path = recordingsDir;
		path.append("/MaxMinAvgSampleDistances.csv");
		bool ok = loadArrayFromTextFile(path,sampleDistance,",",3,data_type_float);
		if (ok)  {
			printf( "Max (%f), Min (%f) and Average (%f) Sample Distances loaded from file %s.\n",getMaxSampleDistance(),getMinSampleDistance(),getAverageSampleDistance(),path.c_str());
		} else {
			//file doesn't exist yet, so need to generate values
			setMaxMinSampleDistances();

			//now write to file
			FILE *f= fopen(path.c_str(),"w");
			fprintf(f,"%f,%f,%f",getMaxSampleDistance(),getMinSampleDistance(),getAverageSampleDistance());
			fclose(f);
			printf( "Max, Min and Average Sample Distances written to file: %s.\n",path.c_str());

		}


	} 
	return this->sampleDistance[max0_min1_avg2];

}

/* --------------------------------------------------------------------------
cache an array of all the individual distinct samples, one per recording
These are used for finding maxMin distance and also for creating subsets of training (only) observation points for passing to VR generator
-------------------------------------------------------------------------- */
bool Enose_classifier::initialiseDistinctSamples()
{
	int totalDistinctSamples = TOTAL_RECORDINGS;// * DISTINCT_SAMPLES_PER_RECORDING;

	//allocate data to load ALL recordings into
	this->allDistinctSamples = new float [totalDistinctSamples * NUM_FEATURES];

	bool ok = loadArrayFromTextFile(recordingsDir + SLASH + FILENAME_ALL_DISTINCT_SAMPLES,allDistinctSamples,COMMA,totalDistinctSamples * NUM_FEATURES,data_type_float);
	if (!ok) {
		cerr << "setMaxMinSampleDistances: unable to load all-in-one recordings file " << FILENAME_ALL_DISTINCT_SAMPLES << endl;
		exit(1);
	}
	return ok;
}

/* --------------------------------------------------------------------------
Load full data set to find  the max and minimum absolute "distance" existing between 2 points in the sensor recording data set
-------------------------------------------------------------------------- */
void Enose_classifier::setMaxMinSampleDistances()
{
	//reset values (negative values imply not set)
	this->setMaxSampleDistance(-1.0);
	this->setMinSampleDistance(-1.0);
	this->setAverageSampleDistance(-1.0);
	this->samplesCompared = 0;
	this->totalledDistance = 0;


	//load all sample data into one giant array (because we need to compare all vs all)
	printf( "Interrogating the full data set to find  the max and minimum distances between sample points..\n");

	int totalDistinctSamples = TOTAL_RECORDINGS;// * DISTINCT_SAMPLES_PER_RECORDING;

	for (int startAtSample = 0; startAtSample < (totalDistinctSamples-1); startAtSample++) {
		cout << "Checking sample " << startAtSample << " of " << totalDistinctSamples << " against " << (totalDistinctSamples-startAtSample) << " samples remaining." << endl;
		findMaxMinSampleDistances(this->allDistinctSamples,startAtSample,totalDistinctSamples);
	}

	//findMaxMinSamplesDistancesOnDevice(this->allDistinctSamples);

	float avg = this->totalledDistance /(double)this->samplesCompared;
	setAverageSampleDistance(avg);

	printf( "Completed Max/min search.\n");

}

/* --------------------------------------------------------------------------
Go through the full data set to find  the max and minimum absolute "distance" existing between 2 points in the sensor recording data set
-------------------------------------------------------------------------- */
void Enose_classifier::findMaxMinSampleDistances(float * samples, UINT startAt, UINT totalSamples)
{

	//we need to do an all vs all distance measurement to find the two closest and the two furthest points

	float * sampleA  = &samples[startAt * NUM_FEATURES]; //remember each sample comprises NUM_FEATURES floats

	for (UINT i = startAt+1; i < totalSamples; i++) {
		float * sampleB  = &samples[i * NUM_FEATURES]; 
		//printf("Sample B (%f,%f,%f,%f)\n",sampleB[0],sampleB[1],sampleB[2],sampleB[3]);
		float distanceAB = getManhattanDistance(sampleA,sampleB,NUM_FEATURES);
		if (distanceAB > this->getMaxSampleDistance() || this->getMaxSampleDistance()<0.0) {
			this->setMaxSampleDistance(distanceAB) ; //update max
			printf("Max updated! sample:%d vs sample:%d    max:%f min:%f\n",startAt,i,this->getMaxSampleDistance(),this->getMinSampleDistance());
			cout << "Total Distance:" << totalledDistance <<endl;
		} else if (distanceAB < this->getMinSampleDistance()|| this->getMinSampleDistance()<0.0) {
			this->setMinSampleDistance(distanceAB) ; //update min
			printf("Min updated! sample:%d vs sample:%d    max:%f min:%f\n",startAt,i,this->getMaxSampleDistance(),this->getMinSampleDistance());
			cout << "Total Distance:" << totalledDistance <<endl;
		}
		this->totalledDistance = this->totalledDistance  + distanceAB;
		this->samplesCompared++;

	}
	//now call fn recursively with next start point (we have done A:B , A:C, A:D etc, now we need to do B:C, B:D, etc)
	//if (startAt+2 < totalSamples) {
	//	findMaxMinSampleDistances(samples, startAt+1);
	//}
}

/* --------------------------------------------------------------------------
Use GPU to go through the full data set to find  the max and minimum and average
 absolute "distance" existing between 2 points in the sensor recording data set
-------------------------------------------------------------------------- */
__device__ volatile unsigned int d_samplesDone;
__device__ volatile unsigned int d_maxDistance;
__device__ volatile unsigned int d_minDistance;
//__device__ volatile unsigned long d_totalledDistance;
//__device__ volatile unsigned long d_distanceCount;

__global__ void cudaFindSampleDistances(float *d_allSamples, UINT totalSamples, UINT scale)
{
	unsigned int startingSample = blockDim.x * blockIdx.x + threadIdx.x;

	if (startingSample<totalSamples-1) {//dont need the last sample or anything beyond (there may be too many threads)

		UINT startIndex  = startingSample * NUM_FEATURES;

		for (UINT otherSample = startingSample +1; otherSample < totalSamples; otherSample++) {
			UINT otherIndex = otherSample * NUM_FEATURES;

			float manhattanDistance = 0.0f;
			for (UINT i=0 ; i < NUM_FEATURES; i++) {
				float a  = d_allSamples[startIndex + i];
				float b = d_allSamples[otherIndex + i];
				float pointDistance = abs(a - b);
				if (pointDistance > 100 * NUM_FEATURES) {
					printf("ERROR? Manhattan distance %f v. large.\n",pointDistance);
				}
				manhattanDistance = manhattanDistance + pointDistance;
			}

			UINT distanceScaled = (UINT)(manhattanDistance * scale);//atomic max/min can only handle integers
			UINT old = atomicMax((unsigned int *) &d_maxDistance,distanceScaled);
			if (distanceScaled > old) printf("Updated (scaled) max from %u to %u\n",old,distanceScaled);

			old = atomicMin((unsigned int *) &d_minDistance, distanceScaled);
			if (distanceScaled < old) printf("Updated (scaled) min from %u to %u\n",old,distanceScaled);

			//long total  = atomicAdd((unsigned long *)&d_totalledDistance,)


		}
		UINT samplesDone = 1 + atomicAdd((unsigned int *) &d_samplesDone, 1);
		//printf("Completed %u of %u\n",samplesDone,totalSamples);
	}
}

void Enose_classifier::findMaxMinSamplesDistancesOnDevice(float * allSamples)
{
	CHECK_CUDA_ERRORS(cudaSetDevice(0));
	UINT totalSamples = TOTAL_RECORDINGS * timestepsPerRecording;
	UINT size  = totalSamples*NUM_FEATURES*sizeof(float);
	float * d_allSamples;
	CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_allSamples, size));
	UINT zero = 0;
	CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_samplesDone,&zero,cudaMemcpyHostToDevice)); //reset count
	CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_maxDistance,&zero,cudaMemcpyHostToDevice)); //reset max
	UINT bigNumber = 10000000;
	CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(d_minDistance,&bigNumber,cudaMemcpyHostToDevice)); //reset min
	CHECK_CUDA_ERRORS(cudaMemcpy(d_allSamples, allSamples,size, cudaMemcpyHostToDevice));
	UINT scale  = 100;//scale up distances found to get some decimal points if needed (max/min atomic operations only support ints)
	cudaFindSampleDistances<<<(totalSamples+255)/256,256>>>(d_allSamples,totalSamples,scale);

	CHECK_CUDA_ERRORS(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
	UINT maxScaled, minScaled;
	void *devPtr;
	CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_maxDistance));
	CHECK_CUDA_ERRORS(cudaMemcpy(&maxScaled, devPtr, sizeof(UINT), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devPtr, d_minDistance));
	CHECK_CUDA_ERRORS(cudaMemcpy(&minScaled, devPtr, sizeof(UINT), cudaMemcpyDeviceToHost));

	cout <<"Max distance found: " << (float)maxScaled/(float)scale << " Min: " << (float)minScaled/(float)scale << endl;
	this->setMaxSampleDistance( (float)maxScaled/(float)scale) ;
	this->setMinSampleDistance( (float)minScaled/(float)scale) ;

	CHECK_CUDA_ERRORS(cudaFree(d_allSamples));

}

/* --------------------------------------------------------------------------
calculate the Manhattan distance metric between two vectors of floats denoting points in for example, feature space
The "Manhattan" distance is simply the sum of all the co-ordinate differences
-------------------------------------------------------------------------- */
float Enose_classifier::getManhattanDistance(float * pointA,float * pointB, UINT numElements)
{
	float totalDistance = 0.0;
	for (UINT i=0 ; i < numElements; i++) {
		float a  = pointA[i];
		float b = pointB[i];
		float pointDistance = abs(a - b);
		if (pointDistance > 100 * numElements) {
			fprintf(stderr,"ERROR? Manhattan distance %f v. large.\n",pointDistance);
		}
		totalDistance = totalDistance + pointDistance;
	}
	return totalDistance;
}


bool Enose_classifier::displayRasterPlot(string srcdir , string srcFilename, float endTime, bool displayActivation, int stretchTime) {
	int totalNeurons = countAN + countPN + countRN;
	stringstream cmd;
	cmd << PYTHON_RUNTIME << SPACE << PYTHON_DIR << "/spikeRasterPlot.py";
	cmd << SPACE << srcdir << SPACE << srcFilename;
	cmd << SPACE << toString(totalNeurons) << SPACE << toString(endTime);

	cmd << SPACE << this->countPN; //used to set the major gridlines to separate RN,PN,AN regions
	cmd << SPACE << CLUST_SIZE_PN; ////used to set the minor gridlines to separate clusters (assumes the same cluster size in all layers)

	cmd << SPACE << (displayActivation?1:0);

	if (!displayActivation) {
		//specify where to draw a box on the heatmap, showing the class evaluation zone used
		float evaluationStartTimeMs = stretchTime * DT  * this->classEvaluationStartTimestep;
		float evaluationEndTimeMs = stretchTime * DT  * this->classEvaluationEndTimestep;
		cmd << SPACE << evaluationStartTimeMs;
		cmd << SPACE << evaluationEndTimeMs;
	}

	cout << "cmd:" << cmd.str() << endl;
	string result;
	invokeLinuxShellCommand(cmd.str().c_str(),result);
	cout << result << endl;
	return true;
}



/*--------------------------------------------------------------------------
   Method for simulating the model for a specified duration in ms
  -------------------------------------------------------------------------- */

void Enose_classifier::run(float runtimeMs, bool createRasterPlot, string filename_rasterPlot,bool usePlasticity, UINT stretchTime)
{

	int timestepsRequired = (int) (runtimeMs/DT);


#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
	ofstream spikeFreqFile;
	string spikeFreqPath = outputDir;
	spikeFreqPath.append(SLASH);
	if (!usePlasticity) {
		#ifdef FLAG_GENERATE_SPIKE_FREQ_DATA_FILE
			spikeFreqPath.append(replace(filename_rasterPlot,"Raster","SpikeRatesHz"));
			spikeFreqFile.open(spikeFreqPath.c_str(),std::ofstream::out);
		#endif

		this->highestWinningSpikeFreqHzAN = -1;
		currClassEvaluationStage = stage_pre_evaluation;
		classEvaluationStartTimestep = 0; classEvaluationEndTimestep = timestepsRequired;
	}
#endif


	//reset spike counts for these layers at start of input presentation
	resetClusterSpikeCountPN();
	resetClusterSpikeCountRN();
	resetClusterSpikeCountPerTimestepAN();

	float startTime = t; //mark when this run began (used to start raster plots' time column at 0)

	FILE *rasterFile = NULL;

	if (createRasterPlot) {
		string rasterPath = outputDir;
		rasterPath.append(SLASH).append(filename_rasterPlot);
		rasterFile = fopen(rasterPath.c_str(),"w");
		if (rasterFile==NULL) {
			fprintf(stderr,"Unable to open raster file for writing %s\n",rasterPath.c_str());
			exit(1);
		}
	}


	unsigned int offset= 0;

	int timestepsBetweenPlasticity = (int) (param_PLASTICITY_INTERVAL_MS/DT);
	int nextPlasticityUpdate = timestepsBetweenPlasticity - 1;

	//t = 0.0f; //reset elapsed time

	//reset spike count totals across the current plasticity window
	resetIndividualSpikeCountPN();
	resetClusterSpikeCountAN();


	for (int timestep= 0; timestep < timestepsRequired; timestep++) {

		offset = timestep * countRN ; //units = num of unsigned ints

		//stretchTime indicates how many time increments to stay on each offset (ie. the saame input data sample)
		//e.g. stretchTime = 1000 would expand ms time into secs time. This is used to apply the sensor recordings in real time, simulating a real enose in action

		for (int repeatOffset = 0; repeatOffset < stretchTime; ++repeatOffset) {

			if(global_RunOnCPU) {
				//step simulation by one timestep on CPU
				stepTimeCPU(inputRates, offset, t);
			} else {
				//step simulation by one timestep on GPU
				stepTimeGPU(d_inputRates, offset, t);
				getSpikesFromGPU(); //need these to calculate winning class etc (and for raster plots)
				//cudaDeviceSynchronize();
			}

			//uncomment this to divert other state data into raster file
			//copyStateFromDevice();
			//fprintf(rasterFile,"%f,%f\n",t,VPN[0]);

			updateIndividualSpikeCountPN(); //total up the spikes on every PN neuron during the run time. Needed for the plasticity rule.
			updateClusterSpikeCountAN(timestep); // total up the spikes witihn each AN cluster during the run. Needed to determine the winning output cluster class
			updateClusterSpikeCountPN();
			updateClusterSpikeCountRN();

			//learning stage
			if (usePlasticity && timestep == nextPlasticityUpdate) {
				UINT currentWinner = calculateCurrentWindowWinner();
					applyLearningRuleSynapses(plasticWeights,currentWinner);
				resetIndividualSpikeCountPN();
				resetClusterSpikeCountAN();
				nextPlasticityUpdate += timestepsBetweenPlasticity;
			}

			if (createRasterPlot) outputSpikes(rasterFile,COMMA, startTime);

			//increment global time elapsed
			t+= DT;
		}

#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
		if (!usePlasticity) {// for test stage only
			//check whether spiking for any class (i.e. in a AN cluster) has passed the detection threshold
			float elapsedTimeMs = stretchTime * DT;
		#ifdef INFER_CLASS_EVALUATION_WINDOW_FROM_AN
				checkSpikeRateThresholdAN(t - startTime, elapsedTimeMs, timestep, timestepsRequired-1, spikeFreqFile);
				resetClusterSpikeCountAN(); // we are hijacking this counter when in test mode to hold the spike counts in the current timestep
		#else
				checkSpikeRateThresholdPN(t - startTime, elapsedTimeMs, timestep, timestepsRequired-1, spikeFreqFile);
				resetClusterSpikeCountPN(); // we are hijacking this counter (when testing using full delayed recordings) to hold the PN spike counts in the current timestep
		#endif
		}
#endif

	} //end of curr time step

	updateWeights_PN_AN(); //at end of presentation, copy plasticity changes over to the actual synapses

	//completed simulation run

	if (createRasterPlot)  {
		fclose(rasterFile);
	}

#ifdef USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE
	#ifdef FLAG_GENERATE_SPIKE_FREQ_DATA_FILE
		if (!usePlasticity) {
			spikeFreqFile.close();
			//cout << highestWinningSpikeFreqHzAN << highestWinningSpikeFreqHzAN << endl;
		}
	#endif
#endif

}

/* --------------------------------------------------------------------------
   output functions
  -------------------------------------------------------------------------- 
   Method for copying all spikes of the last time step from the GPU

  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
  -------------------------------------------------------------------------- */

void Enose_classifier::getSpikesFromGPU()
{
#ifndef DEVICE_MEM_ALLOCATED_ON_DEVICE
	copySpikeNFromDevice(); //GeNN 1.0 does this as part of copySpikesFromDevice();
#endif
	copySpikesFromDevice();

	//cout << " Spikes (RN,PN,AN):" << glbscntRN << SPACE << glbscntPN << SPACE << glbscntAN << endl;
	if (glbscntRN==0) {
		//cerr << "No poisson spikes at time " << t << endl;
		//exit(1);
	}
}

/*--------------------------------------------------------------------------
   Method for writing the spikes occurred in the last time step to a file
           This can be used to create a raster plot (by providing coordinates (t,index) of points to plot 
           File format: time|idx of spiked neuron|\n
  -------------------------------------------------------------------------- */

void Enose_classifier::outputSpikes(FILE *f, string delim, float startTime )
{

	/*
	printf("RN spikes: %u\n",glbscntRN);
	printf("PN spikes: %u\n",glbscntPN);
	printf("AN spikes: %u\n",glbscntAN);
	 */

	float relTime = t - startTime;
	for (int i= 0; i < glbscntRN; i++) {
		fprintf(f, "%f%s%d\n", relTime, delim.c_str(), glbSpkRN[i]);
	}

	for (int i= 0; i < glbscntPN; i++) {
		fprintf(f,  "%f%s%d\n", relTime, delim.c_str(), countRN + glbSpkPN[i] );
	}

	for (int i= 0; i < glbscntAN; i++) {
		fprintf(f, "%f%s%d\n", relTime, delim.c_str(), countRN + countPN + glbSpkAN[i]);
	}
}

/*--------------------------------------------------------------------------
   overwrite actual synapse weights with any plastic changes made
  -------------------------------------------------------------------------- */
void Enose_classifier::updateWeights_PN_AN()
{
	memcpy(gpPNAN,plasticWeights,countPNAN * sizeof(float));

	//update to new weights on the device
	if(!global_RunOnCPU) {
		updateWeights_PN_AN_on_device();
	}
}

/*--------------------------------------------------------------------------
   Method for updating the conductances of the learning synapses between projection neurons PNs and association ANs on the device memory
  -------------------------------------------------------------------------- */
void Enose_classifier::updateWeights_PN_AN_on_device()
{
#ifdef DEVICE_MEM_ALLOCATED_ON_DEVICE
	void *d_ptrPNAN;
	cudaGetSymbolAddress(&d_ptrPNAN, d_gpPNAN);
	CHECK_CUDA_ERRORS(cudaMemcpy(d_ptrPNAN, gpPNAN, countPNAN*sizeof(float), cudaMemcpyHostToDevice));
#else
	//New version. device mem allocated using cudaMalloc from the host side
	CHECK_CUDA_ERRORS(cudaMemcpy(d_gpPNAN, gpPNAN, countPNAN*sizeof(float), cudaMemcpyHostToDevice));
#endif
}


/*--------------------------------------------------------------------------
   Initialise the set of weights for the SPARSE 1:1 subcluster-subcluster synapses RN-PN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Enose_classifier::initialiseWeights_SPARSE_RN_PN()
{

#ifdef USE_SPARSE_ENCODING
	//We use SPARSE set up because the clusters connect only to the corresponding cluster in the second population
	//Because the SPARSE format is unintutive, We will start by creatiing an ALL-TO_ALL array which addresses the task
	// we will then convert this to SPARSE (effectively, this filters out all the zero weight connections)
	float * tmp_gpRNPN = new float[countRN * countPN]; //put it on the heap, might be v large
#else
	float * tmp_gpRNPN = gpRNPN; //for DENSE, reference the predefined all-all-array
#endif

	//for each synapse from population X to pop Y
	for (UINT x= 0; x < countRN; x++) {
		for (UINT y= 0; y < countPN; y++) {
			UINT synapseIdx = x * countPN + y;
			tmp_gpRNPN[synapseIdx] = 0.0f; //default to no connection
			if  (getClusterIndex(x,CLUST_SIZE_RN) == getClusterIndex(y,CLUST_SIZE_PN)) { //same cluster
				if (randomEventOccurred(param_CONNECTIVITY_RN_PN)) {
					tmp_gpRNPN[synapseIdx]  = WEIGHT_RN_PN * param_GLOBAL_WEIGHT_SCALING;
				}
			}
		}
	}
	//checkContents("RNPN Connections",tmp_gpRNPN,countRN * countPN,countPN,data_type_float,2);

#ifdef USE_SPARSE_ENCODING
	createSparseConnectivityFromDense(countRN,countPN,tmp_gpRNPN,&gRNPN, false);
	delete[] tmp_gpRNPN; //don't need now
#endif

	printf("Initialised weights for  SPARSE 1:1 subcluster-subcluster synapses RN-PN.\n");
}



/*--------------------------------------------------------------------------
   initialise the set of weights for the DENSE subcluster-subcluster WTA synapses PN-PN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Enose_classifier::initialiseWeights_WTA_PN_PN()
{
	createWTAConnectivity(gpPNPN, countPN, CLUST_SIZE_PN, param_WEIGHT_WTA_PN_PN * param_GLOBAL_WEIGHT_SCALING, param_CONNECTIVITY_PN_PN);
	//checkContents("PN PN Connections",gpPNPN,countPN*countPN,countPN,data_type_float,1," ");
}

/*  -------------------------------------------------------------------------- 
  initialise the set of weights for the DENSE plastic synapses PN-AN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Enose_classifier::initialiseWeights_DENSE_PN_AN()
{
	//uses a DENSE connection matrix (i.e. all-to-all)
	//The paper specifies an N% connectivity (e.g. 50%) , so N% of the weights will be fixed at zero
	for (UINT i= 0; i < countPN; i++) {
		for (UINT j= 0; j < countAN; j++) {
			if (randomEventOccurred(param_CONNECTIVITY_PN_AN)) {
				//set weight randomly between limits
				float weight  = param_START_WEIGHT_MIN_PN_AN  +  getRand0to1() * (param_START_WEIGHT_MAX_PN_AN - param_START_WEIGHT_MIN_PN_AN);
				gpPNAN[i*countAN + j] = weight * param_GLOBAL_WEIGHT_SCALING;
			} else {
				gpPNAN[i*countAN + j] = 0.0; //zero weighted = no connection
			}
		}
	}
	//checkContents("PN-AN Connections",gpPNAN,countPN*countAN,countAN,data_type_float,3);

	//initialise plastic weights as a copy of PN-AN. These weights are updated periodically during a presention but not used in classifier until end of presentation
	memcpy(plasticWeights,gpPNAN,countPNAN * sizeof(float));

}
/*  --------------------------------------------------------------------------   
  //initialise the set of weights for the DENSE subcluster-subcluster WTA synapses AN-AN (GeNN has no automatic function for what we need)
  -------------------------------------------------------------------------- */
void Enose_classifier::initialiseWeights_WTA_AN_AN()
{
	createWTAConnectivity(gpANAN, countAN, CLUST_SIZE_AN, param_WEIGHT_WTA_AN_AN * param_GLOBAL_WEIGHT_SCALING, param_CONNECTIVITY_AN_AN);
}

/*  --------------------------------------------------------------------------   
  allocate storage on CPU and GPU for the set of input data (rates) to be processed
  -------------------------------------------------------------------------- */
void Enose_classifier::initialiseInputData()
{
	//allocate storage for the set of input data (rates) to be processed
	this->inputRatesSize = countRN * sizeof(unsigned int) * timestepsPerRecording;

	//allocate memory on the CPU to hold the current input dataset
	this->inputRates = new unsigned int[timestepsPerRecording * countRN];

	//allocate corresponding memory on the GPU device to hold the input dataset
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &d_inputRates, inputRatesSize));

	printf("Memory allocated for input rates on CPU and GPU.\n");
}

/* --------------------------------------------------------------------------
set the integer code for the class labelled to the specified recording
-------------------------------------------------------------------------- */
void Enose_classifier::setCorrectClass(UINT recordingIdx)
{
	this->correctClass = this->classLabel[recordingIdx];
}

/* --------------------------------------------------------------------------
Use the results of the classifier to update the weights of to the outputs
-------------------------------------------------------------------------- */
void Enose_classifier::applyLearningRuleSynapses(float * synapsesPNAN, UINT currentWinner)
{

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
						//float polarity  = currentWinner == this->correctClass ? 1.0 : -1.0;
						float weightDelta  = currentWinner == this->correctClass ? param_WEIGHT_INC_DELTA_PN_AN : - param_WEIGHT_DEC_MULTIPLIER_PN_AN * param_WEIGHT_INC_DELTA_PN_AN;
						weightDelta  = weightDelta  * param_PLASTICITY_INTERVAL_MS / RECORDING_TIME_MS; //use only in proportion to the fraction of recoding time being used
						//weightDelta  = weightDelta  * DISTINCT_SAMPLES_PER_RECORDING; //scale up depending on how many different samples are presented in the recording time

						float weightChange  = weightDelta * param_GLOBAL_WEIGHT_SCALING;

						//get new weight 
						currWeight += weightChange;

						//constrain to limits w-min, w-max 
						if (currWeight > param_MAX_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING) currWeight = param_MAX_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING;
						if (currWeight < param_MIN_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING) currWeight = param_MIN_WEIGHT_PN_AN * param_GLOBAL_WEIGHT_SCALING;

						//update the synapse array with new value
						synapsesPNAN[synapseIdx] = currWeight;

						//string direction = weightDelta>0?"increased":"decreased";
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
UINT Enose_classifier::getClassCluster(UINT anIdx)
{
	return getClusterIndex(anIdx,CLUST_SIZE_AN) ;
}

/* --------------------------------------------------------------------------
return the index of the neuron cluster that a specified neuron belongs to
-------------------------------------------------------------------------- */
UINT Enose_classifier::getClusterIndex(UINT neuronIndex, UINT clusterSize)
{
	return neuronIndex / clusterSize ;
}

/* --------------------------------------------------------------------------
fill in the passed connectivity array to create a WTA structure between clusters of a specified size
DENSE Connections are made between neurons (with a specified probability) unless they are in the same cluster
-------------------------------------------------------------------------- */
void Enose_classifier::createWTAConnectivity(float * synapse, UINT populationSize, UINT clusterSize, float synapseWeight, float probability)
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
Use spike counts in AN clusters after a run to decide the winning class , store result in winningClass instance var
-------------------------------------------------------------------------- */
UINT Enose_classifier::calculateWinner(unsigned int * clusterSpikeCount)
{
	UINT winner = getIndexOfHighestEntry(clusterSpikeCount,NUM_CLASSES);

	return winner;
}

/* --------------------------------------------------------------------------
 returns the winning class indicated by the highest cluster spike count in AN during the evaluation window alone
  -------------------------------------------------------------------------- */
int Enose_classifier::calculateEvaluationPeriodWinner()
{
	if (currClassEvaluationStage!=stage_post_evaluation) {//the evaluation window never started or never ended - this is a fail by definition
		return -1; // can never match correct class
	}

	UINT winner = 0;
	UINT max = 0;

	for(UINT clustIdx=0; clustIdx<NUM_CLASSES; clustIdx++) {
		//add up the spikes that occurred in this cluster over the evaluation window demarcated during the recording presentation
		int clusterSpikeCount  = 0;
		for (int timestep = classEvaluationStartTimestep; timestep < classEvaluationEndTimestep; ++timestep) {
			clusterSpikeCount = clusterSpikeCount + clusterSpikeCountPerTimestepAN[timestep*NUM_CLASSES + clustIdx];
		}

		if (clusterSpikeCount > max) {
			max = clusterSpikeCount;
			winner = clustIdx;
		} else if (clusterSpikeCount== max) { //draw
			//toss a coin (we don't want to always favour the same cluster)
			if (randomEventOccurred(0.5)) winner = clustIdx;
		}
	}
	return winner;
}


/* --------------------------------------------------------------------------
 returns the winning class using either the overall spike count over the whole recording presentation or just the evaluation window
  -------------------------------------------------------------------------- */
UINT Enose_classifier::calculateOverallWinner(bool restrictToEvaluationPeriod)
{
	if (restrictToEvaluationPeriod) {
		this->winningClass = calculateEvaluationPeriodWinner();
	} else {
		this->winningClass = calculateWinner(overallWinnerSpikeCountAN);
	}

	return this->winningClass;
}


UINT Enose_classifier::calculateCurrentWindowWinner() {

	UINT winner = calculateWinner(clusterSpikeCountAN);
	return winner;

}

/* --------------------------------------------------------------------------
 provide an average spike rate Hz per neuron for a no. of spikes occurring in a given time across N neurons
  -------------------------------------------------------------------------- */
float Enose_classifier::calculateAvgSpikeFreqHz(int spikeCount,float periodMs,int numNeurons) {
	float avgSpikesPerNeuron =  ((float)spikeCount)/numNeurons;
	float spikeFreqHz = 1000.0f * avgSpikesPerNeuron / periodMs ;
	return spikeFreqHz;

}

/* --------------------------------------------------------------------------
 check whether spiking for any class (i.e. in a AN cluster) has passed the detection threshold
  -------------------------------------------------------------------------- */
void Enose_classifier::checkSpikeRateThresholdAN(float sampleTimeMs, float periodMs, UINT timestep, UINT lastTimestep, ofstream & spikeFreqFile) {


	UINT winner = getIndexOfHighestEntry(clusterSpikeCountAN,NUM_CLASSES);

	float mostSpikesInCluster = clusterSpikeCountAN[winner];

	float maxSpikeFreqHz  = calculateAvgSpikeFreqHz(mostSpikesInCluster,periodMs,CLUST_SIZE_AN);

#ifdef FLAG_GENERATE_SPIKE_FREQ_DATA_FILE
	spikeFreqFile << sampleTimeMs << COMMA << winner << COMMA << periodMs << COMMA << maxSpikeFreqHz << endl;
#endif

	if (maxSpikeFreqHz > highestWinningSpikeFreqHzAN ) {
		highestWinningSpikeFreqHzAN = maxSpikeFreqHz; //track the highest winning rate for this recording in order to work out a good threshold value
	}



	switch (currClassEvaluationStage) {
	case stage_pre_evaluation:
		if (maxSpikeFreqHz  >= param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN) {
			currClassEvaluationStage = stage_evaluation_active;

			//look back in time for this cluster/class to find a better point to start the window
			int lookBack = 0;
			bool spikingTooLow = false;
			while (!spikingTooLow && lookBack < timestep) {
				int pastSpikeCount = clusterSpikeCountPerTimestepAN[(timestep - lookBack)*NUM_CLASSES + winner];
				if (calculateAvgSpikeFreqHz(pastSpikeCount,periodMs,CLUST_SIZE_AN) < param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN / 2)  {
					spikingTooLow = true;
				} else {
					lookBack++;
				}
			}

			classEvaluationStartTimestep = timestep - lookBack; //there will have been some activity in this cluster on previous timestep
			classEvaluationEndTimestep = timestep + param_MIN_CLASS_EVALUATION_TIMESTEPS; //set end of window to be a minimum time away
		}
		break;

	case stage_evaluation_active :
		//reached the end of recording and still in active evaluation. End the window here.
		if (timestep== lastTimestep) {
			classEvaluationEndTimestep = timestep;
			currClassEvaluationStage = stage_post_evaluation;
		}

		//end the window by passing min legth and spiking dropping below specified rate
		if (timestep >= classEvaluationEndTimestep && maxSpikeFreqHz  < param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_AN /2) {
			currClassEvaluationStage = stage_post_evaluation;
			classEvaluationEndTimestep = timestep + 1; //reset the end point to where the spike rate drops low enough
		}
		break;


	default:
		break;
	}

	spikeFreqFile.flush();

}


/* --------------------------------------------------------------------------
 check whether spiking for any class (i.e. in a PN cluster) has passed the detection threshold
  -------------------------------------------------------------------------- */
void Enose_classifier::checkSpikeRateThresholdPN(float sampleTimeMs, float periodMs, UINT timestep, UINT lastTimestep, ofstream & spikeFreqFile) {


	UINT winner = getIndexOfHighestEntry(clusterSpikeCountPN,global_NumVR);

	float mostSpikesInCluster = clusterSpikeCountPN[winner];
	float avgSpikesPerNeuron =  mostSpikesInCluster/CLUST_SIZE_PN;
	float maxSpikeFreqHz = 1000.0f * avgSpikesPerNeuron / periodMs ;

#ifdef FLAG_GENERATE_SPIKE_FREQ_DATA_FILE
	spikeFreqFile << sampleTimeMs << COMMA << winner << COMMA << periodMs << COMMA << maxSpikeFreqHz << endl;
#endif

	switch (currClassEvaluationStage) {
	case stage_pre_evaluation:
		if (maxSpikeFreqHz  >= param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN) {
			currClassEvaluationStage = stage_evaluation_active;
			classEvaluationStartTimestep = timestep - 1; //there will have been some activity on the previous step
			classEvaluationEndTimestep = timestep + param_MIN_CLASS_EVALUATION_TIMESTEPS; //set end of window to be a minimum time away
			//cout << "Class evaluation triggered at "  << sampleTimeMs << "ms" << endl;
		}
		break;

	case stage_evaluation_active :
		if (timestep >= classEvaluationEndTimestep && maxSpikeFreqHz  < param_CLASS_EVALUATION_SPIKING_THRESHOLD_HZ_PN) {
			currClassEvaluationStage = stage_post_evaluation;
			classEvaluationEndTimestep = timestep + 1; //window may be extended beyond min if spike rate still high enough
			//cout << "Class evaluation stopped at "  << sampleTimeMs << "ms" << endl;
		}

		//reached the end of recording and still in active evaluation. End the window here.
		if (timestep== lastTimestep) {
			classEvaluationEndTimestep = timestep;
			currClassEvaluationStage = stage_post_evaluation;
		}

		break;

	default:
		break;
	}

	spikeFreqFile.flush();

}

/* --------------------------------------------------------------------------
total up the spikes on every PN neuron during the full run time. Needed for the plasticity rule.
-------------------------------------------------------------------------- */
void Enose_classifier::updateIndividualSpikeCountPN()
{
	for (int i= 0; i < glbscntPN; i++) {
		UINT neuronIdx = glbSpkPN[i]; //this array lists by index the PN neurons with "spike events" during last timestep
		individualSpikeCountPN[neuronIdx] = individualSpikeCountPN[neuronIdx]  + 1; //increment count for this neuron
	}
}

/* --------------------------------------------------------------------------
CLear down totals in spike count array - for reset between runs
-------------------------------------------------------------------------- */
void Enose_classifier::resetIndividualSpikeCountPN()
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
void Enose_classifier::updateClusterSpikeCountAN(UINT timestep)
{
	for (int i= 0; i < glbscntAN; i++) {
		UINT neuronIdx = glbSpkAN[i]; //this array lists by index the AN neurons that "spiked" during last timestep
		UINT classId = getClassCluster(neuronIdx);
		clusterSpikeCountAN[classId] = clusterSpikeCountAN[classId] + 1; //increment count for this cluster/class
		overallWinnerSpikeCountAN[classId] = overallWinnerSpikeCountAN[classId] + 1;

		clusterSpikeCountPerTimestepAN[timestep*NUM_CLASSES + classId] += 1;
	}
}

/* --------------------------------------------------------------------------
totals up the spikes within each PN cluster during the input presentation. Used to investigate whether WTA in the PN layer is actually indicating the nearest VR(s)
-------------------------------------------------------------------------- */
void Enose_classifier::updateClusterSpikeCountPN()
{
	for (int i= 0; i < glbscntPN; i++) {
		UINT neuronIdx = glbSpkPN[i]; //this array lists by index the PN neurons that "spiked" during last timestep
		UINT vrId = getClusterIndex(neuronIdx,CLUST_SIZE_PN);
		clusterSpikeCountPN[vrId] = clusterSpikeCountPN[vrId] + 1; //increment count for this cluster/VR
	}
}

/* --------------------------------------------------------------------------
totals up the spikes within each RN cluster during the input presentation. Used to investigate whether WTA in the PN layer is actually indicating the nearest VR(s)
-------------------------------------------------------------------------- */
void Enose_classifier::updateClusterSpikeCountRN()
{
	for (int i= 0; i < glbscntRN; i++) {
		UINT neuronIdx = glbSpkRN[i]; //this array lists by index the RN neurons that "spiked" during last timestep
		UINT vrId = getClusterIndex(neuronIdx,CLUST_SIZE_RN);
		clusterSpikeCountRN[vrId] = clusterSpikeCountRN[vrId] + 1; //increment count for this cluster/VR
	}
}

/* --------------------------------------------------------------------------
Clear down totals in spike count array - for reset between plasticity windows
-------------------------------------------------------------------------- */
void Enose_classifier::resetClusterSpikeCountAN()
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
Clear down totals in spike count array - for reset between plasticity windows
-------------------------------------------------------------------------- */
void Enose_classifier::resetClusterSpikeCountPerTimestepAN()
{
	int numTimesteps  = RECORDING_TIME_MS / DT;
	if (clusterSpikeCountPerTimestepAN==NULL) {
		//create array if not done yet
		clusterSpikeCountPerTimestepAN = new UINT [NUM_CLASSES * numTimesteps];
	}

	for (int i= 0; i < NUM_CLASSES * numTimesteps; i++) {
		clusterSpikeCountPerTimestepAN[i] = 0;
	}
}


/* --------------------------------------------------------------------------
Clear down totals in spike count array - for reset between recording presentations
-------------------------------------------------------------------------- */
void Enose_classifier::resetClusterSpikeCountPN()
{
	if (clusterSpikeCountPN==NULL) {
		//create array if not done yet
		clusterSpikeCountPN = new UINT [global_NumVR];
	}

	for (int i= 0; i < global_NumVR; i++) {
		clusterSpikeCountPN[i] = 0;
	}
}
/* --------------------------------------------------------------------------
Clear down totals in spike count array - for reset between recording presentations
-------------------------------------------------------------------------- */
void Enose_classifier::resetClusterSpikeCountRN()
{
	if (clusterSpikeCountRN==NULL) {
		//create array if not done yet
		clusterSpikeCountRN = new UINT [global_NumVR];
	}

	for (int i= 0; i < global_NumVR; i++) {
		clusterSpikeCountRN[i] = 0;
	}
}


/* --------------------------------------------------------------------------
Clear down overall winner 
-------------------------------------------------------------------------- */
void Enose_classifier::resetOverallWinner()
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

