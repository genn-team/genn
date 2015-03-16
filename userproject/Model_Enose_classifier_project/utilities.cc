#ifndef utilities_cc
#define utilities_cc
using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm> //for std:find

//for stat command (file info interrgation
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

typedef unsigned int UINT;
enum data_type {data_type_int, data_type_uint, data_type_float,data_type_double};
const string COMMA = ",";
const string SPACE = " ";
const string TAB = "\t";
const string NL = "\n";
const string SLASH = "/";
const string DOT = ".";


/*--------------------------------------------------------------------------
invoke a shell command in Linux
-------------------------------------------------------------------------- */
bool invokeLinuxShellCommand(const char* cmdLine, string &result) {
	char buf[512];
	FILE *output = popen(cmdLine, "r");
	if (!output) {
		cerr << "Failed to invoke command " << cmdLine;
		return false;
	}
	while(fgets(buf, sizeof(buf), output)!=NULL) {
		result.append(buf);
	}

	return true;
}

//replace every occurence of find in src with replacement
string replace(string src,const string &find, const string &replacement)
{
	int pos  = src.find(find);
	while ((pos!=string::npos)) {
		src = src.replace(pos,find.length(),replacement);
		pos  = src.find(find);
	}
	return src;
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
Utility to check if a file already exists
-----------------------------------------------------------------*/
bool fileExists(string path)
{
	FILE * f = fopen(path.c_str(),"r");
	if (f==NULL) {
		return false;
	} else {
		fclose(f);
		return true;
	}
}

//export an array of floats as a delimted string
string toString(float * data, int count, string delim) {
	string result;
	for (int i = 0; i < count; i++) {
		result.append(toString(data[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export an array of ints as a delimted string
string toString(int * data, int count, string delim) {
	string result;
	for (int i = 0; i < count; i++) {
		result.append(toString(data[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export an array of uints as a delimted string
string toString(UINT * data, int count, string delim) {
	string result;
	for (int i = 0; i < count; i++) {
		result.append(toString(data[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export a vector of floats as a delimted string
string toString(vector<float> &vec, string delim) {
	string result;
	UINT count = vec.size();
	for (int i = 0; i < count; i++) {
		result.append(toString(vec[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export a vector of ints as a delimted string
string toString(vector<int> &vec, string delim) {
	string result;
	UINT count = vec.size();
	for (int i = 0; i < count; i++) {
		result.append(toString(vec[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export a vector of unsigned ints as a delimted string
string toString(vector<UINT> &vec, string delim) {
	string result;
	UINT count = vec.size();
	for (int i = 0; i < count; i++) {
		result.append(toString(vec[i]));
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//export a vector of strings as a single delimted string
string toString(vector<string> &vec, string delim) {
	string result;
	UINT count = vec.size();
	for (int i = 0; i < count; i++) {
		result.append(vec[i]);
		if (i<(count-1)) result.append(delim);
	}
	return result;
}

//add elements provided as a delimeted string to the vector
bool addAllToVector(vector<float> &vec, const char * line ) {
	stringstream elements(line);
	float element;
	while(!elements.eof()) {
		elements >> element;
		vec.push_back(element);
	}
	return true;
}

//add elements provided as a delimeted string to the vector
bool addAllToVector(vector<UINT> &vec, const char * line ) {
	stringstream elements(line);
	UINT element;
	while(!elements.eof()) {
		elements >> element;
		vec.push_back(element);
	}
	return true;
}

string trimWhitespace(string str) {
	std::size_t first = str.find_first_not_of(" \n\t");
	if (first==string::npos) first = 0;
	std::size_t last  = str.find_last_not_of(" \n\t");
	if (last==string::npos) last = str.size();
	return str.substr(first, last-first+1);
}

bool directoryExists(string const &path) {
	struct stat st;
	if (stat(path.c_str(), &st) == -1) {
		return false;
	}
	return  S_ISDIR(st.st_mode);
}



bool createDirectory(string path) {
	if (directoryExists(path)) {
		cerr << "WARNING: instructed to create directory that already exists. Ignoring.." << endl;
		return false;
	} else {
		string cmd  = "mkdir '" + path + "'";
		return system(cmd.c_str());
	}
}


bool promptYN(string prompt) {
	string input;
	cout << prompt  << "[y/n]" << endl;
	cin >> input;
	return (input=="y" || input=="Y");
}

bool wildcardFileDelete(string dir, string fileWildcard, bool doubleCheck) {
	string cmd;
	string result;
	if (doubleCheck) {
		if (!promptYN("Files will be deleted from " + dir +  ". Proceed?")) return false;
	}

	cmd = "rm -f " + dir + "/" + fileWildcard;
	return invokeLinuxShellCommand(cmd.c_str(),result);
}

bool writeArrayToTextFile(FILE * out, void * array, UINT rows, UINT cols, data_type dataType, UINT decimalPoints, bool includeRowNum, string delim, bool closeAfterWrite)
{

	if (out==NULL) {
		cerr << "writeArrayToTextFile: specified file not open or valid";
		return false;
	}

	float * floatArray = (float*)array;
	UINT * 	uintArray = (UINT*)array;
	int * intArray = (int*)array;
	double * doubleArray = (double*)array;

	UINT rowNum = 0;
	if (includeRowNum) fprintf(out, "%d%s",rowNum++, delim.c_str());

	for (UINT i = 0; i < rows*cols; ++i) {

		switch (dataType) {
		case(data_type_float):
											fprintf(out, "%*.*f", 1,decimalPoints, floatArray[i]);
		break;
		case(data_type_uint):
											fprintf(out, "%d", uintArray[i]);
		break;
		case(data_type_double):
											fprintf(out, "%*.*f", 1,decimalPoints, doubleArray[i]);
		break;
		case(data_type_int):
											fprintf(out, "%d", intArray[i]);
		break;
		default:
			fprintf(stderr, "ERROR: Unknown data type:%d\n", dataType);
			return false;

		}
		bool lastOneInColumn = i % cols == cols -1;
		bool lastDataItem = i== rows*cols - 1;

		if (!lastDataItem) {
			if (lastOneInColumn ) {
				fprintf(out, "\n");
				if (includeRowNum) fprintf(out, "%d%s",rowNum++, delim.c_str());
			} else {
				fprintf(out, "%s", delim.c_str());
			}
		}

	}
	if (closeAfterWrite) fclose(out);
	return true;
}

/* --------------------------------------------------------------------------
utility to load an array of the designated data_type from a delimited text file
-------------------------------------------------------------------------- */
bool loadArrayFromTextFile(string path, void * array, string delim, UINT arrayLen, data_type dataType)
{
	using namespace std;

	float * floatArray = (float*)array;
	UINT * 	uintArray = (UINT*)array;
	int * intArray = (int*)array;
	double * doubleArray = (double*)array;

	ifstream file(path.c_str());
	if(!file.is_open()) {
		cerr << "loadArrayFromTextFile: Unable to open file to read (" << path << ")" << endl;
		return false;
	}

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

			if (index<arrayLen) {
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
			}
		} while(pos2!=string::npos);
	}
	file.close();
	if (index>arrayLen) {
		cerr << "loadArrayFromTextFile: WARNING: This file (" << path <<  ") contained "<< index << " values, but the array provided allowed only " << arrayLen << " elements."<< endl;
		cerr << "Import was truncated at array length." << endl;
	}

	if (index<arrayLen){
		cerr << "loadArrayFromTextFile: WARNING: This file (" + path + ") is smaller than the array provided (" << arrayLen << " elements)."<< endl;
		cerr << "Import stopped at end of file, remainder of array will be empty." << endl;
	}

	return true;
}


/* --------------------------------------------------------------------------
debug utility to output a line of dashes
-------------------------------------------------------------------------- */
void printSeparator()
{
	printf("--------------------------------------------------------------\n");
}


/* --------------------------------------------------------------------------
debug utility to check contents of an array
-------------------------------------------------------------------------- */
void checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints, string delim )
{
	printSeparator();
	printf("\n%s\n", title.c_str());
	writeArrayToTextFile(stdout,array,howMany/displayPerLine,displayPerLine,dataType,decimalPoints,false,delim,false);
	printSeparator();
}

void checkContents(string title, void * array, UINT howMany, UINT displayPerLine, data_type dataType, UINT decimalPoints ) {
	checkContents( title, array,howMany, displayPerLine,dataType,decimalPoints,"\t" );//default delim tab
}


/*--------------------------------------------------------------------------
 Utility function to join (append together) a set of text files named by file path in vector, to create the specified result file
 -------------------------------------------------------------------------- */
bool joinTextFiles(vector<string> &filePaths, string outputPath) {
	bool result = true;
	ofstream out(outputPath.c_str());
	if (!out.is_open()) {
		cerr << "joinTextFiles: WARNING, failed to open specified output file : " << outputPath<< endl;
		return false;
	}

	for (int i = 0; i < filePaths.size(); ++i) {
		ifstream in(filePaths[i].c_str());
		if (in.is_open()) {
			while(!in.eof()) {
				string line;
				getline(in,line);
				out << line << endl;
			}
			in.close();
		} else {
			cerr << "joinTextFiles: WARNING, failed to open specified file : " << filePaths[i] << endl;
			result = false;
		}
	}
	out.close();
	cout << "Created merged file : " << outputPath << endl;
	return result;
}

/*--------------------------------------------------------------------------
 Utility function to determine if a passed vector of ints contains the specified value
 -------------------------------------------------------------------------- */
bool vectorContains(vector<int> &vec ,int lookingFor)
{
	vector<int>::iterator it = find(vec.begin(), vec.end(), lookingFor);
	return (it != vec.end());
}

void zeroArray(float * array,UINT size) {
	for (int i = 0; i < size; ++i) {
		array[i] = 0.0f;
	}
}
void zeroArray(UINT * array,UINT size) {
	for (int i = 0; i < size; ++i) {
		array[i] = 0;
	}
}
void zeroArray(int * array,UINT size) {
	for (int i = 0; i < size; ++i) {
		array[i] = 0;
	}
}

bool createDirIfNotExists(string path) {
	if(!directoryExists(path)) {
		return createDirectory(path);
	}
	return true;
}

bool renameFile(string path, string  newPath) {
	string cmd = "mv '" + path + "' '" + newPath + "'";
	cout << cmd << endl;
	int ret  = system(cmd.c_str());
}

bool copyFile(string path, string  newPath) {
	string cmd = "cp '" + path + "' '" + newPath + "'";
	cout << cmd << endl;
	int ret  = system(cmd.c_str());
}

/* --------------------------------------------------------------------------
generate a random number between 0 and 1 inclusive
-------------------------------------------------------------------------- */
float getRand0to1()
{
	return ((float)rand()) / ((float)RAND_MAX) ;
}

/* --------------------------------------------------------------------------
create a pseudorandom event with a specified probability
-------------------------------------------------------------------------- */
bool randomEventOccurred(float probability)
{
	if (probability <0.0 || probability > 1.0)  {
		fprintf(stderr, "randomEventOccurred() fn. ERROR! INVALID PROBABILITY SPECIFIED AS %f.\n", probability);
		exit(1);
	} else {
		return getRand0to1() <= probability;
	}
}

/*--------------------------------------------------------------------------
find the highest entry in the passed int array and return its index
if there are two or more highest then choose randomly between them
-------------------------------------------------------------------------- */
int getIndexOfHighestEntry(unsigned int * data, int size)
{
	UINT winner = 0;
	int max = 0;

	for(UINT i=0; i<size; i++) {
		if (data[i] > max) {
			max = data[i];
			winner = i;
		} else if (data[i] == max) { //draw
			//toss a coin (we don't want to always favour the same index)
			if (randomEventOccurred(0.5)) winner = i;
		}
	}
	return winner;
}

/*--------------------------------------------------------------------------
find the highest entry in the passed float array and return its index
if there are two or more highest then choose randomly between them
-------------------------------------------------------------------------- */
int getIndexOfHighestEntry(float * data, int size)
{
	UINT winner = 0;
	float max = 0;

	for(UINT i=0; i<size; i++) {
		if (data[i] > max) {
			max = data[i];
			winner = i;
		} else if (data[i] == max) { //draw
			//toss a coin (we don't want to always favour the same index)
			if (randomEventOccurred(0.5)) winner = i;
		}
	}
	return winner;
}

#endif
