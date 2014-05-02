
/*--------------------------------------------------------------------------
   Author/Modifier: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef _UTILS_H_
#define _UTILS_H_ //!< macro for avoiding multiple inclusion during compilation


//--------------------------------------------------------------------------
/*! \file utils.h

\brief This file contains standard utility functions provide within the NVIDIA CUDA software development toolkit (SDK). The remainder of the file contains a function that defines the standard neuron models.
*/
//--------------------------------------------------------------------------

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;


//--------------------------------------------------------------------------
/* \brief Macro for wrapping cuda runtime function calls and catching any errors that may be thrown.
 */
//--------------------------------------------------------------------------

#define CHECK_CUDA_ERRORS(call)					           \
{								      	   \
  cudaError_t error = call;						   \
  if (error != cudaSuccess)						   \
  {                                                                        \
    fprintf(stderr, "%s: %i: cuda error %i: %s\n",			   \
	    __FILE__, __LINE__, (int)error, cudaGetErrorString(error));	   \
    exit(EXIT_FAILURE);						           \
  }									   \
}


//--------------------------------------------------------------------------
/* \brief template function for string conversion from const char* to C++ string
 */
//--------------------------------------------------------------------------

template<typename T>
std::string toString(T t)
{
  std::stringstream s;
  s << t;
  return s.str();
} 

#define tS(X) toString(X) //!< Macro providing the abbreviated syntax tS() instead of toString().


//--------------------------------------------------------------------------
/* \brief Function to write the comment header denoting file authorship and contact details into the generated code.
 */
//--------------------------------------------------------------------------

void writeHeader(ostream &os) 
{
  string s;
  ifstream is("header.src");
  getline(is, s);
  while (is.good()) {
    os << s << endl;
    getline(is, s);
  }
  os << endl;
}


//--------------------------------------------------------------------------
/* \brief Tool for substituting strings in the neuron code strings or other templates
 */
//--------------------------------------------------------------------------

void substitute(string &s, const string trg, const string rep)
{
  size_t found= s.find(trg);
  while (found != string::npos) {
    s.replace(found,trg.length(),rep);
    found= s.find(trg);
  }
}


//--------------------------------------------------------------------------
/* \brief Tool for determining the size of variable types on the current architecture
 */
//--------------------------------------------------------------------------

unsigned int theSize(string type) 
{
  unsigned int size = 0;
  if (type == "int") size = sizeof(int);
  if (type == "unsigned int") size = sizeof(unsigned int);
  if (type == "float") size = sizeof(float);
  if (type == "double") size = sizeof(double);
  if (type == "long double") size = sizeof(long double);
  return size;
}


//--------------------------------------------------------------------------
/* \brief Macro for a "safe" output of a parameter into generated code by essentially just
   adding a bracket around the parameter value in the generated code.
 */
//--------------------------------------------------------------------------
 
#define SAVEP(X) "(" << X << ")" 


#endif  // _UTILS_H_
