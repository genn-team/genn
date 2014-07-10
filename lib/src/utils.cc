/*--------------------------------------------------------------------------
   Author/Modifier: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
   
   This file contains neuron model definitions.
  
--------------------------------------------------------------------------*/

#ifndef _UTILS_CC_
#define _UTILS_CC_ //!< macro for avoiding multiple inclusion during compilation


//--------------------------------------------------------------------------
/*! \file utils.cc

\brief This file contains the definitions of standard utility functions provided within the NVIDIA CUDA software development toolkit (SDK).
*/
//--------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include "utils.h"

using namespace std;


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
//! \brief Tool for finding strings in another string
//--------------------------------------------------------------------------

bool find(string &s, const string trg);
{
  size_t found = s.find(trg);
  return (found != string::npos);
}


#endif  // _UTILS_CC_
